import os
import pypsa
import shutil
import networkx as nx
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from pypsa.descriptors import get_switchable_as_dense as as_dense

class RedispatchModel:

    def __init__(self, model, folder_path, solver="gurobi", solver_settings="default"):
        """init redispatch model"""
        self.set_model(folder_path, model)
        self.set_folder_paths(folder_path)
        self.set_solver(solver, solver_settings)
        self.run_network_model()
        self.run_market_model()
        self.run_redispatch_model()

    def set_folder_paths(self, folder_path):
        """set folder path with subfolder structure"""
        self.folder_path = folder_path
        self.results_folder = os.path.join(os.path.join(folder_path, "results_redispatch"))
        self.init_directory(self.results_folder)

    def set_model(self, folder_path, model):
        """set model name"""
        self.model_name = model
        self.model = pypsa.Network(os.path.join(folder_path, "../networks", model))

    def set_solver(self, solver, settings):
        """set solver and solver settings"""
        self.solver = solver
        self.solver_settings = {}
        if solver == "gurobi":
            if settings == "default":
                self.solver_settings = {"Crossover": 0, "Threads": 4, "Method": 2, "BarConvTol": 1.e-6, "Seed": 123, "AggFill": 0, "PreDual": 0, "GURO_PAR_BARDENSETHRESH": 200}

    def init_directory(self, path, empty_directory=False):
        """check if directory exists, if not create it"""
        if os.path.exists(path):
            if empty_directory:
                shutil.rmtree(os.path.join(path))
                os.mkdir(path)
        else:
            os.mkdir(path)

    def export_model_results(self, model, model_name):
        """save model results"""
        # results folder for current redispatch model
        path = os.path.join(self.results_folder, self.model_name[:-3])
        self.init_directory(path)
        path = os.path.join(path, model_name)
        self.init_directory(path, empty_directory=True)
        model.export_to_csv_folder(path)
        # model.export_to_netcdf(path)

    def run_network_model(self):
        """run network model"""
        self.network_model = self.model.copy()
        self.network_model.optimize(solver_name=self.solver, solver_options=self.solver_settings)
        self.export_model_results(self.network_model, "network_model")

    def run_market_model(self):
        """build market model"""
        assert hasattr(self, "network_model"), "run network model first"
        self.market_model = self.model.copy()

        # add bidding zones, where each country is a bidding zone
        zones = self.network_model.buses.country # two zones from minimal example (self.network_model.buses.y > 51).map(lambda x: "North" if x else "South")
        for c in self.market_model.iterate_components(self.market_model.one_port_components):
            c.df.bus = c.df.bus.map(zones)

        # bidding zones are coupled through transmission lines
        for c in self.market_model.iterate_components(self.market_model.branch_components):
            c.df.bus0 = c.df.bus0.map(zones)
            c.df.bus1 = c.df.bus1.map(zones)
            internal = c.df.bus0 == c.df.bus1
            self.market_model.mremove(c.name, c.df.loc[internal].index)

        # remove previous buses and add bidding zones as new buses
        self.market_model.mremove("Bus", self.market_model.buses.index)
        self.market_model.madd("Bus", zones.unique())

        # solve market model
        self.market_model.optimize(solver_name=self.solver, solver_options=self.solver_settings)
        self.export_model_results(self.market_model, "market_model")

        # get market clearing prices for each zone
        # price = self.market_model.buses_t.marginal_price

    def run_redispatch_model(self):
        """build redispatch model"""
        assert hasattr(self, "network_model"), "run network model first"
        assert hasattr(self, "market_model"), "run market model first"
        self.redispatch_model = self.model.copy()

        # fix generator dispatch to market model results
        p = (self.market_model.generators_t.p / self.market_model.generators.p_nom).fillna(0)
        self.redispatch_model.generators_t.p_min_pu = p
        self.redispatch_model.generators_t.p_max_pu = p

        # add generators bidding assuming
        # - generators can reduce dispatch to zero (includes renewable curtailment)
        # - generators can increase dispatch to available/nominal capacity
        # - no changes in marginal costs (i.e. reducing dispatch lowers costs)
        # hence, two stage market should result in same cost as nodal market
        g_up = self.redispatch_model.generators.copy()
        g_down = self.redispatch_model.generators.copy()
        g_up.index = g_up.index.map(lambda x: x + " ramp up")
        g_down.index = g_down.index.map(lambda x: x + " ramp down")

        up = (as_dense(self.market_model, "Generator", "p_max_pu") * self.market_model.generators.p_nom - self.market_model.generators_t.p).clip(0) / self.market_model.generators.p_nom
        down = -self.market_model.generators_t.p / self.market_model.generators.p_nom

        up.columns = up.columns.map(lambda x: x + " ramp up")
        down.columns = down.columns.map(lambda x: x + " ramp down")

        self.redispatch_model.madd("Generator", g_up.index, p_max_pu=up, **g_up.drop("p_max_pu", axis=1))
        self.redispatch_model.madd("Generator", g_down.index, p_min_pu=down, p_max_pu=0, **g_down.drop(["p_max_pu", "p_min_pu"], axis=1))

        # solve redispatch model
        self.redispatch_model.optimize(solver_name=self.solver, solver_options=self.solver_settings)
        self.export_model_results(self.redispatch_model, "redispatch_model")

        ## change bidding strategies
        # ramping up and down is twice as expensive
        # self.redispatch_model.generators.loc[self.redispatch_model.generators.index.str.contains("ramp up"), "marginal_cost"] *= 2
        # generators are compensated for curtailing them or ramping them down by 0.5 of margnial cost
        # self.redispatch_model.generators.loc[self.redispatch_model.index.str.contains("ramp down"), "marginal_cost"] *= -0.5
        # hence, outcome should be more expensive than in ideal nodal market
        # self.redispatch_model.optimize(solver_name=self.solver, solver_options=self.solver_default)

if __name__ == "__main__":
    # model = "elec_s_6_ec_lcopt_Co2L-24H.nc"
    # model = "elec_s_37_ec_lv1.5_CO2L-4380SEG.nc"
    model = "elec_s_37_ec_lv1.5_CO2L-24H.nc"

    folder_path = os.path.join("../..", "results", "elec")
    RedispatchModel(model, folder_path)