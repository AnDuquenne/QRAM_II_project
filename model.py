import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint


class MarkowitzMeanVarOptimization:
    def __init__(self, data, constraints=None):
        # if there is a column date, make it the index
        self.data = data
        if 'date' in data.columns:
            self.data['date'] = pd.to_datetime(data['date'])
            self.data = data.set_index('date')
        self.data = np.log(1 + self.data)
        self.vol = np.array(self.data.std()) * np.sqrt(252)
        self.ticker = np.array(self.data.columns)
        self.mu = np.array((self.data.mean() + 1) ** 252 - 1)
        self.corr_matrix = np.array(self.data.corr())
        self.cov_matrix = self.corr_matrix * (self.vol.reshape(1, -1).T @ self.vol.reshape(1, -1))
        self.x0 = np.ones(self.mu.shape) / self.mu.shape[0]

        self.constraints = []

        if 'Fully Invested' in constraints:
            self.constraints.append(LinearConstraint(np.ones(self.x0.shape), ub=1))
            self.constraints.append(LinearConstraint(-np.ones(self.x0.shape), ub=-1))
        # Add no short selling constraint
        if 'No Short Selling' in constraints:
            self.constraints.append(LinearConstraint(np.eye(self.x0.shape[0]), lb=0))

        print(self.constraints)

    def QP(self, x, gamma):
        v = 0.5 * x.T @ self.cov_matrix @ x - gamma * x.T @ self.mu

        return v

    def efficient_frontier(self, gam):
        res = minimize(self.QP, self.x0, args=(gam),
                       options={'disp': False}, constraints=self.constraints)
        optimized_weights = res.x
        mu_optimized = optimized_weights @ self.mu
        vol_optimized = np.sqrt(optimized_weights @ self.cov_matrix @ optimized_weights)

        return np.array([mu_optimized, vol_optimized]), optimized_weights

    def efficient_frontier_points(self):
        gammas = np.linspace(-0.1, 1, 25)
        frontier = np.array([self.efficient_frontier(gam)[0] for gam in gammas])
        frontier = frontier.T

        return frontier

    # Accessors
    def get_vol(self):
        return self.vol

    def get_cov_matrix(self):
        return self.cov_matrix

    def get_mu(self):
        return self.mu

    def get_data(self):
        return self.data

    def get_tickers(self):
        return self.ticker
