import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

from utils import *

import sys
import os
import io

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


class MarkowitzMeanVarOptimization:
    def __init__(self, data, constraints=None, ratings=None, rf_rate=0.02):
        # if there is a column date, make it the index
        self.data = data
        self.rf_rate = compute_return_daily(rf_rate)
        if 'date' in data.columns:
            self.data['date'] = pd.to_datetime(data['date'])
            self.data = data.set_index('date')
        self.data = np.log(1 + self.data)
        self.vol = np.array(self.data.std()) * np.sqrt(252)
        self.ticker = np.array(self.data.columns)
        self.mean = np.append(self.data.mean().values, self.rf_rate)
        # self.mu = np.array((self.mean + 1) ** 252 - 1)
        self.mu = self.mean
        self.corr_matrix = np.array(self.data.corr())
        self.cov_matrix = extend_cov_matrix(self.corr_matrix * (self.vol.reshape(1, -1).T @ self.vol.reshape(1, -1)))
        self.x0 = np.ones(self.mu.shape[0]) / self.mu.shape[0]

        # Update vol vector
        self.vol = extend_vol_vector(self.vol)

        # Constraints
        self.constraints = []

        if 'Fully Invested' in constraints:
            self.constraints.append(LinearConstraint(np.ones(self.x0.shape), ub=1))
            self.constraints.append(LinearConstraint(-np.ones(self.x0.shape), ub=-1))
        # Add no short selling constraint
        if 'No Short Selling' in constraints:
            self.constraints.append(LinearConstraint(np.eye(self.x0.shape[0]), lb=0))

        self.ratings = ratings

    def QP(self, x, cov, mu, gamma):
        v = 0.5 * x.T @ cov @ x - gamma * x.T @ mu

        return v

    def efficient_frontier(self, gam):

        res = minimize(self.QP, self.x0, args=(self.cov_matrix, self.mu, gam),
                       options={'disp': False}, constraints=self.constraints)
        optimized_weights = res.x
        mu_optimized = optimized_weights @ self.mu
        vol_optimized = np.sqrt(optimized_weights @ self.cov_matrix @ optimized_weights)

        return np.array([mu_optimized, vol_optimized]), optimized_weights

    def efficient_frontier_points(self):
        gammas = np.linspace(-0.1, 1500, 50)
        frontier = np.array([self.efficient_frontier(gam)[0] for gam in gammas])
        frontier = frontier.T

        return frontier

    def set_special_constraints(self, min_, max_, percentage):
        # Create the constraints for the special case
        # 80% in the range of min_ and max_
        tickers_in_range = np.zeros(self.x0.shape[0])
        tickers_out_range = np.zeros(self.x0.shape[0])
        for i in range(self.ticker.shape[0]):
            tick_ = self.ticker[i]
            print(tick_)
            print(self.ratings['Ticker'].values)
            if tick_ in self.ratings['Ticker'].values:
                # get the index of the ticker in the ratings DataFrame
                idx = np.where(self.ratings['Ticker'] == tick_)[0][0]
                print(idx)
                print(type(idx))
                print(self.ratings['S&P Rating Number'].values)
                print(self.ratings['S&P Rating Number'].values[idx])
                if min_ <= self.ratings['S&P Rating Number'].values[idx] <= max_:
                    tickers_in_range[i] = 1
                else:
                    tickers_out_range[i] = 1

        # Create the constraints
        self.constraints.append(LinearConstraint(tickers_in_range, lb=percentage))
        self.constraints.append(LinearConstraint(tickers_out_range, ub=1 - percentage))
        # print_red(tickers_out_range)
        # print_red(tickers_in_range)

    # Accessors functions
    def get_vol(self):
        return self.vol

    def get_cov_matrix(self):
        return self.cov_matrix

    def get_mean(self):
        return self.mean

    def get_mu(self):
        return self.mu

    def get_data(self):
        return self.data

    def get_tickers(self):
        return self.ticker


class BlackLittermanOptimization:
    def __init__(self, data, constraints=None, ratings=None, SR_x0=0.25, rf_rate=0.02, P=None, Q=None, omega=None, tau=0.025):
        # if there is a column date, make it the index
        self.data = data
        if 'date' in data.columns:
            self.data['date'] = pd.to_datetime(data['date'])
            self.data = data.set_index('date')
        self.data = np.log(1 + self.data)
        self.vol = np.array(self.data.std()) * np.sqrt(252)
        self.ticker = np.array(self.data.columns)
        self.mean = self.data.mean().values
        # self.mu = np.array((self.mean + 1) ** 252 - 1)
        self.mu = self.mean.astype(np.float64)
        self.corr_matrix = np.array(self.data.corr())
        self.cov_matrix = self.corr_matrix * (self.vol.reshape(1, -1).T @ self.vol.reshape(1, -1)).astype(np.float64)
        self.x0 = np.ones(self.mu.shape[0]) / self.mu.shape[0]
        self.rf_rate = rf_rate
        self.P = P.astype(np.float64)
        self.Q = Q.astype(np.float64)
        self.omega = omega.astype(np.float64)

        # Update vol vector
        self.vol = extend_vol_vector(self.vol)

        # Black Litterman implied volatility and returns
        self.SR_x0 = SR_x0
        self.vol_x0 = np.sqrt(self.x0 @ self.cov_matrix @ self.x0)
        self.implied_phi = self.SR_x0 / self.vol_x0
        self.implied_gam = 1 / self.implied_phi
        self.implied_mu = (self.rf_rate + self.SR_x0 * (self.cov_matrix @ self.x0)
                           / np.sqrt(self.x0 @ self.cov_matrix @ self.x0))

        if omega is None:
            self.mu_bar = None
        else:
            # print_yellow("Gamma matrix: " + str(self.gamma_matrix(tau)))
            # print_yellow("P matrix: " + str(self.P))
            # print_yellow("Q matrix: " + str(self.Q))
            # print_yellow("Omega matrix: " + str(self.omega))
            self.mu_bar = (self.implied_mu +
                           (self.gamma_matrix(tau) @ self.P.T) @
                           np.linalg.inv(self.P @ self.gamma_matrix(tau) @ self.P.T + self.omega) @
                           (self.Q - self.P @ self.implied_mu))

        # Constraints
        self.constraints = []

        if 'Fully Invested' in constraints:
            self.constraints.append(LinearConstraint(np.ones(self.x0.shape), ub=1))
            self.constraints.append(LinearConstraint(-np.ones(self.x0.shape), ub=-1))
        # Add no short selling constraint
        if 'No Short Selling' in constraints:
            self.constraints.append(LinearConstraint(np.eye(self.x0.shape[0]), lb=0))

        self.ratings = ratings

    def QP(self, x, cov, mu, gamma):
        v = 0.5 * x.T @ cov @ x - gamma * x.T @ mu

        return v

    def efficient_frontier(self, gam):

        # Add the risk-free asset
        cov_ = extend_cov_matrix(self.cov_matrix)
        mu_ = np.append(self.mu, self.rf_rate)
        x0 = np.ones(mu_.shape[0]) / mu_.shape[0]

        res = minimize(self.QP, x0, args=(cov_, mu_, gam),
                       options={'disp': False}, constraints=self.constraints)
        optimized_weights = res.x
        mu_optimized = optimized_weights @ mu_
        vol_optimized = np.sqrt(optimized_weights @ cov_ @ optimized_weights)

        return np.array([mu_optimized, vol_optimized]), optimized_weights

    def efficient_frontier_points(self):
        gammas = np.linspace(-0.1, 1, 25)
        frontier = np.array([self.efficient_frontier(gam)[0] for gam in gammas])
        frontier = frontier.T

        return frontier

    def optimize_black_litterman(self):
        if self.omega is None:
            raise ValueError("Omega is not defined. Please define Omega before running Black Litterman optimization.")
        else:
            res = minimize(self.QP, self.x0, args=(self.cov_matrix, self.mu_bar, self.implied_gam),
                           options={'disp': False}, constraints=self.constraints)
            optimized_weights = res.x
            mu_optimized = optimized_weights @ self.mu
            vol_optimized = np.sqrt(optimized_weights @ self.cov_matrix @ optimized_weights)

            return np.array([mu_optimized, vol_optimized]), optimized_weights

    def set_special_constraints(self, min_, max_, percentage):
        # Create the constraints for the special case
        # 80% in the range of min_ and max_
        tickers_in_range = np.zeros(self.x0.shape[0])
        tickers_out_range = np.zeros(self.x0.shape[0])
        for i in range(self.ticker.shape[0]):
            tick_ = self.ticker[i]
            print(tick_)
            print(self.ratings['Ticker'].values)
            if tick_ in self.ratings['Ticker'].values:
                # get the index of the ticker in the ratings DataFrame
                idx = np.where(self.ratings['Ticker'] == tick_)[0][0]
                print(idx)
                print(type(idx))
                print(self.ratings['S&P Rating Number'].values)
                print(self.ratings['S&P Rating Number'].values[idx])
                if min_ <= self.ratings['S&P Rating Number'].values[idx] <= max_:
                    tickers_in_range[i] = 1
                else:
                    tickers_out_range[i] = 1

        # Create the constraints
        self.constraints.append(LinearConstraint(tickers_in_range, lb=percentage))
        self.constraints.append(LinearConstraint(tickers_out_range, ub=1 - percentage))
        # print_red(tickers_out_range)
        # print_red(tickers_in_range)

    def gamma_matrix(self, tau):
        # Remove the risk-free asset
        return tau * self.cov_matrix

    # Accessors functions
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

    # Set functions
    def set_P(self, P):
        self.P = P

    def set_Q(self, Q):
        self.Q = Q

    def set_omega(self, omega):
        self.omega = omega

    def set_tau(self, tau):
        self.tau = tau