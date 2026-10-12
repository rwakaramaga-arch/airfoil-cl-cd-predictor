"""Model definitions, target handling and metrics.

Every model is wrapped in AeroRegressor, which
  * fits [Cl, log(Cd)] jointly (Cd in log space, inverted at prediction),
  * standardises both targets so neither dominates multi-output splits/losses,
  * puts a StandardScaler in front of the estimator, so the scaler is always
    fitted on the training rows of whatever fold is being fitted.
"""
import copy
import re

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


# ----------------------------------------------------------- target wrapper

def to_z(Y):
    """Physical [Cl, Cd] -> modelling space [Cl, log Cd]."""
    Y = np.asarray(Y, dtype=float)
    return np.column_stack([Y[:, 0], np.log(Y[:, 1])])


def from_z(Z):
    return np.column_stack([Z[:, 0], np.exp(Z[:, 1])])


class AeroRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, base=None):
        self.base = base

    def fit(self, X, Y):
        Z = to_z(Y)
        self.mu_, self.sd_ = Z.mean(axis=0), Z.std(axis=0)
        self.base_ = clone(self.base).fit(np.asarray(X, dtype=float), (Z - self.mu_) / self.sd_)
        return self

    def predict_z(self, X):
        return self.base_.predict(np.asarray(X, dtype=float)) * self.sd_ + self.mu_

    def predict(self, X):
        return from_z(self.predict_z(X))

    @property
    def scaler_(self):
        return self.base_.named_steps["scale"]

    @property
    def estimator_(self):
        return self.base_.named_steps["est"]


# ------------------------------------------------------------------ the MLP

class GroupedEarlyStoppingMLP(RegressorMixin, BaseEstimator):
    """One MLPRegressor with 2 outputs and ReLU hidden layers, trained epoch
    by epoch with early stopping on a validation set of whole airfoils.

    sklearn's built-in early_stopping holds out random ROWS, which puts
    alpha=4.00 in training and alpha=4.25 of the same polar in validation.
    Here the airfoil is recovered from the CST columns (the first
    `n_group_cols` features; one geometry = one airfoil) and the validation
    split is made by airfoil."""

    def __init__(self, hidden_layer_sizes=(128, 128), alpha=1e-4, learning_rate_init=1e-3,
                 batch_size=256, max_epochs=300, patience=15, validation_fraction=0.15,
                 n_group_cols=10, random_state=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.n_group_cols = n_group_cols
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        _, groups = np.unique(np.round(X[:, :self.n_group_cols], 9), axis=0, return_inverse=True)
        gss = GroupShuffleSplit(n_splits=1, test_size=self.validation_fraction,
                                random_state=self.random_state)
        tr, va = next(gss.split(X, groups=groups))
        net = MLPRegressor(hidden_layer_sizes=tuple(self.hidden_layer_sizes), activation="relu",
                           solver="adam", alpha=self.alpha, batch_size=self.batch_size,
                           learning_rate_init=self.learning_rate_init, shuffle=True,
                           random_state=self.random_state)
        best, wait, best_state = np.inf, 0, None
        self.loss_curve_, self.val_curve_ = [], []
        for epoch in range(self.max_epochs):
            net.partial_fit(X[tr], y[tr])
            val = float(np.mean((net.predict(X[va]) - y[va]) ** 2))
            self.loss_curve_.append(float(net.loss_))
            self.val_curve_.append(val)
            if val < best - 1e-7:
                best, wait, self.best_epoch_ = val, 0, epoch
                best_state = (copy.deepcopy(net.coefs_), copy.deepcopy(net.intercepts_))
            else:
                wait += 1
                if wait >= self.patience:
                    break
        net.coefs_, net.intercepts_ = best_state
        self.net_, self.n_epochs_ = net, epoch + 1
        self.n_val_airfoils_ = int(len(np.unique(groups[va])))
        return self

    def predict(self, X):
        return self.net_.predict(np.asarray(X, dtype=float))


# ------------------------------------------------------------------ factory

def _estimator(kind, cfg, seed, n_group_cols):
    n_jobs = cfg["compute"]["n_jobs"]
    if kind == "rf":
        return RandomForestRegressor(n_jobs=n_jobs, random_state=seed)
    if kind == "et":
        return ExtraTreesRegressor(n_jobs=n_jobs, random_state=seed)
    if kind == "xgb":
        return XGBRegressor(tree_method="hist", multi_strategy="one_output_per_tree",
                            n_jobs=n_jobs, random_state=seed)
    if kind == "mlp":
        m = cfg["mlp"]
        return GroupedEarlyStoppingMLP(max_epochs=m["max_epochs"], patience=m["patience"],
                                       validation_fraction=m["validation_fraction"],
                                       n_group_cols=n_group_cols, random_state=seed)
    if kind == "ridge":
        return Ridge()
    if kind == "mean":
        return DummyRegressor(strategy="mean")
    raise ValueError(f"unknown model kind {kind!r}")


def make_model(kind, cfg, feats, params=None, seed=0):
    cst = [f for f in feats if re.fullmatch(r"A[ul]\d+", f)]
    n_group = len(cst)
    if list(feats[:n_group]) != cst:
        raise ValueError("CST coefficients must be the first features (MLP grouping relies on it)")
    model = AeroRegressor(Pipeline([("scale", StandardScaler()),
                                    ("est", _estimator(kind, cfg, seed, n_group))]))
    if params:
        model.set_params(**{f"base__est__{k}": _param(k, v) for k, v in params.items()})
    return model


def _param(name, value):
    return tuple(value) if name == "hidden_layer_sizes" else value


def search_space(kind, cfg):
    grid = cfg["tuning"]["grids"][kind]
    return {f"base__est__{k}": [_param(k, v) for v in vals] for k, vals in grid.items()}


def search_score(model, X, Y):
    """Tuning objective: mean of R2(Cl) and R2(log Cd)."""
    Z, T = model.predict_z(X), to_z(Y)
    return float(np.mean([r2(T[:, i], Z[:, i]) for i in range(2)]))


# ------------------------------------------------------------------ metrics

def r2(y, p):
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - p) ** 2) / ss_tot) if ss_tot > 0 else np.nan


def target_metrics(target, y, p, cfg):
    """RMSE, MSE, MAE, R2 for one target in physical units, plus R2 in log
    space (Cd only) and MAPE (Cl restricted to |Cl| > threshold)."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    err = p - y
    out = {"target": target, "n": int(len(y)), "MSE": float(np.mean(err ** 2)),
           "MAE": float(np.mean(np.abs(err))), "R2": r2(y, p)}
    out["RMSE"] = float(np.sqrt(out["MSE"]))
    if target == "Cd":
        out["R2_log"] = r2(np.log(y), np.log(np.clip(p, 1e-12, None)))
        mask = np.ones(len(y), dtype=bool)
    else:
        out["R2_log"] = np.nan          # log of a signed quantity is undefined
        mask = np.abs(y) > cfg["evaluation"]["cl_mape_min_abs"]
    out["MAPE"] = float(100 * np.mean(np.abs(err[mask] / y[mask]))) if mask.any() else np.nan
    out["MAPE_n"] = int(mask.sum())
    return out


def thin_airfoil_cl(alpha_deg):
    """Thin-airfoil theory, Cl = 2*pi*alpha (alpha in radians)."""
    return 2 * np.pi * np.deg2rad(np.asarray(alpha_deg, dtype=float))
