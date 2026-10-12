"""Streamlit-free application logic for AeroML, so it can be unit-tested.

Chain: inputs -> validation -> training-envelope check -> scaler ->
model -> Cl, Cd -> derived quantities."""
import json
import sys
import time
from datetime import datetime
from math import comb
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent / "model"

# The pickled models are instances of classes in models.py (AeroRegressor,
# GroupedEarlyStoppingMLP). Phase 4 copies it to app/models.py so app/ can be deployed
# on its own; in the full project the root copy is used if the app copy is absent.
if not (Path(__file__).resolve().parent / "models.py").exists():
    _PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
LATENCY_LOG = Path(__file__).resolve().parents[1] / "outputs" / "app_latency.csv"

CST_UPPER = [f"Au{i}" for i in range(5)]
CST_LOWER = [f"Al{i}" for i in range(5)]
GEOMETRY = ["max_thickness", "x_max_thickness", "max_camber", "x_max_camcer"]
N1, N2 = 0.5, 1.0          # Kulfan class function exponents (round nose, sharp trailing edge)


class AppModelError(RuntimeError):
    """Model directory missing or incomplete."""


# ------------------------------------------------------------------ loading

def load_bundle(model_dir=MODEL_DIR):
    model_dir = Path(model_dir)
    card_path = model_dir / "model_card.json"
    if not model_dir.is_dir() or not card_path.exists():
        raise AppModelError(
            f"No trained model found in '{model_dir}'. Build it with:  py pipeline.py --phase 4")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    # Deployments may ship only some models (size limits): keep the ones on disk.
    all_models = card["models"]
    card["models"] = [m for m in all_models if (model_dir / m["file"]).exists()]
    if not card["models"]:
        raise AppModelError(f"The model files are missing from '{model_dir}': none of "
                            f"{[m['file'] for m in all_models]} were found. Build them with:  "
                            "py pipeline.py --phase 4")
    feats = json.loads((model_dir / "feature_columns.json").read_text(encoding="utf-8"))["features"]
    return {"card": card, "features": feats, "dir": model_dir,
            "n_models_trained": len(all_models),
            "missing_models": [m for m in all_models if m not in card["models"]],
            "airfoils": pd.read_csv(model_dir / card["airfoil_table"]),
            "polars": pd.read_csv(model_dir / card["polar_table"])}


def load_model(bundle, kind):
    entry = next(m for m in bundle["card"]["models"] if m["kind"] == kind)
    model = joblib.load(bundle["dir"] / entry["file"])
    if kind in ("rf", "et"):
        model.estimator_.n_jobs = 1     # threaded forests are not bit-reproducible
    return model


# --------------------------------------------------------------- geometry

def cst_surface(coeffs, x):
    """Kulfan CST: y = x^N1 (1-x)^N2 * sum_i A_i * C(n,i) x^i (1-x)^(n-i)."""
    n = len(coeffs) - 1
    shape = sum(a * comb(n, i) * x ** i * (1 - x) ** (n - i) for i, a in enumerate(coeffs))
    return x ** N1 * (1 - x) ** N2 * shape


def airfoil_shape(geom, n=401):
    x = (1 - np.cos(np.linspace(0, np.pi, n))) / 2            # cosine spacing
    yu = cst_surface([geom[c] for c in CST_UPPER], x)
    yl = cst_surface([geom[c] for c in CST_LOWER], x)
    t = yu - yl
    return {"x": x, "y_upper": yu, "y_lower": yl, "camber_line": (yu + yl) / 2,
            "max_thickness": float(t.max()), "x_max_thickness": float(x[t.argmax()])}


# ------------------------------------------------------- input validation

def validate_inputs(geom, re_, alphas):
    """Hard errors: inputs the model cannot be evaluated on at all."""
    errors = []
    vals = [geom[c] for c in CST_UPPER + CST_LOWER + GEOMETRY] + [re_] + list(np.atleast_1d(alphas))
    if not np.all(np.isfinite(np.asarray(vals, dtype=float))):
        errors.append("All inputs must be finite numbers.")
    if re_ <= 0:
        errors.append("Reynolds number must be positive.")
    for c in ("x_max_thickness", "x_max_camcer"):
        if not 0 <= geom[c] <= 1:
            errors.append(f"{c} is a chordwise position and must lie between 0 and 1.")
    if geom["max_thickness"] <= 0:
        errors.append("max_thickness must be positive.")
    shape = airfoil_shape(geom)
    if np.any(shape["y_upper"][1:-1] < shape["y_lower"][1:-1]):
        errors.append("Upper and lower surfaces cross: the CST coefficients do not describe a valid airfoil.")
    return errors


def envelope_warnings(geom, re_, alphas, envelope):
    """Soft warnings: inputs outside the ranges seen in training."""
    out = []
    checks = [("max_thickness", geom["max_thickness"], "thickness (t/c)"),
              ("max_camber", geom["max_camber"], "maximum camber"),
              ("Re", re_, "Reynolds number")]
    for key, v, label in checks:
        lo, hi = envelope[key]
        if v < lo - 1e-12 or v > hi + 1e-12:
            out.append(f"{label} = {v:.4g} is outside the training range [{lo:.4g}, {hi:.4g}]. "
                       "The prediction is an extrapolation and may be unreliable.")
    a = np.atleast_1d(alphas)
    lo, hi = envelope["Alpha"]
    n_out = int(((a < lo) | (a > hi)).sum())
    if n_out:
        out.append(f"{n_out} of {len(a)} angle(s) of attack fall outside the training range "
                   f"[{lo:g}°, {hi:g}°]. Predictions there are extrapolation.")
    return out


# --------------------------------------------------------------- predict

def feature_matrix(geom, re_, alphas, feats):
    alphas = np.atleast_1d(np.asarray(alphas, dtype=float))
    row = {**{k: float(geom[k]) for k in CST_UPPER + CST_LOWER + GEOMETRY}, "Re": float(re_)}
    X = np.empty((len(alphas), len(feats)))
    for j, f in enumerate(feats):
        X[:, j] = alphas if f == "Alpha" else row[f]
    return X


def predict(model, geom, re_, alphas, feats):
    """Returns a DataFrame with Alpha, Cl, Cd, L/D and the latency in seconds."""
    X = feature_matrix(geom, re_, alphas, feats)
    t0 = time.perf_counter()
    Xs = model.scaler_.transform(X)                       # scaler ...
    Z = model.estimator_.predict(Xs) * model.sd_ + model.mu_   # ... -> model -> [Cl, log Cd]
    latency = time.perf_counter() - t0
    cl, cd = Z[:, 0], np.exp(Z[:, 1])
    return pd.DataFrame({"Alpha": X[:, feats.index("Alpha")], "Cl": cl, "Cd": cd, "L/D": cl / cd}), latency


def derived(res):
    """Cl_max and its angle, Cd_min, max L/D and its angle, stall angle."""
    a, cl, cd, ld = (res[c].to_numpy() for c in ("Alpha", "Cl", "Cd", "L/D"))
    i, j, k = int(np.argmax(cl)), int(np.argmax(ld)), int(np.argmin(cd))
    stalled = i < len(a) - 1 and np.any(cl[i + 1:] < cl[i] - 1e-3)
    return {"Cl_max": float(cl[i]), "alpha_Cl_max": float(a[i]), "Cd_min": float(cd[k]),
            "alpha_Cd_min": float(a[k]), "LD_max": float(ld[j]), "alpha_LD_max": float(a[j]),
            "stall_angle": float(a[i]) if stalled else None}


def true_polar(bundle, airfoil, re_):
    p = bundle["polars"]
    return p[(p.airfoil == airfoil) & (p.Re == re_)].sort_values("Alpha")


def sweep_alphas(cfg_app):
    lo, hi, step = cfg_app["alpha_min"], cfg_app["alpha_max"], cfg_app["alpha_step"]
    return np.round(np.arange(lo, hi + step / 2, step), 6)


def log_latency(kind, mode, n_points, seconds, path=LATENCY_LOG):
    """Append one latency record; never let logging break the app."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        row = pd.DataFrame([{"timestamp": datetime.now().isoformat(timespec="seconds"), "source": "app",
                             "model": kind, "mode": mode, "n_points": n_points,
                             "latency_ms": seconds * 1e3, "ms_per_point": seconds * 1e3 / max(n_points, 1)}])
        row.to_csv(path, mode="a", header=not path.exists(), index=False)
    except OSError:
        pass
