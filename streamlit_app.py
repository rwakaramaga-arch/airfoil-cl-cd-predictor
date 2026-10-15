"""AeroML - NACA Airfoil Aerodynamic Predictor (Streamlit app).

Run locally:   streamlit run app/streamlit_app.py
"""
import base64
import io
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402


def show_fig(fig, container=st):
    """Render a figure as an inline PNG. Streamlit Cloud's media endpoint
    returns HTTP 500 for st.pyplot images, so bypass it entirely."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    container.markdown(
        f'<img src="data:image/png;base64,{b64}" style="width:100%;height:auto;" alt="">',
        unsafe_allow_html=True)


st.set_page_config(page_title="AeroML", page_icon="✈️", layout="wide")

MODEL_DIR = Path(os.environ.get("AEROML_MODEL_DIR", core.MODEL_DIR))
TRUE_C, INK2 = "#0b0b0b", "#52514e"
PRED_C = {"Cl": "#2a78d6", "Cd": "#eb6834", "polar": "#1baf7a"}
CUSTOM = "Custom (enter coefficients)"


@st.cache_resource(show_spinner="Loading models...")
def get_bundle(path, model_files=()):
    # model_files is part of the cache key, so adding/removing a model file refreshes the bundle
    return core.load_bundle(path)


def model_files_on_disk(path):
    p = Path(path)
    return tuple(sorted(f.name for f in p.glob("model_*.joblib"))) if p.is_dir() else ()


@st.cache_resource(show_spinner="Loading model...")
def get_model(path, kind):
    return core.load_model(get_bundle(path, model_files_on_disk(path)), kind)


@st.cache_data
def airfoil_table(path):
    return get_bundle(path, model_files_on_disk(path))["airfoils"]


try:
    bundle = get_bundle(str(MODEL_DIR), model_files_on_disk(MODEL_DIR))
except core.AppModelError as e:
    st.title("AeroML — NACA Airfoil Aerodynamic Predictor")
    st.error(f"**Model files are missing, so the app cannot make predictions.** {e}")
    st.stop()

card, feats = bundle["card"], bundle["features"]
airfoils = airfoil_table(str(MODEL_DIR))
geo_by_name = airfoils.set_index("airfoil")
APP = card["app"]
RE_VALUES = card["re_values"]
# Only the models whose files are on disk (core.load_bundle drops the rest).
MODELS = {m["name"]: m["kind"] for m in card["models"]}
PRIMARY_NAME = next((m["name"] for m in card["models"] if m["primary"]), card["models"][0]["name"])
GEO_KEYS = core.CST_UPPER + core.CST_LOWER + core.GEOMETRY


# ---------------------------------------------------------------- state

def preset_values(name):
    r = geo_by_name.loc[name]
    return {k: float(r[k]) for k in GEO_KEYS}


def defaults():
    d = {"airfoil": APP["default_airfoil"] if APP["default_airfoil"] in geo_by_name.index else airfoils.airfoil.iloc[0],
         "model": PRIMARY_NAME, "re": 1_000_000.0 if 1e6 in RE_VALUES else RE_VALUES[-1],
         "mode": "Sweep", "alpha_single": 4.0,
         "alpha_range": (float(APP["alpha_min"]), float(APP["alpha_max"])), "alpha_step": 1.0}
    d.update(preset_values(d["airfoil"]))
    return d


def init_state():
    for k, v in defaults().items():
        st.session_state.setdefault(k, v)
    if st.session_state["model"] not in MODELS:      # e.g. a model file removed mid-session
        st.session_state["model"] = PRIMARY_NAME
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("show_guide", False)


def reset_all():
    for k, v in defaults().items():
        st.session_state[k] = v
    st.session_state["result"] = None


def apply_preset():
    if st.session_state["airfoil"] != CUSTOM:
        st.session_state.update(preset_values(st.session_state["airfoil"]))
    st.session_state["result"] = None


def toggle_guide():
    st.session_state["show_guide"] = not st.session_state["show_guide"]


init_state()

# --------------------------------------------------------------- header

st.title("AeroML — NACA Airfoil Aerodynamic Predictor")
st.caption("Machine-learning prediction of lift and drag coefficients from CST geometry and flow conditions")
c1, c2, c3, _ = st.columns([1, 1, 1, 3])
c1.button("📖 User guide", on_click=toggle_guide, use_container_width=True)
c2.button("↺ Reset", on_click=reset_all, use_container_width=True)
predict_clicked = c3.button("▶ Predict", type="primary", use_container_width=True)


GUIDE_NAMES = {"xgb": "XGBoost", "mlp": "the neural network", "rf": "Random Forest", "et": "Extra Trees"}
NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five"}


def _join(names):
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def deployment_note():
    """Shown in the guide only when some trained models were not shipped."""
    missing = bundle["missing_models"]
    if not missing:
        return ""
    have = _join([GUIDE_NAMES.get(m["kind"], m["name"]) for m in card["models"]])
    gone = _join([GUIDE_NAMES.get(m["kind"], m["name"]) for m in missing])
    n = bundle["n_models_trained"]
    gone = gone[0].upper() + gone[1:]
    return (f"\n\n*This deployment includes {have}. {gone} {'is' if len(missing) == 1 else 'are'} excluded "
            f"because of hosting size limits; all {NUMBER_WORDS.get(n, n)} are compared in the project report.*")


def user_guide():
    left, right = st.columns(2)
    with left:
        st.markdown(f"""
#### 1. About
AeroML predicts the lift coefficient **Cl** and drag coefficient **Cd** of a 2D NACA airfoil from
its shape and the flow conditions. It is a *surrogate* for XFOIL: it was trained on
XFOIL panel-method polars and answers in milliseconds, where XFOIL needs seconds per polar and can
fail to converge. The default model is **{PRIMARY_NAME}**, trained on
{card['training_airfoils']} airfoils.{deployment_note()}

#### 2. Input features (16)
| Feature | Meaning |
|---|---|
| Au0–Au4 | Upper-surface CST coefficients |
| Al0–Al4 | Lower-surface CST coefficients |
| max_thickness | Maximum thickness as a fraction of chord (t/c) |
| x_max_thickness | Chordwise position of maximum thickness (0–1) |
| max_camber | Maximum camber as a fraction of chord |
| x_max_camcer | Chordwise position of maximum camber (0–1) |
| Re | Reynolds number based on chord |
| Alpha | Angle of attack in degrees |

**What the CST coefficients control.** Each surface is
y = √x·(1−x)·Σ Aᵢ·Bᵢ(x), where the Bᵢ are Bernstein polynomials that each peak at a different
chord station. **Au0 / Al0** set the leading-edge region (nose radius), **Au1–Au3** the forward,
mid and aft surface, and **Au4 / Al4** the trailing-edge region (boattail). A larger positive
Au pushes the upper surface up. Al values are usually negative, and a more negative Al pushes the
lower surface down. Upper and lower coefficients that are equal and opposite give a symmetric airfoil.
""")
    with right:
        st.markdown("""
#### 3. How to use
1. In the sidebar, **pick an airfoil**: all 14 geometry fields fill in. Or choose *Custom* and
   enter your own coefficients under the three geometry sections.
2. **Choose a model**. Every trained model is available, so you can compare them.
3. Set the **Reynolds number** and choose **single angle** or **sweep**. In sweep mode, set the
   angle range and the **step** — 0.25°, 0.5°, 1°, 2° or 5° between predicted angles.
4. A note on step size: the training data is sampled every 0.25°, so smaller steps interpolate
   rather than add information. A coarse step can also step over the lift peak, so Cl_max and the
   stall angle are only resolved to about half the step you choose.
5. Press **Predict** and read the results below.

#### 4. Reading the results
- **Cl**: lift per unit span. **Cd**: drag. **L/D = Cl/Cd** is aerodynamic efficiency; its
  maximum marks the best cruise angle.
- **Cl–α curve**: linear at moderate angles. Its peak is **Cl_max**, and lift falls beyond the
  **stall angle**.
- **Drag polar (Cd vs Cl)**: the low-drag "bucket" is the range of Cl where drag stays near Cd_min.
  6-series laminar airfoils show a pronounced bucket.
- If the airfoil is in the dataset, the true XFOIL polar is drawn **dashed**, so you can see the
  accuracy directly.

#### 5. Limitations
- **Extrapolation**: outside the training envelope (warnings appear), predictions are unreliable.
- **Panel-method data**: XFOIL models separated flow poorly, so post-stall values are less
  trustworthy than pre-stall values.
- **Preliminary design only**: this is not CFD or wind-tunnel data. Verify final designs with
  higher-fidelity methods.
""")
    st.divider()


if st.session_state["show_guide"]:
    user_guide()

# --------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Inputs")
    st.selectbox("Airfoil preset", [CUSTOM] + list(airfoils.airfoil), key="airfoil", on_change=apply_preset,
                 help="Picking an airfoil fills all 14 geometry fields.")
    st.selectbox("Model", list(MODELS), key="model",
                 help="All trained models. The default is the deployment model.")
    st.select_slider("Reynolds number", options=RE_VALUES, key="re", format_func=lambda v: f"{v:,.0f}")
    st.radio("Mode", ["Single angle", "Sweep"], key="mode", horizontal=True)
    if st.session_state["mode"] == "Single angle":
        st.number_input("Angle of attack (deg)", min_value=-90.0, max_value=90.0, step=0.25, key="alpha_single")
    else:
        st.slider("Alpha range (deg)", float(APP["alpha_min"]), float(APP["alpha_max"]),
                  step=float(APP["alpha_step"]), key="alpha_range")
        # default (1.0) comes from defaults() via session state, so no value= here
        st.select_slider("Step (deg)", options=[0.25, 0.5, 1.0, 2.0, 5.0], key="alpha_step",
                         help="Spacing between predicted angles. The training data is sampled at 0.25 deg, "
                              "so finer steps interpolate rather than add information.")
    with st.expander("Upper surface (Au0–Au4)"):
        for k in core.CST_UPPER:
            st.number_input(k, step=0.001, format="%.6f", key=k)
    with st.expander("Lower surface (Al0–Al4)"):
        for k in core.CST_LOWER:
            st.number_input(k, step=0.001, format="%.6f", key=k)
    with st.expander("Thickness and camber"):
        st.number_input("max_thickness (t/c)", step=0.001, format="%.5f", key="max_thickness")
        st.number_input("x_max_thickness", step=0.01, format="%.4f", key="x_max_thickness")
        st.number_input("max_camber", step=0.001, format="%.5f", key="max_camber")
        st.number_input("x_max_camcer", step=0.01, format="%.4f", key="x_max_camcer")


# --------------------------------------------------------------- predict

def run_prediction():
    s = st.session_state
    geom = {k: float(s[k]) for k in GEO_KEYS}
    re_ = float(s["re"])
    notices, step = [], None
    if s["mode"] == "Single angle":
        alphas = np.array([float(s["alpha_single"])])
    else:
        lo, hi = s["alpha_range"]
        step = float(s["alpha_step"])
        # + step/2 includes the upper bound; rounding removes float drift (e.g. 1e-16)
        alphas = np.round(np.arange(lo, hi + step / 2, step), 6)
        n = len(alphas)
        if n < 2:
            return {"errors": ["Range is smaller than one step — widen the range or reduce the step."]}
        if n > 500:
            notices.append(f"{n} points requested; capping at 500.")
            alphas = alphas[np.unique(np.linspace(0, n - 1, 500).round().astype(int))]
    errors = core.validate_inputs(geom, re_, alphas)
    if errors:
        return {"errors": errors}
    warnings = core.envelope_warnings(geom, re_, alphas, card["training_envelope"])
    kind = MODELS[s["model"]]
    model = get_model(str(MODEL_DIR), kind)
    res, latency = core.predict(model, geom, re_, alphas, feats)
    mode = "single" if len(alphas) == 1 else "sweep"
    core.log_latency(kind, mode, len(alphas), latency)
    preset = s["airfoil"]
    unchanged = preset != CUSTOM and all(np.isclose(geom[k], v, rtol=0, atol=1e-9)
                                         for k, v in preset_values(preset).items())
    truth = core.true_polar(bundle, preset, re_) if unchanged else None
    return {"errors": [], "warnings": warnings, "res": res, "latency": latency, "geom": geom, "re": re_,
            "kind": kind, "model_name": s["model"], "airfoil": preset if unchanged else "custom geometry",
            "truth": truth, "derived": core.derived(res) if len(res) >= 5 else None,
            "notices": notices, "step": step, "n": len(res)}


if predict_clicked:
    st.session_state["result"] = run_prediction()

result = st.session_state["result"]


def plot_airfoil(geom, title):
    sh = core.airfoil_shape(geom)
    fig, ax = plt.subplots(figsize=(8, 1.9))
    ax.fill_between(sh["x"], sh["y_lower"], sh["y_upper"], color="#cde2fb")
    ax.plot(sh["x"], sh["y_upper"], color="#184f95", linewidth=1.5)
    ax.plot(sh["x"], sh["y_lower"], color="#184f95", linewidth=1.5)
    ax.plot(sh["x"], sh["camber_line"], color=INK2, linewidth=0.8, linestyle="--")
    ax.set_aspect("equal")
    ax.set_xlabel("x/c")
    ax.set_title(f"{title} — reconstructed from CST (N1=0.5, N2=1.0, order 4): "
                 f"t/c {sh['max_thickness']:.4f} at x/c {sh['x_max_thickness']:.3f}", loc="left", fontsize=9)
    ax.grid(color="#e1e0d9", linewidth=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return fig


if result is None:
    st.info("Choose an airfoil and conditions in the sidebar, then press **Predict**.")
    s = st.session_state
    show_fig(plot_airfoil({k: float(s[k]) for k in GEO_KEYS}, s["airfoil"]))
elif result["errors"]:
    for e in result["errors"]:
        st.error(e)
else:
    for w in result.get("notices", []):
        st.warning(w)
    for w in result["warnings"]:
        st.warning("⚠️ **Extrapolation:** " + w)
    res, d = result["res"], result["derived"]
    st.subheader(f"{result['airfoil']} · Re = {result['re']:,.0f} · {result['model_name']}")
    if len(res) == 1:
        r = res.iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cl", f"{r.Cl:.4f}")
        m2.metric("Cd", f"{r.Cd:.5f}")
        m3.metric("L/D", f"{r['L/D']:.1f}")
        m4.metric("Latency", f"{result['latency'] * 1e3:.2f} ms")
    else:
        m = st.columns(5)
        if d is None:
            m[0].info("Cl_max and stall angle need at least 5 points — reduce the step.")
        else:
            m[0].metric("Cl_max", f"{d['Cl_max']:.3f}", f"at {d['alpha_Cl_max']:.2f}°", delta_color="off")
            m[1].metric("Cd_min", f"{d['Cd_min']:.5f}", f"at {d['alpha_Cd_min']:.2f}°", delta_color="off")
            m[2].metric("Max L/D", f"{d['LD_max']:.1f}", f"at {d['alpha_LD_max']:.2f}°", delta_color="off")
            m[3].metric("Stall angle", "none in range" if d["stall_angle"] is None else f"{d['stall_angle']:.2f}°")
        m[4].metric("Latency", f"{result['latency'] * 1e3:.1f} ms", f"{len(res)} points", delta_color="off")

        truth = result["truth"]
        has_truth = truth is not None and len(truth) > 0
        cols = st.columns(3)
        specs = [("Lift curve", "Alpha", "Cl", "Angle of attack (deg)", "Cl", PRED_C["Cl"]),
                 ("Drag curve", "Alpha", "Cd", "Angle of attack (deg)", "Cd", PRED_C["Cd"]),
                 ("Drag polar", "Cd", "Cl", "Cd", "Cl", PRED_C["polar"])]
        for col, (title, xc, yc, xl, yl, c) in zip(cols, specs):
            fig, ax = plt.subplots(figsize=(4.4, 3.4))
            ax.plot(res[xc], res[yc], color=c, linewidth=2, label="Prediction")
            if has_truth:
                ax.plot(truth[xc], truth[yc], color=TRUE_C, linestyle="--", linewidth=1.4, label="XFOIL (true)")
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_title(title, loc="left")
            ax.grid(color="#e1e0d9", linewidth=0.5)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            show_fig(fig, col)
        if result.get("step") is not None:
            st.caption(f"{result['n']} points at {result['step']}° spacing, "
                       f"predicted in {result['latency'] * 1e3:.1f} ms")
        if not has_truth:
            st.caption("No XFOIL reference shown: the geometry is custom/modified, or this airfoil has no "
                       "polar at this Reynolds number.")
    show_fig(plot_airfoil(result["geom"], result["airfoil"]))
    with st.expander("Predicted values", expanded=len(res) == 1):
        st.dataframe(res.style.format({"Alpha": "{:.2f}", "Cl": "{:.4f}", "Cd": "{:.5f}", "L/D": "{:.1f}"}),
                     use_container_width=True, hide_index=True)
    out = res.assign(airfoil=result["airfoil"], Re=result["re"], model=result["model_name"])
    st.download_button("⬇ Download predictions (CSV)", out.to_csv(index=False).encode(),
                       file_name=f"aeroml_{result['airfoil']}_Re{int(result['re'])}.csv", mime="text/csv")

# ---------------------------------------------------------------- footer

st.divider()
sel = next(m for m in card["models"] if m["name"] == st.session_state["model"])
tm = sel["test_metrics"]
st.caption(
    f"**Model:** {sel['name']}{' (deployment build)' if sel['deployment_compressed'] else ''} · "
    f"**Test R²** Cl {tm['Cl']['R2']:.4f}, Cd {tm['Cd']['R2']:.4f} (log-space {tm['Cd']['R2_log']:.4f}) on "
    f"{len(card['test_airfoils'])} held-out airfoils · **Trained on** {card['training_airfoils']} airfoils · "
    f"**Data:** XFOIL panel-method results; limited fidelity beyond stall. Preliminary design tool, not CFD.")
