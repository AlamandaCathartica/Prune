"""
    pip install flask numpy matplotlib scipy pillow
    python app.py
Then open http://127.0.0.1:5000
"""

from matplotlib import axes
import base64
import io
import os
import re

import matplotlib
matplotlib.use("Agg")  # headless rendering — no display needed on the server
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
import numpy as np
from scipy.special import gamma as _gamma, digamma as _digamma
from PIL import Image
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

N_DEFAULT = 60
N_MIN = 10
N_MAX = 150
RANGE = 2.0
RANGE_MIN = -50.0
RANGE_MAX = 50.0
FRAMES_MIN = 4
FRAMES_MAX = 60
FRAME_MS_MIN = 30
FRAME_MS_MAX = 1000
ALL_KEYS = ["x", "y", "a", "b", "r", "p"]
LABELS = {
    "x": "x  (Re input)", "y": "y  (Re output)",
    "a": "im(a)  (Im input)", "b": "im(b)  (Im output)",
    "r": "|w|  (magnitude)", "p": "arg(w)  (phase)",
}
CMAP_LINEAR = plt.get_cmap("viridis")
CMAP_CYCLIC = plt.get_cmap("hsv")

# Names allowed inside an equation, e.g. "sin(z)**2 + exp(1/z)"
SAFE_NAMES = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
    "abs": np.abs, "conj": np.conj,
    "gamma": _gamma, "digamma": _digamma,
    "pi": np.pi, "e": np.e, "j": 1j,
}


def compile_equation(text):
    text = (text or "").strip()
    if not text:
        raise ValueError("Enter an equation.")

    expr = text.replace("^", "**")

    identifiers = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr))
    allowed = set(SAFE_NAMES) | {"z"}
    unknown = identifiers - allowed
    if unknown:
        raise ValueError(f"Unknown name(s) in equation: {', '.join(sorted(unknown))}")

    code = compile(expr, "<equation>", "eval")

    def f(z):
        return eval(code, {"__builtins__": {}}, {**SAFE_NAMES, "z": z})

    # smoke-test it before trusting it on the full grid
    try:
        f(np.complex128(1.0 + 1.0j))
    except Exception as e:
        raise ValueError(f"Couldn't evaluate that equation ({e}).")

    return f


def compute_fields(equation_text, re_min, re_max, im_min, im_max, n):
    f = compile_equation(equation_text)

    x_lin = np.linspace(re_min, re_max, n)
    a_lin = np.linspace(im_min, im_max, n)
    X, A = np.meshgrid(x_lin, a_lin)
    Z = X + 1j * A

    with np.errstate(all="ignore"):
        W = f(Z)
        W = np.asarray(W, dtype=np.complex128)
        W = np.where(np.isfinite(W), W, 0)

    # clip the output so wild singularities don't blow up the plot scale;
    # the clip bound scales with the input domain so it stays sensible
    # whether the person zoomed in tight or way out.
    span = max(abs(re_min), abs(re_max), abs(im_min), abs(im_max), 1e-9)
    clip = span * 6

    Y = np.clip(np.real(W), -clip, clip)
    B = np.clip(np.imag(W), -clip, clip)
    R = np.clip(np.abs(W), 0, clip)
    P = np.angle(W)  # already bounded to [-pi, pi], no clipping needed

    return {"x": X, "y": Y, "a": A, "b": B, "r": R, "p": P}


def build_figure(fields, x_key, y_key, z_key, color_key, m, n, figsize=(8, 6), dpi=160):
    sx, sy, sz = fields[x_key], fields[y_key], fields[z_key]
    cfield = fields[color_key]
    cmap = CMAP_CYCLIC if color_key == "p" else CMAP_LINEAR

    if color_key == "p":
        norm = plt.Normalize(vmin=-np.pi, vmax=np.pi)
    else:
        norm = plt.Normalize(vmin=np.nanmin(cfield), vmax=np.nanmax(cfield))
    facecolors = cmap(norm(cfield))

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

    ax.plot_surface(
        sx, sy, sz,
        facecolors=facecolors,
        rstride=1, cstride=1,
        linewidth=0, antialiased=False, shade=False,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)

    ax.set_box_aspect(None, zoom=0.85)
    ax.view_init(elev=n, azim=m)
    ax.set_xlabel(LABELS[x_key], fontsize=8)
    ax.set_ylabel(LABELS[y_key], fontsize=8)
    ax.set_zlabel(LABELS[z_key], fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_facecolor("#FFFBFE")
    fig.patch.set_facecolor("#FFFBFE")
    ax.xaxis.pane.set_facecolor("#FFFFFF")
    ax.yaxis.pane.set_facecolor("#FFFFFF")
    ax.zaxis.pane.set_facecolor("#FFFFFF")
    # ax.grid(True)

    return fig


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def render_png(fields, x_key, y_key, z_key, color_key, m, n):
    fig = build_figure(fields, x_key, y_key, z_key, color_key, m, n)
    return base64.b64encode(fig_to_png_bytes(fig)).decode("ascii")


def render_gif(fields, x_key, y_key, z_key, color_key, m0, n0, m1, n1, frame_count, frame_ms):
    frames = []
    for i in range(frame_count):
        t = i / (frame_count - 1) if frame_count > 1 else 0.0
        m = m0 + (m1 - m0) * t
        n = n0 + (n1 - n0) * t
        fig = build_figure(fields, x_key, y_key, z_key, color_key, m, n)
        png_bytes = fig_to_png_bytes(fig)
        frames.append(Image.open(io.BytesIO(png_bytes)).convert("RGB"))

    buf = io.BytesIO()
    frames[0].save(
        buf, format="GIF", save_all=True, append_images=frames[1:],
        duration=frame_ms, loop=0, disposal=2,
    )
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")




def parse_common_payload(payload):
    """Shared parsing/validation for equation, axes, domain range, and resolution.
    Raises ValueError with a user-facing message on bad input."""
    equation = payload.get("equation", "z**2")
    x_key = payload.get("x_axis", "x")
    y_key = payload.get("y_axis", "y")
    z_key = payload.get("z_axis", "a")
    color_key = payload.get("color", "b")

    try:
        re_min = float(payload.get("re_min", -RANGE))
        re_max = float(payload.get("re_max", RANGE))
        im_min = float(payload.get("im_min", -RANGE))
        im_max = float(payload.get("im_max", RANGE))
        resolution = int(float(payload.get("resolution", N_DEFAULT)))
    except (TypeError, ValueError):
        raise ValueError("Range and resolution values must be numbers.")

    axis_keys = {"x": x_key, "y": y_key, "z": z_key, "color": color_key}
    for label, key in axis_keys.items():
        if key not in ALL_KEYS:
            raise ValueError(f"Invalid {label} axis selection.")
    if len({x_key, y_key, z_key}) < 3:
        raise ValueError("X, Y, and Z axes must all be different.")

    re_min = max(RANGE_MIN, min(RANGE_MAX, re_min))
    re_max = max(RANGE_MIN, min(RANGE_MAX, re_max))
    im_min = max(RANGE_MIN, min(RANGE_MAX, im_min))
    im_max = max(RANGE_MIN, min(RANGE_MAX, im_max))
    if re_max - re_min < 1e-6:
        raise ValueError("x2 must be greater than x1.")
    if im_max - im_min < 1e-6:
        raise ValueError("y2 must be greater than y1.")

    resolution = max(N_MIN, min(N_MAX, resolution))

    return {
        "equation": equation, "x_key": x_key, "y_key": y_key, "z_key": z_key,
        "color_key": color_key, "re_min": re_min, "re_max": re_max,
        "im_min": im_min, "im_max": im_max, "resolution": resolution,
    }


def normalize_angle(v, lo, hi):
    """Wrap v into [lo, hi) using true modulo arithmetic, so e.g. 720 deg
    wraps to the same angle as 0 deg, -450 wraps the same as -90, etc."""
    period = hi - lo
    return ((v - lo) % period) + lo


def parse_view(payload, m_field="m", n_field="n", m_default=45, n_default=45):
    try:
        m = float(payload.get(m_field, m_default))
        n_elev = float(payload.get(n_field, n_default))
    except (TypeError, ValueError):
        raise ValueError("Rotation values must be numbers.")
    m = normalize_angle(m, -180.0, 180.0)
    n_elev = normalize_angle(n_elev, -90.0, 90.0)
    return m, n_elev


def parse_animation_angle(payload, field, default, lo, hi):
    """Same modulo-wrap as parse_view, applied per-field so azimuth (m0/m1)
    and elevation (n0/n1) each wrap into their own range. A huge sweep like
    0 -> 720 deg is mathematically identical to 0 -> 0 deg, so it wraps down
    to that -- and the explicit start==end check below then reports a clear
    error instead of silently producing a 1-frame GIF."""
    try:
        v = float(payload.get(field, default))
    except (TypeError, ValueError):
        raise ValueError("Start/end angles must be numbers.")
    return normalize_angle(v, lo, hi)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/render", methods=["POST"])
def api_render():
    payload = request.get_json(force=True, silent=True) or {}
    try:
        common = parse_common_payload(payload)
        m, n_elev = parse_view(payload)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        fields = compute_fields(
            common["equation"], common["re_min"], common["re_max"],
            common["im_min"], common["im_max"], common["resolution"],
        )
        image_b64 = render_png(
            fields, common["x_key"], common["y_key"], common["z_key"],
            common["color_key"], m, n_elev,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Render failed: {e}"}), 500

    return jsonify({
        "image": f"data:image/png;base64,{image_b64}",
        "m": m, "n": n_elev,
        "re_min": common["re_min"], "re_max": common["re_max"],
        "im_min": common["im_min"], "im_max": common["im_max"],
        "resolution": common["resolution"],
    })


@app.route("/api/animate", methods=["POST"])
def api_animate():
    payload = request.get_json(force=True, silent=True) or {}
    try:
        common = parse_common_payload(payload)
        m0 = parse_animation_angle(payload, "m0", -60, -180.0, 180.0)
        n0 = parse_animation_angle(payload, "n0", 20, -90.0, 90.0)
        m1 = parse_animation_angle(payload, "m1", 60, -180.0, 180.0)
        n1 = parse_animation_angle(payload, "n1", 20, -90.0, 90.0)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if abs(m1 - m0) < 1e-6 and abs(n1 - n0) < 1e-6:
        return jsonify({"error": "Start and end angles are the same — pick a different end angle so the frames actually differ."}), 400

    try:
        frame_count = int(float(payload.get("frames", 20)))
        frame_ms = int(float(payload.get("frame_ms", 80)))
    except (TypeError, ValueError):
        return jsonify({"error": "Frame count and duration must be numbers."}), 400

    frame_count = max(FRAMES_MIN, min(FRAMES_MAX, frame_count))
    frame_ms = max(FRAME_MS_MIN, min(FRAME_MS_MAX, frame_ms))

    try:
        fields = compute_fields(
            common["equation"], common["re_min"], common["re_max"],
            common["im_min"], common["im_max"], common["resolution"],
        )
        gif_b64 = render_gif(
            fields, common["x_key"], common["y_key"], common["z_key"],
            common["color_key"], m0, n0, m1, n1, frame_count, frame_ms,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Animation failed: {e}"}), 500

    return jsonify({
        "image": f"data:image/gif;base64,{gif_b64}",
        "m0": m0, "n0": n0, "m1": m1, "n1": n1,
        "frames": frame_count, "frame_ms": frame_ms,
        "re_min": common["re_min"], "re_max": common["re_max"],
        "im_min": common["im_min"], "im_max": common["im_max"],
        "resolution": common["resolution"],
    })


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(debug=debug_mode, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
