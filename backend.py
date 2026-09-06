"""
backend.py

Local Flask server for Data Lens. Run this from the same folder as
dataset_preprocessor.ipynb and dataset_preprocessor_gui.html:

    pip install -r requirements.txt
    python backend.py

Then open http://127.0.0.1:5000 in your browser.

The HTML/JS front end is now a thin client: every analysis step (missing
values, outliers, correlation, encoding, scaling, splitting, ...) is
computed here in Python using the exact same pandas / numpy / scikit-learn
functions that live in dataset_preprocessor.ipynb (ported into
preprocessing_lib.py). No computation happens in JavaScript anymore.
"""

import io
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_file, send_from_directory

import preprocessing_lib as lib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=None)

# --------------------------------------------------------------------------
# In-memory application state.
# This is a local, single-user desktop tool (mirrors the notebook's single
# `df` variable), so a simple module-level dict is enough - no need for a
# database or per-user sessions.
# --------------------------------------------------------------------------
STATE = {
    "filename": None,
    "original_df": None,   # df exactly as uploaded (Overview/Missing/etc. tabs use this)
    "df": None,             # working copy (the "Prepare" tab mutates this)
    "target_col": None,
    "pipeline_log": [],
    "split": None,          # (X_train, X_val, X_test, y_train, y_val, y_test)
    "fitted_preprocessor": None,
}


def require_data():
    if STATE["df"] is None:
        return jsonify({"error": "No dataset loaded yet. Upload a file first."}), 400
    return None


def col_types(df):
    lookup, _ = lib.df_column_types(df)
    return lookup


# --------------------------------------------------------------------------
# Front end
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "dataset_preprocessor_gui.html")


# --------------------------------------------------------------------------
# 1. Load dataset
# --------------------------------------------------------------------------
@app.route("/api/load", methods=["POST"])
def api_load():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    try:
        df = lib.load_data(f, f.filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    if df.empty:
        return jsonify({"error": "That file has no readable rows."}), 400

    STATE["filename"] = f.filename
    STATE["original_df"] = df
    STATE["df"] = df.copy()
    STATE["target_col"] = df.columns[-1]
    STATE["pipeline_log"] = []
    STATE["split"] = None
    STATE["fitted_preprocessor"] = None

    return jsonify(_dataset_payload())


def _dataset_payload():
    df = STATE["original_df"]
    types = col_types(df)
    return {
        "filename": STATE["filename"],
        "rows": len(df),
        "columns": list(df.columns),
        "types": types,
        "target_col": STATE["target_col"],
        "overview": lib.dataset_overview(df),
    }


@app.route("/api/sample", methods=["POST"])
def api_sample():
    """Load a small built-in sample dataset (used by the 'Try sample data' button)."""
    import numpy as np

    rng = np.random.default_rng(42)
    n = 300
    age = rng.normal(40, 12, n).round(1)
    income = (rng.normal(60000, 20000, n) + age * 300).round(2)
    income[rng.choice(n, 10, replace=False)] = np.nan
    city = rng.choice(["Jakarta", "Surabaya", "Bandung", "Medan", " jakarta "], n)
    signup_date = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 700, n), unit="D")
    churned = (income < 55000).astype(int)
    df = pd.DataFrame(
        {
            "customer_id": range(1, n + 1),
            "age": age,
            "annual_income": income,
            "city": city,
            "signup_date": signup_date.astype(str),
            "churned": churned,
        }
    )
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)  # a few duplicate rows on purpose

    STATE["filename"] = "sample_customers.csv"
    STATE["original_df"] = df
    STATE["df"] = df.copy()
    STATE["target_col"] = "churned"
    STATE["pipeline_log"] = []
    STATE["split"] = None
    STATE["fitted_preprocessor"] = None
    return jsonify(_dataset_payload())


# --------------------------------------------------------------------------
# 2/3. Overview & column types (always computed on the ORIGINAL upload)
# --------------------------------------------------------------------------
@app.route("/api/overview")
def api_overview():
    err = require_data()
    if err:
        return err
    return jsonify(lib.dataset_overview(STATE["original_df"]))


# --------------------------------------------------------------------------
# 4. Missing values
# --------------------------------------------------------------------------
@app.route("/api/missing")
def api_missing():
    err = require_data()
    if err:
        return err
    return jsonify(lib.missing_data_report(STATE["original_df"]))


# --------------------------------------------------------------------------
# 5/6. Duplicates, invalid values, categorical quality
# --------------------------------------------------------------------------
@app.route("/api/quality")
def api_quality():
    err = require_data()
    if err:
        return err
    df = STATE["original_df"]
    types = col_types(df)
    numeric_cols = [c for c, t in types.items() if t == "numeric"]
    categorical_cols = [c for c, t in types.items() if t in ("categorical", "boolean")]
    n_dupes = int(df.duplicated().sum())
    invalid = lib.detect_invalid_values(df, numeric_cols, categorical_cols)
    cat_report, label_issues = lib.categorical_quality_report(df, categorical_cols)
    return jsonify(
        {
            "n_duplicates": n_dupes,
            "n_rows": len(df),
            "invalid_values": invalid,
            "categorical_quality": cat_report,
            "inconsistent_labels": label_issues,
        }
    )


@app.route("/api/quality/remove_duplicates", methods=["POST"])
def api_remove_duplicates():
    err = require_data()
    if err:
        return err
    STATE["df"], removed = lib.remove_duplicates(STATE["df"])
    _log({"op": "remove_duplicates", "removed": removed})
    return jsonify({"removed": removed, "rows_left": len(STATE["df"])})


# --------------------------------------------------------------------------
# 7. Numerical skewness / distributions
# --------------------------------------------------------------------------
@app.route("/api/skewness")
def api_skewness():
    err = require_data()
    if err:
        return err
    df = STATE["original_df"]
    types = col_types(df)
    numeric_cols = [c for c, t in types.items() if t == "numeric"]
    return jsonify(lib.numerical_skewness(df, numeric_cols))


@app.route("/api/distribution")
def api_distribution():
    err = require_data()
    if err:
        return err
    col = request.args.get("col")
    df = STATE["original_df"]
    if col not in df.columns:
        return jsonify({"error": f"Unknown column {col}"}), 400
    types = col_types(df)
    col_type = "numeric" if types.get(col) == "numeric" else "categorical"
    chart = lib.plot_distribution(df, col, col_type)
    payload = {"chart": chart, "type": col_type}
    if col_type == "numeric":
        vals = df[col].dropna()
        payload["stats"] = {
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "count": int(vals.count()),
        }
        sk = lib.numerical_skewness(df, [col])[0]
        payload["skewness"] = sk
    return jsonify(payload)


# --------------------------------------------------------------------------
# 8. Outliers
# --------------------------------------------------------------------------
@app.route("/api/outliers")
def api_outliers():
    err = require_data()
    if err:
        return err
    col = request.args.get("col")
    method = request.args.get("method", "iqr")
    df = STATE["original_df"]
    if col not in df.columns:
        return jsonify({"error": f"Unknown column {col}"}), 400
    if method == "zscore":
        return jsonify(lib.detect_outliers_zscore(df, col))
    return jsonify(lib.detect_outliers_iqr(df, col))


# --------------------------------------------------------------------------
# 9. Target analysis
# --------------------------------------------------------------------------
@app.route("/api/target")
def api_target():
    err = require_data()
    if err:
        return err
    col = request.args.get("col") or STATE["target_col"]
    STATE["target_col"] = col
    df = STATE["original_df"]
    if col not in df.columns:
        return jsonify({"error": f"Unknown column {col}"}), 400
    return jsonify(lib.analyze_target(df, col))


# --------------------------------------------------------------------------
# 10. Relationships
# --------------------------------------------------------------------------
@app.route("/api/relationship")
def api_relationship():
    err = require_data()
    if err:
        return err
    col = request.args.get("col")
    target = request.args.get("target") or STATE["target_col"]
    df = STATE["original_df"]
    if col not in df.columns or target not in df.columns:
        return jsonify({"error": "Unknown column"}), 400
    chart = lib.relationship_plot(df, col, target)
    return jsonify({"chart": chart})


# --------------------------------------------------------------------------
# 11. Correlation / redundancy / leakage
# --------------------------------------------------------------------------
@app.route("/api/correlation")
def api_correlation():
    err = require_data()
    if err:
        return err
    df = STATE["original_df"]
    result = lib.correlation_analysis(df)
    types = col_types(df)
    numeric_cols = [c for c, t in types.items() if t == "numeric"]
    redundant, identical = lib.find_redundant_features(df, numeric_cols)
    leakage = []
    if STATE["target_col"]:
        leakage = lib.check_data_leakage(df, STATE["target_col"], numeric_cols)
    result["redundant_pairs"] = redundant
    result["identical_columns"] = identical
    result["leakage_flags"] = leakage
    return jsonify(result)


# --------------------------------------------------------------------------
# Prepare tab: everything below mutates the WORKING COPY (STATE["df"])
# --------------------------------------------------------------------------
def _log(step):
    STATE["pipeline_log"].append(step)


@app.route("/api/prepare/state")
def api_prepare_state():
    err = require_data()
    if err:
        return err
    df = STATE["df"]
    types = col_types(df)
    missing_cols = [c for c in df.columns if df[c].isna().any()]
    date_like = [c for c, t in types.items() if t in ("categorical",) and lib.is_date_like_column(df, c)]
    text_like = [
        c
        for c, t in types.items()
        if t == "categorical" and c not in date_like and lib.is_text_like_column(df, c)
    ]
    skewed = []
    for c, t in types.items():
        if t == "numeric":
            vals = df[c].dropna()
            if len(vals) > 2:
                sk = lib.numerical_skewness(df, [c])[0]["skewness"]
                if sk is not None and abs(sk) >= 1 and (vals >= 0).all():
                    skewed.append(c)
    return jsonify(
        {
            "rows": len(df),
            "columns": list(df.columns),
            "types": types,
            "missing_columns": missing_cols,
            "date_like_columns": date_like,
            "text_like_columns": text_like,
            "skewed_columns": skewed,
            "target_col": STATE["target_col"],
            "pipeline_log": STATE["pipeline_log"],
            "has_split": STATE["split"] is not None,
        }
    )


@app.route("/api/prepare/set_target", methods=["POST"])
def api_set_target():
    err = require_data()
    if err:
        return err
    STATE["target_col"] = request.json.get("target_col")
    return jsonify({"target_col": STATE["target_col"]})


@app.route("/api/prepare/fill_missing", methods=["POST"])
def api_fill_missing():
    err = require_data()
    if err:
        return err
    body = request.json
    col, strategy = body["column"], body["strategy"]
    constant = body.get("constant")
    STATE["df"], value = lib.fill_missing(STATE["df"], col, strategy, constant)
    _log({"op": f"fill_{strategy}", "column": col, "value": str(value)})
    return jsonify({"filled_with": str(value)})


@app.route("/api/prepare/drop_rows_missing", methods=["POST"])
def api_drop_rows_missing():
    err = require_data()
    if err:
        return err
    col = request.json["column"]
    STATE["df"], removed = lib.drop_rows_with_missing(STATE["df"], col)
    _log({"op": "drop_rows_missing", "column": col, "removed": removed})
    return jsonify({"removed": removed, "rows_left": len(STATE["df"])})


@app.route("/api/prepare/drop_column", methods=["POST"])
def api_drop_column():
    err = require_data()
    if err:
        return err
    col = request.json["column"]
    STATE["df"] = STATE["df"].drop(columns=[col])
    _log({"op": "drop_column", "column": col})
    return jsonify({"columns": list(STATE["df"].columns)})


@app.route("/api/prepare/extract_datetime", methods=["POST"])
def api_extract_datetime():
    err = require_data()
    if err:
        return err
    col = request.json["column"]
    STATE["df"] = lib.engineer_datetime_features(STATE["df"], col)
    _log({"op": "extract_datetime_features", "column": col})
    return jsonify({"columns": list(STATE["df"].columns)})


@app.route("/api/prepare/extract_text", methods=["POST"])
def api_extract_text():
    err = require_data()
    if err:
        return err
    col = request.json["column"]
    STATE["df"] = lib.engineer_text_features(STATE["df"], col)
    _log({"op": "extract_text_features", "column": col})
    return jsonify({"columns": list(STATE["df"].columns)})


@app.route("/api/prepare/bool_to_int", methods=["POST"])
def api_bool_to_int():
    err = require_data()
    if err:
        return err
    col = request.json["column"]
    STATE["df"][col] = STATE["df"][col].astype("boolean").astype("Int64")
    _log({"op": "boolean_to_int", "column": col})
    return jsonify({"ok": True})


@app.route("/api/prepare/encode", methods=["POST"])
def api_encode():
    err = require_data()
    if err:
        return err
    body = request.json
    col, method = body["column"], body["method"]
    if method == "onehot":
        STATE["df"] = lib.encode_categorical(STATE["df"], onehot_cols=[col])
    else:
        STATE["df"] = lib.encode_categorical(STATE["df"], ordinal_cols=[col])
    _log({"op": f"{method}_encode", "column": col})
    return jsonify({"columns": list(STATE["df"].columns)})


@app.route("/api/prepare/scale", methods=["POST"])
def api_scale():
    err = require_data()
    if err:
        return err
    method = request.json.get("method", "standard")
    types = col_types(STATE["df"])
    numeric_cols = [c for c, t in types.items() if t == "numeric"]
    if not numeric_cols:
        return jsonify({"error": "No numeric columns to scale"}), 400
    STATE["df"], _scaler = lib.scale_numerical(STATE["df"], numeric_cols, method)
    _log({"op": "scale_numeric", "method": method, "columns": numeric_cols})
    return jsonify({"scaled_columns": numeric_cols})


@app.route("/api/prepare/log_transform", methods=["POST"])
def api_log_transform():
    err = require_data()
    if err:
        return err
    types = col_types(STATE["df"])
    numeric_cols = [c for c, t in types.items() if t == "numeric"]
    STATE["df"], transformed = lib.transform_skewed(STATE["df"], numeric_cols)
    _log({"op": "log_transform", "columns": transformed})
    return jsonify({"transformed_columns": transformed})


@app.route("/api/prepare/split", methods=["POST"])
def api_split():
    err = require_data()
    if err:
        return err
    body = request.json
    test_size = float(body.get("test_size", 0.2))
    val_size = float(body.get("val_size", 0.1))
    target_col = STATE["target_col"]
    if target_col not in STATE["df"].columns:
        return jsonify({"error": f"Target column '{target_col}' not in working copy"}), 400

    df = STATE["df"]
    is_classification = (not pd.api.types.is_numeric_dtype(df[target_col])) or df[target_col].nunique() <= 20
    X_train, X_val, X_test, y_train, y_val, y_test = lib.split_data(
        df, target_col, test_size=test_size, val_size=val_size, stratify=is_classification
    )
    STATE["split"] = (X_train, X_val, X_test, y_train, y_val, y_test)
    _log({"op": "train_val_test_split", "test_size": test_size, "val_size": val_size})
    return jsonify(
        {
            "train": len(X_train),
            "val": 0 if X_val is None else len(X_val),
            "test": 0 if X_test is None else len(X_test),
            "task": "classification" if is_classification else "regression",
        }
    )


@app.route("/api/prepare/build_pipeline", methods=["POST"])
def api_build_pipeline():
    """Fit the actual sklearn ColumnTransformer, exactly like the notebook's
    build_preprocessing_pipeline + fit_transform step."""
    err = require_data()
    if err:
        return err
    if STATE["split"] is None:
        return jsonify({"error": "Split the dataset first."}), 400
    X_train, X_val, X_test, y_train, y_val, y_test = STATE["split"]
    body = request.json or {}
    scale_method = body.get("scale_method", "standard")
    types = col_types(X_train)
    numeric_cols = [c for c, t in types.items() if t in ("numeric", "boolean") and c in X_train.columns]
    categorical_cols = [c for c, t in types.items() if t == "categorical" and c in X_train.columns]
    preprocessor = lib.build_preprocessing_pipeline(numeric_cols, categorical_cols, scale_method=scale_method)
    preprocessor.fit(X_train)
    STATE["fitted_preprocessor"] = preprocessor
    _log({"op": "build_preprocessing_pipeline", "scale_method": scale_method,
          "numeric_cols": numeric_cols, "categorical_cols": categorical_cols})
    X_train_t = preprocessor.transform(X_train)
    return jsonify({"output_shape": list(X_train_t.shape), "numeric_cols": numeric_cols, "categorical_cols": categorical_cols})


@app.route("/api/prepare/reset", methods=["POST"])
def api_reset():
    err = require_data()
    if err:
        return err
    STATE["df"] = STATE["original_df"].copy()
    STATE["split"] = None
    STATE["fitted_preprocessor"] = None
    STATE["pipeline_log"] = []
    return jsonify({"columns": list(STATE["df"].columns)})


# --------------------------------------------------------------------------
# Downloads
# --------------------------------------------------------------------------
@app.route("/api/download/processed")
def download_processed():
    err = require_data()
    if err:
        return err
    buf = io.StringIO()
    STATE["df"].to_csv(buf, index=False)
    mem = io.BytesIO(buf.getvalue().encode("utf-8"))
    return send_file(mem, mimetype="text/csv", as_attachment=True, download_name="processed_dataset.csv")


@app.route("/api/download/split/<part>")
def download_split(part):
    if STATE["split"] is None:
        return jsonify({"error": "No split available"}), 400
    X_train, X_val, X_test, y_train, y_val, y_test = STATE["split"]
    mapping = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }
    if part not in mapping or mapping[part][0] is None:
        return jsonify({"error": f"No '{part}' split available"}), 400
    X, y = mapping[part]
    out = X.copy()
    out[STATE["target_col"]] = y
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    mem = io.BytesIO(buf.getvalue().encode("utf-8"))
    return send_file(mem, mimetype="text/csv", as_attachment=True, download_name=f"{part}.csv")


@app.route("/api/download/pipeline")
def download_pipeline():
    """Saves the fitted sklearn ColumnTransformer with joblib, like the
    notebook's save_pipeline(), plus a JSON log of every step taken."""
    if STATE["fitted_preprocessor"] is not None:
        buf = io.BytesIO()
        joblib.dump(STATE["fitted_preprocessor"], buf)
        buf.seek(0)
        return send_file(buf, mimetype="application/octet-stream",
                          as_attachment=True, download_name="preprocessing_pipeline.joblib")
    # fall back to the JSON step log if no sklearn pipeline has been fitted yet
    import json
    payload = {
        "source_file": STATE["filename"],
        "target_column": STATE["target_col"],
        "final_columns": list(STATE["df"].columns) if STATE["df"] is not None else [],
        "steps": STATE["pipeline_log"],
    }
    mem = io.BytesIO(json.dumps(payload, indent=2).encode("utf-8"))
    return send_file(mem, mimetype="application/json", as_attachment=True, download_name="preprocessing_pipeline.json")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
