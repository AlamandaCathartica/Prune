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

import csv
import io
import os
import time
import uuid

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_file, send_from_directory

import preprocessing_lib as lib
import ml_lib as ml

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
    "undo_stack": [],        # snapshots of STATE["df"] for undo
    "redo_stack": [],        # snapshots for redo
    # --- Baseline machine learning (Model tab) ---
    "ml_task": None,         # "classification" | "regression", set at train time
    "ml_trained": {},        # model_id -> {"model": fitted estimator, "label": str, "result": {...}}
    "ml_preprocessor": None, # ColumnTransformer fit on the split used for training
    "ml_label_encoder": None,
    "ml_class_names": None,
    "ml_eval_name": None,    # "test" | "validation" - which split was used for evaluation
    "ml_Xeval": None,        # transformed eval features, kept for feature importance
    "ml_yeval": None,        # eval target, kept for feature importance
    "ml_feature_cols": [],   # raw (pre-encoding) feature columns, in model order
    "ml_numeric_cols": [],
    "ml_categorical_cols": [],
    "ml_feature_meta": [],   # per-feature type/range/options, for the Predict form
    # --- Experiment tracking (persists across trainings/resets until cleared) ---
    "experiments": [],
}

UNDO_LIMIT = 25


def _snapshot():
    """Call before any working-copy mutation so it can be undone."""
    if STATE["df"] is not None:
        STATE["undo_stack"].append(STATE["df"].copy())
        if len(STATE["undo_stack"]) > UNDO_LIMIT:
            STATE["undo_stack"].pop(0)
        STATE["redo_stack"] = []


def require_data():
    if STATE["df"] is None:
        return jsonify({"error": "No dataset loaded yet. Upload a file first."}), 400
    return None


def col_types(df):
    lookup, _ = lib.df_column_types(df)
    return lookup


def _ml_reset():
    """Clear any trained models - called whenever the dataset or the
    train/val/test split changes underneath them, since old models
    reference a preprocessor fit on a split that no longer exists."""
    STATE["ml_task"] = None
    STATE["ml_trained"] = {}
    STATE["ml_preprocessor"] = None
    STATE["ml_label_encoder"] = None
    STATE["ml_class_names"] = None
    STATE["ml_eval_name"] = None
    STATE["ml_Xeval"] = None
    STATE["ml_yeval"] = None
    STATE["ml_feature_cols"] = []
    STATE["ml_numeric_cols"] = []
    STATE["ml_categorical_cols"] = []
    STATE["ml_feature_meta"] = []


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
    STATE["undo_stack"] = []
    STATE["redo_stack"] = []
    _ml_reset()

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
    STATE["undo_stack"] = []
    STATE["redo_stack"] = []
    _ml_reset()
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
    _snapshot()
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
    method = request.args.get("method", "pearson")
    df = STATE["original_df"]
    result = lib.correlation_analysis(df, method=method)
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
# Data preview
# --------------------------------------------------------------------------
@app.route("/api/preview")
def api_preview():
    err = require_data()
    if err:
        return err
    n = int(request.args.get("n", 10))
    mode = request.args.get("mode", "head")
    return jsonify(lib.data_preview(STATE["original_df"], n=n, mode=mode))


# --------------------------------------------------------------------------
# Automatic dataset profiling
# --------------------------------------------------------------------------
@app.route("/api/profile")
def api_profile():
    err = require_data()
    if err:
        return err
    return jsonify(lib.generate_profile(STATE["original_df"]))


# --------------------------------------------------------------------------
# Smart warnings
# --------------------------------------------------------------------------
@app.route("/api/warnings")
def api_warnings():
    err = require_data()
    if err:
        return err
    return jsonify({"warnings": lib.generate_warnings(STATE["original_df"], STATE["target_col"])})


# --------------------------------------------------------------------------
# Improved missing analysis
# --------------------------------------------------------------------------
@app.route("/api/missing/extra")
def api_missing_extra():
    err = require_data()
    if err:
        return err
    df = STATE["original_df"]
    return jsonify({
        "row_stats": lib.missing_row_stats(df),
        "nullity_correlation": lib.missing_correlation(df),
    })


# --------------------------------------------------------------------------
# Improved outlier analysis
# --------------------------------------------------------------------------
@app.route("/api/outliers/summary")
def api_outliers_summary():
    err = require_data()
    if err:
        return err
    df = STATE["original_df"]
    types = col_types(df)
    numeric_cols = [c for c, t in types.items() if t == "numeric"]
    return jsonify({"summary": lib.outliers_summary_all(df, numeric_cols)})


@app.route("/api/outliers/mad")
def api_outliers_mad():
    err = require_data()
    if err:
        return err
    col = request.args.get("col")
    df = STATE["original_df"]
    if col not in df.columns:
        return jsonify({"error": f"Unknown column {col}"}), 400
    return jsonify(lib.detect_outliers_mad(df, col))


# --------------------------------------------------------------------------
# Improved relationships
# --------------------------------------------------------------------------
@app.route("/api/relationships/summary")
def api_relationships_summary():
    err = require_data()
    if err:
        return err
    target = request.args.get("target") or STATE["target_col"]
    df = STATE["original_df"]
    if target not in df.columns:
        return jsonify({"error": f"Unknown column {target}"}), 400
    return jsonify({"target": target, "rows": lib.relationship_summary_all(df, target)})


@app.route("/api/relationship/categorical")
def api_relationship_categorical():
    err = require_data()
    if err:
        return err
    col = request.args.get("col")
    target = request.args.get("target") or STATE["target_col"]
    df = STATE["original_df"]
    if col not in df.columns or target not in df.columns:
        return jsonify({"error": "Unknown column"}), 400
    chart = lib.categorical_heatmap(df, col, target)
    v = lib.cramers_v(df, col, target)
    return jsonify({"chart": chart, "cramers_v": v})


# --------------------------------------------------------------------------
# Improved correlation
# --------------------------------------------------------------------------
@app.route("/api/correlation/target")
def api_correlation_target():
    err = require_data()
    if err:
        return err
    method = request.args.get("method", "pearson")
    target = request.args.get("target") or STATE["target_col"]
    df = STATE["original_df"]
    return jsonify({"target": target, "ranking": lib.target_correlation_ranking(df, target, method)})


# --------------------------------------------------------------------------
# Extended quality checks
# --------------------------------------------------------------------------
@app.route("/api/quality/extra")
def api_quality_extra():
    err = require_data()
    if err:
        return err
    return jsonify(lib.extended_quality_checks(STATE["original_df"]))


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
    _snapshot()
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
    _snapshot()
    col = request.json["column"]
    STATE["df"], removed = lib.drop_rows_with_missing(STATE["df"], col)
    _log({"op": "drop_rows_missing", "column": col, "removed": removed})
    return jsonify({"removed": removed, "rows_left": len(STATE["df"])})


@app.route("/api/prepare/drop_column", methods=["POST"])
def api_drop_column():
    err = require_data()
    if err:
        return err
    _snapshot()
    col = request.json["column"]
    STATE["df"] = STATE["df"].drop(columns=[col])
    _log({"op": "drop_column", "column": col})
    return jsonify({"columns": list(STATE["df"].columns)})


@app.route("/api/prepare/extract_datetime", methods=["POST"])
def api_extract_datetime():
    err = require_data()
    if err:
        return err
    _snapshot()
    col = request.json["column"]
    STATE["df"] = lib.engineer_datetime_features(STATE["df"], col)
    _log({"op": "extract_datetime_features", "column": col})
    return jsonify({"columns": list(STATE["df"].columns)})


@app.route("/api/prepare/extract_text", methods=["POST"])
def api_extract_text():
    err = require_data()
    if err:
        return err
    _snapshot()
    col = request.json["column"]
    STATE["df"] = lib.engineer_text_features(STATE["df"], col)
    _log({"op": "extract_text_features", "column": col})
    return jsonify({"columns": list(STATE["df"].columns)})


@app.route("/api/prepare/bool_to_int", methods=["POST"])
def api_bool_to_int():
    err = require_data()
    if err:
        return err
    _snapshot()
    col = request.json["column"]
    STATE["df"][col] = STATE["df"][col].astype("boolean").astype("Int64")
    _log({"op": "boolean_to_int", "column": col})
    return jsonify({"ok": True})


@app.route("/api/prepare/encode", methods=["POST"])
def api_encode():
    err = require_data()
    if err:
        return err
    _snapshot()
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
    _snapshot()
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
    _snapshot()
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
    _ml_reset()
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
    STATE["undo_stack"] = []
    STATE["redo_stack"] = []
    _ml_reset()
    return jsonify({"columns": list(STATE["df"].columns)})


# --------------------------------------------------------------------------
# Undo / redo
# --------------------------------------------------------------------------
@app.route("/api/prepare/undo", methods=["POST"])
def api_undo():
    err = require_data()
    if err:
        return err
    if not STATE["undo_stack"]:
        return jsonify({"error": "Nothing to undo."}), 400
    STATE["redo_stack"].append(STATE["df"].copy())
    STATE["df"] = STATE["undo_stack"].pop()
    if STATE["pipeline_log"]:
        STATE["pipeline_log"].append({"op": "undo"})
    return jsonify({"columns": list(STATE["df"].columns), "rows": len(STATE["df"]),
                     "can_undo": len(STATE["undo_stack"]) > 0, "can_redo": True})


@app.route("/api/prepare/redo", methods=["POST"])
def api_redo():
    err = require_data()
    if err:
        return err
    if not STATE["redo_stack"]:
        return jsonify({"error": "Nothing to redo."}), 400
    STATE["undo_stack"].append(STATE["df"].copy())
    STATE["df"] = STATE["redo_stack"].pop()
    STATE["pipeline_log"].append({"op": "redo"})
    return jsonify({"columns": list(STATE["df"].columns), "rows": len(STATE["df"]),
                     "can_undo": True, "can_redo": len(STATE["redo_stack"]) > 0})


@app.route("/api/prepare/history")
def api_history():
    err = require_data()
    if err:
        return err
    return jsonify({
        "log": STATE["pipeline_log"],
        "can_undo": len(STATE["undo_stack"]) > 0,
        "can_redo": len(STATE["redo_stack"]) > 0,
    })


@app.route("/api/prepare/history/clear", methods=["POST"])
def api_history_clear():
    err = require_data()
    if err:
        return err
    STATE["pipeline_log"] = []
    return jsonify({"ok": True})


# --------------------------------------------------------------------------
# Before / after comparison
# --------------------------------------------------------------------------
@app.route("/api/prepare/compare")
def api_compare():
    err = require_data()
    if err:
        return err
    return jsonify(lib.compare_before_after(STATE["original_df"], STATE["df"]))


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
@app.route("/api/prepare/feature/interaction", methods=["POST"])
def api_feature_interaction():
    err = require_data()
    if err:
        return err
    _snapshot()
    body = request.json
    STATE["df"], name = lib.create_interaction_feature(STATE["df"], body["col1"], body["col2"], body.get("op", "multiply"))
    _log({"op": "interaction_feature", "col1": body["col1"], "col2": body["col2"], "operator": body.get("op", "multiply"), "new_column": name})
    return jsonify({"new_column": name, "columns": list(STATE["df"].columns)})


@app.route("/api/prepare/feature/polynomial", methods=["POST"])
def api_feature_polynomial():
    err = require_data()
    if err:
        return err
    _snapshot()
    body = request.json
    degree = int(body.get("degree", 2))
    STATE["df"], name = lib.create_polynomial_feature(STATE["df"], body["column"], degree)
    _log({"op": "polynomial_feature", "column": body["column"], "degree": degree, "new_column": name})
    return jsonify({"new_column": name, "columns": list(STATE["df"].columns)})


@app.route("/api/prepare/feature/bin", methods=["POST"])
def api_feature_bin():
    err = require_data()
    if err:
        return err
    _snapshot()
    body = request.json
    n_bins = int(body.get("bins", 4))
    strategy = body.get("strategy", "quantile")
    try:
        STATE["df"], name = lib.create_binned_feature(STATE["df"], body["column"], n_bins, strategy)
    except Exception as e:
        STATE["undo_stack"].pop()
        return jsonify({"error": str(e)}), 400
    _log({"op": "binned_feature", "column": body["column"], "bins": n_bins, "strategy": strategy, "new_column": name})
    return jsonify({"new_column": name, "columns": list(STATE["df"].columns)})


# --------------------------------------------------------------------------
# Feature selection
# --------------------------------------------------------------------------
@app.route("/api/prepare/feature_selection/variance")
def api_fs_variance():
    err = require_data()
    if err:
        return err
    threshold = float(request.args.get("threshold", 0.0))
    types = col_types(STATE["df"])
    numeric_cols = [c for c, t in types.items() if t == "numeric"]
    keep, drop = lib.variance_threshold_select(STATE["df"], numeric_cols, threshold)
    return jsonify({"keep": keep, "drop": drop})


@app.route("/api/prepare/feature_selection/correlation")
def api_fs_correlation():
    err = require_data()
    if err:
        return err
    threshold = float(request.args.get("threshold", 0.95))
    types = col_types(STATE["df"])
    numeric_cols = [c for c, t in types.items() if t == "numeric"]
    keep, drop = lib.correlation_based_selection(STATE["df"], numeric_cols, threshold)
    return jsonify({"keep": keep, "drop": drop})


@app.route("/api/prepare/feature_selection/kbest")
def api_fs_kbest():
    err = require_data()
    if err:
        return err
    k = int(request.args.get("k", 10))
    target = STATE["target_col"]
    if target not in STATE["df"].columns:
        return jsonify({"error": "Target column not in working copy"}), 400
    return jsonify({"ranking": lib.select_k_best_report(STATE["df"], target, k)})


@app.route("/api/prepare/feature_selection/apply", methods=["POST"])
def api_fs_apply():
    err = require_data()
    if err:
        return err
    _snapshot()
    cols = request.json.get("drop_columns", [])
    cols = [c for c in cols if c in STATE["df"].columns]
    STATE["df"] = STATE["df"].drop(columns=cols)
    _log({"op": "feature_selection_drop", "columns": cols})
    return jsonify({"dropped": cols, "columns": list(STATE["df"].columns)})


# --------------------------------------------------------------------------
# Advanced encoding
# --------------------------------------------------------------------------
@app.route("/api/prepare/encode/advanced", methods=["POST"])
def api_encode_advanced():
    err = require_data()
    if err:
        return err
    _snapshot()
    body = request.json
    col, method = body["column"], body["method"]
    if method == "frequency":
        STATE["df"] = lib.frequency_encode(STATE["df"], col)
    elif method == "target":
        target = STATE["target_col"]
        if target not in STATE["df"].columns:
            STATE["undo_stack"].pop()
            return jsonify({"error": "Target column not in working copy"}), 400
        STATE["df"] = lib.target_encode(STATE["df"], col, target)
    elif method == "binary":
        STATE["df"], n_bits = lib.binary_encode(STATE["df"], col)
    else:
        STATE["undo_stack"].pop()
        return jsonify({"error": f"Unknown method {method}"}), 400
    _log({"op": f"{method}_encode", "column": col})
    return jsonify({"columns": list(STATE["df"].columns)})


# --------------------------------------------------------------------------
# Advanced scaling / transformation
# --------------------------------------------------------------------------
@app.route("/api/prepare/scale/advanced", methods=["POST"])
def api_scale_advanced():
    err = require_data()
    if err:
        return err
    _snapshot()
    body = request.json
    method = body.get("method", "power")
    types = col_types(STATE["df"])
    numeric_cols = [c for c, t in types.items() if t == "numeric"]
    if not numeric_cols:
        STATE["undo_stack"].pop()
        return jsonify({"error": "No numeric columns to scale"}), 400
    if method == "power":
        STATE["df"], _ = lib.power_transform_cols(STATE["df"], numeric_cols, method="yeo-johnson")
    elif method == "normalize":
        STATE["df"], _ = lib.normalize_rows(STATE["df"], numeric_cols)
    else:
        STATE["undo_stack"].pop()
        return jsonify({"error": f"Unknown method {method}"}), 400
    _log({"op": f"{method}_transform", "columns": numeric_cols})
    return jsonify({"columns": numeric_cols})


# --------------------------------------------------------------------------
# Better train / validation / test split
# --------------------------------------------------------------------------
@app.route("/api/prepare/split/time", methods=["POST"])
def api_split_time():
    err = require_data()
    if err:
        return err
    body = request.json
    order_col = body["order_column"]
    test_size = float(body.get("test_size", 0.2))
    val_size = float(body.get("val_size", 0.1))
    df = STATE["df"]
    if order_col not in df.columns:
        return jsonify({"error": f"Unknown column {order_col}"}), 400
    train, val, test = lib.time_based_split(df, order_col, test_size, val_size)
    target = STATE["target_col"]

    def split_xy(part):
        if part is None or target not in part.columns:
            return None, None
        return part.drop(columns=[target]), part[target]

    X_train, y_train = split_xy(train)
    X_val, y_val = split_xy(val)
    X_test, y_test = split_xy(test)
    STATE["split"] = (X_train, X_val, X_test, y_train, y_val, y_test)
    _ml_reset()
    _log({"op": "time_based_split", "order_column": order_col, "test_size": test_size, "val_size": val_size})
    return jsonify({"train": len(train), "val": 0 if val is None else len(val), "test": 0 if test is None else len(test)})


@app.route("/api/prepare/split/kfold")
def api_split_kfold():
    err = require_data()
    if err:
        return err
    n_splits = int(request.args.get("n_splits", 5))
    stratify = request.args.get("stratify", "false") == "true"
    df = STATE["df"]
    target = STATE["target_col"]
    stratify_col = target if (stratify and target in df.columns) else None
    try:
        folds = lib.kfold_indices(df, n_splits, stratify_col)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"folds": folds})


# --------------------------------------------------------------------------
# Baseline machine learning (Model tab)
#
# Trains real scikit-learn (and optionally XGBoost/LightGBM) models on the
# train/val/test split from the Prepare tab, evaluates them on the held-out
# split, and returns a comparison table plus detailed per-model evaluation
# (confusion matrix, ROC/PR curves, residuals, ...).
# --------------------------------------------------------------------------
def _current_task():
    """Classification vs. regression, inferred from the target column of the
    *working copy* the same way /api/prepare/split does."""
    target = STATE["target_col"]
    df = STATE["df"]
    if df is None or target not in df.columns:
        return None
    y = df[target]
    is_classification = (not pd.api.types.is_numeric_dtype(y)) or y.nunique() <= 20
    return "classification" if is_classification else "regression"


def _metric_info(task):
    return ml.CLASSIFICATION_METRIC_INFO if task == "classification" else ml.REGRESSION_METRIC_INFO


def _comparison_rows():
    rows = []
    for model_id, entry in STATE["ml_trained"].items():
        row = {"model_id": model_id, "label": entry["label"]}
        row.update(entry["result"]["metrics"])
        rows.append(row)
    return rows


@app.route("/api/model/task")
def api_model_task():
    err = require_data()
    if err:
        return err
    task = _current_task()
    if task is None:
        return jsonify({"error": "Target column not in working copy"}), 400
    return jsonify({"task": task, "target_col": STATE["target_col"], "has_split": STATE["split"] is not None})


@app.route("/api/model/list")
def api_model_list():
    err = require_data()
    if err:
        return err
    task = request.args.get("task") or _current_task()
    if task not in ("classification", "regression"):
        return jsonify({"error": "task must be 'classification' or 'regression'"}), 400
    return jsonify({
        "task": task,
        "models": ml.available_models(task),
        "optional_deps": ml.optional_deps_status(),
        "metric_info": _metric_info(task),
        "trained": [mid for mid, entry in STATE["ml_trained"].items()] if STATE.get("ml_task") == task else [],
    })


@app.route("/api/model/train", methods=["POST"])
def api_model_train():
    err = require_data()
    if err:
        return err
    if STATE["split"] is None:
        return jsonify({"error": "Split the dataset into train/val/test first, in the Prepare tab."}), 400

    body = request.json or {}
    model_ids = body.get("model_ids", [])
    if not model_ids:
        return jsonify({"error": "Choose at least one model to train."}), 400

    X_train, X_val, X_test, y_train, y_val, y_test = STATE["split"]
    task = "classification" if ((not pd.api.types.is_numeric_dtype(y_train)) or y_train.nunique() <= 20) else "regression"

    if STATE.get("ml_task") != task:
        _ml_reset()
    STATE["ml_task"] = task

    X_eval, y_eval, eval_name = ml.pick_eval_set(X_val, X_test, y_val, y_test)
    if X_eval is None:
        return jsonify({"error": "No validation or test split available. Split the data first."}), 400

    X_train_c, y_train_c = ml.drop_na_target(X_train, y_train)
    X_eval_c, y_eval_c = ml.drop_na_target(X_eval, y_eval)

    types = col_types(pd.concat([X_train_c, X_eval_c]))
    numeric_cols = [c for c, t in types.items() if t in ("numeric", "boolean") and c in X_train_c.columns]
    categorical_cols = [c for c, t in types.items() if t == "categorical" and c in X_train_c.columns]

    try:
        preprocessor, Xtr, Xev = ml.fit_transform_features(X_train_c, X_eval_c, numeric_cols, categorical_cols)
    except Exception as e:
        return jsonify({"error": f"Could not prepare features for modeling: {e}"}), 400
    STATE["ml_preprocessor"] = preprocessor

    ytr, yev, class_names, label_encoder = ml.prepare_targets(task, y_train_c, y_eval_c)
    STATE["ml_label_encoder"] = label_encoder
    STATE["ml_class_names"] = class_names
    STATE["ml_eval_name"] = eval_name
    STATE["ml_Xeval"] = Xev
    STATE["ml_yeval"] = yev
    STATE["ml_feature_cols"] = numeric_cols + categorical_cols
    STATE["ml_numeric_cols"] = numeric_cols
    STATE["ml_categorical_cols"] = categorical_cols
    STATE["ml_feature_meta"] = ml.feature_metadata(X_train_c, numeric_cols, categorical_cols)

    trained_ok = []
    errors = {}
    for model_id in model_ids:
        try:
            model, result = ml.train_and_evaluate(model_id, task, Xtr, ytr, Xev, yev, class_names)
            label = result.pop("label")
            STATE["ml_trained"][model_id] = {"model": model, "label": label, "result": result}
            trained_ok.append(model_id)
            STATE.setdefault("experiments", []).append({
                "id": uuid.uuid4().hex[:8],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": STATE.get("filename"),
                "target": STATE.get("target_col"),
                "task": task,
                "eval_split": eval_name,
                "model_id": model_id,
                "label": label,
                "metrics": result["metrics"],
            })
        except Exception as e:
            errors[model_id] = str(e)

    return jsonify({
        "task": task,
        "eval_split": eval_name,
        "trained": trained_ok,
        "comparison": _comparison_rows(),
        "metric_info": _metric_info(task),
        "errors": errors,
    })


@app.route("/api/model/compare")
def api_model_compare():
    err = require_data()
    if err:
        return err
    task = STATE.get("ml_task")
    if not task:
        return jsonify({"task": None, "comparison": []})
    return jsonify({
        "task": task,
        "eval_split": STATE.get("ml_eval_name"),
        "comparison": _comparison_rows(),
        "metric_info": _metric_info(task),
    })


@app.route("/api/model/result/<model_id>")
def api_model_result(model_id):
    err = require_data()
    if err:
        return err
    entry = STATE.get("ml_trained", {}).get(model_id)
    if entry is None:
        return jsonify({"error": "Train this model first."}), 400
    payload = {"model_id": model_id, "label": entry["label"], "task": STATE["ml_task"], "eval_split": STATE.get("ml_eval_name")}
    payload.update(entry["result"])
    return jsonify(payload)


@app.route("/api/model/feature_importance/<model_id>")
def api_model_feature_importance(model_id):
    err = require_data()
    if err:
        return err
    entry = STATE.get("ml_trained", {}).get(model_id)
    if entry is None:
        return jsonify({"error": "Train this model first."}), 400
    Xeval, yeval = STATE.get("ml_Xeval"), STATE.get("ml_yeval")
    if Xeval is None:
        return jsonify({"error": "No evaluation data available. Retrain the model."}), 400
    result = ml.compute_feature_importance(entry["model"], STATE["ml_preprocessor"], Xeval, yeval, STATE["ml_task"])
    if result is None:
        return jsonify({"error": "Feature importance isn't available for this model."}), 400
    return jsonify({"model_id": model_id, "label": entry["label"], **result})


@app.route("/api/model/predict_form/<model_id>")
def api_model_predict_form(model_id):
    err = require_data()
    if err:
        return err
    entry = STATE.get("ml_trained", {}).get(model_id)
    if entry is None:
        return jsonify({"error": "Train this model first."}), 400
    return jsonify({
        "model_id": model_id,
        "label": entry["label"],
        "task": STATE["ml_task"],
        "features": STATE.get("ml_feature_meta", []),
    })


@app.route("/api/model/predict/<model_id>", methods=["POST"])
def api_model_predict(model_id):
    err = require_data()
    if err:
        return err
    entry = STATE.get("ml_trained", {}).get(model_id)
    if entry is None:
        return jsonify({"error": "Train this model first."}), 400

    body = request.json or {}
    raw_values = body.get("values", {})
    feature_cols = STATE.get("ml_feature_cols", [])
    numeric_cols = set(STATE.get("ml_numeric_cols", []))

    row = {}
    for c in feature_cols:
        v = raw_values.get(c, None)
        if v == "":
            v = None
        if v is not None and c in numeric_cols:
            try:
                v = float(v)
            except (TypeError, ValueError):
                pass
        row[c] = v

    try:
        result = ml.predict_single(
            entry["model"], STATE["ml_preprocessor"], STATE.get("ml_label_encoder"),
            STATE["ml_task"], STATE.get("ml_class_names"), row, feature_cols,
        )
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 400

    return jsonify({"model_id": model_id, "label": entry["label"], **result})


@app.route("/api/model/download/<model_id>")
def api_model_download(model_id):
    entry = STATE.get("ml_trained", {}).get(model_id)
    if entry is None:
        return jsonify({"error": "Train this model first."}), 400
    buf = io.BytesIO()
    joblib.dump(
        {
            "preprocessor": STATE["ml_preprocessor"],
            "model": entry["model"],
            "label_encoder": STATE.get("ml_label_encoder"),
            "task": STATE["ml_task"],
            "target_col": STATE["target_col"],
        },
        buf,
    )
    buf.seek(0)
    return send_file(buf, mimetype="application/octet-stream",
                      as_attachment=True, download_name=f"model_{model_id}.joblib")


@app.route("/api/model/reset", methods=["POST"])
def api_model_reset():
    err = require_data()
    if err:
        return err
    _ml_reset()
    return jsonify({"ok": True})


# --------------------------------------------------------------------------
# Experiment tracking
# Every model trained via /api/model/train is logged here (timestamp,
# dataset, target, model, metrics), independent of ml_trained / ml_reset,
# so switching models or splits doesn't erase the run history.
# --------------------------------------------------------------------------
@app.route("/api/experiments")
def api_experiments():
    return jsonify({"experiments": STATE.get("experiments", [])})


@app.route("/api/experiments/clear", methods=["POST"])
def api_experiments_clear():
    STATE["experiments"] = []
    return jsonify({"ok": True})


@app.route("/api/experiments/download")
def api_experiments_download():
    exps = STATE.get("experiments", [])
    if not exps:
        return jsonify({"error": "No experiments logged yet."}), 400
    metric_keys = sorted({k for e in exps for k in e["metrics"].keys()})
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "timestamp", "dataset", "target", "task", "eval_split", "model_id", "label"] + metric_keys)
    for e in exps:
        writer.writerow(
            [e["id"], e["timestamp"], e["dataset"], e["target"], e["task"], e["eval_split"], e["model_id"], e["label"]]
            + [e["metrics"].get(k) for k in metric_keys]
        )
    mem = io.BytesIO(buf.getvalue().encode("utf-8"))
    return send_file(mem, mimetype="text/csv", as_attachment=True, download_name="experiments.csv")


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
