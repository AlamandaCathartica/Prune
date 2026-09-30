"""
ml_lib.py

Baseline machine learning. This module trains and evaluates classic
scikit-learn models (plus optional XGBoost / LightGBM, if installed) on the
train/validation/test split produced by the Prepare tab.

Like preprocessing_lib.py, this is a plain, importable module: backend.py
exposes it as a handful of JSON API routes and the HTML front end just
renders whatever comes back (including server-rendered matplotlib charts).
"""

import base64
import io

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless, server-side rendering
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

# Optional dependencies. Prune works fine without these; they just don't
# show up in the model list unless installed (`pip install xgboost` /
# `pip install lightgbm`).
try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False


# --------------------------------------------------------------------------
# Metric definitions, shown in the UI as tooltips/legends next to the
# comparison table so a number is never shown without an explanation.
# --------------------------------------------------------------------------
CLASSIFICATION_METRIC_INFO = {
    "accuracy": "Share of all predictions that were exactly correct.",
    "precision": "Of everything the model predicted positive, the share that really was positive. Low precision means false alarms.",
    "recall": "Of everything that really was positive, the share the model actually caught. Low recall means missed cases.",
    "f1": "Harmonic mean of precision and recall. A single number that balances both, useful when they trade off against each other.",
    "roc_auc": "Probability the model ranks a random positive example above a random negative one. 0.5 = no better than a coin flip, 1.0 = perfect separation.",
    "pr_auc": "Area under the precision-recall curve (a.k.a. average precision). More informative than ROC-AUC when the classes are imbalanced.",
}

REGRESSION_METRIC_INFO = {
    "mae": "Mean Absolute Error - the average absolute gap between predicted and actual values, in the target's own units.",
    "mse": "Mean Squared Error - the average squared gap between predicted and actual values. Squaring makes big misses count much more.",
    "rmse": "Root Mean Squared Error - the square root of MSE, which brings the units back to the target's original scale.",
    "r2": "R-squared (coefficient of determination) - the share of the target's variance the model explains. 1.0 = perfect fit, 0 = no better than always predicting the average, negative = worse than that.",
    "mape": "Mean Absolute Percentage Error - the average absolute error expressed as a percentage of the actual value. Undefined/unstable when actual values are near zero.",
}


# --------------------------------------------------------------------------
# Model registries
# --------------------------------------------------------------------------
def _clf_registry():
    reg = {
        "logistic_regression": ("Logistic Regression", lambda: LogisticRegression(max_iter=1000)),
        "decision_tree": ("Decision Tree", lambda: DecisionTreeClassifier(random_state=42)),
        "random_forest": ("Random Forest", lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        "extra_trees": ("Extra Trees", lambda: ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        "knn": ("KNN", lambda: KNeighborsClassifier(n_jobs=-1)),
        "svm": ("SVM", lambda: SVC(probability=True, random_state=42)),
        "naive_bayes": ("Naive Bayes", lambda: GaussianNB()),
        "gradient_boosting": ("Gradient Boosting", lambda: GradientBoostingClassifier(random_state=42)),
        "hist_gradient_boosting": ("HistGradientBoosting", lambda: HistGradientBoostingClassifier(random_state=42)),
    }
    if _HAS_XGB:
        reg["xgboost"] = ("XGBoost", lambda: XGBClassifier(eval_metric="logloss", random_state=42))
    if _HAS_LGBM:
        reg["lightgbm"] = ("LightGBM", lambda: LGBMClassifier(random_state=42, verbosity=-1))
    return reg


def _reg_registry():
    reg = {
        "linear_regression": ("Linear Regression", lambda: LinearRegression()),
        "ridge": ("Ridge", lambda: Ridge(random_state=42)),
        "lasso": ("Lasso", lambda: Lasso(random_state=42)),
        "elastic_net": ("ElasticNet", lambda: ElasticNet(random_state=42)),
        "decision_tree": ("Decision Tree", lambda: DecisionTreeRegressor(random_state=42)),
        "random_forest": ("Random Forest", lambda: RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        "extra_trees": ("Extra Trees", lambda: ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        "gradient_boosting": ("Gradient Boosting", lambda: GradientBoostingRegressor(random_state=42)),
        "hist_gradient_boosting": ("HistGradientBoosting", lambda: HistGradientBoostingRegressor(random_state=42)),
        "svr": ("SVR", lambda: SVR()),
        "knn": ("KNN Regression", lambda: KNeighborsRegressor(n_jobs=-1)),
    }
    if _HAS_XGB:
        reg["xgboost"] = ("XGBoost", lambda: XGBRegressor(random_state=42))
    if _HAS_LGBM:
        reg["lightgbm"] = ("LightGBM", lambda: LGBMRegressor(random_state=42, verbosity=-1))
    return reg


def available_models(task):
    """List of {id, label} model options for the given task, in a stable
    (insertion) order, for the front end's checkbox list."""
    reg = _clf_registry() if task == "classification" else _reg_registry()
    return [{"id": model_id, "label": label} for model_id, (label, _factory) in reg.items()]


def optional_deps_status():
    return {"xgboost": _HAS_XGB, "lightgbm": _HAS_LGBM}


# --------------------------------------------------------------------------
# Feature / target preparation
# --------------------------------------------------------------------------
def pick_eval_set(X_val, X_test, y_val, y_test):
    """Prefer the held-out test split for evaluation; fall back to the
    validation split if no test split was carved out."""
    if X_test is not None and len(X_test) > 0:
        return X_test, y_test, "test"
    if X_val is not None and len(X_val) > 0:
        return X_val, y_val, "validation"
    return None, None, None


def drop_na_target(X, y):
    """Drop rows where the target is missing, keeping X/y aligned."""
    if y is None:
        return X, y
    mask = y.notna()
    return X.loc[mask], y.loc[mask]


def _build_preprocessor(numeric_cols, categorical_cols):
    """A dense (never sparse) impute+scale / impute+one-hot ColumnTransformer.
    Dense output is required because some baseline models (Naive Bayes, KNN
    with certain metrics) don't accept sparse matrices."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))
    return ColumnTransformer(transformers, remainder="drop")


def fit_transform_features(X_train, X_eval, numeric_cols, categorical_cols):
    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)
    preprocessor.fit(X_train)
    Xtr = np.asarray(preprocessor.transform(X_train))
    Xev = np.asarray(preprocessor.transform(X_eval))
    return preprocessor, Xtr, Xev


def prepare_targets(task, y_train, y_eval):
    """Returns (y_train_encoded, y_eval_encoded, class_names, label_encoder).
    class_names / label_encoder are None for regression."""
    if task == "regression":
        return y_train.astype(float).values, y_eval.astype(float).values, None, None

    if pd.api.types.is_numeric_dtype(y_train):
        classes = sorted(pd.concat([y_train, y_eval]).unique().tolist())
        class_names = [str(c) for c in classes]
        return y_train.values, y_eval.values, class_names, None

    le = LabelEncoder()
    le.fit(pd.concat([y_train, y_eval]).astype(str))
    y_train_enc = le.transform(y_train.astype(str))
    y_eval_enc = le.transform(y_eval.astype(str))
    return y_train_enc, y_eval_enc, list(le.classes_), le


# --------------------------------------------------------------------------
# Training + evaluation
# --------------------------------------------------------------------------
def train_and_evaluate(model_id, task, Xtr, ytr, Xeval, yeval, class_names=None):
    registry = _clf_registry() if task == "classification" else _reg_registry()
    if model_id not in registry:
        raise ValueError(f"Unknown model '{model_id}' for task '{task}'")
    label, factory = registry[model_id]
    model = factory()
    model.fit(Xtr, ytr)
    if task == "classification":
        result = _evaluate_classification(model, Xeval, yeval, class_names)
    else:
        result = _evaluate_regression(model, Xeval, yeval)
    result["label"] = label
    return model, result


def _evaluate_classification(model, X, y, class_names=None):
    y_pred = model.predict(X)
    classes_present = np.unique(np.concatenate([np.asarray(y), np.asarray(y_pred)]))
    n_classes = len(classes_present)
    average = "binary" if n_classes <= 2 else "macro"

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y, y_pred, average=average, zero_division=0)),
        "roc_auc": None,
        "pr_auc": None,
    }

    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X)
        except Exception:
            y_proba = None
    elif hasattr(model, "decision_function"):
        try:
            y_proba = model.decision_function(X)
        except Exception:
            y_proba = None

    roc_chart = None
    pr_chart = None
    if y_proba is not None:
        try:
            if n_classes <= 2:
                scores = y_proba[:, 1] if getattr(y_proba, "ndim", 1) == 2 else y_proba
                metrics["roc_auc"] = float(roc_auc_score(y, scores))
                metrics["pr_auc"] = float(average_precision_score(y, scores))
                fpr, tpr, _ = roc_curve(y, scores)
                prec, rec, _ = precision_recall_curve(y, scores)
                roc_chart = _plot_roc_curve(fpr, tpr, metrics["roc_auc"])
                pr_chart = _plot_pr_curve(rec, prec, metrics["pr_auc"])
            elif getattr(y_proba, "ndim", 1) == 2:
                metrics["roc_auc"] = float(roc_auc_score(y, y_proba, multi_class="ovr", average="macro"))
        except Exception:
            pass

    cm = confusion_matrix(y, y_pred, labels=classes_present)
    # class_names lines up with classes_present by position only when the
    # classes are the contiguous ints 0..n-1 (i.e. a LabelEncoder was used
    # upstream in prepare_targets). Otherwise fall back to the raw values
    # so we never mislabel a row/column.
    is_contiguous_ints = all(float(v) == i for i, v in enumerate(classes_present))
    if class_names and len(class_names) == len(classes_present) and is_contiguous_ints:
        labels = [class_names[i] for i in range(len(classes_present))]
    else:
        labels = [str(c) for c in classes_present]

    report = classification_report(y, y_pred, labels=classes_present, target_names=labels, output_dict=True, zero_division=0)
    cm_chart = _plot_confusion_matrix(cm, labels)

    return {
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": labels,
        "confusion_matrix_chart": cm_chart,
        "classification_report": report,
        "roc_chart": roc_chart,
        "pr_chart": pr_chart,
    }


def feature_metadata(X, numeric_cols, categorical_cols):
    """Describes each input feature so the front end can build a prediction
    form: range/mean for numeric columns, option list for categorical ones."""
    meta = []
    for c in numeric_cols:
        s = pd.to_numeric(X[c], errors="coerce").dropna()
        meta.append({
            "name": c, "type": "numeric",
            "min": float(s.min()) if len(s) else None,
            "max": float(s.max()) if len(s) else None,
            "mean": float(s.mean()) if len(s) else None,
        })
    for c in categorical_cols:
        vals = sorted(X[c].dropna().astype(str).unique().tolist())[:50]
        meta.append({"name": c, "type": "categorical", "options": vals})
    return meta


def get_feature_names(preprocessor):
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return None


def compute_feature_importance(model, preprocessor, Xeval, yeval, task, top_n=20):
    """Ranks transformed (post-encoding) features by importance, using the
    model's own attributes when available and falling back to permutation
    importance (which works for any fitted estimator, at the cost of
    re-scoring the model several times)."""
    names = get_feature_names(preprocessor)
    importances = None
    method = None

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
        method = "Built-in feature importance (impurity-based)"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
        method = "Absolute coefficient magnitude"
    else:
        try:
            scoring = "accuracy" if task == "classification" else "r2"
            r = permutation_importance(model, Xeval, yeval, n_repeats=8, random_state=42, scoring=scoring, n_jobs=-1)
            importances = r.importances_mean
            method = "Permutation importance (drop in score when a feature is shuffled)"
        except Exception:
            return None

    if importances is None or len(importances) == 0:
        return None
    if names is None or len(names) != len(importances):
        names = [f"feature_{i}" for i in range(len(importances))]

    order = np.argsort(importances)[::-1][:top_n]
    top_names = [names[i] for i in order]
    top_vals = [float(importances[i]) for i in order]

    return {
        "method": method,
        "features": [{"feature": n, "importance": v} for n, v in zip(top_names, top_vals)],
        "chart": _plot_feature_importance(top_names, top_vals, method),
    }


def predict_single(model, preprocessor, label_encoder, task, class_names, input_row, feature_cols):
    """Runs one raw (un-preprocessed) row of feature values through the
    fitted preprocessor + model and returns a JSON-safe prediction, plus
    class probabilities for classifiers that support predict_proba."""
    row = {c: input_row.get(c) for c in feature_cols}
    X = pd.DataFrame([row], columns=feature_cols)
    Xt = np.asarray(preprocessor.transform(X))

    pred = model.predict(Xt)[0]
    result = {}

    if task == "classification":
        label = label_encoder.inverse_transform([int(pred)])[0] if label_encoder is not None else pred
        if isinstance(label, np.generic):
            label = label.item()
        result["prediction"] = label
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(Xt)[0]
                classes = class_names if class_names and len(class_names) == len(proba) else [str(i) for i in range(len(proba))]
                result["probabilities"] = {str(c): float(p) for c, p in zip(classes, proba)}
            except Exception:
                pass
    else:
        result["prediction"] = float(pred)

    return result


def _plot_feature_importance(names, values, method):
    fig, ax = plt.subplots(figsize=(6.2, max(3, 0.34 * len(names))))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color="#6750A4")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(method)
    ax.set_title("Feature importance")
    plt.tight_layout()
    return fig_to_base64(fig)


def _evaluate_regression(model, X, y):
    y_pred = model.predict(X)
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = float(mean_squared_error(y, y_pred))
    metrics = {
        "mae": float(mean_absolute_error(y, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y, y_pred)),
    }
    try:
        nonzero = y != 0
        metrics["mape"] = float(mean_absolute_percentage_error(y[nonzero], y_pred[nonzero]) * 100) if nonzero.any() else None
    except Exception:
        metrics["mape"] = None

    return {
        "metrics": metrics,
        "residual_chart": _plot_residuals(y, y_pred),
        "actual_vs_predicted_chart": _plot_actual_vs_predicted(y, y_pred),
        "error_distribution_chart": _plot_error_distribution(y, y_pred),
    }


# --------------------------------------------------------------------------
# Charts (matplotlib, server-rendered, same convention as preprocessing_lib)
# --------------------------------------------------------------------------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def _plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(cm, cmap="Purples")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    thresh = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                     color="white" if cm[i, j] > thresh else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    plt.tight_layout()
    return fig_to_base64(fig)


def _plot_roc_curve(fpr, tpr, auc):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(fpr, tpr, color="#6750A4", linewidth=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#CAC4D0")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    return fig_to_base64(fig)


def _plot_pr_curve(recall, precision, ap):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(recall, precision, color="#6750A4", linewidth=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    return fig_to_base64(fig)


def _plot_residuals(y_true, y_pred):
    resid = y_true - y_pred
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.scatter(y_pred, resid, alpha=0.6, s=18, color="#6750A4")
    ax.axhline(0, color="#B3261E", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Residual plot")
    plt.tight_layout()
    return fig_to_base64(fig)


def _plot_actual_vs_predicted(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.scatter(y_true, y_pred, alpha=0.6, s=18, color="#6750A4")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "--", color="#B3261E")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs. predicted")
    plt.tight_layout()
    return fig_to_base64(fig)


def _plot_error_distribution(y_true, y_pred):
    resid = y_true - y_pred
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.hist(resid, bins=30, color="#6750A4", edgecolor="white")
    ax.axvline(0, color="#B3261E", linestyle="--")
    ax.set_xlabel("Prediction error (actual - predicted)")
    ax.set_ylabel("Count")
    ax.set_title("Prediction error distribution")
    plt.tight_layout()
    return fig_to_base64(fig)
