"""
preprocessing_lib.py

This module is a direct, importable port of the logic from
dataset_preprocessor.ipynb. The Flask backend (backend.py) calls these
functions so that the HTML front end is backed by the *real* pandas /
numpy / scikit-learn pipeline instead of a JavaScript re-implementation.
"""

import base64
import io

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.feature_selection import VarianceThreshold, f_classif, f_regression, mutual_info_classif, mutual_info_regression

import matplotlib
matplotlib.use("Agg")  # headless, server-side rendering
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------------------------------
def load_data(file_storage, filename):
    """Load a dataset from an uploaded file (csv / excel / json / parquet)."""
    name = filename.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file_storage)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(file_storage)
    elif name.endswith(".json"):
        df = pd.read_json(file_storage)
    elif name.endswith(".parquet"):
        df = pd.read_parquet(file_storage)
    else:
        raise ValueError(f"Unsupported file type: {filename}")
    return df


def load_data_sql(connection_string, query):
    import sqlalchemy

    engine = sqlalchemy.create_engine(connection_string)
    return pd.read_sql(query, engine)


# --------------------------------------------------------------------------
# 2 & 3. Understand the data / check data types
# --------------------------------------------------------------------------
def classify_columns(df, text_unique_ratio=0.5, text_avg_len=30):
    numeric_cols, categorical_cols, boolean_cols, datetime_cols, text_cols = [], [], [], [], []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            boolean_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(s):
            datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(col)
        else:
            non_null = s.dropna().astype(str)
            if len(non_null) == 0:
                categorical_cols.append(col)
                continue
            unique_ratio = non_null.nunique() / len(non_null)
            avg_len = non_null.str.len().mean()
            if unique_ratio > text_unique_ratio and avg_len > text_avg_len:
                text_cols.append(col)
            else:
                categorical_cols.append(col)
    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "boolean": boolean_cols,
        "datetime": datetime_cols,
        "text": text_cols,
    }


def get_column_types(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c not in datetime_cols]
    return numeric_cols, categorical_cols, datetime_cols


def dataset_overview(df):
    n_rows, n_cols = df.shape
    n_dupes = int(df.duplicated().sum())
    numeric_cols, categorical_cols, datetime_cols = get_column_types(df)
    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null": df.notnull().sum(),
            "missing": df.isnull().sum(),
            "missing_pct": (df.isnull().mean() * 100).round(2),
            "unique": df.nunique(),
        }
    )
    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_dupes": n_dupes,
        "n_numeric": len(numeric_cols),
        "n_categorical": len(categorical_cols),
        "n_datetime": len(datetime_cols),
        "summary": summary.reset_index().rename(columns={"index": "column"}).to_dict(orient="records"),
    }


# --------------------------------------------------------------------------
# 4. Check missing values
# --------------------------------------------------------------------------
def missing_data_report(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        return {"report": [], "chart": None}
    report = pd.DataFrame(
        {"missing_count": missing, "missing_pct": (missing / len(df) * 100).round(2)}
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, max(3, 0.4 * len(missing))))
    report["missing_pct"].sort_values().plot(kind="barh", ax=axes[0], color="#6750A4")
    axes[0].set_xlabel("% missing")
    axes[0].set_title("Missing values by column")
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="mako", ax=axes[1])
    axes[1].set_title("Missing value map")
    plt.tight_layout()
    chart = fig_to_base64(fig)

    return {
        "report": report.reset_index().rename(columns={"index": "column"}).to_dict(orient="records"),
        "chart": chart,
    }


# --------------------------------------------------------------------------
# 5. Check duplicate data
# --------------------------------------------------------------------------
def remove_duplicates(df):
    before = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    return cleaned, before - len(cleaned)


# --------------------------------------------------------------------------
# 6. Check invalid / inconsistent values
# --------------------------------------------------------------------------
def detect_invalid_values(df, numeric_cols=None, categorical_cols=None):
    numeric_cols = numeric_cols or df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = categorical_cols or [c for c in df.columns if c not in numeric_cols]
    issues = []
    for col in numeric_cols:
        n_inf = int(np.isinf(df[col]).sum())
        n_neg = int((df[col] < 0).sum())
        if n_inf > 0:
            issues.append({"column": col, "issue": "infinite values", "count": n_inf})
        if n_neg > 0:
            issues.append({"column": col, "issue": "negative values", "count": n_neg})
    for col in categorical_cols:
        s = df[col].dropna().astype(str)
        n_blank = int((s.str.strip() == "").sum())
        n_whitespace_variants = int(s.nunique() - s.str.strip().str.lower().nunique())
        if n_blank > 0:
            issues.append({"column": col, "issue": "blank/whitespace-only strings", "count": n_blank})
        if n_whitespace_variants > 0:
            issues.append({"column": col, "issue": "casing/whitespace inconsistencies", "count": n_whitespace_variants})
    return issues


def categorical_quality_report(df, categorical_cols=None, rare_threshold=0.01):
    categorical_cols = categorical_cols or df.select_dtypes(exclude=[np.number]).columns.tolist()
    rows = []
    label_issues = {}
    for col in categorical_cols:
        s = df[col].dropna().astype(str)
        if s.empty:
            continue
        freq = s.value_counts(normalize=True)
        rare = freq[freq < rare_threshold]
        rows.append(
            {
                "column": col,
                "n_categories": int(s.nunique()),
                "top_category": str(freq.index[0]),
                "top_pct": round(freq.iloc[0] * 100, 2),
                "n_rare_categories": int(len(rare)),
            }
        )
        lowered = s.str.strip().str.lower()
        groups = {}
        for orig, low in zip(s, lowered):
            groups.setdefault(low, set()).add(orig)
        inconsistent = {k: v for k, v in groups.items() if len(v) > 1}
        if inconsistent:
            label_issues[col] = {k: list(v) for k, v in inconsistent.items()}
    return rows, label_issues


# --------------------------------------------------------------------------
# 7. Analyze numerical features / distributions
# --------------------------------------------------------------------------
def numerical_skewness(df, numeric_cols=None):
    numeric_cols = numeric_cols or df.select_dtypes(include=[np.number]).columns.tolist()
    rows = []
    for col in numeric_cols:
        vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        sk = stats.skew(vals) if len(vals) > 2 else np.nan
        label = skew_label(sk)
        rows.append(
            {
                "column": col,
                "skewness": None if pd.isna(sk) else round(float(sk), 3),
                "interpretation": label,
            }
        )
    rows.sort(key=lambda r: abs(r["skewness"]) if r["skewness"] is not None else -1, reverse=True)
    return rows


def skew_label(sk):
    if sk is None or pd.isna(sk):
        return "n/a"
    if abs(sk) < 0.5:
        return "approximately symmetric"
    if abs(sk) < 1:
        return "moderately skewed"
    return "highly skewed"


def plot_distribution(df, col, col_type):
    if col_type == "numeric":
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#6750A4")
        ax.set_title(f"Distribution of {col}")
        stats_ = df[col].describe()
        ax.axvline(stats_["mean"], color="#B3261E", linestyle="--", linewidth=1, label="mean")
        ax.axvline(stats_["50%"], color="#146C2E", linestyle="--", linewidth=1, label="median")
        ax.legend()
        plt.tight_layout()
        return fig_to_base64(fig)
    else:
        top = df[col].value_counts().nlargest(20)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        top.sort_values().plot(kind="barh", ax=ax, color="#7D5260")
        ax.set_title(f"Distribution of {col}")
        plt.tight_layout()
        return fig_to_base64(fig)


# --------------------------------------------------------------------------
# 8. Check outliers
# --------------------------------------------------------------------------
def detect_outliers_iqr(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (df[col] < lower) | (df[col] > upper)
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.boxplot(x=df[col], ax=ax, color="#EADDFF")
    ax.set_title(col)
    plt.tight_layout()
    chart = fig_to_base64(fig)
    return {
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "n_outliers": int(mask.sum()),
        "outlier_pct": round(float(mask.mean() * 100), 2),
        "chart": chart,
    }


def detect_outliers_zscore(df, col, threshold=3.0):
    vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    if vals.std() == 0 or len(vals) < 2:
        return {"n_outliers": 0, "outlier_pct": 0.0}
    z = np.abs(stats.zscore(vals))
    n_out = int((z > threshold).sum())
    return {"n_outliers": n_out, "outlier_pct": round(n_out / len(vals) * 100, 2)}


# --------------------------------------------------------------------------
# 9. Analyze the target variable
# --------------------------------------------------------------------------
def analyze_target(df, target_col, task=None):
    s = df[target_col].dropna()
    is_numeric = pd.api.types.is_numeric_dtype(s)
    if task is None:
        task = "regression" if is_numeric and s.nunique() > 20 else "classification"

    if task == "classification":
        counts = s.value_counts()
        pct = (counts / counts.sum() * 100).round(2)
        report = [{"value": str(k), "count": int(v), "pct": float(pct[k])} for k, v in counts.items()]
        imbalance_ratio = float(counts.max() / counts.min())
        fig, ax = plt.subplots(figsize=(6, 3.5))
        counts.sort_values().plot(kind="barh", ax=ax, color="#6750A4")
        ax.set_title(f"Class distribution: {target_col}")
        plt.tight_layout()
        chart = fig_to_base64(fig)
        return {"task": task, "report": report, "imbalance_ratio": imbalance_ratio, "chart": chart}
    else:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.histplot(s, kde=True, ax=ax, color="#6750A4")
        ax.set_title(f"Distribution of target: {target_col}")
        plt.tight_layout()
        chart = fig_to_base64(fig)
        sk = stats.skew(s.replace([np.inf, -np.inf], np.nan).dropna())
        outliers = detect_outliers_iqr(df, target_col)
        return {"task": task, "skewness": float(sk), "chart": chart, "outliers": outliers}


# --------------------------------------------------------------------------
# 10. Relationships between variables
# --------------------------------------------------------------------------
def relationship_plot(df, col, target_col):
    numeric_cols, categorical_cols, _ = get_column_types(df)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    if target_col in numeric_cols and col in numeric_cols:
        sns.scatterplot(x=df[col], y=df[target_col], ax=ax, color="#6750A4", alpha=0.6)
        ax.set_title(f"{col} vs {target_col}")
    elif target_col in numeric_cols and col in categorical_cols:
        sns.boxplot(data=df, x=col, y=target_col, ax=ax, color="#EADDFF")
        ax.set_title(f"{target_col} by {col}")
        plt.xticks(rotation=45, ha="right")
    elif target_col in categorical_cols and col in numeric_cols:
        sns.boxplot(data=df, x=target_col, y=col, ax=ax, color="#EADDFF")
        ax.set_title(f"{col} by {target_col}")
        plt.xticks(rotation=45, ha="right")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Both columns are categorical", ha="center", va="center")
    plt.tight_layout()
    return fig_to_base64(fig)


# --------------------------------------------------------------------------
# 11. Correlation / redundant features / data leakage
# --------------------------------------------------------------------------
def correlation_analysis(df, method="pearson", threshold=0.7):
    numeric_cols, _, _ = get_column_types(df)
    if len(numeric_cols) < 2:
        return {"chart": None, "strong_pairs": [], "matrix": None, "columns": numeric_cols}
    corr = df[numeric_cols].replace([np.inf, -np.inf], np.nan).corr(method=method)
    fig, ax = plt.subplots(figsize=(0.6 * len(numeric_cols) + 3, 0.6 * len(numeric_cols) + 2))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdPu", center=0, ax=ax, square=True)
    ax.set_title(f"{method.title()} correlation matrix")
    plt.tight_layout()
    chart = fig_to_base64(fig)

    rows = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            c1, c2 = numeric_cols[i], numeric_cols[j]
            val = corr.loc[c1, c2]
            if pd.notna(val) and abs(val) >= threshold:
                rows.append({"feature_1": c1, "feature_2": c2, "correlation": float(val)})
    rows.sort(key=lambda r: abs(r["correlation"]), reverse=True)
    return {
        "chart": chart,
        "strong_pairs": rows,
        "matrix": corr.round(3).to_dict(),
        "columns": numeric_cols,
    }


def find_redundant_features(df, numeric_cols=None, corr_threshold=0.95):
    numeric_cols = list(dict.fromkeys(numeric_cols or df.select_dtypes(include=[np.number]).columns.tolist()))
    if len(numeric_cols) < 2:
        return [], []
    clean = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    corr = clean.corr().abs()
    rows = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            c1, c2 = numeric_cols[i], numeric_cols[j]
            val = corr.loc[c1, c2]
            if pd.notna(val) and val >= corr_threshold:
                rows.append({"feature_1": c1, "feature_2": c2, "correlation": float(val)})
    rows.sort(key=lambda r: r["correlation"], reverse=True)

    identical_cols = []
    cols = list(df.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            try:
                if df[c1].equals(df[c2]):
                    identical_cols.append([c1, c2])
            except Exception:
                pass
    return rows, identical_cols


def check_data_leakage(df, target_col, numeric_cols=None, corr_threshold=0.98, suspicious_keywords=None):
    suspicious_keywords = suspicious_keywords or ["target", "label", "outcome", "future", "after", "result", "post_"]
    numeric_cols = numeric_cols or df.select_dtypes(include=[np.number]).columns.tolist()
    flags = []
    if target_col in numeric_cols:
        others = [c for c in numeric_cols if c != target_col]
        if others:
            corr = df[others + [target_col]].replace([np.inf, -np.inf], np.nan).corr()[target_col].drop(target_col)
            suspicious_corr = corr[corr.abs() >= corr_threshold]
            for col, val in suspicious_corr.items():
                flags.append({"column": col, "reason": f"correlation {val:.3f} with target"})
    for col in df.columns:
        if col == target_col:
            continue
        if any(k in col.lower() for k in suspicious_keywords):
            flags.append({"column": col, "reason": "suspicious column name"})
    seen = set()
    deduped = []
    for f in flags:
        key = (f["column"], f["reason"])
        if key not in seen:
            seen.add(key)
            deduped.append(f)
    return deduped


# --------------------------------------------------------------------------
# 12. Feature engineering
# --------------------------------------------------------------------------
def engineer_datetime_features(df, col):
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    df[f"{col}_year"] = df[col].dt.year
    df[f"{col}_month"] = df[col].dt.month
    df[f"{col}_day"] = df[col].dt.day
    df[f"{col}_dayofweek"] = df[col].dt.dayofweek
    df[f"{col}_hour"] = df[col].dt.hour
    return df


def engineer_text_features(df, col):
    df = df.copy()
    s = df[col].astype(str)
    df[f"{col}_length"] = s.str.len()
    df[f"{col}_word_count"] = s.str.split().str.len()
    return df


def is_date_like_column(df, col, sample=200):
    s = df[col].dropna().astype(str).head(sample)
    if s.empty:
        return False
    parsed = pd.to_datetime(s, errors="coerce")
    return parsed.notna().mean() > 0.8


def is_text_like_column(df, col, unique_ratio=0.5, avg_len=30):
    s = df[col].dropna().astype(str)
    if s.empty:
        return False
    return (s.nunique() / len(s) > unique_ratio) and (s.str.len().mean() > avg_len)


# --------------------------------------------------------------------------
# 13. Encode categorical variables
# --------------------------------------------------------------------------
def encode_categorical(df, onehot_cols=None, ordinal_cols=None):
    df = df.copy()
    if onehot_cols:
        df = pd.get_dummies(df, columns=onehot_cols, drop_first=False)
    if ordinal_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[ordinal_cols] = enc.fit_transform(df[ordinal_cols].astype(str))
    return df


# --------------------------------------------------------------------------
# 14. Scale / transform numerical features
# --------------------------------------------------------------------------
def scale_numerical(df, numeric_cols, method="standard"):
    df = df.copy()
    scaler = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler()}[method]
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, scaler


def transform_skewed(df, numeric_cols=None, skew_threshold=1.0):
    numeric_cols = numeric_cols or df.select_dtypes(include=[np.number]).columns.tolist()
    df = df.copy()
    transformed = []
    for col in numeric_cols:
        vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        sk = stats.skew(vals) if len(vals) > 2 else 0
        if abs(sk) >= skew_threshold and (df[col].dropna() >= 0).all():
            df[col] = np.log1p(df[col])
            transformed.append(col)
    return df, transformed


# --------------------------------------------------------------------------
# 15. Split the data
# --------------------------------------------------------------------------
def split_data(df, target_col, test_size=0.2, val_size=0.1, stratify=False, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    strat = y if stratify else None
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, stratify=strat, random_state=random_state
    )
    if val_size > 0:
        rel_val = val_size / (test_size + val_size)
        strat2 = y_temp if stratify else None
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1 - rel_val, stratify=strat2, random_state=random_state
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
    return X_train, X_temp, None, y_train, y_temp, None


# --------------------------------------------------------------------------
# 16. Build the preprocessing pipeline
# --------------------------------------------------------------------------
def build_preprocessing_pipeline(numeric_cols, categorical_cols, scale_method="standard",
                                  numeric_impute="median", categorical_impute="most_frequent"):
    scaler = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler()}[scale_method]
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy=numeric_impute)), ("scaler", scaler)])
    categorical_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy=categorical_impute)), ("encoder", OneHotEncoder(handle_unknown="ignore"))]
    )
    preprocessor = ColumnTransformer(
        [("num", numeric_pipeline, numeric_cols), ("cat", categorical_pipeline, categorical_cols)]
    )
    return preprocessor


# --------------------------------------------------------------------------
# 17. Save processed dataset / pipeline
# --------------------------------------------------------------------------
def save_processed_data(df, path):
    df.to_csv(path, index=False)


def save_pipeline(pipeline, path):
    joblib.dump(pipeline, path)


# --------------------------------------------------------------------------
# Missing-value imputation actions (used by the "Prepare" tab)
# --------------------------------------------------------------------------
def fill_missing(df, col, strategy, constant=None):
    df = df.copy()
    if strategy == "mean":
        value = df[col].astype(float).mean()
    elif strategy == "median":
        value = df[col].astype(float).median()
    elif strategy == "mode":
        m = df[col].mode(dropna=True)
        value = m.iloc[0] if not m.empty else None
    elif strategy == "constant":
        value = constant
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    df[col] = df[col].fillna(value)
    return df, value


def drop_rows_with_missing(df, col):
    before = len(df)
    df = df[df[col].notna()].reset_index(drop=True)
    return df, before - len(df)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def df_column_types(df):
    """JSON-friendly per-column type classification (numeric/categorical/boolean/datetime/text)."""
    types = classify_columns(df)
    lookup = {}
    for kind, cols in types.items():
        for c in cols:
            lookup[c] = kind
    return lookup, types


# ==========================================================================
# PHASE 1 additions — richer EDA
# ==========================================================================

# --------------------------------------------------------------------------
# Data preview
# --------------------------------------------------------------------------
def data_preview(df, n=10, mode="head"):
    """Return a small, JSON-safe preview of the dataframe."""
    if mode == "tail":
        sub = df.tail(n)
    elif mode == "sample":
        sub = df.sample(min(n, len(df)), random_state=42) if len(df) else df
    else:
        sub = df.head(n)
    safe = sub.copy()
    for c in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[c]):
            safe[c] = safe[c].astype(str)
    safe = safe.where(pd.notnull(safe), None)
    return {
        "columns": list(df.columns),
        "rows": safe.to_dict(orient="records"),
        "n_rows_total": len(df),
        "n_cols_total": len(df.columns),
    }


# --------------------------------------------------------------------------
# Automatic dataset profiling
# --------------------------------------------------------------------------
def generate_profile(df):
    """A lightweight, self-contained profiling report (pandas-profiling-lite)."""
    types = classify_columns(df)
    lookup, _ = df_column_types(df)
    n_rows = len(df)
    columns = []
    for col in df.columns:
        s = df[col]
        kind = lookup.get(col, "categorical")
        entry = {
            "column": col,
            "type": kind,
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "unique": int(s.nunique(dropna=True)),
            "unique_pct": round(float(s.nunique(dropna=True) / n_rows * 100), 2) if n_rows else 0.0,
            "is_constant": bool(s.nunique(dropna=True) <= 1),
        }
        if kind == "numeric":
            vals = s.replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals):
                entry.update(
                    {
                        "mean": round(float(vals.mean()), 4),
                        "std": round(float(vals.std()), 4) if len(vals) > 1 else 0.0,
                        "min": round(float(vals.min()), 4),
                        "q25": round(float(vals.quantile(0.25)), 4),
                        "median": round(float(vals.median()), 4),
                        "q75": round(float(vals.quantile(0.75)), 4),
                        "max": round(float(vals.max()), 4),
                        "zeros_pct": round(float((vals == 0).mean() * 100), 2),
                        "skewness": round(float(stats.skew(vals)), 3) if len(vals) > 2 else None,
                        "kurtosis": round(float(stats.kurtosis(vals)), 3) if len(vals) > 2 else None,
                    }
                )
        elif kind in ("categorical", "boolean"):
            vc = s.dropna().astype(str).value_counts()
            if len(vc):
                entry["top_value"] = str(vc.index[0])
                entry["top_pct"] = round(float(vc.iloc[0] / len(s.dropna()) * 100), 2)
        columns.append(entry)

    n_dupes = int(df.duplicated().sum())
    total_cells = n_rows * len(df.columns)
    total_missing = int(df.isna().sum().sum())
    alerts = generate_warnings(df)
    return {
        "n_rows": n_rows,
        "n_cols": len(df.columns),
        "n_dupes": n_dupes,
        "dupes_pct": round(n_dupes / n_rows * 100, 2) if n_rows else 0.0,
        "total_missing": total_missing,
        "missing_pct": round(total_missing / total_cells * 100, 2) if total_cells else 0.0,
        "memory_kb": round(float(df.memory_usage(deep=True).sum() / 1024), 1),
        "type_counts": {k: len(v) for k, v in types.items()},
        "columns": columns,
        "n_alerts": len(alerts),
    }


# --------------------------------------------------------------------------
# Smart warnings — automatic red flags across the whole dataset
# --------------------------------------------------------------------------
def generate_warnings(df, target_col=None):
    warnings = []
    n_rows = len(df)
    lookup, types = df_column_types(df)

    n_dupes = int(df.duplicated().sum())
    if n_dupes > 0:
        warnings.append({"severity": "warning", "column": None, "type": "duplicates",
                          "message": f"{n_dupes} duplicate row(s) ({round(n_dupes/n_rows*100,1)}%)."})

    for col in df.columns:
        s = df[col]
        miss_pct = float(s.isna().mean() * 100)
        if miss_pct >= 50:
            warnings.append({"severity": "high", "column": col, "type": "missing",
                              "message": f"'{col}' is {round(miss_pct,1)}% missing."})
        elif miss_pct > 0:
            warnings.append({"severity": "info", "column": col, "type": "missing",
                              "message": f"'{col}' has {round(miss_pct,1)}% missing values."})

        nunique = s.nunique(dropna=True)
        if nunique <= 1:
            warnings.append({"severity": "high", "column": col, "type": "constant",
                              "message": f"'{col}' is constant (only {nunique} unique value) — little/no signal."})
        elif n_rows and nunique / n_rows > 0.95 and lookup.get(col) != "numeric":
            warnings.append({"severity": "warning", "column": col, "type": "high_cardinality",
                              "message": f"'{col}' looks ID-like ({nunique} unique values in {n_rows} rows)."})
        elif lookup.get(col) == "categorical" and nunique > 50:
            warnings.append({"severity": "info", "column": col, "type": "high_cardinality",
                              "message": f"'{col}' has high cardinality ({nunique} categories) — consider grouping rare ones."})

        if lookup.get(col) == "numeric":
            vals = s.replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals) > 2:
                sk = stats.skew(vals)
                if abs(sk) >= 1:
                    warnings.append({"severity": "info", "column": col, "type": "skew",
                                      "message": f"'{col}' is highly skewed (skew={round(float(sk),2)}) — consider a transform."})
            if np.isinf(s.replace([np.nan], 0)).sum() > 0:
                warnings.append({"severity": "warning", "column": col, "type": "infinite",
                                  "message": f"'{col}' contains infinite values."})

    numeric_cols = [c for c, t in lookup.items() if t == "numeric"]
    if len(numeric_cols) >= 2:
        redundant, identical = find_redundant_features(df, numeric_cols, corr_threshold=0.95)
        for pair in redundant[:5]:
            warnings.append({"severity": "warning", "column": None, "type": "redundant",
                              "message": f"'{pair['feature_1']}' and '{pair['feature_2']}' are highly correlated (r={round(pair['correlation'],2)})."})
        for c1, c2 in identical[:5]:
            warnings.append({"severity": "high", "column": None, "type": "identical",
                              "message": f"'{c1}' and '{c2}' are identical columns."})

    if target_col and target_col in df.columns:
        s = df[target_col].dropna()
        if lookup.get(target_col) != "numeric" or s.nunique() <= 20:
            counts = s.value_counts()
            if len(counts) >= 2:
                ratio = counts.max() / counts.min()
                if ratio >= 5:
                    warnings.append({"severity": "warning", "column": target_col, "type": "imbalance",
                                      "message": f"Target '{target_col}' is imbalanced (majority/minority ratio {round(float(ratio),1)}x)."})

    order = {"high": 0, "warning": 1, "info": 2}
    warnings.sort(key=lambda w: order.get(w["severity"], 3))
    return warnings


# --------------------------------------------------------------------------
# Improved quality checks
# --------------------------------------------------------------------------
def extended_quality_checks(df):
    lookup, _ = df_column_types(df)
    n_rows = len(df)
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    id_like_cols = [
        c for c in df.columns
        if n_rows and df[c].nunique(dropna=True) / n_rows > 0.95 and lookup.get(c) != "numeric"
    ]
    high_cardinality_cols = [
        c for c, t in lookup.items()
        if t == "categorical" and df[c].nunique(dropna=True) > 50
    ]
    mixed_type_cols = []
    for c in df.columns:
        if lookup.get(c) in ("categorical",):
            kinds = df[c].dropna().map(lambda v: type(v).__name__).unique()
            if len(kinds) > 1:
                mixed_type_cols.append({"column": c, "types": list(kinds)})
    single_use_cols = [c for c in df.columns if df[c].nunique(dropna=True) == n_rows and n_rows > 0]
    return {
        "constant_columns": constant_cols,
        "id_like_columns": id_like_cols,
        "high_cardinality_columns": high_cardinality_cols,
        "mixed_type_columns": mixed_type_cols,
        "all_unique_columns": single_use_cols,
    }


# --------------------------------------------------------------------------
# Improved missing-value analysis
# --------------------------------------------------------------------------
def missing_correlation(df):
    """Correlation between the *nullity* of columns — reveals whether values
    tend to go missing together (useful for guessing MCAR vs MAR/MNAR)."""
    miss_cols = [c for c in df.columns if df[c].isna().any()]
    if len(miss_cols) < 2:
        return {"columns": miss_cols, "matrix": None, "pairs": []}
    null_df = df[miss_cols].isna().astype(int)
    corr = null_df.corr()
    pairs = []
    for i in range(len(miss_cols)):
        for j in range(i + 1, len(miss_cols)):
            c1, c2 = miss_cols[i], miss_cols[j]
            val = corr.loc[c1, c2]
            if pd.notna(val) and abs(val) >= 0.5:
                pairs.append({"col_1": c1, "col_2": c2, "correlation": round(float(val), 3)})
    pairs.sort(key=lambda r: abs(r["correlation"]), reverse=True)
    return {"columns": miss_cols, "matrix": corr.round(3).to_dict(), "pairs": pairs}


def missing_row_stats(df):
    n_rows = len(df)
    per_row_missing = df.isna().sum(axis=1)
    rows_with_any = int((per_row_missing > 0).sum())
    rows_complete = n_rows - rows_with_any
    worst_pct = float(per_row_missing.max() / len(df.columns) * 100) if n_rows and len(df.columns) else 0.0
    return {
        "rows_with_any_missing": rows_with_any,
        "rows_complete": rows_complete,
        "rows_with_any_missing_pct": round(rows_with_any / n_rows * 100, 2) if n_rows else 0.0,
        "max_missing_pct_in_a_row": round(worst_pct, 2),
    }


# --------------------------------------------------------------------------
# Improved outlier analysis
# --------------------------------------------------------------------------
def detect_outliers_mad(df, col, threshold=3.5):
    """Modified Z-score using median absolute deviation — robust to skew,
    unlike a plain z-score."""
    vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    if len(vals) < 2:
        return {"n_outliers": 0, "outlier_pct": 0.0}
    median = vals.median()
    mad = (vals - median).abs().median()
    if mad == 0:
        return {"n_outliers": 0, "outlier_pct": 0.0}
    modified_z = 0.6745 * (vals - median) / mad
    mask = modified_z.abs() > threshold
    return {"n_outliers": int(mask.sum()), "outlier_pct": round(float(mask.mean() * 100), 2)}


def outliers_summary_all(df, numeric_cols=None):
    """One-row-per-column IQR + MAD outlier summary across every numeric column."""
    numeric_cols = numeric_cols or df.select_dtypes(include=[np.number]).columns.tolist()
    rows = []
    for col in numeric_cols:
        vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals) < 4:
            continue
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_mask = (vals < lower) | (vals > upper)
        mad_res = detect_outliers_mad(df, col)
        rows.append(
            {
                "column": col,
                "iqr_outliers": int(iqr_mask.sum()),
                "iqr_pct": round(float(iqr_mask.mean() * 100), 2),
                "mad_outliers": mad_res["n_outliers"],
                "mad_pct": mad_res["outlier_pct"],
            }
        )
    rows.sort(key=lambda r: r["iqr_pct"], reverse=True)
    return rows


# --------------------------------------------------------------------------
# Improved relationships — categorical vs categorical, and a summary table
# --------------------------------------------------------------------------
def cramers_v(df, col1, col2):
    confusion = pd.crosstab(df[col1], df[col2])
    if confusion.size == 0 or confusion.shape[0] < 2 or confusion.shape[1] < 2:
        return None
    chi2 = stats.chi2_contingency(confusion, correction=False)[0]
    n = confusion.sum().sum()
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1)) if n > 1 else 0
    rcorr = r - ((r - 1) ** 2) / (n - 1) if n > 1 else r
    kcorr = k - ((k - 1) ** 2) / (n - 1) if n > 1 else k
    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return None
    return float(np.sqrt(phi2corr / denom))


def categorical_heatmap(df, col1, col2):
    confusion = pd.crosstab(df[col1], df[col2])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="RdPu", ax=ax)
    ax.set_title(f"{col1} vs {col2}")
    plt.tight_layout()
    return fig_to_base64(fig)


def relationship_summary_all(df, target_col):
    """Association strength between every other column and the target:
    Pearson r for numeric-numeric, correlation ratio (eta) for numeric-categorical,
    Cramer's V for categorical-categorical."""
    numeric_cols, categorical_cols, _ = get_column_types(df)
    target_is_numeric = target_col in numeric_cols
    rows = []
    for col in df.columns:
        if col == target_col:
            continue
        try:
            if target_is_numeric and col in numeric_cols:
                val = df[[col, target_col]].replace([np.inf, -np.inf], np.nan).corr().iloc[0, 1]
                rows.append({"column": col, "metric": "pearson r", "value": None if pd.isna(val) else round(float(val), 3)})
            elif target_is_numeric and col not in numeric_cols:
                groups = [g[target_col].dropna().values for _, g in df.groupby(col, observed=True) if len(g) > 0]
                eta = correlation_ratio(df[col], df[target_col])
                rows.append({"column": col, "metric": "correlation ratio", "value": None if eta is None else round(eta, 3)})
            elif not target_is_numeric and col in numeric_cols:
                eta = correlation_ratio(df[target_col], df[col])
                rows.append({"column": col, "metric": "correlation ratio", "value": None if eta is None else round(eta, 3)})
            else:
                v = cramers_v(df, col, target_col)
                rows.append({"column": col, "metric": "Cramér's V", "value": None if v is None else round(v, 3)})
        except Exception:
            rows.append({"column": col, "metric": "n/a", "value": None})
    rows.sort(key=lambda r: (r["value"] is None, -(r["value"] or 0)))
    return rows


def correlation_ratio(categories, values):
    """Eta (correlation ratio) between a categorical column and a numeric column."""
    df_ = pd.DataFrame({"cat": categories, "val": values}).dropna()
    if df_["cat"].nunique() < 2 or len(df_) < 2:
        return None
    cat_means = df_.groupby("cat", observed=True)["val"].mean()
    overall_mean = df_["val"].mean()
    ss_between = sum(len(g) * (cat_means[k] - overall_mean) ** 2 for k, g in df_.groupby("cat", observed=True))
    ss_total = ((df_["val"] - overall_mean) ** 2).sum()
    if ss_total == 0:
        return 0.0
    return float(np.sqrt(ss_between / ss_total))


# --------------------------------------------------------------------------
# Improved correlation — multiple methods + target ranking
# --------------------------------------------------------------------------
def target_correlation_ranking(df, target_col, method="pearson"):
    numeric_cols, _, _ = get_column_types(df)
    if target_col not in numeric_cols:
        return []
    others = [c for c in numeric_cols if c != target_col]
    if not others:
        return []
    corr = df[others + [target_col]].replace([np.inf, -np.inf], np.nan).corr(method=method)[target_col].drop(target_col)
    rows = [{"column": c, "correlation": None if pd.isna(v) else round(float(v), 3)} for c, v in corr.items()]
    rows.sort(key=lambda r: abs(r["correlation"]) if r["correlation"] is not None else -1, reverse=True)
    return rows


# ==========================================================================
# PHASE 2 additions — more powerful preprocessing
# ==========================================================================

# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
def create_interaction_feature(df, col1, col2, op="multiply"):
    df = df.copy()
    name = f"{col1}_{op}_{col2}"
    a, b = df[col1].astype(float), df[col2].astype(float)
    if op == "multiply":
        df[name] = a * b
    elif op == "add":
        df[name] = a + b
    elif op == "subtract":
        df[name] = a - b
    elif op == "divide":
        df[name] = a / b.replace(0, np.nan)
    else:
        raise ValueError(f"Unknown op: {op}")
    return df, name


def create_polynomial_feature(df, col, degree=2):
    df = df.copy()
    name = f"{col}_pow{degree}"
    df[name] = df[col].astype(float) ** degree
    return df, name


def create_binned_feature(df, col, n_bins=4, strategy="quantile"):
    df = df.copy()
    name = f"{col}_binned"
    if strategy == "quantile":
        df[name] = pd.qcut(df[col], q=n_bins, duplicates="drop").astype(str)
    else:
        df[name] = pd.cut(df[col], bins=n_bins).astype(str)
    return df, name


def create_ratio_features_report(df, numeric_cols):
    """Just a helper to list sensible candidate pairs (both positive columns)."""
    candidates = []
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            if i == j:
                continue
            c1, c2 = numeric_cols[i], numeric_cols[j]
            if (df[c2].dropna() != 0).all():
                candidates.append((c1, c2))
    return candidates


# --------------------------------------------------------------------------
# Feature selection
# --------------------------------------------------------------------------
def variance_threshold_select(df, numeric_cols, threshold=0.0):
    if not numeric_cols:
        return [], []
    clean = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(df[numeric_cols].median(numeric_only=True))
    vt = VarianceThreshold(threshold=threshold)
    try:
        vt.fit(clean)
    except ValueError:
        return [], list(numeric_cols)
    keep = [c for c, k in zip(numeric_cols, vt.get_support()) if k]
    drop = [c for c in numeric_cols if c not in keep]
    return keep, drop


def correlation_based_selection(df, numeric_cols, threshold=0.95):
    """Greedy: for each pair above threshold, drop the second (keeps the first
    occurrence), same idea as a typical redundant-feature pruning step."""
    numeric_cols = list(numeric_cols)
    if len(numeric_cols) < 2:
        return numeric_cols, []
    corr = df[numeric_cols].replace([np.inf, -np.inf], np.nan).corr().abs()
    to_drop = set()
    for i in range(len(numeric_cols)):
        c1 = numeric_cols[i]
        if c1 in to_drop:
            continue
        for j in range(i + 1, len(numeric_cols)):
            c2 = numeric_cols[j]
            if c2 in to_drop:
                continue
            val = corr.loc[c1, c2]
            if pd.notna(val) and val >= threshold:
                to_drop.add(c2)
    keep = [c for c in numeric_cols if c not in to_drop]
    return keep, sorted(to_drop)


def select_k_best_report(df, target_col, k=10, task=None):
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
    if not numeric_cols:
        return []
    X = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(df[numeric_cols].median(numeric_only=True))
    y = df[target_col]
    mask = y.notna()
    X, y = X[mask], y[mask]
    if task is None:
        task = "classification" if (not pd.api.types.is_numeric_dtype(y)) or y.nunique() <= 20 else "regression"
    try:
        if task == "classification":
            y_enc = y.astype("category").cat.codes
            scores = mutual_info_classif(X, y_enc, random_state=42)
        else:
            scores = mutual_info_regression(X, y, random_state=42)
    except Exception:
        return []
    rows = [{"column": c, "score": round(float(s), 4)} for c, s in zip(numeric_cols, scores)]
    rows.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(rows):
        r["selected"] = i < k
    return rows


# --------------------------------------------------------------------------
# Advanced encoding
# --------------------------------------------------------------------------
def frequency_encode(df, col):
    df = df.copy()
    freq = df[col].value_counts(normalize=True)
    df[f"{col}_freq"] = df[col].map(freq).fillna(0.0)
    return df


def target_encode(df, col, target_col, smoothing=10.0):
    """Smoothed mean target encoding (a simplified, in-memory version of the
    classic Bayesian target encoder — global mean shrinkage for rare categories)."""
    df = df.copy()
    global_mean = df[target_col].astype(float).mean()
    stats_ = df.groupby(col, observed=True)[target_col].agg(["mean", "count"])
    smoothed = (stats_["mean"] * stats_["count"] + global_mean * smoothing) / (stats_["count"] + smoothing)
    df[f"{col}_target_enc"] = df[col].map(smoothed).fillna(global_mean)
    return df


def binary_encode(df, col):
    """Manual binary encoding: integer-code each category, then expand to
    base-2 digit columns (no extra dependency required)."""
    df = df.copy()
    codes = df[col].astype("category").cat.codes + 1  # 0 reserved for NaN
    max_code = int(codes.max()) if len(codes) else 0
    n_bits = max(1, int(np.ceil(np.log2(max_code + 1)))) if max_code > 0 else 1
    for b in range(n_bits):
        df[f"{col}_bin{b}"] = ((codes >> b) & 1).astype(int)
    return df, n_bits


# --------------------------------------------------------------------------
# Advanced scaling / transformation
# --------------------------------------------------------------------------
def power_transform_cols(df, numeric_cols, method="yeo-johnson"):
    df = df.copy()
    pt = PowerTransformer(method=method)
    df[numeric_cols] = pt.fit_transform(df[numeric_cols])
    return df, pt


def normalize_rows(df, numeric_cols, norm="l2"):
    df = df.copy()
    normalizer = Normalizer(norm=norm)
    df[numeric_cols] = normalizer.fit_transform(df[numeric_cols])
    return df, normalizer


# --------------------------------------------------------------------------
# Better train / validation / test split
# --------------------------------------------------------------------------
def time_based_split(df, order_col, test_size=0.2, val_size=0.1):
    """Chronological split: earliest rows go to train, latest to test — for
    time series / any dataset where a random split would leak the future."""
    df_sorted = df.sort_values(order_col).reset_index(drop=True)
    n = len(df_sorted)
    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    n_train = n - n_test - n_val
    train = df_sorted.iloc[:n_train]
    val = df_sorted.iloc[n_train:n_train + n_val] if n_val > 0 else None
    test = df_sorted.iloc[n_train + n_val:] if n_test > 0 else None
    return train, val, test


def kfold_indices(df, n_splits=5, stratify_col=None, random_state=42):
    from sklearn.model_selection import KFold, StratifiedKFold

    n = len(df)
    if stratify_col is not None:
        y = df[stratify_col]
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = list(skf.split(np.zeros(n), y))
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = list(kf.split(np.zeros(n)))
    return [{"fold": i + 1, "train_size": len(tr), "val_size": len(va)} for i, (tr, va) in enumerate(folds)]


# --------------------------------------------------------------------------
# Before / after comparison
# --------------------------------------------------------------------------
def compare_before_after(original_df, working_df):
    common_cols = [c for c in original_df.columns if c in working_df.columns]
    rows = []
    for col in common_cols:
        o, w = original_df[col], working_df[col]
        entry = {
            "column": col,
            "missing_before": int(o.isna().sum()),
            "missing_after": int(w.isna().sum()),
        }
        if pd.api.types.is_numeric_dtype(o) and pd.api.types.is_numeric_dtype(w):
            ov, wv = o.dropna(), w.dropna()
            entry["mean_before"] = round(float(ov.mean()), 4) if len(ov) else None
            entry["mean_after"] = round(float(wv.mean()), 4) if len(wv) else None
            entry["std_before"] = round(float(ov.std()), 4) if len(ov) > 1 else None
            entry["std_after"] = round(float(wv.std()), 4) if len(wv) > 1 else None
        rows.append(entry)
    new_cols = [c for c in working_df.columns if c not in original_df.columns]
    dropped_cols = [c for c in original_df.columns if c not in working_df.columns]
    return {
        "rows_before": len(original_df),
        "rows_after": len(working_df),
        "cols_before": len(original_df.columns),
        "cols_after": len(working_df.columns),
        "new_columns": new_cols,
        "dropped_columns": dropped_cols,
        "columns": rows,
    }
