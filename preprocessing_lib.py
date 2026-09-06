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
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

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
