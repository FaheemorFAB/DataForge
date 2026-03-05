"""
Module: AutoML with FLAML
Pure Python — no Streamlit. Returns dicts/DataFrames for Flask API.
"""

import pandas as pd
import numpy as np
import pickle
import time


def _detect_task(y: pd.Series) -> str:
    if y.dtype == object or str(y.dtype) == "category":
        return "classification"
    n_unique = y.nunique()
    if n_unique <= 20 and n_unique / len(y) < 0.05:
        return "classification"
    return "regression"


def _encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode all non-numeric columns to numeric codes suitable for FLAML.

    FIX 1: select_dtypes now includes both 'object' AND 'category'.
            After pyjanitor cleaning, string columns become category dtype —
            the old code only matched 'object', so cleaned DataFrames arrived
            at FLAML with raw category columns, causing crashes.

    FIX 2: cat.codes represents NaN as -1 (an int), not as np.nan.
            The later fillna(median) never touched -1, poisoning every
            tree split with a spurious label.  We now cast to float and
            replace -1 with np.nan so the median imputation works correctly.
    """
    df = df.copy()

    # Drop datetime columns
    for col in df.select_dtypes(include=["datetime64", "datetimetz"]).columns:
        df.drop(columns=[col], inplace=True)

    # FIX 1 + FIX 2: encode both object AND category cols; preserve NaN
    for col in df.select_dtypes(include=["object", "category"]).columns:
        codes = df[col].astype("category").cat.codes.astype(float)  # float to allow NaN
        codes[codes == -1] = np.nan                                  # FIX 2: -1 → real NaN
        df[col] = codes

    return df


def _build_leaderboard(automl) -> list:
    """
    FIX: automl.model_history does not exist in FLAML — it was a fabricated
    attribute that always raised AttributeError (silently caught), leaving only
    the single-best-model fallback row in the output.

    The real FLAML API is:
      automl.best_loss_per_estimator   → {estimator_name: best_val_loss, ...}
      automl.best_config_per_estimator → {estimator_name: config_dict, ...}

    We build one row per estimator FLAML actually evaluated, mark the overall
    winner, and sort ascending by metric (lower loss = better).
    """
    rows = []

    try:
        loss_map   = automl.best_loss_per_estimator   # always present after fit()
        config_map = automl.best_config_per_estimator or {}
        best_name  = automl.best_estimator

        for estimator, loss in loss_map.items():
            try:
                metric_val = round(float(loss), 6)
            except (TypeError, ValueError):
                metric_val = None

            rows.append({
                "model":       estimator,
                "metric":      metric_val,
                "best_config": str(config_map.get(estimator, "")),
                "best":        estimator == best_name,
            })

        # Sort: best (lowest loss) first
        rows.sort(key=lambda r: (r["metric"] is None, r["metric"] or 0))

    except Exception:
        # Absolute fallback: at least return the single best model
        try:
            rows = [{
                "model":       automl.best_estimator,
                "metric":      round(float(automl.best_loss), 6),
                "best_config": str(automl.best_config),
                "best":        True,
            }]
        except Exception:
            pass

    return rows


def _compute_metrics(automl, X_test, y_test, task: str) -> dict:
    y_pred = automl.predict(X_test)
    metrics = {}
    if task == "classification":
        from sklearn.metrics import accuracy_score, f1_score
        metrics["Accuracy"]       = round(float(accuracy_score(y_test, y_pred)), 4)
        metrics["F1 (weighted)"]  = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
        try:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(y_test)) == 2:
                proba = automl.predict_proba(X_test)[:, 1]
                metrics["ROC-AUC"] = round(float(roc_auc_score(y_test, proba)), 4)
        except Exception:
            pass
    else:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        metrics["MAE"]  = round(float(mean_absolute_error(y_test, y_pred)), 4)
        metrics["RMSE"] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
        metrics["R²"]   = round(float(r2_score(y_test, y_pred)), 4)
    return metrics


def run_automl(
    df: pd.DataFrame,
    target_col: str,
    task_choice: str = "auto-detect",
    time_budget: int = 120,
    test_size: float = 0.2,
) -> dict:
    """
    Run FLAML AutoML.
    Returns dict with metrics, leaderboard, feature_importance, model_pkl_bytes, error.
    """
    try:
        from flaml import AutoML
    except ImportError:
        return {"error": "FLAML is not installed. Run: pip install flaml[default]"}

    from sklearn.model_selection import train_test_split

    auto_task = _detect_task(df[target_col])
    task      = auto_task if task_choice == "auto-detect" else task_choice

    df_model = _encode_features(df.drop(columns=[target_col]))
    y = df[target_col].copy()

    # FIX 1 also applies to the target: if it's still category, encode it
    if task == "classification" and (y.dtype == object or str(y.dtype) == "category"):
        y = y.astype("category").cat.codes

    valid_mask = y.notna()
    df_model   = df_model[valid_mask]
    y          = y[valid_mask]

    df_model = df_model.dropna(axis=1, how="all")
    # Now fillna works correctly because FIX 2 ensured -1 → np.nan above
    df_model = df_model.fillna(df_model.median(numeric_only=True))

    stratify = y if task == "classification" and y.nunique() < 50 else None
    X_train, X_test, y_train, y_test = train_test_split(
        df_model, y,
        test_size=test_size,
        random_state=42,
        shuffle=True,
        stratify=stratify,
    )

    automl = AutoML()
    start  = time.time()
    try:
        automl.fit(X_train, y_train, time_budget=time_budget, task=task,
                   log_type="all", verbose=0, seed=42)
    except Exception as e:
        return {"error": f"FLAML training failed: {str(e)}"}

    elapsed = round(time.time() - start, 1)
    metrics = _compute_metrics(automl, X_test, y_test, task)
    leaderboard = _build_leaderboard(automl)

    # Feature importance
    feature_importance = []
    try:
        model      = automl.model.estimator
        feat_names = list(df_model.columns)
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).flatten()

        if importances is not None and len(importances) == len(feat_names):
            fi_df = (
                pd.DataFrame({"feature": feat_names, "importance": importances})
                .sort_values("importance", ascending=False)
                .head(15)
            )
            feature_importance = fi_df.to_dict("records")
            for r in feature_importance:
                r["importance"] = round(float(r["importance"]), 6)
    except Exception:
        pass

    model_pkl = pickle.dumps(automl.model)

    return {
        "error":              None,
        "task":               task,
        "best_estimator":     automl.best_estimator,
        "best_loss":          round(float(automl.best_loss), 6),
        "best_config":        str(automl.best_config),
        "elapsed_s":          elapsed,
        "train_rows":         len(X_train),
        "test_rows":          len(X_test),
        "metrics":            metrics,
        "leaderboard":        leaderboard,
        "feature_importance": feature_importance,
        "model_pkl":          model_pkl,          # bytes — stored in session, not serialized to JSON
    }
