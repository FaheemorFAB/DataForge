"""
Module: Data Cleaning
Pure Python — no Streamlit. Returns dicts/DataFrames for Flask API consumption.
"""

import pandas as pd
import numpy as np

try:
    import janitor
    JANITOR_OK = True
except ImportError:
    JANITOR_OK = False


def _infer_fill_strategy(series: pd.Series):
    """Return (strategy_label, filled_series)."""
    if series.dtype == object or str(series.dtype) == "category":
        mode_val = series.mode().iloc[0] if not series.mode().empty else "Unknown"
        return f"mode ('{mode_val}')", series.fillna(mode_val)
    else:
        median_val = series.median()
        mean_val   = series.mean()
        skew       = abs(series.skew()) if series.notna().sum() > 2 else 0
        if skew > 1:
            return f"median ({median_val:.4g})", series.fillna(median_val)
        else:
            return f"mean ({mean_val:.4g})", series.fillna(mean_val)


def auto_fix_missing(df: pd.DataFrame) -> tuple:
    """
    Impute missing values column-by-column.
    Returns (cleaned_df, log_of_changes as list[dict]).
    """
    df = df.copy()
    log = []
    for col in df.columns:
        n_missing = df[col].isnull().sum()
        if n_missing == 0:
            continue
        pct = n_missing / len(df) * 100
        if pct > 60:
            df.drop(columns=[col], inplace=True)
            log.append({
                "column": col,
                "missing": int(n_missing),
                "pct_missing": round(pct, 1),
                "action": "Dropped (>60% missing)",
                "type": "drop"
            })
            continue
        strategy, filled = _infer_fill_strategy(df[col])
        df[col] = filled
        log.append({
            "column": col,
            "missing": int(n_missing),
            "pct_missing": round(pct, 1),
            "action": f"Filled with {strategy}",
            "type": "fill"
        })
    return df, log


def structural_clean(df: pd.DataFrame) -> tuple:
    """
    Apply pyjanitor + structural fixes.
    Returns (cleaned_df, list[str] of actions).
    """
    actions = []
    original_cols = df.columns.tolist()

    if JANITOR_OK:
        df = df.janitor.clean_names(strip_underscores=True, case_type="snake")
    else:
        df.columns = (
            df.columns.str.strip()
              .str.lower()
              .str.replace(r"[^\w]", "_", regex=True)
              .str.replace(r"_+", "_", regex=True)
              .str.strip("_")
        )

    new_cols = df.columns.tolist()
    renamed = [(o, n) for o, n in zip(original_cols, new_cols) if o != n]
    if renamed:
        actions.append(f"Renamed {len(renamed)} column(s) to snake_case")

    n_dupes = df.duplicated().sum()
    if n_dupes:
        df = df.drop_duplicates()
        actions.append(f"Removed {int(n_dupes):,} duplicate row(s)")

    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()
    if len(str_cols):
        actions.append(f"Stripped whitespace from {len(str_cols)} string column(s)")

    empty_rows = df.isnull().all(axis=1).sum()
    if empty_rows:
        df = df[~df.isnull().all(axis=1)]
        actions.append(f"Removed {int(empty_rows)} fully-empty row(s)")

    for col in df.select_dtypes(include="object").columns:
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().mean() > 0.85:
            df[col] = coerced
            actions.append(f"Coerced '{col}' to numeric")

    for col in df.select_dtypes(include="object").columns:
        try:
            parsed = pd.to_datetime(df[col], infer_format=True, errors="coerce")
            if parsed.notna().mean() > 0.85:
                df[col] = parsed
                actions.append(f"Parsed '{col}' as datetime")
        except Exception:
            pass

    if not actions:
        actions.append("No structural issues found — data looks clean!")

    return df, actions


def run_cleaning_pipeline(df_raw: pd.DataFrame) -> dict:
    """
    Run full cleaning pipeline.
    Returns dict with cleaned df, stats, logs.
    """
    df_step1, missing_log = auto_fix_missing(df_raw)
    df_clean, struct_actions = structural_clean(df_step1)

    return {
        "df_clean": df_clean,
        "missing_log": missing_log,
        "struct_actions": struct_actions,
        "stats": {
            "original_rows": int(len(df_raw)),
            "cleaned_rows":  int(len(df_clean)),
            "original_cols": int(df_raw.shape[1]),
            "cleaned_cols":  int(df_clean.shape[1]),
            "rows_removed":  int(len(df_raw) - len(df_clean)),
            "cols_removed":  int(df_raw.shape[1] - df_clean.shape[1]),
        }
    }
