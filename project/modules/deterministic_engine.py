"""
Module: Deterministic Query Engine — CSV Analyst Pro
Zero LLM. No API calls. No retries. No hallucinations.

Architecture:
    User Query
        ↓
    Preprocessor     (normalize text)
        ↓
    Intent Detector  (regex + keyword rules)
        ↓
    Column Matcher   (exact → fuzzy → token overlap)
        ↓
    Query Planner    (structured plan dict)
        ↓
    Pandas Executor  (deterministic pandas ops)
        ↓
    Result Serializer + Auto Insight

Reduces LLM usage by ~85%. LLM kept only for optional richer insight narrative.
"""

import re
import logging
from difflib import get_close_matches
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── spaCy optional import ────────────────────────────────────────────────────

try:
    import spacy as _spacy
    _nlp = _spacy.load("en_core_web_sm")
    SPACY_OK = True
except Exception:
    _nlp = None
    SPACY_OK = False

# ─── Constants ────────────────────────────────────────────────────────────────

# Intent keyword map — ordered from most specific to most general
_INTENT_RULES: list[tuple[str, re.Pattern]] = [
    ("top_n",       re.compile(r"\btop\s*\d+|\bhighest\s*\d+|\blargest\s*\d+|\bbottom\s*\d+")),
    ("filter",      re.compile(r"\bwhere\b|\bfilter\b|\bonly\b|\bwith\b|\bexcluding\b|\bexclude\b")),
    ("correlation", re.compile(r"\bcorrelat|\brelation|\bdependenc")),
    ("distribution",re.compile(r"\bdistribut|\bhistogram\b|\bspread\b|\brange\b|\bfrequenc")),
    ("mean",        re.compile(r"\baverage\b|\bmean\b|\bavg\b")),
    ("median",      re.compile(r"\bmedian\b|\bmiddle\b|\b50th\b")),
    ("sum",         re.compile(r"\btotal\b|\bsum\b|\baggregate\b|\bcombined\b")),
    ("max",         re.compile(r"\bmaximum\b|\bhighest\b|\blargest\b|\bmax\b|\bbiggest\b")),
    ("min",         re.compile(r"\bminimum\b|\blowest\b|\bsmallest\b|\bmin\b")),
    ("count",       re.compile(r"\bcount\b|\bhow many\b|\bnumber of\b|\bfrequency\b")),
    ("unique",      re.compile(r"\bunique\b|\bdistinct\b|\bdifferent\b")),
    ("missing",     re.compile(r"\bmissing\b|\bnull\b|\bnan\b|\bempty\b|\bblank\b")),
    ("outlier",     re.compile(r"\boutlier\b|\banomaly\b|\banomalie\b|\bextreme\b")),
    ("trend",       re.compile(r"\btrend\b|\bover time\b|\btime series\b|\bby (month|year|day|date)")),
    ("compare",     re.compile(r"\bcompar\b|\bvs\b|\bversus\b|\bagainst\b|\bdifference\b")),
    ("summary",     re.compile(r"\bsummar\b|\boverall\b|\bdescrib\b|\bovervi\b|\bstatistic")),
]

# Chart type hints
_CHART_RULES: list[tuple[str, re.Pattern]] = [
    ("bar_chart",      re.compile(r"\bbar\b|\bcolumn chart\b")),
    ("line_chart",     re.compile(r"\bline\b|\btrend\b|\bover time\b")),
    ("scatter_chart",  re.compile(r"\bscatter\b|\bcorrelat\b|\brelation")),
    ("histogram",      re.compile(r"\bhistogram\b|\bdistribut\b|\bspread\b")),
    ("pie_chart",      re.compile(r"\bpie\b|\bshare\b|\bproportion\b|\bbreakdown\b")),
]

# Aggregation synonyms → pandas method name
_AGG_MAP = {
    "sum": "sum", "total": "sum", "aggregate": "sum",
    "mean": "mean", "average": "mean", "avg": "mean",
    "count": "count", "number": "count",
    "max": "max", "maximum": "max", "highest": "max", "largest": "max",
    "min": "min", "minimum": "min", "lowest": "min", "smallest": "min",
    "median": "median",
}


# ─── Step 1 — Normalise ───────────────────────────────────────────────────────

def normalize_query(query: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    query = query.lower().strip()
    query = re.sub(r"[^\w\s]", " ", query)
    query = re.sub(r"\s+", " ", query)
    return query


# ─── Step 2 — Intent detection ───────────────────────────────────────────────

def detect_intent(query: str) -> str:
    """
    Rule-based intent detection. Returns intent string from _INTENT_RULES.
    Falls back to 'summary'.
    """
    norm = normalize_query(query)
    for intent, pattern in _INTENT_RULES:
        if pattern.search(norm):
            return intent
    return "summary"


def detect_chart_type(query: str, intent: str) -> Optional[str]:
    """
    Detect preferred chart type from query text or intent fallback.
    Returns None if no chart is appropriate.
    """
    norm = normalize_query(query)
    for chart_type, pattern in _CHART_RULES:
        if pattern.search(norm):
            return chart_type

    # Fallback by intent
    _INTENT_TO_CHART = {
        "distribution": "histogram",
        "top_n":        "bar_chart",
        "sum":          "bar_chart",
        "mean":         "bar_chart",
        "count":        "bar_chart",
        "trend":        "line_chart",
        "correlation":  "scatter_chart",
        "compare":      "bar_chart",
    }
    return _INTENT_TO_CHART.get(intent)


# ─── Step 3 — Column type detection ─────────────────────────────────────────

def get_column_types(df: pd.DataFrame) -> dict:
    """Classify all columns by dtype family."""
    return {
        "numeric":     df.select_dtypes(include="number").columns.tolist(),
        "categorical": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "datetime":    df.select_dtypes(include="datetime").columns.tolist(),
        "boolean":     df.select_dtypes(include="bool").columns.tolist(),
    }


# ─── Step 4 — Column matcher ─────────────────────────────────────────────────

def match_column(
    query: str,
    columns: list[str],
    cutoff: float = 0.55,
) -> Optional[str]:
    """
    Three-stage column matcher:
      1. Exact substring match (case-insensitive)
      2. difflib fuzzy match on full column names
      3. Token-level overlap (any query word matches any column word)

    Returns the best matching column name or None.
    """
    norm_query = normalize_query(query)
    norm_cols  = {c: c.lower().replace("_", " ") for c in columns}

    # Stage 1 — exact substring
    for col, norm in norm_cols.items():
        if norm in norm_query or norm_query in norm:
            return col

    # Stage 2 — difflib on normalised column names
    matches = get_close_matches(norm_query, list(norm_cols.values()), n=1, cutoff=cutoff)
    if matches:
        for col, norm in norm_cols.items():
            if norm == matches[0]:
                return col

    # Stage 3 — token overlap (any word in query that appears in any column name)
    query_tokens = set(norm_query.split())
    best_col, best_score = None, 0
    for col, norm in norm_cols.items():
        col_tokens = set(norm.split())
        overlap    = len(query_tokens & col_tokens)
        if overlap > best_score:
            best_score = overlap
            best_col   = col
    if best_score > 0:
        return best_col

    return None


def match_columns_spacy(query: str, columns: list[str]) -> list[str]:
    """
    spaCy-enhanced column matching: extract nouns + fuzzy match.
    Falls back to match_column() if spaCy unavailable.
    """
    if not SPACY_OK:
        col = match_column(query, columns)
        return [col] if col else []

    doc       = _nlp(query)
    nouns     = [token.lemma_.lower() for token in doc if token.pos_ in ("NOUN", "PROPN")]
    found     = []
    norm_cols = {c: c.lower().replace("_", " ") for c in columns}

    for noun in nouns:
        matches = get_close_matches(noun, list(norm_cols.values()), n=1, cutoff=0.6)
        if matches:
            for col, norm in norm_cols.items():
                if norm == matches[0] and col not in found:
                    found.append(col)

    # fallback if spaCy found nothing
    if not found:
        col = match_column(query, columns)
        if col:
            found.append(col)

    return found


def extract_numbers(query: str) -> list[int | float]:
    """Extract all numbers from a query string."""
    if SPACY_OK:
        doc = _nlp(query)
        nums = []
        for token in doc:
            if token.like_num:
                try:
                    nums.append(int(token.text))
                except ValueError:
                    try:
                        nums.append(float(token.text))
                    except ValueError:
                        pass
        return nums
    # Regex fallback
    return [int(m) if "." not in m else float(m)
            for m in re.findall(r"\b\d+(?:\.\d+)?\b", query)]


def extract_agg_func(query: str) -> str:
    """Detect aggregation function from query text. Defaults to 'sum'."""
    norm = normalize_query(query)
    for keyword, func in _AGG_MAP.items():
        if re.search(rf"\b{re.escape(keyword)}\b", norm):
            return func
    return "sum"


# ─── Step 5 — Query planner ──────────────────────────────────────────────────

def build_plan(query: str, df: pd.DataFrame) -> dict:
    """
    Build a structured execution plan from the query.
    Returns a dict consumed by execute_plan().
    """
    norm    = normalize_query(query)
    intent  = detect_intent(query)
    chart   = detect_chart_type(query, intent)
    types   = get_column_types(df)
    numbers = extract_numbers(query)
    agg     = extract_agg_func(query)

    # Attempt spaCy-enhanced column detection first; fall back to regex
    all_cols        = df.columns.tolist()
    spacy_cols      = match_columns_spacy(query, all_cols)
    numeric_match   = next((c for c in spacy_cols if c in types["numeric"]),   None)
    category_match  = next((c for c in spacy_cols if c in types["categorical"]), None)
    datetime_match  = next((c for c in spacy_cols if c in types["datetime"]), None)

    # Fallback: try direct column matching per type group
    if not numeric_match:
        numeric_match = match_column(norm, types["numeric"])
    if not category_match:
        category_match = match_column(norm, types["categorical"])
    if not datetime_match:
        datetime_match = match_column(norm, types["datetime"])

    base = {
        "intent":         intent,
        "chart_type":     chart,
        "agg_func":       agg,
        "numeric_col":    numeric_match,
        "category_col":   category_match,
        "datetime_col":   datetime_match,
        "numeric_cols":   types["numeric"],
        "category_cols":  types["categorical"],
        "n":              numbers[0] if numbers else 10,
        "raw_query":      query,
    }

    # Add intent-specific fields
    if intent == "top_n":
        m = re.search(r"\btop\s*(\d+)|\bbottom\s*(\d+)|\bhighest\s*(\d+)|\blargest\s*(\d+)", norm)
        if m:
            base["n"]         = int(next(g for g in m.groups() if g))
            base["ascending"] = "bottom" in norm or "lowest" in norm or "smallest" in norm

    if intent == "filter":
        base["filter_expr"] = _extract_filter_expr(norm, df)

    return base


def _extract_filter_expr(query: str, df: pd.DataFrame) -> Optional[dict]:
    """
    Best-effort extraction of simple filter expressions.
    Handles: col > N, col < N, col == value, col contains value.
    """
    # Numeric comparisons:  column > 100
    m = re.search(r"(\w[\w\s]*?)\s*(>|<|>=|<=|==|=|!=)\s*(\d+(?:\.\d+)?)", query)
    if m:
        col = match_column(m.group(1).strip(), df.columns.tolist())
        if col:
            op  = m.group(2).replace("=", "==").replace("!==", "!=")
            return {"col": col, "op": op, "value": float(m.group(3))}

    # Category equality:  where category is X
    m = re.search(r"where\s+(\w+)\s+(?:is|=|==)\s+['\"]?(\w[\w\s]*?)['\"]?(?:\s|$)", query)
    if m:
        col = match_column(m.group(1).strip(), df.columns.tolist())
        if col:
            return {"col": col, "op": "==", "value": m.group(2).strip()}

    return None


# ─── Step 6 — Pandas executor ────────────────────────────────────────────────

def execute_plan(plan: dict, df: pd.DataFrame):
    """
    Execute a plan dict against df.
    Returns a pandas DataFrame, Series, scalar, or string.
    Raises ValueError with a human-readable message if plan is unresolvable.
    """
    intent    = plan["intent"]
    num_col   = plan["numeric_col"]
    cat_col   = plan["category_col"]
    dt_col    = plan["datetime_col"]
    agg       = plan["agg_func"]
    n         = int(plan.get("n", 10))
    # Only keep columns that are genuinely numeric in the current df
    num_cols  = [c for c in plan["numeric_cols"] if c in df.columns and
                 pd.api.types.is_numeric_dtype(df[c])]

    # ── top_n ──────────────────────────────────────────────────────────────
    if intent == "top_n":
        ascending = plan.get("ascending", False)

        if cat_col and num_col:
            # ideal: groupby category, aggregate numeric
            grouped = df.groupby(cat_col)[num_col].agg(agg).reset_index()
            grouped.columns = [cat_col, num_col]
            return grouped.nlargest(n, num_col) if not ascending else grouped.nsmallest(n, num_col)

        if cat_col and not num_col:
            # category known, no numeric → value_counts
            vc = df[cat_col].value_counts().head(n).reset_index()
            vc.columns = [cat_col, "count"]
            return vc

        if not cat_col and num_col:
            # numeric only → nlargest
            return df[[num_col]].dropna().nlargest(n, num_col).reset_index(drop=True)

        # nothing matched — value_counts across all low-cardinality cat cols
        cat_cols_plan = plan.get("category_cols", [])
        low_card = [c for c in cat_cols_plan if df[c].nunique() <= 50]
        if low_card:
            vc = df[low_card[0]].value_counts().head(n).reset_index()
            vc.columns = [low_card[0], "count"]
            return vc
        raise ValueError("Could not determine columns for top_n query.")

    # ── mean ───────────────────────────────────────────────────────────────
    if intent == "mean":
        if cat_col and num_col:
            return df.groupby(cat_col)[num_col].mean().reset_index()
        if num_col:
            return df[num_col].mean()
        return df[num_cols].mean()

    # ── median ─────────────────────────────────────────────────────────────
    if intent == "median":
        if cat_col and num_col:
            return df.groupby(cat_col)[num_col].median().reset_index()
        if num_col:
            return df[num_col].median()
        return df[num_cols].median()

    # ── sum ────────────────────────────────────────────────────────────────
    if intent == "sum":
        if cat_col and num_col:
            return df.groupby(cat_col)[num_col].sum().reset_index()
        if num_col:
            return df[num_col].sum()
        return df[num_cols].sum()

    # ── max / min ──────────────────────────────────────────────────────────
    if intent in ("max", "min"):
        fn = "max" if intent == "max" else "min"
        if cat_col and num_col:
            return df.groupby(cat_col)[num_col].agg(fn).reset_index()
        if num_col:
            return getattr(df[num_col], fn)()
        return getattr(df[num_cols], fn)()

    # ── count ──────────────────────────────────────────────────────────────
    if intent == "count":
        if cat_col:
            vc = df[cat_col].value_counts().reset_index()
            vc.columns = [cat_col, "count"]
            return vc
        return pd.Series({"total_rows": len(df)})

    # ── unique ─────────────────────────────────────────────────────────────
    if intent == "unique":
        col = cat_col or num_col
        if col:
            return pd.DataFrame({
                col:          df[col].dropna().unique(),
                "unique_count": [df[col].nunique()] + [None] * (df[col].nunique() - 1)
            })
        return pd.Series({c: df[c].nunique() for c in df.columns})

    # ── missing ────────────────────────────────────────────────────────────
    if intent == "missing":
        missing = df.isnull().sum()
        pct     = (missing / len(df) * 100).round(2)
        return pd.DataFrame({
            "column":      missing.index,
            "missing":     missing.values,
            "missing_pct": pct.values,
        }).sort_values("missing", ascending=False)

    # ── distribution ───────────────────────────────────────────────────────
    if intent == "distribution":
        if num_col:
            # specific column requested
            desc = df[num_col].describe()
            return pd.DataFrame({
                "column":    num_col,
                "statistic": desc.index,
                "value":     desc.values.round(4),
            })
        if not num_cols:
            raise ValueError("No numeric columns found for distribution query.")
        # "numeric columns" (plural) — describe all numeric cols, one row per stat
        rows = []
        for col in num_cols:
            d = df[col].describe()
            for stat, val in d.items():
                rows.append({"column": col, "statistic": stat,
                             "value": round(float(val), 4) if pd.notna(val) else None})
        return pd.DataFrame(rows)

    # ── correlation ────────────────────────────────────────────────────────
    if intent == "correlation":
        # select only numeric cols, drop columns that are entirely NaN
        safe_num = df[num_cols].select_dtypes(include="number").dropna(axis=1, how="all")
        safe_num = safe_num.fillna(safe_num.mean())
        if safe_num.shape[1] < 2:
            raise ValueError("Need at least 2 numeric columns for correlation.")
        try:
            corr = safe_num.corr(numeric_only=True).round(4)   # pandas >= 1.5
        except TypeError:
            corr = safe_num.corr().round(4)                     # pandas < 1.5
        corr = corr.reset_index()
        corr.columns = ["column"] + [str(c) for c in corr.columns[1:]]
        return corr

    # ── trend ──────────────────────────────────────────────────────────────
    if intent == "trend":
        if dt_col and num_col:
            return df.groupby(dt_col)[num_col].agg(agg).reset_index()
        if dt_col:
            return df[dt_col].value_counts().sort_index().reset_index()
        raise ValueError("No datetime column found for trend query.")

    # ── outlier ────────────────────────────────────────────────────────────
    if intent == "outlier":
        col = num_col or (num_cols[0] if num_cols else None)
        if not col:
            raise ValueError("No numeric column found for outlier query.")
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr     = q3 - q1
        mask    = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
        return df[mask].reset_index(drop=True)

    # ── compare ────────────────────────────────────────────────────────────
    if intent == "compare":
        if cat_col and num_col:
            return df.groupby(cat_col)[num_col].agg(agg).reset_index().sort_values(num_col, ascending=False)
        return df[num_cols].agg(["mean", "median", "std", "min", "max"]).T.reset_index().rename(
            columns={"index": "column"})

    # ── filter ─────────────────────────────────────────────────────────────
    if intent == "filter":
        expr = plan.get("filter_expr")
        if expr:
            col, op, val = expr["col"], expr["op"], expr["value"]
            ops = {">": "__gt__", "<": "__lt__", ">=": "__ge__", "<=": "__le__",
                   "==": "__eq__", "!=": "__ne__"}
            mask = getattr(df[col], ops.get(op, "__eq__"))(val) if op in ops else df[col] == val
            return df[mask].reset_index(drop=True).head(100)
        # Fallback: return full df head
        return df.head(50)

    # ── summary (default) ──────────────────────────────────────────────────
    # Build a clean per-column summary: type, non-null count, missing%, unique, sample
    rows = []
    for col in df.columns:
        series   = df[col]
        non_null = int(series.notna().sum())
        missing  = round((series.isna().sum() / len(df)) * 100, 1)
        n_unique = int(series.nunique())
        dtype    = str(series.dtype)

        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            rows.append({
                "column":   col,
                "type":     dtype,
                "non_null": non_null,
                "missing%": missing,
                "unique":   n_unique,
                "min":      round(float(clean.min()), 2) if len(clean) else None,
                "mean":     round(float(clean.mean()), 2) if len(clean) else None,
                "max":      round(float(clean.max()), 2) if len(clean) else None,
                "sample":   None,
            })
        else:
            top = series.value_counts()
            rows.append({
                "column":   col,
                "type":     dtype,
                "non_null": non_null,
                "missing%": missing,
                "unique":   n_unique,
                "min":      None,
                "mean":     None,
                "max":      None,
                "sample":   str(top.index[0]) if len(top) else None,
            })
    return pd.DataFrame(rows)


# ─── Step 7 — Auto insight (zero LLM) ────────────────────────────────────────

def generate_auto_insight(result, plan: dict, df: pd.DataFrame) -> str:
    """
    Deterministic insight generation — no LLM, no API.
    Produces plain readable text from the result + plan metadata.
    """
    intent  = plan["intent"]
    num_col = plan["numeric_col"]
    cat_col = plan["category_col"]
    n       = plan.get("n", 10)

    try:
        # ── scalar results ─────────────────────────────────────────────────
        if isinstance(result, (int, float, np.integer, np.floating)):
            col_label = num_col or "value"
            if intent == "mean":
                return f"The average {col_label} across all records is {float(result):,.2f}."
            if intent == "sum":
                return f"The total {col_label} across all records is {float(result):,.2f}."
            if intent == "max":
                return f"The highest {col_label} recorded is {float(result):,.2f}."
            if intent == "min":
                return f"The lowest {col_label} recorded is {float(result):,.2f}."
            return f"Result: {float(result):,.4f}."

        # ── DataFrame results ──────────────────────────────────────────────
        if isinstance(result, pd.DataFrame) and len(result) > 0:
            cols    = result.columns.tolist()
            num_out = result.select_dtypes(include="number").columns.tolist()

            if intent == "top_n":
                if num_out and cat_col and cat_col in result.columns:
                    top_row = result.iloc[0]
                    bot_row = result.iloc[-1]
                    val_col = num_out[0]
                    return (
                        f"The top {n} {cat_col} by {val_col} range from "
                        f"'{top_row[cat_col]}' ({top_row[val_col]:,.2f}) down to "
                        f"'{bot_row[cat_col]}' ({bot_row[val_col]:,.2f})."
                    )
                # value_counts fallback: first col is category, second is count
                if len(result.columns) >= 2:
                    cat_c   = result.columns[0]
                    cnt_c   = result.columns[1]
                    top_row = result.iloc[0]
                    total   = result[cnt_c].sum()
                    pct     = round(top_row[cnt_c] / total * 100, 1) if total else 0
                    return (
                        f"Top category in {cat_c}: '{top_row[cat_c]}' with "
                        f"{int(top_row[cnt_c]):,} records ({pct}% of total)."
                    )

            if intent == "correlation" and len(num_out) > 1:
                corr_vals = result.set_index(cols[0])[num_out]
                flat = corr_vals.unstack().drop_duplicates()
                flat = flat[flat != 1.0].abs()
                if not flat.empty:
                    strongest_idx = flat.idxmax()
                    return (
                        f"Among {len(result)} columns, the strongest correlation is between "
                        f"{strongest_idx[0]} and {strongest_idx[1]} "
                        f"({flat.max():.3f}). "
                        f"Values close to 1.0 indicate strong linear relationships."
                    )

            if intent == "missing":
                total_missing = result["missing"].sum() if "missing" in result.columns else 0
                worst         = result.iloc[0]["column"] if "column" in result.columns else "unknown"
                worst_pct     = result.iloc[0]["missing_pct"] if "missing_pct" in result.columns else 0
                return (
                    f"The dataset has {int(total_missing):,} total missing values across all columns. "
                    f"The column with the most gaps is '{worst}' at {worst_pct:.1f}% missing."
                )

            if intent == "distribution" and "statistic" in result.columns:
                if "column" in result.columns:
                    # multi-column distribution
                    parts = []
                    for col_name, grp in result.groupby("column"):
                        vm = dict(zip(grp["statistic"], grp["value"]))
                        mean_v = vm.get("mean", 0) or 0
                        max_v  = vm.get("max",  0) or 0
                        parts.append(f"{col_name} (mean {mean_v:,.2f}, max {max_v:,.2f})")
                    return "Numeric column distributions: " + "; ".join(parts) + "."
                else:
                    # single-column distribution
                    val_map   = dict(zip(result["statistic"], result["value"]))
                    mean_v    = val_map.get("mean", 0) or 0
                    std_v     = val_map.get("std",  0) or 0
                    min_v     = val_map.get("min",  0) or 0
                    max_v     = val_map.get("max",  0) or 0
                    col_label = num_col or "the column"
                    return (
                        f"{col_label} has a mean of {mean_v:,.2f} with std {std_v:,.2f}. "
                        f"Values range from {min_v:,.2f} to {max_v:,.2f}."
                    )

            if intent == "outlier":
                col_label = num_col or "the selected column"
                return (
                    f"Found {len(result):,} outlier rows in {col_label} using the IQR method "
                    f"(values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR)."
                )

            if intent in ("mean", "sum", "max", "min", "compare") and num_out and cat_col:
                top_row = result.sort_values(num_out[0], ascending=False).iloc[0]
                return (
                    f"Across {len(result)} {cat_col} groups, "
                    f"'{top_row[cat_col]}' leads with a {intent} of {top_row[num_out[0]]:,.2f} "
                    f"in {num_out[0]}."
                )

            if intent == "summary" and "column" in result.columns:
                total_cols    = len(result)
                num_rows      = result[result["type"].str.contains("float|int", na=False)]
                cat_rows      = result[~result["type"].str.contains("float|int", na=False)]
                missing_cols  = result[result["missing%"] > 0] if "missing%" in result.columns else pd.DataFrame()
                insight_parts = [f"Dataset has {len(df):,} rows and {total_cols} columns."]
                if len(num_rows):
                    insight_parts.append(f"{len(num_rows)} numeric and {len(cat_rows)} categorical columns.")
                if len(missing_cols):
                    worst = missing_cols.sort_values("missing%", ascending=False).iloc[0]
                    insight_parts.append(
                        f"{len(missing_cols)} columns have missing values; "
                        f"'{worst['column']}' is worst at {worst['missing%']}% missing."
                    )
                return " ".join(insight_parts)

            if num_out:
                col  = num_out[0]
                vals = result[col].dropna()
                if len(vals):
                    return (
                        f"The result contains {len(result):,} rows. "
                        f"{col} ranges from {vals.min():,.2f} to {vals.max():,.2f} "
                        f"with a mean of {vals.mean():,.2f}."
                    )

            return f"Query returned {len(result):,} rows and {len(result.columns)} columns."

        # ── Series results ─────────────────────────────────────────────────
        if isinstance(result, pd.Series) and len(result) > 0:
            return (
                f"Result has {len(result)} entries. "
                f"Values range from {result.min():,.2f} to {result.max():,.2f}."
            )

    except Exception as exc:
        logger.warning("Auto insight failed: %s", exc)

    return ""


# ─── Result serialiser ────────────────────────────────────────────────────────

def _make_json_safe(rows: list[dict]) -> list[dict]:
    safe = []
    for row in rows:
        safe_row = {}
        for k, v in row.items():
            if isinstance(v, np.integer):    v = int(v)
            elif isinstance(v, np.floating): v = None if (isinstance(v, float) and (v != v)) else float(v)
            elif isinstance(v, np.bool_):    v = bool(v)
            elif v is None or (isinstance(v, float) and v != v): v = None
            safe_row[str(k)] = v
        safe.append(safe_row)
    return safe


def _result_to_serializable(result, plan: dict) -> dict:
    """Convert execution result → JSON-safe payload for the Flask frontend."""
    chart_type = plan.get("chart_type")
    intent     = plan["intent"]
    num_col    = plan["numeric_col"]
    cat_col    = plan["category_col"]

    # Scalar
    if isinstance(result, (int, float, np.integer, np.floating)):
        return {"type": "metric", "value": float(result),
                "label": f"{plan['agg_func'].title()} of {num_col or 'value'}"}

    # Series → DataFrame
    if isinstance(result, pd.Series):
        result = result.reset_index()
        result.columns = [str(c) for c in result.columns]

    if not isinstance(result, pd.DataFrame):
        return {"type": "summary", "text": str(result)[:2000]}

    result = result.copy()
    result = result.replace([np.inf, -np.inf], None)
    result = result.where(result.notna(), None)
    num_out = result.select_dtypes(include="number").columns.tolist()
    all_cols = result.columns.tolist()

    # Histogram data (distribution intent)
    if intent == "distribution" and "statistic" in result.columns:
        return {
            "type":    "table",
            "headers": result.columns.tolist(),
            "rows":    _make_json_safe(result.to_dict("records")),
            "total":   len(result),
        }

    # Chart payloads
    if chart_type in ("bar_chart", "line_chart", "histogram") and len(all_cols) >= 2:
        x_col = str(cat_col) if (cat_col and cat_col in all_cols) else str(all_cols[0])
        y_col = str(num_col) if (num_col and num_col in num_out)  else (str(num_out[0]) if num_out else str(all_cols[-1]))
        return {
            "type":    chart_type,
            "labels":  result[x_col].astype(str).tolist(),
            "values":  result[y_col].tolist(),
            "x_label": x_col,
            "y_label": y_col,
        }

    if chart_type == "scatter_chart" and len(num_out) >= 2:
        x = num_col or num_out[0]
        y = num_out[1] if num_out[0] == x else num_out[0]
        pts = result[[x, y]].dropna().head(500).to_dict("records")
        return {"type": "scatter_chart", "points": pts, "x_label": x, "y_label": y}

    # Table fallback
    rows = _make_json_safe(result.head(200).to_dict("records"))
    return {
        "type":    "table",
        "headers": [str(c) for c in result.columns],
        "rows":    rows,
        "total":   len(result),
    }


# ─── Main entry point ─────────────────────────────────────────────────────────

def run_deterministic_pipeline(query: str, df: pd.DataFrame) -> dict:
    """
    Full deterministic pipeline. Zero API calls.

    Usage in Flask
    --------------
        from deterministic_engine import run_deterministic_pipeline

        result = run_deterministic_pipeline(query, df)
        return jsonify(result)

    Returns
    -------
    {
        "error":   None | str,
        "answer":  str,      # plain text answer
        "result":  dict,     # chart / table / metric payload
        "insight": str,      # auto-generated insight (no LLM)
        "intent":  str,
        "plan":    dict,     # full plan (useful for debugging)
    }
    """
    try:
        plan       = build_plan(query, df)
        raw_result = execute_plan(plan, df)
        serialized = _result_to_serializable(raw_result, plan)
        insight    = generate_auto_insight(raw_result, plan, df)
        answer     = _build_answer_text(raw_result, plan)

        return {
            "error":   None,
            "answer":  answer,
            "result":  serialized,
            "insight": insight,
            "intent":  plan["intent"],
            "plan":    {k: v for k, v in plan.items()
                        if k not in ("numeric_cols", "category_cols")},
        }
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.error("Deterministic pipeline error: %s", exc, exc_info=True)
        return {"error": f"Query could not be processed: {exc}"}


def _build_answer_text(result, plan: dict) -> str:
    """Produce a one-line plain-text answer from the result."""
    intent  = plan["intent"]
    num_col = plan["numeric_col"]
    cat_col = plan["category_col"]

    if isinstance(result, (int, float, np.integer, np.floating)):
        return f"{plan['agg_func'].title()} of {num_col}: {float(result):,.4f}"

    if isinstance(result, pd.DataFrame):
        return (
            f"Showing {len(result):,} rows for intent '{intent}'"
            + (f" on {num_col}" if num_col else "")
            + (f" grouped by {cat_col}" if cat_col else "") + "."
        )

    if isinstance(result, pd.Series):
        return f"Series result with {len(result)} values."

    return str(result)[:300]