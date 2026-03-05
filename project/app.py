"""
CSV Analyst Pro — Flask Application
Routes:
  GET  /                          → Upload page
  GET  /workspace                 → Main workspace
  POST /api/upload                → Upload CSV, start session
  GET  /api/profile               → Dataset profile JSON
  GET  /api/preview               → First N rows as JSON
  POST /api/query                 → AI query via Gemini
  POST /api/clean                 → Run cleaning pipeline
  GET  /api/clean/download        → Download cleaned CSV
  POST /api/eda                   → Generate ydata-profiling report
  GET  /api/eda/report            → Serve EDA HTML report
  POST /api/automl/train          → Train FLAML AutoML
  GET  /api/automl/download       → Download best model (.pkl)
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os, io, uuid, json, pickle, tempfile, traceback
from pathlib import Path
from functools import wraps

import pandas as pd
import numpy as np
from flask import (Flask, render_template, request, jsonify, session,
                   send_file, redirect, url_for, Response)
from dotenv import load_dotenv

load_dotenv(override=True)

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", uuid.uuid4().hex)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

# Temp storage dir for large objects (DataFrames, models)
STORE_DIR = Path(tempfile.gettempdir()) / "csv_analyst_pro"
STORE_DIR.mkdir(exist_ok=True)

# ── Module imports ─────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from modules.data_cleaner   import run_cleaning_pipeline
from modules.eda_report     import generate_eda_report
from modules.automl_trainer import run_automl, _detect_task
from modules.gemini_pipeline import run_query_pipeline, is_available as gemini_available


# ══════════════════════════════════════════════════════════════════════════════
# SESSION HELPERS  — DataFrames too large for cookie; store on disk
# ══════════════════════════════════════════════════════════════════════════════
def _sid() -> str:
    """Get or create a persistent session storage ID."""
    if "store_id" not in session:
        session["store_id"] = uuid.uuid4().hex
    return session["store_id"]


def _path(key: str) -> Path:
    return STORE_DIR / f"{_sid()}_{key}"


def _save(key: str, obj):
    """Pickle an object to disk."""
    with open(_path(key), "wb") as f:
        pickle.dump(obj, f)


def _load(key: str):
    """Unpickle an object from disk. Returns None if not found."""
    p = _path(key)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def _clear_store():
    """Delete all stored objects for this session."""
    sid = _sid()
    for p in STORE_DIR.glob(f"{sid}_*"):
        p.unlink(missing_ok=True)


def _require_df(fn):
    """Decorator: return 400 if no uploaded dataset in session."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if _load("df_raw") is None:
            return jsonify({"error": "No dataset uploaded. Please upload a CSV first."}), 400
        return fn(*args, **kwargs)
    return wrapper


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _df_profile(df: pd.DataFrame, filename: str = "") -> dict:
    """Build dataset profile dict from a DataFrame."""
    missing     = int(df.isnull().sum().sum())
    numeric_cnt = int(len(df.select_dtypes(include=np.number).columns))
    total_cells = df.shape[0] * df.shape[1]
    miss_pct    = round(missing / max(total_cells, 1) * 100, 1)

    columns = []
    for col, dtype in zip(df.columns, df.dtypes):
        null_pct = round(df[col].isnull().mean() * 100, 1)
        columns.append({
            "name":     col,
            "dtype":    str(dtype),
            "null_pct": null_pct,
            "quality":  round(100 - null_pct, 1),
        })

    return {
        "filename":    filename,
        "rows":        df.shape[0],
        "cols":        df.shape[1],
        "numeric":     numeric_cnt,
        "missing":     missing,
        "missing_pct": miss_pct,
        "columns":     columns,
    }


def _safe_json_value(v):
    if isinstance(v, (np.integer,)):   return int(v)
    if isinstance(v, (np.floating,)):  return None if np.isnan(v) else float(v)
    if isinstance(v, np.bool_):        return bool(v)
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)): return None
    if pd.isna(v):                     return None
    return v


def _df_to_json_rows(df: pd.DataFrame, limit: int = 500) -> dict:
    df = df.head(limit).replace([np.inf, -np.inf], None)
    headers = [str(c) for c in df.columns]
    rows = []
    for _, row in df.iterrows():
        rows.append([_safe_json_value(v) for v in row])
    return {"headers": headers, "rows": rows, "total": len(df)}


# ══════════════════════════════════════════════════════════════════════════════
# PAGE ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/workspace")
def workspace():
    profile = _load("profile")
    if not profile:
        return redirect(url_for("index"))
    return render_template(
        "workspace.html",
        profile=profile,
        gemini_ok=gemini_available(),
    )


# ══════════════════════════════════════════════════════════════════════════════
# API: UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported"}), 400

    try:
        df = pd.read_csv(f)
    except Exception as e:
        return jsonify({"error": f"Could not parse CSV: {str(e)}"}), 400

    _clear_store()
    _save("df_raw", df)
    profile = _df_profile(df, filename=f.filename)
    _save("profile", profile)
    session["filename"] = f.filename

    return jsonify({"ok": True, "profile": profile})


# ══════════════════════════════════════════════════════════════════════════════
# API: PROFILE + PREVIEW
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/profile")
@_require_df
def api_profile():
    profile = _load("profile") or {}
    clean_profile = _load("clean_profile")
    return jsonify({
        "raw":     profile,
        "cleaned": clean_profile,
    })


@app.route("/api/preview")
@_require_df
def api_preview():
    use_clean = request.args.get("clean", "false").lower() == "true"
    df = _load("df_clean") if use_clean else _load("df_raw")
    if df is None:
        df = _load("df_raw")
    limit = int(request.args.get("limit", 200))
    return jsonify(_df_to_json_rows(df, limit=limit))


# ══════════════════════════════════════════════════════════════════════════════
# API: AI QUERY
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/query", methods=["POST"])
@_require_df
def api_query():
    body  = request.get_json(force=True)
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    df_clean = _load("df_clean")
    df_raw   = _load("df_raw")
    df       = df_clean if df_clean is not None else df_raw

    result = run_query_pipeline(query, df)
    return jsonify(result)


# ══════════════════════════════════════════════════════════════════════════════
# API: DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/clean", methods=["POST"])
@_require_df
def api_clean():
    df_raw = _load("df_raw")
    try:
        result = run_cleaning_pipeline(df_raw)
        df_clean = result["df_clean"]
        _save("df_clean", df_clean)
        clean_profile = _df_profile(df_clean, filename=session.get("filename", ""))
        _save("clean_profile", clean_profile)

        return jsonify({
            "ok":             True,
            "stats":          result["stats"],
            "missing_log":    result["missing_log"],
            "struct_actions": result["struct_actions"],
            "clean_profile":  clean_profile,
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/clean/download")
@_require_df
def api_clean_download():
    df = _load("df_clean")
    if df is None:
        return jsonify({"error": "Run cleaning first"}), 400
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    fname = "cleaned_" + session.get("filename", "data.csv")
    return send_file(buf, mimetype="text/csv",
                     as_attachment=True, download_name=fname)


# ══════════════════════════════════════════════════════════════════════════════
# API: EDA REPORT
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/eda", methods=["POST"])
@_require_df
def api_eda():
    body     = request.get_json(force=True) or {}
    minimal  = bool(body.get("minimal", True))
    sample_n = int(body.get("sample_n", 5000))

    df_clean = _load("df_clean")
    df_raw   = _load("df_raw")
    df       = df_clean if df_clean is not None else df_raw

    result = generate_eda_report(df, minimal=minimal, sample_n=sample_n)
    if result["error"]:
        return jsonify({"error": result["error"]}), 500

    _save("eda_html", result["html"])
    return jsonify({"ok": True, "rows_profiled": result["rows_profiled"]})


@app.route("/api/eda/report")
def api_eda_report():
    html = _load("eda_html")
    if not html:
        return Response("""<!DOCTYPE html><html data-theme="dark"><head><style>
          html,body{background:#0A0A0B;color:#66666a;font-family:monospace;margin:0;
            display:flex;align-items:center;justify-content:center;height:100vh}
          </style></head><body><p>No EDA report yet — click "Generate Report".</p></body></html>
        """, mimetype="text/html", status=404)
    return Response(html, mimetype="text/html")


# ══════════════════════════════════════════════════════════════════════════════
# API: AUTOML
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/api/automl/detect-task", methods=["POST"])
@_require_df
def api_automl_detect():
    body       = request.get_json(force=True) or {}
    target_col = body.get("target_col")
    df_clean   = _load("df_clean")
    df_raw     = _load("df_raw")
    df         = df_clean if df_clean is not None else df_raw

    if not target_col or target_col not in df.columns:
        return jsonify({"error": "Invalid target column"}), 400

    task      = _detect_task(df[target_col])
    n_unique  = int(df[target_col].nunique())
    return jsonify({"task": task, "n_unique": n_unique})


@app.route("/api/automl/train", methods=["POST"])
@_require_df
def api_automl_train():
    body        = request.get_json(force=True) or {}
    target_col  = body.get("target_col")
    task_choice = body.get("task_choice", "auto-detect")
    time_budget = int(body.get("time_budget", 120))
    test_size   = int(body.get("test_size", 20)) / 100.0

    df_clean = _load("df_clean")
    df_raw   = _load("df_raw")
    df       = df_clean if df_clean is not None else df_raw

    if not target_col or target_col not in df.columns:
        return jsonify({"error": f"Target column '{target_col}' not found"}), 400

    result = run_automl(df, target_col, task_choice, time_budget, test_size)
    if result.get("error"):
        return jsonify({"error": result["error"]}), 500

    # Store model bytes separately; don't send over JSON
    model_pkl = result.pop("model_pkl", None)
    if model_pkl:
        _save("model_pkl", model_pkl)
        _save("best_estimator", result["best_estimator"])

    return jsonify({"ok": True, **result})


@app.route("/api/automl/download")
def api_automl_download():
    model_pkl = _load("model_pkl")
    if not model_pkl:
        return jsonify({"error": "No trained model available"}), 400
    best_estimator = _load("best_estimator") or "model"
    buf = io.BytesIO(model_pkl)
    buf.seek(0)
    return send_file(buf, mimetype="application/octet-stream",
                     as_attachment=True,
                     download_name=f"best_model_{best_estimator}.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)