"""
Module: Gemini AI Query Pipeline — CSV Analyst Pro
Flask-ready. Uses LangChain + Gemini. No PandasAI.

Architecture:
    User Query
        ↓
    LangChain Agent
        ↓
    Gemini LLM  (gemini-2.5-flash → 2.5-flash-lite → 2.0-flash)
        ↓
    Python Pandas Tool  (StructuredTool with retry + logging)
        ↓
    Result
        ↓
    Insight Summary  (separate LLM call)

Fixes applied:
  - function_response.name empty  → explicit name via StructuredTool
  - 429 rate-limit                → parse retry_delay, sleep, fall-through models
  - <p> tags visible in frontend  → strip_markdown() returns plain text
"""

import os, sys, re, json, time, traceback, logging
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Ensure the project modules directory is always importable regardless of CWD
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# Also add the parent, in case modules/ is a sub-package
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

_MODEL_FALLBACKS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]
PRIMARY_MODEL = _MODEL_FALLBACKS[0]

INTENT_TYPES = [
    "bar_chart", "line_chart", "scatter_chart",
    "histogram", "table", "metric", "summary",
]
MAX_RETRIES = 2
SAMPLE_ROWS = 5

# ─── Availability ─────────────────────────────────────────────────────────────

_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_OK       = bool(_GEMINI_API_KEY)


def is_available() -> bool:
    return GEMINI_OK


def _require_key():
    if not GEMINI_OK:
        raise EnvironmentError(
            "GEMINI_API_KEY not configured. Add it to your .env file."
        )


# ─── Rate-limit aware LLM caller ─────────────────────────────────────────────

def _parse_retry_seconds(err_str: str) -> int:
    """Extract retry delay from a 429 error string. Returns 60 as safe default."""
    m = re.search(r"retry[_\s]+delay\s*\{\s*seconds:\s*(\d+)", err_str, re.IGNORECASE)
    if m:
        return int(m.group(1)) + 5   # +5s buffer
    m = re.search(r"retry in ([\d.]+)s", err_str, re.IGNORECASE)
    if m:
        return int(float(m.group(1))) + 5
    return 60   # safe default


def _build_llm(model: str = PRIMARY_MODEL, temperature: float = 0):
    """
    Return a patched ChatGoogleGenerativeAI that NEVER streams.

    Root cause: LangChain agents call ._stream() directly regardless of
    streaming=False on the LLM, which triggers the finish_reason int/enum
    crash in langchain-google-genai <= 1.0.10.

    Fix: subclass and override _stream() to delegate to _generate(), returning
    a single-chunk iterator. This bypasses the broken streaming code path
    entirely while remaining fully compatible with AgentExecutor.
    """
    _require_key()
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.outputs import ChatGenerationChunk
        from langchain_core.messages import AIMessageChunk
    except ImportError as exc:
        raise ImportError("Run: pip install langchain-google-genai") from exc

    class _NoStreamGemini(ChatGoogleGenerativeAI):
        """ChatGoogleGenerativeAI with _stream() replaced by _generate().
        Prevents the finish_reason AttributeError on free-tier Gemini models."""

        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            # Call the non-streaming path and yield result as a single chunk
            result = self._generate(messages, stop=stop,
                                    run_manager=run_manager, **kwargs)
            for gen in result.generations:
                content = gen.message.content if hasattr(gen, "message") else str(gen)
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content=content)
                )

    return _NoStreamGemini(
        model=model,
        temperature=temperature,
        google_api_key=_GEMINI_API_KEY,
        streaming=False,
    )


def _llm_with_fallback(temperature: float = 0):
    """
    Try each model in _MODEL_FALLBACKS.
    On 429, parse the retry delay, sleep, then try the next (cheaper) model.
    """
    last_err = None
    for model in _MODEL_FALLBACKS:
        try:
            return _build_llm(model, temperature)
        except Exception as exc:
            err_str = str(exc)
            last_err = exc
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                delay = _parse_retry_seconds(err_str)
                logger.warning(
                    "Model %s hit rate limit. Sleeping %ds then trying next model.",
                    model, delay
                )
                time.sleep(delay)
            else:
                logger.warning("Model %s unavailable: %s", model, exc)
    raise last_err


def _invoke_with_retry(llm, prompt: str, max_attempts: int = 3) -> str:
    """
    Invoke an LLM with automatic 429 retry + model fallback.
    Accepts a plain string prompt.
    """
    for attempt in range(max_attempts):
        try:
            response = llm.invoke(prompt)
            return response.content.strip()
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                delay = _parse_retry_seconds(err_str)
                logger.warning(
                    "429 on invoke attempt %d/%d. Sleeping %ds.",
                    attempt + 1, max_attempts, delay
                )
                if attempt < max_attempts - 1:
                    time.sleep(delay)
                    # try next fallback model
                    llm = _llm_with_fallback(temperature=0)
                    continue
            raise
    raise RuntimeError("Max invoke attempts exceeded")


# ─── Custom Pandas Tool (StructuredTool — fixes empty function_response.name) ─

def _make_pandas_tool(df: pd.DataFrame):
    """
    Production-grade tool wrapping sandboxed pandas execution.

    Uses StructuredTool.from_function() with an explicit `name` so that
    LangChain always populates function_response.name in the Gemini request,
    fixing the 400 'Name cannot be empty' error.
    """
    from langchain.tools import StructuredTool
    from pydantic import BaseModel

    class PandasInput(BaseModel):
        code: str

    def _run_pandas(code: str) -> str:
        current_code = _clean_code(code)
        local_ns: dict = {"df": df.copy(), "pd": pd, "np": np, "result": None}

        for attempt in range(MAX_RETRIES + 1):
            try:
                exec(current_code, {}, local_ns)  # noqa: S102
                return _result_to_str(local_ns.get("result"))
            except Exception:
                err = traceback.format_exc()
                logger.warning(
                    "[pandas_query] attempt %d/%d failed:\n%s",
                    attempt + 1, MAX_RETRIES + 1, err,
                )
                if attempt == MAX_RETRIES:
                    return f"ERROR after {MAX_RETRIES + 1} attempts:\n{err}"
                current_code = _fix_code_via_llm(current_code, err)

        return "ERROR: max retries exceeded"

    return StructuredTool.from_function(
        func=_run_pandas,
        name="pandas_query",          # explicit name → fixes function_response.name bug
        description=(
            "Execute Python/pandas code against the dataframe `df`. "
            "Always assign the final answer to a variable named `result`."
        ),
        args_schema=PandasInput,
    )


def _fix_code_via_llm(code: str, error: str) -> str:
    """Ask Gemini to fix broken pandas code."""
    llm  = _llm_with_fallback()
    prompt = (
        f"Fix this Python/pandas code:\n```python\n{code}\n```\n"
        f"Error:\n```\n{error}\n```\n"
        "Rules: assign answer to `result`. Use only pd/np. "
        "No print/plt/markdown. Return ONLY executable Python."
    )
    return _clean_code(_invoke_with_retry(llm, prompt))


# ─── LangChain Agent ──────────────────────────────────────────────────────────

def create_agent(df: pd.DataFrame, use_custom_tool: bool = True):
    """
    Build a LangChain AgentExecutor backed by Gemini.
    Returns AgentExecutor — call .invoke({"input": query}).
    """
    _require_key()
    llm = _llm_with_fallback()
    return (
        _build_custom_agent(llm, df)
        if use_custom_tool
        else _build_builtin_agent(llm, df)
    )


def _build_custom_agent(llm, df: pd.DataFrame):
    """Custom StructuredTool agent with rate-limit handling."""
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate

    pandas_tool = _make_pandas_tool(df)

    col_info = ", ".join(
        f"{c} ({dt})" for c, dt in zip(df.columns, df.dtypes)
    )
    system_prompt = (
        "You are a senior data analyst. "
        f"The dataframe `df` has {df.shape[0]:,} rows × {df.shape[1]} columns: {col_info}.\n\n"
        "Use the pandas_query tool to answer the user's question. "
        "Always assign the computed answer to `result` inside the tool. "
        "Summarise findings clearly. "
        "Plain text only — no markdown, no asterisks, no bullet symbols, no bold formatting."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human",  "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, [pandas_tool], prompt)
    return AgentExecutor(
        agent=agent,
        tools=[pandas_tool],
        verbose=False,
        max_iterations=6,
        handle_parsing_errors=True,
    )


def _build_builtin_agent(llm, df: pd.DataFrame):
    """LangChain built-in pandas agent (fallback path)."""
    from langchain_experimental.agents import create_pandas_dataframe_agent
    return create_pandas_dataframe_agent(
        llm, df,
        verbose=False,
        allow_dangerous_code=True,
        max_iterations=6,
        handle_parsing_errors=True,
    )


# ─── Insight summary ─────────────────────────────────────────────────────────

def generate_insight(query: str, raw_result: str) -> str:
    """Separate LLM call: 2-3 plain-text insight sentences."""
    if not raw_result or raw_result.startswith("ERROR"):
        return ""
    try:
        llm = _llm_with_fallback(temperature=0.3)
        prompt = (
            f"A data analyst answered: \"{query}\"\n"
            f"Result:\n{raw_result[:1500]}\n\n"
            "Write 2-3 concise insight sentences with specific numbers. "
            "Plain prose only — no markdown, no asterisks, no bullet points, no bold."
        )
        return _invoke_with_retry(llm, prompt)
    except Exception as exc:
        logger.warning("Insight generation failed: %s", exc)
        return ""


# ─── Intent classifier ────────────────────────────────────────────────────────

def classify_intent(query: str, df: pd.DataFrame) -> dict:
    """JSON intent classifier for chart-type routing."""
    _require_key()
    col_info = "\n".join(
        f"  {c}: {dt} | sample: {df[c].dropna().head(3).tolist()}"
        for c, dt in zip(df.columns, df.dtypes)
    )
    llm = _llm_with_fallback()
    prompt = (
        "Classify the query intent.\n"
        f"DATASET: {df.shape[0]} rows × {df.shape[1]} cols\n"
        f"COLUMNS:\n{col_info}\n"
        f"QUERY: \"{query}\"\n\n"
        f"Return ONLY valid JSON — keys: type (one of {INTENT_TYPES}), "
        "x_col, y_col, agg_func (sum|mean|count|max|min|none), "
        "top_n, filter_col, reasoning. No markdown fences."
    )
    try:
        raw = _invoke_with_retry(llm, prompt)
        return _clean_json(raw)
    except Exception:
        return {
            "type": "summary", "x_col": None, "y_col": None,
            "agg_func": "none", "top_n": None, "filter_col": None,
            "reasoning": "fallback",
        }



# ─── Deterministic engine loader (bulletproof, call-time path resolution) ────

_DET_MODULE = None   # cached after first successful load

def _load_deterministic_engine():
    """
    Load deterministic_engine.py using 4 fallback strategies so it works
    regardless of CWD, Flask watchdog restarts, or module/ sub-package layout.
    """
    global _DET_MODULE
    if _DET_MODULE is not None:
        return _DET_MODULE

    import importlib.util as ilu, inspect

    # Strategy 1 — normal sys.path import
    try:
        import deterministic_engine as m
        _DET_MODULE = m
        return m
    except ModuleNotFoundError:
        pass

    # Candidate directories to search
    candidates = [
        os.path.dirname(os.path.abspath(__file__)),          # same dir as gemini_ai.py
        os.path.dirname(os.path.abspath(inspect.getfile(    # absolute caller dir
            inspect.currentframe()))),
        os.getcwd(),                                          # Flask working dir
        os.path.join(os.getcwd(), "modules"),                 # project/modules/
    ]

    for directory in candidates:
        path = os.path.join(directory, "deterministic_engine.py")
        if os.path.isfile(path):
            spec = ilu.spec_from_file_location("deterministic_engine", path)
            mod  = ilu.module_from_spec(spec)
            # inject into sys.modules so subsequent imports resolve correctly
            sys.modules["deterministic_engine"] = mod
            spec.loader.exec_module(mod)
            _DET_MODULE = mod
            logger.info("Loaded deterministic_engine from %s", path)
            return mod

    raise ImportError(
        "deterministic_engine.py not found. "
        "Place it in the same folder as gemini_ai.py (e.g. modules/)."
    )


# ─── Main pipeline (Flask entry point) ───────────────────────────────────────

def run_query_pipeline(query: str, df: pd.DataFrame) -> dict:
    """
    Hybrid pipeline: Deterministic engine first, LangChain+Gemini as fallback.

    Flow:
        1. Try deterministic_engine (zero LLM, zero API, instant)
        2. If deterministic fails → fall back to LangChain + Gemini
        3. Gemini used only for richer insight narrative (optional, suppressed on 429)

    Usage:
        from gemini_ai import run_query_pipeline
        result = run_query_pipeline(query, df)
        return jsonify(result)
    """
    # ── Step 1: Deterministic engine (zero API calls) ──────────────────────
    try:
        det_mod = _load_deterministic_engine()
        det_result = det_mod.run_deterministic_pipeline(query, df)

        if not det_result.get("error"):
            # Deterministic succeeded — optionally enrich insight with LLM
            insight = det_result.get("insight", "")
            if GEMINI_OK and not insight:
                try:
                    insight = generate_insight(query, det_result.get("answer", ""))
                except Exception:
                    pass   # 429 or quota — auto insight already filled in

            return {
                "error":   None,
                "answer":  det_result["answer"],
                "result":  det_result["result"],
                "insight": insight or det_result.get("insight", ""),
                "intent":  det_result["intent"],
                "engine":  "deterministic",
            }
    except Exception as det_exc:
        logger.warning("Deterministic engine failed (%s), falling back to LLM.", det_exc)

    # ── Step 2: LangChain + Gemini fallback ────────────────────────────────
    if not GEMINI_OK:
        return {"error": "Query could not be resolved. Set GEMINI_API_KEY for LLM fallback."}

    try:
        intent   = classify_intent(query, df)
        agent    = create_agent(df, use_custom_tool=True)
        response = agent.invoke({"input": query})
        if isinstance(response, dict):
            # .get("output", "") avoids falsy "" falling through to str(response)
            answer = response.get("output", "").strip()
            if not answer:
                answer = response.get("result", "").strip() if response.get("result") else ""
            if not answer:
                # last resort: remove input key and stringify remaining
                leftover = {k: v for k, v in response.items() if k != "input"}
                answer = str(leftover.get("output", leftover)) if leftover else "No result."
        else:
            answer = str(response).strip()

        serialized = _answer_to_serializable(answer, intent, df)
        insight    = generate_insight(query, answer)

        return {
            "error":   None,
            "answer":  strip_markdown(answer),
            "result":  serialized,
            "insight": strip_markdown(insight),
            "intent":  intent.get("type", "summary"),
            "engine":  "llm",
        }

    except Exception as exc:
        logger.error("LLM pipeline error: %s", exc, exc_info=True)
        return {"error": str(exc), "traceback": traceback.format_exc()}


# ─── Serialisation helpers ────────────────────────────────────────────────────

def _answer_to_serializable(answer: str, intent: dict, df: pd.DataFrame) -> dict:
    itype = intent.get("type", "summary")

    if itype == "metric":
        m = re.search(r"[-+]?\d[\d,]*\.?\d*", answer.replace(",", ""))
        if m:
            return {"type": "metric", "value": float(m.group().replace(",", ""))}

    if itype in ("bar_chart", "histogram", "line_chart", "scatter_chart"):
        payload = _build_chart_payload(intent, df)
        if payload:
            return payload

    if itype == "table":
        rows = df.head(100).to_dict("records")
        return {
            "type":    "table",
            "headers": df.columns.tolist(),
            "rows":    _make_json_safe(rows),
            "total":   len(df),
        }

    return {"type": "summary", "text": strip_markdown(answer[:2000])}


def _build_chart_payload(intent: dict, df: pd.DataFrame) -> dict | None:
    try:
        x_col    = intent.get("x_col")
        y_col    = intent.get("y_col")
        agg_func = intent.get("agg_func", "sum")
        top_n    = intent.get("top_n")
        itype    = intent.get("type")

        if not x_col or x_col not in df.columns:
            return None

        if itype == "scatter_chart":
            nc = df.select_dtypes(include=np.number).columns.tolist()
            x  = x_col or (nc[0] if nc else None)
            y  = y_col or (nc[1] if len(nc) > 1 else None)
            if x and y and x in df.columns and y in df.columns:
                pts = df[[x, y]].dropna().head(500).to_dict("records")
                return {"type": "scatter_chart", "points": pts, "x_label": x, "y_label": y}
            return None

        if y_col and y_col in df.columns and agg_func != "none":
            agg_map = {"sum":"sum","mean":"mean","count":"count","max":"max","min":"min"}
            grouped = df.groupby(x_col)[y_col].agg(agg_map.get(agg_func,"sum")).reset_index()
        else:
            grouped         = df[x_col].value_counts().reset_index()
            grouped.columns = [x_col, "count"]
            y_col           = "count"

        grouped = grouped.dropna()
        if top_n:
            grouped = grouped.nlargest(int(top_n), y_col)

        chart_type = "bar_chart" if itype in ("bar_chart","histogram") else "line_chart"
        return {
            "type":    chart_type,
            "labels":  grouped[x_col].astype(str).tolist(),
            "values":  grouped[y_col].tolist(),
            "x_label": x_col,
            "y_label": y_col,
        }
    except Exception as exc:
        logger.warning("Chart payload build failed: %s", exc)
        return None


def _result_to_str(result) -> str:
    if result is None:                                        return "No result produced."
    if isinstance(result, (int, float, np.integer, np.floating)): return str(float(result))
    if isinstance(result, pd.DataFrame):                     return result.head(50).to_string(index=False)
    if isinstance(result, pd.Series):                        return result.head(50).to_string()
    return str(result)[:2000]


def _make_json_safe(rows: list[dict]) -> list[dict]:
    safe = []
    for row in rows:
        safe_row = {}
        for k, v in row.items():
            if isinstance(v, np.integer):    v = int(v)
            elif isinstance(v, np.floating): v = None if np.isnan(v) else float(v)
            elif isinstance(v, np.bool_):    v = bool(v)
            safe_row[str(k)] = v
        safe.append(safe_row)
    return safe


def _clean_code(t: str) -> str:
    t = re.sub(r"^```(?:python)?", "", t.strip(), flags=re.MULTILINE)
    return re.sub(r"```$", "", t.strip()).strip()


def _clean_json(t: str) -> dict:
    t = re.sub(r"^```(?:json)?", "", t.strip(), flags=re.MULTILINE)
    return json.loads(re.sub(r"```$", "", t.strip()).strip())


# ─── Markdown stripper (plain text — works with textContent AND innerHTML) ───

def strip_markdown(text: str) -> str:
    """
    Strip markdown syntax from LLM output → plain readable text.
    Works safely whether the frontend uses textContent or innerHTML.
    """
    if not text:
        return text
    # code fences
    text = re.sub(r"```[\s\S]*?```", "", text)
    # inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # bold / italic  ** __ * _
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__",     r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text)
    text = re.sub(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)",       r"\1", text)
    # headings
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # bullet symbols at line start
    text = re.sub(r"^\s*[*\-]\s+", "", text, flags=re.MULTILINE)
    # numbered list markers
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    # collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()