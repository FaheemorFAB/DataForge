"""
Module: EDA Report
Pure Python — no Streamlit. Returns theme-aware HTML string for Flask to serve.

Fixes applied
─────────────
FIX A  visions/imghdr DeprecationWarning fires the MOMENT ydata_profiling is
        imported (not just when to_html() runs), because visions eagerly imports
        imghdr_patch.py at module load time.  Flask's watchdog sees that file
        being "touched" and schedules a server restart — killing the in-progress
        ProfileReport generation.

        Solution: suppress ALL DeprecationWarning / FutureWarning globally at
        process startup using PYTHONWARNINGS or warnings.filterwarnings() called
        before the first import of ydata_profiling.  Here we guard both the
        import AND the generation inside a warnings context, and additionally
        patch the imghdr module so it never raises again.

FIX B  ydata-profiling HTML is a self-contained Bootstrap 4 document with
        hardcoded white backgrounds.  We inject a <style> block driven by a
        CSS custom-property theme (data-theme="dark"|"light" on <html>) and a
        tiny <script> that listens for postMessage from the parent workspace
        page so the iframe can switch themes live without reloading.

FIX C  The navbar remains white in dark mode because ydata-profiling renders it
        with a hardcoded class="navbar navbar-light bg-light" (or bg-white) and
        often an inline style="background-color:#fff".  Both survive generic CSS
        overrides.  Solution: a pre-injection HTML pass that rewrites those
        navbar class tokens and strips its inline style, THEN we also add
        ultra-high-specificity CSS targeting every Bootstrap bg-* utility class.
"""

from __future__ import annotations
import warnings
import pandas as pd


# ── Suppress the imghdr deprecation warning permanently ───────────────────────
# This must happen before ANY import of visions / ydata_profiling.
# We do it at module-load time so even the first import in the process is clean.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="imghdr")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="visions")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ── dtype sanitiser ───────────────────────────────────────────────────────────
def _sanitise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast category → object and bool → int so ydata-profiling v4 doesn't
    crash on chi-squared / correlation matrix steps.
    """
    df = df.copy()
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].astype(object)
    for col in df.select_dtypes(include="bool").columns:
        df[col] = df[col].astype(int)
    return df


# ── Theme CSS block ───────────────────────────────────────────────────────────
_THEME_CSS = """
<style id="analyst-pro-theme">
/* ── variables ──────────────────────────────────── */
html[data-theme="dark"] {
  --bg:       #0A0A0B;
  --surface:  #111113;
  --border:   #1f1f22;
  --txt:      #e8e8ea;
  --txt-m:    #66666a;
  --accent:   #2E5BFF;
  --card:     #16161a;
  --head-bg:  #0d0d0f;
  --code-bg:  #0d1117;
  --code-txt: #34d399;
}
html[data-theme="light"] {
  --bg:       #f2f2f5;
  --surface:  #ffffff;
  --border:   #e2e2e6;
  --txt:      #0a0a0b;
  --txt-m:    #73737a;
  --accent:   #2E5BFF;
  --card:     #ffffff;
  --head-bg:  #f8f8fb;
  --code-bg:  #f3f4f6;
  --code-txt: #1d4ed8;
}

/* ── base ───────────────────────────────────────── */
html, body, .container-fluid, .container {
  background-color: var(--bg)  !important;
  color:            var(--txt) !important;
}

/* ══════════════════════════════════════════════════
   NAVBAR — FIX C
   Bootstrap's bg-light / bg-white / navbar-light
   utility classes all set background-color via their
   own rules.  We need higher specificity than those
   AND we need to survive the inline style strip.
   Using html[data-theme] prefix raises specificity
   above Bootstrap's 0-1-0 selectors.
   ══════════════════════════════════════════════════ */

/* Kill Bootstrap bg-* utility overrides */
html[data-theme] .bg-light,
html[data-theme] .bg-white,
html[data-theme] .bg-dark,
html[data-theme] .bg-secondary,
html[data-theme] .bg-primary {
  background-color: var(--head-bg) !important;
}

/* Kill navbar-light's own background + text colours */
html[data-theme] .navbar,
html[data-theme] .navbar-light,
html[data-theme] .navbar-dark,
html[data-theme] nav,
html[data-theme] header,
html[data-theme] .report-header,
html[data-theme] .report-title {
  background-color: var(--head-bg) !important;
  border:           none          !important;
  border-bottom:    none          !important;
  border-color:     transparent   !important;
  box-shadow:       none          !important;
  outline:          none          !important;
}

/* ── hamburger expanded menu (.navbar-collapse) ─────────────
   NO html[data-theme] prefix — Bootstrap adds .show via JS
   after paint so a theme-prefixed rule loses the race.
   Hardcoded fallback values guarantee dark even if CSS vars
   haven't resolved yet.                                      */
.navbar-collapse,
.navbar-collapse.show,
.navbar-collapse.collapsing {
  background-color: var(--head-bg, #0d0d0f)        !important;
  border:           1px solid var(--border, #1f1f22) !important;
  border-top:       none                            !important;
  border-radius:    0 0 .5rem .5rem                 !important;
  padding:          .5rem 1rem                      !important;
}
.navbar-collapse .nav-item,
.navbar-collapse .nav-link,
.navbar-collapse .navbar-nav,
.navbar-collapse * {
  background-color: transparent              !important;
  color:            var(--txt, #e8e8ea)      !important;
  border-color:     transparent              !important;
}
.navbar-collapse .nav-link:hover {
  color: var(--accent, #2E5BFF) !important;
}

html[data-theme] .navbar *,
html[data-theme] .navbar-brand,
html[data-theme] .nav-link,
html[data-theme] .navbar-text,
html[data-theme] .report-header *,
html[data-theme] .report-title * {
  color:        var(--txt)       !important;
  border-color: transparent      !important;
}

/* ydata <hr> separator after navbar */
html[data-theme] .navbar + hr,
html[data-theme] header + hr,
html[data-theme] .navbar ~ hr,
html[data-theme] hr,
html[data-theme] .container > hr,
html[data-theme] .container-fluid > hr { display: none !important; }

html[data-theme] .navbar::after,  html[data-theme] header::after,
html[data-theme] .navbar::before, html[data-theme] header::before {
  display: none !important; content: none !important;
}

/* ── cards ───────────────────────────────────────── */
.card { background-color: var(--card)   !important; border-color: var(--border) !important; }
.card-header { background-color: var(--head-bg) !important; color: var(--txt) !important; border-color: var(--border) !important; }
.card-body { background-color: var(--card) !important; color: var(--txt) !important; }

/* ── tables ──────────────────────────────────────── */
table, .table, .table * { color: var(--txt) !important; border-color: var(--border) !important; }
table, .table { background-color: var(--surface) !important; }
thead, .table thead th, th {
  background-color: var(--head-bg) !important;
  color: var(--txt-m) !important;
}
.table-striped tbody tr:nth-of-type(odd) td {
  background-color: rgba(255,255,255,.025) !important;
}
html[data-theme="light"] .table-striped tbody tr:nth-of-type(odd) td {
  background-color: rgba(0,0,0,.025) !important;
}
.table-hover tbody tr:hover td { background-color: rgba(46,91,255,.06) !important; }

/* ── tabs ────────────────────────────────────────── */
.nav-tabs { border-color: var(--border) !important; }
.nav-tabs .nav-link { color: var(--txt-m) !important; background: transparent !important; border-color: transparent !important; }
.nav-tabs .nav-link.active { color: var(--txt) !important; background-color: var(--card) !important; border-color: var(--border) var(--border) var(--card) !important; }
.nav-tabs .nav-link:hover { color: var(--txt) !important; border-color: var(--border) !important; }

/* ── badges ──────────────────────────────────────── */
.badge, .badge-pill { background-color: var(--accent) !important; color: #fff !important; }
.badge-warning  { background-color: #f59e0b !important; }
.badge-danger   { background-color: #ef4444 !important; }
.badge-success  { background-color: #10b981 !important; }
.badge-secondary { background-color: var(--border) !important; color: var(--txt-m) !important; }

/* ── alerts & progress ───────────────────────────── */
.alert { background-color: var(--card) !important; border-color: var(--border) !important; color: var(--txt) !important; }
.progress { background-color: var(--border) !important; }
.progress-bar { background-color: var(--accent) !important; }

/* ── code / pre ──────────────────────────────────── */
code, pre, kbd { background-color: var(--code-bg) !important; color: var(--code-txt) !important; border-color: var(--border) !important; }

/* ── panels / collapse / accordion ──────────────── */
.panel, .panel-body, .panel-heading, .accordion-body, .accordion-header,
[data-toggle="collapse"] { background-color: var(--card) !important; border-color: var(--border) !important; color: var(--txt) !important; }

/* ── buttons ─────────────────────────────────────── */
.btn-default, .btn-secondary, .btn-outline-secondary {
  background-color: var(--surface) !important; border-color: var(--border) !important; color: var(--txt) !important;
}
.btn-primary { background-color: var(--accent) !important; border-color: var(--accent) !important; color: #fff !important; }

/* ── ydata-profiling specific ────────────────────── */
.anchor, .anchor-icon { color: var(--txt-m) !important; }
.stat, .stat * { color: var(--txt) !important; }
.freq-table td, .freq-table th { color: var(--txt) !important; border-color: var(--border) !important; }
.mini-freq-table td { background-color: var(--surface) !important; }
.col-wrap { background-color: var(--surface) !important; }

/* ── typography ──────────────────────────────────── */
h1,h2,h3,h4,h5,h6,.h1,.h2,.h3,.h4,.h5,.h6 { color: var(--txt) !important; }
p, li, label, small, span, td, th, div { color: var(--txt) !important; }
a { color: var(--accent) !important; }
hr { border-color: var(--border) !important; }

/* ── scrollbars ──────────────────────────────────── */
::-webkit-scrollbar { width:4px; height:4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius:10px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
"""

# ── Theme script ──────────────────────────────────────────────────────────────
_THEME_SCRIPT = """
<script id="analyst-pro-theme-script">
(function () {
  function applyTheme(t) {
    document.documentElement.setAttribute('data-theme', t === 'light' ? 'light' : 'dark');
  }

  // Read parent localStorage synchronously (same origin)
  var saved = 'dark';
  try { saved = window.parent.localStorage.getItem('analyst-theme') || 'dark'; } catch(e) {}
  applyTheme(saved);

  // Live updates from parent via postMessage
  window.addEventListener('message', function (e) {
    if (e.data && e.data.type === 'analyst-theme') {
      applyTheme(e.data.theme);
    }
  });
})();
</script>
"""


import re as _re

_EXTRA_CSS = """
<style id="analyst-pro-cells">
/* ── nuke any element-level white backgrounds ── */
* { background-color: inherit; }

/* ── stat value boxes ─────────────────────────── */
.col-sm-3, .col-md-3, .col-lg-3,
.result-col, .result-item,
[class*="col-"] {
  background-color: var(--surface) !important;
  color: var(--txt) !important;
  border-color: var(--border) !important;
}

/* ── every td and th absolutely ──────────────── */
td, th {
  background-color: var(--surface) !important;
  color: var(--txt) !important;
  border-color: var(--border) !important;
}
tr:nth-child(even) td { background-color: var(--card) !important; }
tr:hover td           { background-color: rgba(46,91,255,.05) !important; }

/* ── Bootstrap row / container backgrounds ───── */
.row, .col, [class^="col-"] { background-color: transparent !important; }
.container, .container-fluid { background-color: var(--bg) !important; }

/* ── input / form controls inside the report ─── */
input, select, textarea {
  background-color: var(--surface) !important;
  color: var(--txt) !important;
  border-color: var(--border) !important;
}

/* ── SVG / plotly charts: axis text ─────────── */
.gtitle, .xtick text, .ytick text,
text.legendtext, .annotation-text {
  fill: var(--txt) !important;
}
.gridlayer path, .zerolinelayer path {
  stroke: var(--border) !important;
}
</style>
"""


# ── FIX C: Rewrite navbar classes in raw HTML ─────────────────────────────────
# Bootstrap's bg-light / bg-white / navbar-light are utility classes whose
# specificity equals our overrides, so the cascade order decides the winner —
# and Bootstrap's stylesheet ships AFTER ours inside the ydata bundle.
# The safest fix is to remove those class tokens from the <nav>/<header> element
# directly in the HTML string before we inject anything.

_NAVBAR_TAG = _re.compile(
    r'(<(?:nav|header)\b[^>]*?)'   # opening tag up to (but not including) >
    r'(>)',
    _re.IGNORECASE | _re.DOTALL,
)

# Targets the Bootstrap collapse <div> — the mobile hamburger panel
_COLLAPSE_TAG = _re.compile(
    r'(<div\b[^>]*?\bnavbar-collapse\b[^>]*?)(>)',
    _re.IGNORECASE | _re.DOTALL,
)

_BG_CLASSES = _re.compile(
    r'\b(bg-light|bg-white|bg-secondary|navbar-light|navbar-dark)\b',
    _re.IGNORECASE,
)

_EXISTING_STYLE = _re.compile(
    r'\s*style\s*=\s*["\'][^"\']*["\']',
    _re.IGNORECASE,
)


def _fix_navbar_classes(html: str) -> str:
    """
    1. Strip Bootstrap bg-* utility classes from every <nav>/<header> tag.
    2. Stamp a hardcoded dark inline style onto every .navbar-collapse <div>
       so it is dark from byte-zero, before Bootstrap JS adds .show and before
       any external stylesheet or CSS variable has resolved.
    """
    def _scrub_tag(m: _re.Match) -> str:
        return _BG_CLASSES.sub('', m.group(1)) + m.group(2)

    def _darken_collapse(m: _re.Match) -> str:
        tag_open = m.group(1)
        # Wipe any existing style attr so we don't double-up
        tag_open = _EXISTING_STYLE.sub('', tag_open)
        # Hardcoded colours — no CSS-var dependency, wins regardless of timing
        tag_open += ' style="background-color:#0d0d0f;color:#e8e8ea;"'
        return tag_open + m.group(2)

    html = _NAVBAR_TAG.sub(_scrub_tag, html)
    html = _COLLAPSE_TAG.sub(_darken_collapse, html)
    return html


def _strip_inline_bg(html: str) -> str:
    """
    Remove inline background-color / background / color style declarations
    that Bootstrap + ydata-profiling bake directly onto elements.

    Preserves the inline style we stamped on .navbar-collapse in
    _fix_navbar_classes so it isn't wiped before it can take effect.
    """
    _STRIP_PROPS = _re.compile(
        r'\s*(?:background(?:-color)?|color'
        r'|border(?:-(?:top|bottom|left|right|color|style|width))?'
        r'|outline(?:-color)?'
        r'|box-shadow)\s*:\s*[^;"}}]+;?',
        _re.IGNORECASE,
    )
    _STYLE_ATTR = _re.compile(r'(\bstyle\s*=\s*["\'])([^"\']*?)(["\'])', _re.IGNORECASE)

    # Protect tags that contain navbar-collapse — we stamped a sentinel style on
    # those in _fix_navbar_classes and must not strip it here.
    _COLLAPSE_FULL_TAG = _re.compile(
        r'(<[^>]*?\bnavbar-collapse\b[^>]*?>)',
        _re.IGNORECASE | _re.DOTALL,
    )
    sentinel_map: dict = {}

    def _protect_collapse(m):
        key = f'\x00COLLAPSE_{len(sentinel_map)}\x00'
        sentinel_map[key] = m.group(1)
        return key

    def _clean_style(m):
        quote   = m.group(1)
        content = m.group(2)
        end     = m.group(3)
        cleaned = _STRIP_PROPS.sub('', content).strip().strip(';').strip()
        if not cleaned:
            return ''
        return f'{quote}{cleaned}{end}'

    # Protect collapse tags, strip everything else, then restore
    html = _COLLAPSE_FULL_TAG.sub(_protect_collapse, html)
    html = _STYLE_ATTR.sub(_clean_style, html)
    for key, val in sentinel_map.items():
        html = html.replace(key, val)
    return html
def _inject_theme(html: str) -> str:
    """
    Full post-processing pipeline:
      1. Set data-theme="dark" directly on <html> tag (no flash, no script race).
      2. Rewrite navbar class tokens (FIX C) — must happen before CSS injection.
      3. Strip inline background/color styles so CSS vars can take effect.
      4. Inject theme CSS (variables + component overrides).
      5. Inject extra CSS targeting stat-value cells and Bootstrap columns.
      6. Inject the postMessage / localStorage theme script (for live switching).
    """
    # Step 1 — stamp data-theme="dark" onto <html> immediately so CSS vars
    # resolve correctly from the very first paint, before any script runs.
    # The tiny inline script in <head> will then correct it to "light" if needed.
    _HTML_TAG = _re.compile(r'(<html\b[^>]*?)(>)', _re.IGNORECASE | _re.DOTALL)

    def _add_data_theme(m: _re.Match) -> str:
        tag = m.group(1)
        # Remove any existing data-theme so we don't double-add
        tag = _re.sub(r'\s*data-theme\s*=\s*["\'][^"\']*["\']', '', tag)
        return tag + ' data-theme="dark"' + m.group(2)

    html = _HTML_TAG.sub(_add_data_theme, html, count=1)

    # Step 2 — rewrite navbar classes (FIX C)
    html = _fix_navbar_classes(html)

    # Step 3 — strip inline bg/color overrides
    html = _strip_inline_bg(html)

    # Step 4+5 — inject CSS before </head>
    # Blocking inline script reads ?theme=dark|light from the iframe's own URL
    # (set by the parent when it builds the src).  This is race-free: it runs
    # synchronously before Bootstrap's stylesheet is parsed, so there is zero
    # flash of white.  Falls back to 'dark' if the param is absent.
    _EARLY_THEME_SCRIPT = """
<script>
(function(){
  var t='dark';
  try{
    var m=location.search.match(/[?&]theme=(dark|light)/);
    if(m) t=m[1];
    else t=window.parent.localStorage.getItem('analyst-theme')||'dark';
  }catch(e){}
  document.documentElement.setAttribute('data-theme',t);
})();
</script>"""

    combined_css = _THEME_CSS + _EXTRA_CSS
    if "<head>" in html:
        html = html.replace("<head>", "<head>" + _EARLY_THEME_SCRIPT, 1)
    if "</head>" in html:
        html = html.replace("</head>", combined_css + "\n</head>", 1)
    else:
        html = combined_css + html

    # Step 6 — inject live-switching script before </body>
    if "</body>" in html:
        html = html.replace("</body>", _THEME_SCRIPT + "\n</body>", 1)
    else:
        html = html + _THEME_SCRIPT

    return html


# ── Public API ────────────────────────────────────────────────────────────────

def generate_eda_report(
    df: pd.DataFrame,
    minimal: bool = True,
    sample_n: int = 5000,
) -> dict:
    """
    Generate a theme-aware ydata-profiling EDA report.

    Returns
    -------
    {
        "html":          str | None,
        "error":         str | None,
        "rows_profiled": int,
    }
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            from ydata_profiling import ProfileReport          # noqa: PLC0415
        except ImportError:
            return {
                "html": None,
                "error": "ydata-profiling not installed. Run: pip install ydata-profiling",
                "rows_profiled": 0,
            }

        n = int(sample_n) if sample_n else 0
        if n and n < len(df):
            profile_df = df.sample(n=n, random_state=42)
        else:
            profile_df = df.copy()

        profile_df = _sanitise_dtypes(profile_df)

        try:
            report = ProfileReport(
                profile_df,
                title="Dataset EDA Report",
                minimal=minimal,
                explorative=not minimal,
                progress_bar=False,
            )
            html_str = report.to_html()
        except Exception as exc:
            return {"html": None, "error": str(exc), "rows_profiled": 0}

    html_str = _inject_theme(html_str)

    return {
        "html":          html_str,
        "error":         None,
        "rows_profiled": len(profile_df),
    }