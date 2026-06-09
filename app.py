"""
HydroStar — Rainwater Harvesting Estimator
==========================================
A simple, robust Streamlit dashboard that estimates how much rainwater a site
can collect, using historic Cardiff climate data as the baseline.

It is built directly on the manager's analysis. Her core formula is:

    Volume (litres) = Rainfall (mm) x Catchment Area (m2) x Runoff Coefficient

(1 mm of rain falling on 1 m2 of roof = 1 litre, before losses. The runoff
coefficient accounts for what is lost to splashing, evaporation and the roof
surface.)

DATA SOURCES (the only two inputs — no live API calls)
  - rainwater_station_raw.csv ...... Cardiff monthly station data
                                     (Year, Month, Frost_Days, Rain_mm,
                                      Month_Name, Season)
  - rainwater_monthly_summary.csv .. monthly aggregates her notebook produced
                                     (average / min / max / std rainfall per
                                      month, plus her harvesting metrics)

The app RE-COMPUTES every volume live from the values the user types in, so the
defaults below reproduce the manager's numbers, but the user can model any site.

Files expected alongside this app.py:
  app.py · requirements.txt · logo.png · rainwater_station_raw.csv ·
  rainwater_monthly_summary.csv

Run with:  streamlit run app.py
"""

import base64
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================================
# 1. CONSTANTS — file names, brand colours, and modelling defaults
# ============================================================================

STATION_FILE = "rainwater_station_raw.csv"
SUMMARY_FILE = "rainwater_monthly_summary.csv"
LOGO_FILE = "logo.png"

# --- HydroStar brand palette (from the brand guidelines / style reference) ---
PRIMARY = "#a7d730"      # primary green
SECONDARY = "#499823"    # secondary green
WATER_BLUE = "#4ea8de"   # used for rainfall (reads as "water")
AMBER = "#f6a609"        # warnings / highlights

# Season ordering and month helpers, shared by the filters and charts.
SEASON_ORDER = ["Spring", "Summer", "Autumn", "Winter"]
MONTH_ORDER = list(range(1, 13))

# Standard rainwater-harvesting runoff coefficients for common roof surfaces.
# The two the manager used are marked; "Custom..." reveals a slider instead.
ROOF_TYPES = {
    "Metal / steel roof": 0.90,          # manager's 40ft container roof
    "Tiled / pitched roof": 0.85,
    "Concrete roof": 0.75,               # manager's Welsh Water building roof
    "Asphalt / bitumen flat roof": 0.70,
    "Gravel flat roof": 0.60,
    "Green / vegetated roof": 0.30,
    "Custom...": None,                    # None => ask for a coefficient directly
}

# Manager's site defaults, so the app opens on numbers that match her summary.
# A 40 ft ISO shipping container footprint is 12.19 m x 2.44 m.
CONTAINER_AREA_DEFAULT = round(12.19 * 2.44, 2)   # ~29.74 m2
BUILDING_AREA_DEFAULT = 500.0


# ============================================================================
# 2. PURE CALCULATION HELPERS (no Streamlit here, so they are easy to reason
#    about and test). These implement the manager's harvesting maths.
# ============================================================================

def surface_factor(area_m2: float, coeff: float, count: int, included: bool) -> float:
    """Litres collected per 1 mm of rain for one surface type.

    Volume = Rainfall(mm) x Area(m2) x Coefficient, so the "per-mm" factor for
    `count` identical surfaces is simply area x coefficient x count.
    """
    return float(area_m2) * float(coeff) * int(count) if included else 0.0


def compute_monthly(
    summary: pd.DataFrame,
    station: pd.DataFrame,
    mode: str,
    year: int | None,
    month_names: list[str],
    f_container: float,
    f_building: float,
) -> pd.DataFrame:
    """Build one row per selected month with rainfall and collectable volume.

    `mode` chooses the rainfall baseline:
      - "summary": long-term monthly AVERAGE rainfall (the "typical year").
      - "station": the ACTUAL rainfall of a single chosen `year`.

    Historical min/max/avg (used for the range bands) always come from the
    summary, so the bands mean the same thing in both modes.
    """
    f_total = f_container + f_building

    cols = ["Month", "Month_Name", "Season", "Avg_Rain_mm", "Min_Rain_mm",
            "Max_Rain_mm", "Std_Rain_mm", "Days_in_Month"]
    base = summary.loc[summary["Month_Name"].isin(month_names), cols].copy()
    base = base.rename(columns={
        "Avg_Rain_mm": "rain_avg", "Min_Rain_mm": "rain_min",
        "Max_Rain_mm": "rain_max", "Std_Rain_mm": "rain_std",
        "Days_in_Month": "days",
    })

    if mode == "station" and year is not None:
        # Pull this year's real monthly rainfall; inner-join drops any month
        # that year has no reading for (handles partial years gracefully).
        yr = station.loc[station["Year"] == year, ["Month", "Rain_mm"]]
        base = base.merge(yr, on="Month", how="inner")
        base["rain_mm"] = base["Rain_mm"]
        base["rain_err"] = 0.0          # no std band for a single year
        base = base.drop(columns=["Rain_mm"])
    else:
        base["rain_mm"] = base["rain_avg"]
        base["rain_err"] = base["rain_std"]

    base = base.sort_values("Month").reset_index(drop=True)

    # Collectable volume in litres for each rainfall figure of interest.
    base["container_litres"] = base["rain_mm"] * f_container
    base["building_litres"] = base["rain_mm"] * f_building
    base["total_litres"] = base["rain_mm"] * f_total
    base["avg_litres"] = base["rain_avg"] * f_total   # long-term typical
    base["min_litres"] = base["rain_min"] * f_total   # historical driest month
    base["max_litres"] = base["rain_max"] * f_total   # historical wettest month

    # Cubic-metre versions of everything (1 m3 = 1000 litres).
    for c in ["container", "building", "total", "avg", "min", "max"]:
        base[f"{c}_m3"] = base[f"{c}_litres"] / 1000.0

    return base


def summarise(view: pd.DataFrame, f_total: float, area_eff: float) -> dict:
    """Headline numbers for the summary panel, computed over the selected months."""
    if view.empty:
        return {"annual_litres": 0.0, "annual_m3": 0.0, "daily_litres": 0.0,
                "env_min_m3": 0.0, "env_max_m3": 0.0, "peak": ("-", 0.0),
                "low": ("-", 0.0), "area_eff": area_eff, "blended": 0.0,
                "n_months": 0}

    annual_litres = float(view["total_litres"].sum())
    total_days = float(view["days"].sum())
    peak_row = view.loc[view["total_m3"].idxmax()]
    low_row = view.loc[view["total_m3"].idxmin()]

    return {
        "annual_litres": annual_litres,
        "annual_m3": annual_litres / 1000.0,
        "daily_litres": annual_litres / total_days if total_days else 0.0,
        # Theoretical envelope: every month at its historical driest / wettest.
        "env_min_m3": float(view["min_m3"].sum()),
        "env_max_m3": float(view["max_m3"].sum()),
        "peak": (peak_row["Month_Name"], float(peak_row["total_m3"])),
        "low": (low_row["Month_Name"], float(low_row["total_m3"])),
        "area_eff": area_eff,
        "blended": (f_total / area_eff) if area_eff else 0.0,
        "n_months": len(view),
    }


# Tiny display formatters used throughout the UI.
def fmt_l(x: float) -> str:
    return f"{x:,.0f} L"


def fmt_m3(x: float) -> str:
    return f"{x:,.1f} m\u00b3"


# ============================================================================
# 3. THEME — HydroStar colours for dark and light mode, injected as CSS
# ============================================================================

# One token dictionary per theme; the CSS template below is filled from it.
DARK = {
    "app_bg": ("radial-gradient(circle at top right, rgba(167,215,48,0.11) 0%, rgba(14,17,23,0) 35%),"
               "radial-gradient(circle at bottom left, rgba(78,168,222,0.08) 0%, rgba(14,17,23,0) 40%), #0e1117"),
    "text": "#f2f4f7", "subtext": "#8c919a",
    "sidebar_grad": "linear-gradient(180deg, rgba(48,52,60,0.98) 0%, rgba(28,34,43,0.98) 72%, rgba(20,25,32,0.98) 100%)",
    "sidebar_border": "rgba(255,255,255,0.10)", "section_label": "rgba(255,255,255,0.66)",
    "card_bg": "linear-gradient(135deg, rgba(167,215,48,0.16), rgba(78,168,222,0.10))",
    "card_border": "rgba(255,255,255,0.12)",
    "input_bg": "rgba(255,255,255,0.06)", "input_border": "rgba(255,255,255,0.16)",
    "metric_bg": "linear-gradient(180deg, rgba(27,34,43,0.96) 0%, rgba(22,29,37,0.96) 100%)",
    "metric_border": "rgba(255,255,255,0.10)",
    "table_bg": "rgba(27,34,43,0.96)", "table_border": "rgba(255,255,255,0.08)",
    "chart_bg": "rgba(27,34,43,0.45)", "chart_border": "rgba(255,255,255,0.06)",
    "hero_bg": "linear-gradient(90deg, rgba(12,16,24,0.92) 0%, rgba(18,30,22,0.88) 70%, rgba(29,52,33,0.78) 100%)",
    "hero_border": "rgba(255,255,255,0.12)", "logo_filter": "drop-shadow(0 6px 14px rgba(0,0,0,0.35))",
    "soft_bg": "rgba(27,34,43,0.7)", "soft_border": "rgba(255,255,255,0.08)",
    "divider": "rgba(255,255,255,0.12)",
}
LIGHT = {
    "app_bg": ("radial-gradient(circle at top right, rgba(167,215,48,0.09) 0%, rgba(247,249,244,0) 35%),"
               "radial-gradient(circle at bottom left, rgba(73,152,35,0.06) 0%, rgba(247,249,244,0) 40%), #f7f9f4"),
    "text": "#1a2010", "subtext": "#4a5240",
    "sidebar_grad": "linear-gradient(180deg, #eef3e6 0%, #e6f0da 100%)",
    "sidebar_border": "rgba(73,152,35,0.20)", "section_label": "#4a5240",
    "card_bg": "linear-gradient(135deg, rgba(167,215,48,0.18), rgba(73,152,35,0.10))",
    "card_border": "rgba(73,152,35,0.25)",
    "input_bg": "rgba(255,255,255,0.85)", "input_border": "rgba(73,152,35,0.30)",
    "metric_bg": "linear-gradient(180deg, #ffffff 0%, #f7f9f4 100%)",
    "metric_border": "rgba(73,152,35,0.15)",
    "table_bg": "#ffffff", "table_border": "rgba(73,152,35,0.15)",
    "chart_bg": "rgba(255,255,255,0.80)", "chart_border": "rgba(73,152,35,0.14)",
    "hero_bg": "linear-gradient(90deg, rgba(247,249,244,0.97) 0%, rgba(230,240,218,0.95) 70%, rgba(210,235,185,0.90) 100%)",
    "hero_border": "rgba(73,152,35,0.20)", "logo_filter": "none",
    "soft_bg": "rgba(240,247,224,0.80)", "soft_border": "rgba(73,152,35,0.15)",
    "divider": "rgba(73,152,35,0.20)",
}

# CSS uses ${name} placeholders (string.Template) so we never have to escape the
# many literal { } braces that CSS itself needs.
_CSS = Template("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Hind:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Hind', sans-serif; }

.stApp { background: ${app_bg}; color: ${text}; }
.block-container { max-width: 1480px; padding: 1.6rem clamp(1rem,2.2vw,2.4rem) 2rem; }
h1,h2,h3,h4,h5,h6 { color: ${text} !important; font-weight: 700; letter-spacing: .1px; }
p, span, label, li { color: ${text} !important; }
.stCaption, .stMarkdown small { color: ${subtext} !important; }

/* Sidebar shell */
section[data-testid="stSidebar"] > div {
    background: ${sidebar_grad}; border-right: 1px solid ${sidebar_border}; }
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span { color: ${text} !important; }
section[data-testid="stSidebar"] hr { margin: .9rem 0; border-color: ${divider}; }

.sidebar-card {
    padding: 1rem 1rem .9rem; border-radius: 14px; margin-bottom: .8rem;
    background: ${card_bg}; border: 1px solid ${card_border}; }
.sidebar-kicker { color: ${primary} !important; font-size:.78rem; font-weight:700;
    letter-spacing:.06em; text-transform:uppercase; margin:0 0 .25rem; }
.sidebar-title { color: ${text} !important; font-size:1.3rem; font-weight:700;
    line-height:1.1; margin:0; }
.sidebar-sub { color: ${subtext} !important; font-size:.9rem; margin:.35rem 0 0; }
.sidebar-label { color: ${section_label} !important; font-size:.76rem; font-weight:700;
    letter-spacing:.07em; text-transform:uppercase; margin:.4rem 0 .2rem; }

/* Inputs */
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div,
.stSelectbox > div > div, .stMultiSelect > div > div,
input[type="number"], .stTextInput input {
    background-color: ${input_bg} !important; border-color: ${input_border} !important;
    color: ${text} !important; }
div[data-baseweb="select"] span, ul[data-testid="stSelectboxVirtualDropdown"] li,
ul[data-testid="stSelectboxVirtualDropdown"] span { color: ${text} !important; }
.stSlider [data-testid="stTickBar"] > div { background-color: rgba(167,215,48,.40); }
.stButton > button, .stDownloadButton > button {
    background-color: ${primary}; color:#1d2430; font-weight:700; border:none; border-radius:8px; }
.stButton > button:hover, .stDownloadButton > button:hover {
    background-color: ${secondary}; color:#fff; }

/* Metric tiles (custom grid so they wrap nicely) */
.metric-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr));
    gap:.85rem; margin:.25rem 0 1rem; }
.metric-tile { min-width:0; background:${metric_bg}; border:1px solid ${metric_border};
    border-left:5px solid ${primary}; border-radius:12px; padding:.85rem 1rem;
    box-shadow:0 6px 16px rgba(0,0,0,.12); }
.metric-label { display:block; color:${subtext} !important; font-size:.76rem; font-weight:700;
    letter-spacing:.04em; text-transform:uppercase; margin-bottom:.35rem; }
.metric-value { display:block; color:${text} !important;
    font-size:clamp(1.2rem,1.7vw,1.7rem); font-weight:700; line-height:1.1; overflow-wrap:anywhere; }

/* Cards, table, hero */
div[data-testid="stDataFrame"] { background-color:${table_bg}; border:1px solid ${table_border};
    border-radius:12px; padding:.2rem; }
.chart-card { background-color:${chart_bg}; border:1px solid ${chart_border};
    border-radius:12px; padding:.55rem 1.2rem .25rem .55rem; margin-bottom:1rem; }
.hero { display:flex; justify-content:space-between; align-items:center; gap:1.2rem;
    padding:1.1rem 1.4rem; border-radius:14px; margin-bottom:1.1rem;
    background:${hero_bg}; border:1px solid ${hero_border}; }
.hero-copy { max-width:72%; }
.hero-title { margin:0; font-size:clamp(1.7rem,2.6vw,2.5rem); line-height:1.1; font-weight:700; }
.hero-sub { margin:.4rem 0 0; color:${subtext} !important; font-size:1rem; }
.hero-logo img { height:88px; width:auto; object-fit:contain; filter:${logo_filter}; }
.hero-wordmark { font-size:1.6rem; font-weight:700; color:${primary} !important; letter-spacing:.5px; }
.soft-card { background:${soft_bg}; border:1px solid ${soft_border}; border-left:4px solid ${primary};
    border-radius:12px; padding:.85rem 1.1rem; margin-bottom:.9rem; }
.soft-card p { margin:0; color:${subtext} !important; font-size:.95rem; }

@media (max-width:1080px){ .hero{flex-direction:column;align-items:flex-start;}
    .hero-copy{max-width:100%;} .metric-grid{grid-template-columns:repeat(2,minmax(0,1fr));} }
@media (max-width:760px){ .metric-grid{grid-template-columns:1fr;} .hero-logo img{height:60px;} }
</style>
""")


def inject_css(theme: str) -> None:
    """Write the HydroStar CSS for the active theme into the page."""
    tokens = LIGHT if theme == "light" else DARK
    tokens = {**tokens, "primary": PRIMARY, "secondary": SECONDARY}
    st.markdown(_CSS.substitute(tokens), unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def encode_logo() -> str:
    """Base64-encode logo.png if present (returns "" so a missing logo is fine)."""
    p = Path(LOGO_FILE)
    return base64.b64encode(p.read_bytes()).decode("utf-8") if p.exists() else ""


def is_light() -> bool:
    return st.session_state.get("theme", "dark") == "light"


def apply_chart_layout(fig: go.Figure, height: int = 380) -> go.Figure:
    """Consistent, theme-aware Plotly styling.

    Solid (not transparent) plot backgrounds are intentional: they keep axis
    labels visible if the user exports a chart to PNG.
    """
    if is_light():
        tmpl, panel, font_c = "plotly_white", "#ffffff", "#1a2010"
        grid, line = "rgba(0,0,0,0.08)", "rgba(0,0,0,0.20)"
    else:
        tmpl, panel, font_c = "plotly_dark", "#1b222b", "#f2f4f7"
        grid, line = "rgba(255,255,255,0.08)", "rgba(255,255,255,0.18)"
    axis_font = dict(color=font_c, family="Hind, sans-serif")

    fig.update_layout(
        template=tmpl, height=height, plot_bgcolor=panel, paper_bgcolor=panel,
        font=dict(color=font_c, family="Hind, sans-serif"),
        margin=dict(l=58, r=24, t=28, b=52),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)", font=axis_font),
        hoverlabel=dict(font=dict(family="Hind, sans-serif")),
    )
    fig.update_xaxes(gridcolor=grid, linecolor=line, zeroline=False,
                     tickfont=axis_font, title_font=axis_font, automargin=True)
    fig.update_yaxes(gridcolor=grid, linecolor=line, zeroline=False,
                     tickfont=axis_font, title_font=axis_font, automargin=True)
    return fig


PLOTLY_CONFIG = {"displaylogo": False, "responsive": True,
                 "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def render_chart(fig: go.Figure) -> None:
    """Draw a chart inside the styled card."""
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    st.markdown('</div>', unsafe_allow_html=True)


def metric_grid(items: list[tuple[str, str]]) -> None:
    """Render the summary numbers as responsive tiles."""
    tiles = "".join(
        f'<div class="metric-tile"><span class="metric-label">{lbl}</span>'
        f'<span class="metric-value">{val}</span></div>'
        for lbl, val in items
    )
    st.markdown(f'<div class="metric-grid">{tiles}</div>', unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    """Top banner with the HydroStar logo (or a text wordmark fallback)."""
    logo_b64 = encode_logo()
    logo = (f'<div class="hero-logo"><img src="data:image/png;base64,{logo_b64}" '
            f'alt="HydroStar logo"></div>' if logo_b64
            else '<div class="hero-wordmark">HydroStar</div>')
    st.markdown(
        f'<div class="hero"><div class="hero-copy">'
        f'<h1 class="hero-title">{title}</h1><p class="hero-sub">{subtitle}</p>'
        f'</div>{logo}</div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# 4. DATA LOADING — the two CSVs, read once from the current directory
# ============================================================================

@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read both CSVs with pandas. Stops the app with a clear message if missing."""
    for f in (STATION_FILE, SUMMARY_FILE):
        if not Path(f).exists():
            st.error(
                f"Could not find **{f}**. Place it in the same folder as app.py "
                "and refresh."
            )
            st.stop()
    station = pd.read_csv(STATION_FILE)
    summary = pd.read_csv(SUMMARY_FILE).sort_values("Month").reset_index(drop=True)
    return station, summary


# ============================================================================
# 5. SIDEBAR CONTROL — one reusable block for a catchment surface
# ============================================================================

def surface_controls(title: str, key: str, default_label: str,
                      default_roof: str, default_area: float) -> dict:
    """Inputs for one surface (container or building) and its per-mm factor."""
    on = st.checkbox(f"Include {title}", value=True, key=f"{key}_on")
    label = st.text_input("Label", value=default_label, key=f"{key}_label",
                          disabled=not on)
    roof = st.selectbox("Roof type", list(ROOF_TYPES),
                        index=list(ROOF_TYPES).index(default_roof),
                        key=f"{key}_roof", disabled=not on,
                        help="Sets the runoff coefficient. Pick 'Custom...' to type your own.")
    if ROOF_TYPES[roof] is None:
        coeff = st.slider("Runoff coefficient", 0.05, 1.00, 0.80, 0.01,
                          key=f"{key}_coeff", disabled=not on)
    else:
        coeff = ROOF_TYPES[roof]
        st.caption(f"Runoff coefficient: **{coeff:.2f}**")
    area = st.number_input("Catchment area (m\u00b2)", min_value=0.0,
                           value=float(default_area), step=1.0,
                           key=f"{key}_area", disabled=not on)
    count = st.number_input("How many of these", min_value=1, value=1, step=1,
                            key=f"{key}_count", disabled=not on)
    return {
        "on": on, "label": label or title, "coeff": coeff, "area": area,
        "count": count, "factor": surface_factor(area, coeff, count, on),
        "eff_area": area * count if on else 0.0,
    }


# ============================================================================
# 6. MAIN APP
# ============================================================================

def main() -> None:
    # set_page_config must be the first Streamlit call.
    st.set_page_config(page_title="HydroStar Rainwater Estimator",
                       page_icon=LOGO_FILE if Path(LOGO_FILE).exists() else "\U0001F4A7",
                       layout="wide", initial_sidebar_state="expanded")

    station, summary = load_data()
    year_min, year_max = int(station["Year"].min()), int(station["Year"].max())

    # ---- SIDEBAR -----------------------------------------------------------
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-card"><p class="sidebar-kicker">HydroStar</p>'
            '<p class="sidebar-title">Rainwater Estimator</p>'
            '<p class="sidebar-sub">Cardiff climate baseline</p></div>',
            unsafe_allow_html=True,
        )

        # Theme toggle — controls which HydroStar CSS palette is applied.
        light = st.toggle("Light mode", value=st.session_state.get("theme", "dark") == "light")
        st.session_state["theme"] = "light" if light else "dark"

        # Which dataset feeds the rainfall baseline.
        st.markdown('<p class="sidebar-label">Rainfall baseline</p>', unsafe_allow_html=True)
        source = st.radio(
            "Data source",
            ["Monthly summary (1978-2025 averages)", "Station records (one year)"],
            label_visibility="collapsed",
            help="Use the long-term average (a typical year) or one specific year's actual rainfall.",
        )
        mode = "summary" if source.startswith("Monthly") else "station"
        year = None
        if mode == "station":
            years = sorted(station["Year"].unique(), reverse=True)
            year = st.selectbox("Year", years, index=0)
            n_yr = int((station["Year"] == year).sum())
            if n_yr < 12:
                st.caption(f"\u26a0\ufe0f {year} has only {n_yr} month(s) of data.")

        # Month / season filter. Months are derived from the chosen seasons.
        st.markdown('<p class="sidebar-label">Period</p>', unsafe_allow_html=True)
        sel_seasons = st.multiselect("Seasons", SEASON_ORDER, default=SEASON_ORDER)
        cand = (summary[summary["Season"].isin(sel_seasons)]
                .sort_values("Month")["Month_Name"].tolist())
        sel_months = st.multiselect("Months", cand, default=cand)

        # Site setup — two configurable catchment surfaces.
        st.markdown('<p class="sidebar-label">Site setup</p>', unsafe_allow_html=True)
        with st.expander("Container surface", expanded=True):
            container = surface_controls("container", "container",
                                         "40 ft container", "Metal / steel roof",
                                         CONTAINER_AREA_DEFAULT)
        with st.expander("Building surface", expanded=True):
            building = surface_controls("building", "building",
                                        "Welsh Water building", "Concrete roof",
                                        BUILDING_AREA_DEFAULT)

        f_container, f_building = container["factor"], building["factor"]
        f_total = f_container + f_building
        area_eff = container["eff_area"] + building["eff_area"]
        st.markdown(
            '<div class="soft-card"><p>'
            f'Effective catchment: <b>{area_eff:,.1f} m\u00b2</b><br>'
            f'Collects <b>{f_total:,.1f} L</b> per mm of rain</p></div>',
            unsafe_allow_html=True,
        )

    inject_css("light" if is_light() else "dark")

    # ---- HEADER ------------------------------------------------------------
    period_note = ("long-term average rainfall (1978-2025)" if mode == "summary"
                   else f"actual rainfall from {year}")
    hero("Rainwater Harvesting Estimator",
         f"Estimating collectable rainwater for your site using {period_note}.")

    st.markdown(
        '<div class="soft-card"><p>Enter your site in the sidebar. Every figure '
        'below is computed live as <b>Rainfall (mm) × Catchment area (m²) '
        '× Runoff coefficient</b>.</p></div>',
        unsafe_allow_html=True,
    )

    # ---- GUARD RAILS -------------------------------------------------------
    if not sel_months:
        st.warning("Select at least one month in the sidebar to see results.")
        st.stop()
    if f_total == 0:
        st.warning("Add at least one catchment surface (tick *Include* and set an "
                   "area greater than 0) to estimate a volume.")
        st.stop()

    # ---- CALCULATE ---------------------------------------------------------
    view = compute_monthly(summary, station, mode, year, sel_months,
                           f_container, f_building)
    s = summarise(view, f_total, area_eff)
    months_label = "full year" if s["n_months"] == 12 else f"{s['n_months']} selected month(s)"

    # ---- SUMMARY PANEL -----------------------------------------------------
    st.subheader("Summary")
    metric_grid([
        (f"Collectable ({months_label})", fmt_m3(s["annual_m3"])),
        ("In litres", fmt_l(s["annual_litres"])),
        ("Peak month", f"{s['peak'][0]} \u00b7 {s['peak'][1]:,.1f} m\u00b3"),
        ("Quietest month", f"{s['low'][0]} \u00b7 {s['low'][1]:,.1f} m\u00b3"),
    ])
    st.caption(
        f"Averages about **{s['daily_litres']:,.0f} litres/day**. "
        f"Across history each month has ranged from its driest to wettest on record, "
        f"a theoretical envelope of **{s['env_min_m3']:,.1f}-{s['env_max_m3']:,.1f} m\u00b3** "
        f"over this period. Effective catchment **{s['area_eff']:,.1f} m\u00b2** at a "
        f"blended runoff coefficient of **{s['blended']:.2f}**."
    )

    err_col = "rgba(0,0,0,0.45)" if is_light() else "rgba(255,255,255,0.55)"
    names = view["Month_Name"].tolist()

    # ---- CHART 1: monthly rainfall (Bar Chart + Error Bars) ----------------
    st.subheader("Monthly rainfall")
    st.caption("Bar chart with error bars \u2014 Cardiff station data.")
    fig1 = go.Figure()
    if mode == "summary":
        fig1.add_bar(
            x=names, y=view["rain_avg"], marker_color=WATER_BLUE,
            name="Average rainfall",
            error_y=dict(type="data", array=view["rain_std"],
                         color=err_col, thickness=1.3, width=4),
            customdata=view["rain_std"].values.reshape(-1, 1),
            hovertemplate="<b>%{x}</b><br>Avg: %{y:.1f} mm<br>\u00b1%{customdata[0]:.1f} mm (1\u03c3)<extra></extra>",
        )
    else:
        fig1.add_bar(
            x=names, y=view["rain_mm"], marker_color=WATER_BLUE,
            name=f"{year} rainfall",
            hovertemplate=f"<b>%{{x}}</b><br>{year}: %{{y:.1f}} mm<extra></extra>",
        )
        fig1.add_scatter(
            x=names, y=view["rain_avg"], mode="lines+markers",
            name="1978-2025 average",
            line=dict(color=PRIMARY, width=2, dash="dash"),
            hovertemplate="<b>%{x}</b><br>Long-term avg: %{y:.1f} mm<extra></extra>",
        )
    fig1.update_yaxes(title_text="Rainfall (mm)")
    render_chart(apply_chart_layout(fig1))

    # ---- CHART 2: collectable volume (Bar Chart + min/max range) -----------
    st.subheader("Predicted collectable volume")
    st.caption("Bar chart with historical min/max range \u2014 your site parameters.")
    lower = np.clip(view["total_m3"] - view["min_m3"], 0, None)
    upper = np.clip(view["max_m3"] - view["total_m3"], 0, None)
    fig2 = go.Figure()
    fig2.add_bar(
        x=names, y=view["total_m3"], marker_color=PRIMARY,
        name="Collectable volume",
        error_y=dict(type="data", symmetric=False, array=upper,
                     arrayminus=lower, color=err_col, thickness=1.3, width=4),
        customdata=np.stack([view["min_m3"], view["max_m3"]], axis=1),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Baseline: %{y:.1f} m\u00b3<br>"
            "Hist. min: %{customdata[0]:.1f} m\u00b3<br>"
            "Hist. max: %{customdata[1]:.1f} m\u00b3"
            "<extra></extra>"
        ),
    )
    fig2.update_yaxes(title_text="Collectable volume (m\u00b3)")
    render_chart(apply_chart_layout(fig2))
    st.caption("Bars use the chosen rainfall baseline; whiskers span the driest to "
               "wettest that month has been since 1978.")

    # ---- CHART 3: where the water comes from (Stacked Bar) -----------------
    if container["on"] and container["area"] > 0 and building["on"] and building["area"] > 0:
        st.subheader("Where the water comes from")
        st.caption("Stacked bar \u2014 each surface's share of the monthly total.")
        fig3 = go.Figure()
        fig3.add_bar(
            x=names, y=view["container_m3"], name=container["label"],
            marker_color=PRIMARY,
            hovertemplate="<b>%{x}</b><br>" + container["label"] + ": %{y:.1f} m\u00b3<extra></extra>",
        )
        fig3.add_bar(
            x=names, y=view["building_m3"], name=building["label"],
            marker_color=WATER_BLUE,
            hovertemplate="<b>%{x}</b><br>" + building["label"] + ": %{y:.1f} m\u00b3<extra></extra>",
        )
        fig3.update_layout(barmode="stack")
        fig3.update_yaxes(title_text="Collectable volume (m\u00b3)")
        render_chart(apply_chart_layout(fig3))

    # ---- CHART 4: cumulative collection (running total) --------------------
    st.subheader("Cumulative collection across the period")
    st.caption("Running total \u2014 useful for estimating tank sizing and demand coverage.")
    cumulative = view["total_m3"].cumsum()
    cum_avg = view["avg_m3"].cumsum()
    cum_min = view["min_m3"].cumsum()
    cum_max = view["max_m3"].cumsum()
    fig4 = go.Figure()
    fig4.add_scatter(
        x=names, y=cum_min,
        mode="lines", line=dict(width=0), showlegend=False,
        hoverinfo="skip",
    )
    fig4.add_scatter(
        x=names, y=cum_max,
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(167,215,48,0.15)",
        name="Hist. min-max range", showlegend=True,
        hoverinfo="skip",
    )
    fig4.add_scatter(
        x=names, y=cum_avg, mode="lines+markers",
        name="Long-term avg",
        line=dict(color=WATER_BLUE, width=2, dash="dot"),
        hovertemplate="<b>%{x}</b><br>Cumulative avg: %{y:.1f} m\u00b3<extra></extra>",
    )
    fig4.add_scatter(
        x=names, y=cumulative, mode="lines+markers",
        name="This baseline",
        line=dict(color=PRIMARY, width=2.5),
        marker=dict(size=7),
        hovertemplate="<b>%{x}</b><br>Cumulative: %{y:.1f} m\u00b3<extra></extra>",
    )
    fig4.update_yaxes(title_text="Cumulative volume (m\u00b3)")
    render_chart(apply_chart_layout(fig4))
    st.caption(
        "The green band is the envelope of cumulative totals if every month sat at "
        "its historical driest or wettest. The dotted blue line is the long-term "
        "average trajectory; the solid green line is your current baseline."
    )

    # ---- CHART 5: seasonal summary (collectable volume by season) ----------
    st.subheader("Collection by season")
    st.caption("Total collectable volume grouped by season \u2014 spot your wet vs dry seasons at a glance.")
    season_view = (
        view.groupby("Season", sort=False)
        .agg(total_m3=("total_m3", "sum"),
             min_m3=("min_m3", "sum"),
             max_m3=("max_m3", "sum"))
        .reindex([s for s in SEASON_ORDER if s in view["Season"].values])
        .reset_index()
    )
    season_colours = {
        "Spring": "#7ec850", "Summer": AMBER,
        "Autumn": "#e07b30", "Winter": WATER_BLUE,
    }
    fig5 = go.Figure()
    fig5.add_bar(
        x=season_view["Season"],
        y=season_view["total_m3"],
        marker_color=[season_colours.get(s, PRIMARY) for s in season_view["Season"]],
        name="Collectable (baseline)",
        customdata=np.stack([season_view["min_m3"], season_view["max_m3"]], axis=1),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Baseline: %{y:.1f} m\u00b3<br>"
            "Hist. min: %{customdata[0]:.1f} m\u00b3<br>"
            "Hist. max: %{customdata[1]:.1f} m\u00b3"
            "<extra></extra>"
        ),
    )
    fig5.update_yaxes(title_text="Collectable volume (m\u00b3)")
    render_chart(apply_chart_layout(fig5, height=320))

    # ---- CHART 6: year-over-year annual rainfall heatmap -------------------
    st.subheader("Year-over-year rainfall heatmap")
    st.caption("Cardiff monthly rainfall (mm) across all station years \u2014 warmer = wetter.")
    _month_order = ["January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"]
    hm_data = (station.pivot_table(index="Year", columns="Month_Name",
                                   values="Rain_mm", aggfunc="sum")
               .reindex(columns=_month_order))
    hm_years = hm_data.index.tolist()
    hm_zero = "#f7f9f4" if is_light() else "#1b222b"
    fig6 = go.Figure(go.Heatmap(
        z=hm_data.values,
        x=_month_order,
        y=hm_years,
        colorscale=[[0, hm_zero], [0.35, WATER_BLUE], [0.7, PRIMARY], [1, "#ffffff"]],
        colorbar=dict(title="mm", thickness=14, len=0.8),
        hovertemplate="<b>%{y} %{x}</b><br>Rainfall: %{z:.1f} mm<extra></extra>",
        zmin=0,
    ))
    fig6.update_yaxes(title_text="Year", autorange="reversed", tickmode="auto", nticks=20)
    fig6.update_xaxes(title_text="Month")
    render_chart(apply_chart_layout(fig6, height=max(320, min(900, 14 * len(hm_years)))))
    st.caption(
        "Each cell is one month's total rainfall at the Cardiff station. "
        "Blank cells are months with no reading (partial years at the start/end of the record)."
    )

    # ---- MONTHLY DETAIL TABLE (+ download) ---------------------------------
    st.subheader("Monthly detail")
    table = view[["Month_Name", "Season", "rain_mm", "container_litres",
                  "building_litres", "total_litres", "total_m3",
                  "min_m3", "max_m3"]].copy()
    table.columns = ["Month", "Season", "Rainfall (mm)", "Container (L)",
                     "Building (L)", "Total (L)", "Total (m\u00b3)",
                     "Hist. min (m\u00b3)", "Hist. max (m\u00b3)"]
    st.dataframe(
        table.style.format({
            "Rainfall (mm)": "{:.1f}", "Container (L)": "{:,.0f}",
            "Building (L)": "{:,.0f}", "Total (L)": "{:,.0f}",
            "Total (m\u00b3)": "{:.2f}", "Hist. min (m\u00b3)": "{:.2f}",
            "Hist. max (m\u00b3)": "{:.2f}",
        }),
        use_container_width=True, hide_index=True,
    )
    st.download_button("Download this table (CSV)",
                       table.to_csv(index=False).encode("utf-8"),
                       file_name="rainwater_estimate.csv", mime="text/csv")

    # ---- EXPLAINER ---------------------------------------------------------
    with st.expander("How this is calculated"):
        st.markdown(
            f"""
**The formula** (from the source analysis)

> Volume (litres) = Rainfall (mm) × Catchment area (m²) × Runoff coefficient

One millimetre of rain on one square metre of roof is one litre before losses;
the runoff coefficient (0-1) accounts for what the roof loses.

**Your current site**
- {container['label']}: {container['area']:,.1f} m² × {container['count']} × runoff {container['coeff']:.2f}
- {building['label']}: {building['area']:,.1f} m² × {building['count']} × runoff {building['coeff']:.2f}
- Combined, the site collects **{f_total:,.1f} litres per mm of rain**.

**The data**
- *Station records*: monthly Cardiff rainfall, {year_min}-{year_max}.
- *Monthly summary*: the average, minimum, maximum and standard deviation of
  rainfall for each calendar month, computed over complete years (1978-2025).
- The min/max whiskers reuse those historical extremes, so they mean the same
  thing whichever baseline you pick.

The default container (40 ft, metal roof) and building (500 m², concrete roof)
reproduce the source analysis, a starting point you can change freely. This is
a first version; a predictive model can be layered on later.
            """
        )


if __name__ == "__main__":
    main()