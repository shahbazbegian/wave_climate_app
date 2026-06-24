"""
Wave Climate Projection Tool — Streamlit Edition
Professional web app for Wave Climate Analysis under RCP Scenarios
Ported from a PyQt5 desktop app to native Streamlit widgets.
Author: Based on research paper "Wave Climate Projection under Climate Change Scenarios"
"""

import os
import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import genpareto, linregress

import analysis

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Page config & style
# ============================================================================

st.set_page_config(
    page_title="Wave Climate Projection Tool",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOT_BG = '#ffffff'
PLOT_GRID = '#dddddd'
PLOT_SPINE = '#999999'
PLOT_LEGEND_BG = '#f0f0f0'

# Path to bundled default datasets, relative to this file (works regardless of
# the working directory Streamlit Cloud launches from).
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Registry of built-in regional datasets. Each region has its own folder
# under data/, its own zip fallback locations, and the same 6 expected base
# filenames (baseline/rcp45/rcp85 × swh/tm).
REGIONS = {
    'mediterranean': {
        'label': 'Mediterranean Sea (default)',
        'dir': os.path.join(APP_DIR, "data", "mediterranean"),
    },
    'brisbane': {
        'label': 'Brisbane (test dataset)',
        'dir': os.path.join(APP_DIR, "data", "brisbane"),
    },
}

# Maps (kind, variable) -> base filename (without extension), shared across
# all regions. Each base name may exist as a plain "<base>.csv" file, a
# gzip-compressed "<base>.csv.gz" file, or as a member inside a zip archive
# (matched by basename, ignoring any folder prefix inside the zip).
DEFAULT_FILES = {
    ('baseline', 'swh'): 'baseline_swh',
    ('baseline', 'tm'): 'baseline_tm',
    ('rcp45', 'swh'): 'rcp45_swh',
    ('rcp45', 'tm'): 'rcp45_tm',
    ('rcp85', 'swh'): 'rcp85_swh',
    ('rcp85', 'tm'): 'rcp85_tm',
}


def _candidate_zip_paths(region):
    """Common places people might drop a zip archive for this region."""
    region_dir = REGIONS[region]['dir']
    return [
        os.path.join(region_dir, "data.zip"),
        os.path.join(APP_DIR, "data", f"{region}.zip"),
        os.path.join(APP_DIR, f"{region}.zip"),
        os.path.join(APP_DIR, "data", "data.zip") if region == 'mediterranean' else None,
        os.path.join(APP_DIR, "data.zip") if region == 'mediterranean' else None,
    ]


def _find_zip_member(region, base):
    """Search this region's candidate zip paths for a zip containing
    <base>.csv or <base>.csv.gz at any depth. Returns (zip_path, member_name)
    or None."""
    import zipfile
    for zip_path in _candidate_zip_paths(region):
        if zip_path is None or not os.path.isfile(zip_path):
            continue
        try:
            with zipfile.ZipFile(zip_path) as zf:
                for name in zf.namelist():
                    bn = os.path.basename(name)
                    if bn == f"{base}.csv" or bn == f"{base}.csv.gz":
                        return zip_path, name
        except zipfile.BadZipFile:
            continue
    return None


def default_file_source(region, kind, variable):
    """Resolve where a default dataset actually lives for the given region.
    Returns one of:
      ('path', <filesystem path>)       — plain .csv or .csv.gz on disk
      ('zip', <zip path>, <member>)     — inside a zip archive
      None                              — not found anywhere
    Preference order: plain .csv > .csv.gz > zip archive."""
    region_dir = REGIONS[region]['dir']
    base = DEFAULT_FILES[(kind, variable)]
    plain_path = os.path.join(region_dir, f"{base}.csv")
    gz_path = os.path.join(region_dir, f"{base}.csv.gz")
    if os.path.isfile(plain_path):
        return ('path', plain_path)
    if os.path.isfile(gz_path):
        return ('path', gz_path)
    in_zip = _find_zip_member(region, base)
    if in_zip is not None:
        zip_path, member = in_zip
        return ('zip', zip_path, member)
    return None


def default_files_available(region):
    """Check which of the 6 expected default datasets exist anywhere for
    this region (plain file, .gz file, or inside a zip). Returns
    {base_name: bool}."""
    status = {}
    for (kind, variable), base in DEFAULT_FILES.items():
        status[base] = default_file_source(region, kind, variable) is not None
    return status


@st.cache_data(show_spinner=False)
def load_default_dataset(region, kind, variable):
    """Load one of the bundled regional datasets from the repo, using the
    same parsing/column-standardization logic as uploaded files.
    Transparently handles plain .csv, gzip .csv.gz, or a member inside a zip
    archive. Cached so repeated tab switches don't re-read from disk."""
    import zipfile

    source = default_file_source(region, kind, variable)
    if source is None:
        base = DEFAULT_FILES[(kind, variable)]
        region_dirname = os.path.basename(REGIONS[region]['dir'])
        return None, (
            f"Default file not found: data/{region_dirname}/{base}.csv "
            f"(or .csv.gz, or inside a data.zip)"
        )

    if source[0] == 'path':
        target = source[1]
    else:
        _, zip_path, member = source
        with zipfile.ZipFile(zip_path) as zf:
            raw = zf.read(member)
        # Wrap the extracted bytes as a file-like object; pandas will
        # auto-detect gzip compression from the member's .gz suffix if present.
        target = io.BytesIO(raw)
        if member.endswith('.gz'):
            # read_csv needs an explicit hint when reading from a buffer
            # rather than a path, since it can't infer from a filename.
            target = ('gzip_buffer', io.BytesIO(raw))

    if isinstance(target, tuple) and target[0] == 'gzip_buffer':
        import gzip
        buf = gzip.GzipFile(fileobj=target[1])
        if kind == 'baseline':
            df, warning = analysis.load_baseline_csv(buf, variable)
        else:
            df, warning = analysis.load_projection_csv(buf, variable)
    else:
        if kind == 'baseline':
            df, warning = analysis.load_baseline_csv(target, variable)
        else:
            df, warning = analysis.load_projection_csv(target, variable)
    return df, warning


def style_plot_axes(ax, fig):
    """Apply light-theme styling to a matplotlib axes/figure."""
    ax.set_facecolor(PLOT_BG)
    fig.patch.set_facecolor(PLOT_BG)
    ax.tick_params(colors='#333333')
    ax.xaxis.label.set_color('#333333')
    ax.yaxis.label.set_color('#333333')
    ax.title.set_color('#222222')
    for spine in ax.spines.values():
        spine.set_color(PLOT_SPINE)


st.markdown(
    """
    <style>
    .stApp { background-color: #ffffff; color: #222222; }
    section[data-testid="stSidebar"] { background-color: #f5f5f5; }
    .stApp p, .stApp span, .stApp label, .stApp div, .stApp h1, .stApp h2,
    .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp li {
        color: #222222 !important;
    }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #555555 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# Session state initialization (mirrors the original app's instance attrs)
# ============================================================================

def init_state():
    defaults = {
        'data_source': None,  # 'mediterranean' or 'custom'
        'baseline_data': {'swh': None, 'tm': None},
        'projection_data': {
            'swh': {'rcp45': None, 'rcp85': None},
            'tm': {'rcp45': None, 'rcp85': None},
        },
        'gpd_results': {
            'swh': {'rcp45': None, 'rcp85': None},
            'tm': {'rcp45': None, 'rcp85': None},
        },
        'projected_series': {
            'swh': {'rcp45': None, 'rcp85': None},
            'tm': {'rcp45': None, 'rcp85': None},
        },
        'test_data': None,
        'projection_results': [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()

# ============================================================================
# Helper: footer credit
# ============================================================================

def show_footer():
    """Developer credit + copyright notice, shown at the bottom of every tab."""
    st.markdown(
        "<hr style='margin-top:2rem; margin-bottom:0.6rem; border-color:#dddddd;'>"
        "<div style='text-align:center; color:#555555; font-size:0.85rem;'>"
        "Developed by Amirhussein Shahbazbegian — All rights reserved."
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================================
# Helper: data preview
# ============================================================================

def show_data_preview(df, title):
    """Show a dataframe preview + summary stats, mirroring update_data_preview."""
    if df is None or len(df) == 0:
        st.info("No data loaded")
        return

    st.markdown(f"**📊 {title}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    if 'time' in df.columns:
        c3.metric("Date range", f"{pd.to_datetime(df['time']).min().date()} → {pd.to_datetime(df['time']).max().date()}")

    if 'swh' in df.columns:
        s = df['swh']
        st.caption(f"SWH — mean: {s.mean():.4f}  |  std: {s.std():.4f}  |  min: {s.min():.4f}  |  max: {s.max():.4f}")
    elif 'tm' in df.columns:
        s = df['tm']
        st.caption(f"Tm — mean: {s.mean():.4f}  |  std: {s.std():.4f}  |  min: {s.min():.4f}  |  max: {s.max():.4f}")

    st.dataframe(df.head(100), use_container_width=True, height=280)
    st.caption(f"Showing first 100 of {len(df):,} rows")


# ============================================================================
# Header
# ============================================================================

col_title, col_version = st.columns([4, 1])
with col_title:
    st.markdown(
        "<h2 style='color:#111111;'>🌊 Wave Climate Projection Tool</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#333333;'>"
        "Analyze historical wave data and project future wave climate (significant wave height "
        "and mean wave period) under the RCP 4.5 and RCP 8.5 climate change scenarios. The tool "
        "fits extreme-value (GPD) statistics to peak events, scales baseline observations onto "
        "future projection years, and visualizes long-term trends — load your own data or use the "
        "built-in Mediterranean Sea dataset to get started."
        "</p>",
        unsafe_allow_html=True,
    )
with col_version:
    st.markdown(
        "<div style='text-align:right; color:#555555; padding-top:14px;'>"
        "v1.0 | Streamlit Edition | RCP4.5/RCP8.5</div>",
        unsafe_allow_html=True,
    )

tab_data, tab_gpd, tab_proj, tab_results = st.tabs(
    ["📁 Data Loading", "🎯 GPD Analysis", "🔄 Projections", "📊 Results"]
)

# ============================================================================
# TAB 1 — Data Loading
# ============================================================================

with tab_data:
    st.markdown("### 🗺️ Choose your data source")

    source_choice = st.radio(
        "Data source",
        options=["Mediterranean Sea (default)", "Brisbane (test dataset)", "Upload my own data"],
        horizontal=True,
        key="data_source_radio",
        label_visibility="collapsed",
    )
    if source_choice.startswith("Mediterranean"):
        st.session_state.data_source = 'mediterranean'
    elif source_choice.startswith("Brisbane"):
        st.session_state.data_source = 'brisbane'
    else:
        st.session_state.data_source = 'custom'

    # ------------------------------------------------------------------
    # Built-in regional datasets (Mediterranean Sea / Brisbane) — load
    # straight from the repo. Accepts a plain .csv, a gzip-compressed
    # .csv.gz, or a member inside a data.zip archive — whichever is found
    # first, per file.
    # ------------------------------------------------------------------
    if st.session_state.data_source in REGIONS:
        region = st.session_state.data_source
        region_label = REGIONS[region]['label']
        region_dirname = os.path.basename(REGIONS[region]['dir'])

        st.caption(f"Reading bundled files from `data/{region_dirname}/` in the repo "
                   "— plain `.csv`, compressed `.csv.gz`, or a `data.zip` containing them all also works.")

        availability = default_files_available(region)
        missing = [label for (label), ok in availability.items() if not ok]

        if missing:
            st.error(
                f"Some default dataset files are missing from the repo for {region_label}:\n\n"
                + "\n".join(f"- `{m}.csv` (or `.csv.gz`, or inside `data.zip`)" for m in missing)
                + f"\n\nUpload them (see the README in `data/{region_dirname}/`), then reload this app."
            )
        else:
            st.success(f"All 6 default {region_label} files found.")

        if st.button(f"📥 Load {region_label} dataset", type="primary", key=f"btn_load_{region}",
                      disabled=bool(missing)):
            load_plan = [
                ('baseline', 'swh', st.session_state.baseline_data, 'swh'),
                ('baseline', 'tm', st.session_state.baseline_data, 'tm'),
                ('rcp45', 'swh', st.session_state.projection_data['swh'], 'rcp45'),
                ('rcp45', 'tm', st.session_state.projection_data['tm'], 'rcp45'),
                ('rcp85', 'swh', st.session_state.projection_data['swh'], 'rcp85'),
                ('rcp85', 'tm', st.session_state.projection_data['tm'], 'rcp85'),
            ]
            messages = []
            with st.spinner(f"Loading {region_label} dataset..."):
                for kind, variable, target_dict, target_key in load_plan:
                    df, warning = load_default_dataset(region, kind, variable)
                    target_dict[target_key] = df
                    label = f"{kind.upper()} {variable.upper()}"
                    if warning:
                        messages.append(("warning", f"{label}: {warning}"))
                    elif df is not None:
                        messages.append(("success", f"{label}: {len(df):,} records loaded"))
                    else:
                        messages.append(("error", f"{label}: failed to load"))

            for level, msg in messages:
                getattr(st, level)(msg)

        st.divider()
        st.markdown("##### 📤 Upload an additional validation file (any format)")
        f = st.file_uploader("Validation CSV", type="csv", key=f"up_test_{region}")
        if f is not None and st.button("Load Validation Data", key=f"btn_test_{region}", type="primary"):
            df, error = analysis.load_test_data(f)
            if error:
                st.error(error)
            else:
                st.session_state.test_data = df
                st.success(f"Test data loaded: {len(df)} records from {df['time'].min()} to {df['time'].max()}")

    # ------------------------------------------------------------------
    # Custom / uploaded data — original manual upload workflow
    # ------------------------------------------------------------------
    else:
        st.markdown(
            """
Load your CSV data in order:
1. **Baseline Data (2005)** — SWH and Tm
2. **RCP 4.5 Projections (2041–2100)** — SWH and Tm
3. **RCP 8.5 Projections (2041–2100)** — SWH and Tm

Load a **test dataset** (e.g. Brisbane format) for validation.
"""
        )

        # ---------------- Baseline ----------------
        with st.expander("📊 Baseline Data (2005)", expanded=True):
            bc1, bc2 = st.columns(2)
            with bc1:
                st.markdown("**Significant Wave Height (SWH)**")
                f = st.file_uploader("Baseline SWH CSV", type="csv", key="up_baseline_swh")
                if f is not None and st.button("Load Baseline SWH", key="btn_baseline_swh", type="primary"):
                    df, warning = analysis.load_baseline_csv(f, 'swh')
                    st.session_state.baseline_data['swh'] = df
                    if warning:
                        st.warning(warning)
                    else:
                        st.success(f"Baseline SWH loaded: {len(df)} records")
            with bc2:
                st.markdown("**Mean Wave Period (Tm)**")
                f = st.file_uploader("Baseline Tm CSV", type="csv", key="up_baseline_tm")
                if f is not None and st.button("Load Baseline Tm", key="btn_baseline_tm", type="primary"):
                    df, warning = analysis.load_baseline_csv(f, 'tm')
                    st.session_state.baseline_data['tm'] = df
                    if warning:
                        st.warning(warning)
                    else:
                        st.success(f"Baseline Tm loaded: {len(df)} records")

        # ---------------- RCP 4.5 ----------------
        with st.expander("🔵 RCP 4.5 Projection Data (2041–2100)"):
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown("**Significant Wave Height (SWH)**")
                f = st.file_uploader("RCP 4.5 SWH CSV (single file)", type="csv", key="up_rcp45_swh")
                if f is not None and st.button("Load RCP 4.5 SWH", key="btn_rcp45_swh", type="primary"):
                    df, warning = analysis.load_projection_csv(f, 'swh')
                    st.session_state.projection_data['swh']['rcp45'] = df
                    if warning:
                        st.warning(warning)
                    else:
                        st.success(f"RCP 4.5 SWH loaded: {len(df)} records")
                files = st.file_uploader("...or merge multiple CSVs (one folder's worth)", type="csv",
                                          accept_multiple_files=True, key="up_rcp45_swh_multi")
                if files and st.button("Merge CSVs", key="btn_merge_rcp45_swh"):
                    progress = st.progress(0)
                    merged = analysis.merge_csv_files([(f.name, f) for f in files], 'swh', progress.progress)
                    progress.empty()
                    if merged is not None:
                        st.session_state.projection_data['swh']['rcp45'] = merged
                        st.success(f"Merged {len(files)} files → {len(merged)} records")
                    else:
                        st.error("No data files found")

            with rc2:
                st.markdown("**Mean Wave Period (Tm)**")
                f = st.file_uploader("RCP 4.5 Tm CSV (single file)", type="csv", key="up_rcp45_tm")
                if f is not None and st.button("Load RCP 4.5 Tm", key="btn_rcp45_tm", type="primary"):
                    df, warning = analysis.load_projection_csv(f, 'tm')
                    st.session_state.projection_data['tm']['rcp45'] = df
                    if warning:
                        st.warning(warning)
                    else:
                        st.success(f"RCP 4.5 Tm loaded: {len(df)} records")
                files = st.file_uploader("...or merge multiple CSVs (one folder's worth)", type="csv",
                                          accept_multiple_files=True, key="up_rcp45_tm_multi")
                if files and st.button("Merge CSVs", key="btn_merge_rcp45_tm"):
                    progress = st.progress(0)
                    merged = analysis.merge_csv_files([(f.name, f) for f in files], 'tm', progress.progress)
                    progress.empty()
                    if merged is not None:
                        st.session_state.projection_data['tm']['rcp45'] = merged
                        st.success(f"Merged {len(files)} files → {len(merged)} records")
                    else:
                        st.error("No data files found")

        # ---------------- RCP 8.5 ----------------
        with st.expander("🔴 RCP 8.5 Projection Data (2041–2100)"):
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown("**Significant Wave Height (SWH)**")
                f = st.file_uploader("RCP 8.5 SWH CSV (single file)", type="csv", key="up_rcp85_swh")
                if f is not None and st.button("Load RCP 8.5 SWH", key="btn_rcp85_swh", type="primary"):
                    df, warning = analysis.load_projection_csv(f, 'swh')
                    st.session_state.projection_data['swh']['rcp85'] = df
                    if warning:
                        st.warning(warning)
                    else:
                        st.success(f"RCP 8.5 SWH loaded: {len(df)} records")
                files = st.file_uploader("...or merge multiple CSVs (one folder's worth)", type="csv",
                                          accept_multiple_files=True, key="up_rcp85_swh_multi")
                if files and st.button("Merge CSVs", key="btn_merge_rcp85_swh"):
                    progress = st.progress(0)
                    merged = analysis.merge_csv_files([(f.name, f) for f in files], 'swh', progress.progress)
                    progress.empty()
                    if merged is not None:
                        st.session_state.projection_data['swh']['rcp85'] = merged
                        st.success(f"Merged {len(files)} files → {len(merged)} records")
                    else:
                        st.error("No data files found")

            with rc2:
                st.markdown("**Mean Wave Period (Tm)**")
                f = st.file_uploader("RCP 8.5 Tm CSV (single file)", type="csv", key="up_rcp85_tm")
                if f is not None and st.button("Load RCP 8.5 Tm", key="btn_rcp85_tm", type="primary"):
                    df, warning = analysis.load_projection_csv(f, 'tm')
                    st.session_state.projection_data['tm']['rcp85'] = df
                    if warning:
                        st.warning(warning)
                    else:
                        st.success(f"RCP 8.5 Tm loaded: {len(df)} records")
                files = st.file_uploader("...or merge multiple CSVs (one folder's worth)", type="csv",
                                          accept_multiple_files=True, key="up_rcp85_tm_multi")
                if files and st.button("Merge CSVs", key="btn_merge_rcp85_tm"):
                    progress = st.progress(0)
                    merged = analysis.merge_csv_files([(f.name, f) for f in files], 'tm', progress.progress)
                    progress.empty()
                    if merged is not None:
                        st.session_state.projection_data['tm']['rcp85'] = merged
                        st.success(f"Merged {len(files)} files → {len(merged)} records")
                    else:
                        st.error("No data files found")

        # ---------------- Test data ----------------
        with st.expander("🧪 Test Data (e.g., Brisbane 2024)"):
            f = st.file_uploader("Test CSV", type="csv", key="up_test")
            if f is not None and st.button("Load Test Data", key="btn_test", type="primary"):
                df, error = analysis.load_test_data(f)
                if error:
                    st.error(error)
                else:
                    st.session_state.test_data = df
                    st.success(f"Test data loaded: {len(df)} records from {df['time'].min()} to {df['time'].max()}")

    # ---------------- Preview ----------------
    st.markdown("### 📋 Data Preview")
    preview_options = {
        "Baseline SWH": st.session_state.baseline_data['swh'],
        "Baseline Tm": st.session_state.baseline_data['tm'],
        "RCP 4.5 SWH": st.session_state.projection_data['swh']['rcp45'],
        "RCP 4.5 Tm": st.session_state.projection_data['tm']['rcp45'],
        "RCP 8.5 SWH": st.session_state.projection_data['swh']['rcp85'],
        "RCP 8.5 Tm": st.session_state.projection_data['tm']['rcp85'],
        "Test Data": st.session_state.test_data,
    }
    loaded_options = {k: v for k, v in preview_options.items() if v is not None}
    if loaded_options:
        choice = st.selectbox("Choose dataset to preview", list(loaded_options.keys()))
        show_data_preview(loaded_options[choice], choice)
    else:
        st.info("No data loaded yet — upload CSVs above to get started.")

    show_footer()

# ============================================================================
# TAB 2 — GPD Analysis
# ============================================================================

with tab_gpd:
    st.caption(
        "Identifies extreme wave events by fitting a Generalized Pareto Distribution (GPD) to "
        "peaks above a chosen threshold, after declustering nearby peaks from the same storm."
    )
    st.markdown("### ⚙️ GPD Analysis Settings")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        gpd_variable_label = st.selectbox(
            "Variable", ["Significant Wave Height (SWH)", "Mean Wave Period (Tm)"], key="gpd_var"
        )
    gpd_variable = 'swh' if gpd_variable_label.startswith("Significant") else 'tm'

    with c2:
        gpd_scenario_label = st.selectbox("Scenario", ["RCP 4.5", "RCP 8.5"], key="gpd_scen")
    gpd_scenario = 'rcp45' if gpd_scenario_label == "RCP 4.5" else 'rcp85'

    with c3:
        decluster_hours = st.number_input("Decluster Hours", min_value=0, max_value=72, value=12, step=1)
    with c4:
        threshold_percentile = st.number_input(
            "Threshold Percentile (%)", min_value=90.0, max_value=99.9, value=99.5, step=0.1, format="%.1f"
        )

    run_gpd = st.button("🎯 Run GPD Analysis", type="primary", key="run_gpd_btn")

    if run_gpd:
        df = st.session_state.projection_data[gpd_variable][gpd_scenario]
        if df is None:
            st.error(f"Please load {gpd_scenario.upper()} {gpd_variable.upper()} data first")
        elif gpd_variable not in df.columns:
            st.error(f"Column '{gpd_variable}' not found in the data. Available columns: {list(df.columns)}")
        else:
            data = pd.Series(df[gpd_variable].values, index=pd.to_datetime(df['time']))
            with st.spinner("Running GPD analysis..."):
                result = analysis.run_gpd_analysis(data, int(decluster_hours), float(threshold_percentile))
            st.session_state.gpd_results[gpd_variable][gpd_scenario] = result
            st.session_state['_last_gpd'] = (result, gpd_variable, gpd_scenario, threshold_percentile)
            st.success(f"GPD analysis complete for {gpd_scenario.upper()} {gpd_variable.upper()}!")

    # Render last result (persists across reruns using session state)
    last = st.session_state.get('_last_gpd')
    if last is not None:
        result, variable, scenario, pct = last
        data_name = f"{scenario.upper()} {variable.upper()}"
        ylabel = 'Hs [m]' if variable == 'swh' else 'Tm [s]'

        st.markdown("#### 🎯 GPD Analysis Results")
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Threshold", f"{result['threshold']:.4f} {ylabel}")
        rc2.metric("Peaks found", result['n_peaks'])
        rc3.metric("ξ (shape)", f"{result['xi']:.4f}")
        rc4.metric("β (scale)", f"{result['beta']:.4f}")

        fig1, ax1 = plt.subplots(figsize=(10, 3.5))
        style_plot_axes(ax1, fig1)

        full_data = st.session_state.projection_data[variable][scenario]
        if full_data is not None:
            time_series = pd.to_datetime(full_data['time'])
            ax1.plot(time_series, full_data[variable], color='#888888', alpha=0.3, linewidth=0.5, label='Full data')

        decl = result['declustered']
        ax1.plot(decl.index, decl.values, 'o', color='#0099cc', markersize=4, alpha=0.8, label='Peaks')
        ax1.axhline(y=result['threshold'], color='#ff6b6b', linestyle='--', linewidth=2,
                    label=f'Threshold ({pct}%)')

        textstr = f'Peaks: {result["n_peaks"]}'
        props = dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                  verticalalignment='top', bbox=props, color='#222222')

        ax1.set_xlabel('Time')
        ax1.set_ylabel(ylabel)
        ax1.set_title(f'Declustered Peaks - {data_name}')
        ax1.legend(facecolor='#f0f0f0', edgecolor='#999999', labelcolor='#222222', fontsize=8)
        ax1.grid(True, alpha=0.4, color='#dddddd')
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(10, 3.5))
        style_plot_axes(ax2, fig2)

        exceedances = result['exceedances']
        if len(exceedances) > 0:
            ax2.hist(exceedances, bins=30, density=True, alpha=0.5, color='#0099cc', label='Exceedances')
            x = np.linspace(0, exceedances.max(), 200)
            pdf = genpareto.pdf(x, result['xi'], result['loc'], result['beta'])
            ax2.plot(x, pdf, 'r-', lw=2, label=f'GPD fit (ξ={result["xi"]:.2f}, β={result["beta"]:.2f})')

        ax2.set_xlabel(f'Exceedance ({ylabel})')
        ax2.set_ylabel('Density')
        ax2.set_title('GPD Fit to Exceedances')
        ax2.legend(facecolor='#f0f0f0', edgecolor='#999999', labelcolor='#222222')
        ax2.grid(True, alpha=0.4, color='#dddddd')
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
    else:
        st.info("Run a GPD analysis to see results here.")

    show_footer()

# ============================================================================
# TAB 3 — Projections
# ============================================================================

with tab_proj:
    st.caption(
        "Projects future wave conditions (through year 2099) by scaling baseline observations "
        "onto each year of the selected RCP scenario, then plots the resulting trend."
    )
    st.markdown("### ⚙️ Generate Projections")

    pc1, pc2 = st.columns(2)
    with pc1:
        proj_variable_label = st.selectbox(
            "Variable", ["Significant Wave Height (SWH)", "Mean Wave Period (Tm)"], key="proj_var"
        )
    proj_variable = 'swh' if proj_variable_label.startswith("Significant") else 'tm'

    with pc2:
        proj_scenario_text = st.selectbox("Scenario", ["RCP 4.5", "RCP 8.5", "Both"], key="proj_scen")

    generate_clicked = st.button("🔄 Generate Projections", type="primary", key="gen_proj_btn")

    if generate_clicked:
        baseline_df = st.session_state.baseline_data[proj_variable]
        if baseline_df is None:
            st.error(f"Please load baseline {proj_variable.upper()} data first")
        elif proj_variable not in baseline_df.columns:
            st.error(f"Column '{proj_variable}' not found in baseline data. Available columns: {list(baseline_df.columns)}")
        else:
            if proj_scenario_text == "Both":
                scenarios_to_run = ['rcp45', 'rcp85']
            elif proj_scenario_text == "RCP 4.5":
                scenarios_to_run = ['rcp45']
            else:
                scenarios_to_run = ['rcp85']

            missing = False
            for sc in scenarios_to_run:
                proj_df = st.session_state.projection_data[proj_variable][sc]
                if proj_df is None:
                    st.error(f"Please load {sc.upper()} {proj_variable.upper()} data first")
                    missing = True
                    break
                if proj_variable not in proj_df.columns:
                    st.error(f"Column '{proj_variable}' not found in {sc.upper()} {proj_variable.upper()} data. "
                             f"Available columns: {list(proj_df.columns)}")
                    missing = True
                    break

            if not missing:
                projection_results = []
                info_lines = []
                progress = st.progress(0)
                status = st.empty()

                for sc in scenarios_to_run:
                    status.text(f"Generating projections for {sc.upper()} {proj_variable.upper()}...")
                    proj_df, error = analysis.generate_projection(
                        baseline_df,
                        st.session_state.projection_data[proj_variable][sc],
                        sc, proj_variable,
                        progress_callback=progress.progress,
                    )
                    if proj_df is not None:
                        # Ensure time is datetime
                        proj_df['time'] = pd.to_datetime(proj_df['time'])
                        
                        # FILTER: Keep only data up to 2099-12-31
                        proj_df = proj_df[proj_df['time'] <= '2099-12-31']
                        
                        # Also filter the original projection data in session state
                        orig_df = st.session_state.projection_data[proj_variable][sc]
                        if orig_df is not None:
                            orig_df['time'] = pd.to_datetime(orig_df['time'])
                            st.session_state.projection_data[proj_variable][sc] = orig_df[orig_df['time'] <= '2099-12-31']
                        
                        st.session_state.projected_series[proj_variable][sc] = proj_df
                        projection_results.append(proj_df)
                        info_lines.append(
                            f"✅ Generated {sc.upper()} {proj_variable.upper()}: {len(proj_df)} records\n"
                            f"   Range: [{proj_df[proj_variable].min():.4f}, {proj_df[proj_variable].max():.4f}]\n"
                            f"   Mean: {proj_df[proj_variable].mean():.4f}, Std: {proj_df[proj_variable].std():.4f}"
                        )
                    else:
                        info_lines.append(f"❌ Failed to generate projections for {sc.upper()} {proj_variable.upper()}: {error}")

                progress.empty()
                status.empty()

                if projection_results:
                    st.session_state.projection_results = projection_results
                    st.session_state['_last_proj'] = (
                        projection_results, proj_variable, proj_scenario_text, "\n\n".join(info_lines)
                    )

                    combined_df = pd.concat(projection_results, ignore_index=True)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    variable_name = "SWH" if proj_variable == 'swh' else "Tm"
                    out_name = f"projection_{variable_name}_{proj_scenario_text.replace(' ', '_')}_{timestamp}.csv"
                    st.session_state['_last_proj_csv'] = (out_name, combined_df.to_csv(index=False))

                    st.success(f"Projections generated successfully for {variable_name}")
                else:
                    st.error("No projections were generated successfully")

    last_proj = st.session_state.get('_last_proj')
    if last_proj is not None:
        projection_results, variable, scenario_text, info_text = last_proj
        variable_name = "SWH" if variable == 'swh' else "Tm"
        ylabel = 'Hs [m]' if variable_name == 'SWH' else 'Tm [s]'
        title_var = 'Significant Wave Height' if variable_name == 'SWH' else 'Mean Wave Period'
        colors = {'rcp45': '#0066cc', 'rcp85': '#ff4444'}
        labels = {'rcp45': 'RCP 4.5', 'rcp85': 'RCP 8.5'}

        # Filter the projection data in session state before plotting
        for scenario in ['rcp45', 'rcp85']:
            if st.session_state.projection_data[variable][scenario] is not None:
                df = st.session_state.projection_data[variable][scenario]
                df['time'] = pd.to_datetime(df['time'])
                st.session_state.projection_data[variable][scenario] = df[df['time'] <= '2099-12-31']

        # Combined projections plot
        fig, ax = plt.subplots(figsize=(12, 5))
        style_plot_axes(ax, fig)

        # Also filter the projection_results list
        filtered_results = []
        for df in projection_results:
            df = df.copy()
            df['time'] = pd.to_datetime(df['time'])
            df = df[df['time'] <= '2099-12-31']
            if len(df) > 0:
                filtered_results.append(df)
        
        # Use filtered results for plotting
        for df in filtered_results:
            if 'scenario' not in df.columns:
                continue
            scenario = df['scenario'].iloc[0]
            df = df.copy()
            df['time'] = pd.to_datetime(df['time'])
            
            # Additional safety filter
            df = df[df['time'] <= '2099-12-31']
            
            df['year'] = df['time'].dt.year
            yearly_avg = df.groupby('year')[variable].mean().reset_index()
            ax.plot(pd.to_datetime(yearly_avg['year'], format='%Y'), yearly_avg[variable],
                    'o-', color=colors.get(scenario, '#888888'), linewidth=2.5,
                    markersize=6, label=labels.get(scenario, scenario))

        # Set x-axis limit to 2099-12-31
        ax.set_xlim([pd.Timestamp('2000-01-01'), pd.Timestamp('2099-12-31')])
        
        ax.set_xlabel('Time')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Projected {title_var} - {scenario_text}')
        ax.legend(facecolor='#f0f0f0', edgecolor='#999999', labelcolor='#222222')
        ax.grid(True, alpha=0.4, color='#dddddd')
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Long-term trend plot
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        style_plot_axes(ax2, fig2)

        for df in filtered_results:
            if 'scenario' not in df.columns:
                continue
            scenario = df['scenario'].iloc[0]
            df = df.copy()
            df['time'] = pd.to_datetime(df['time'])
            
            # Safety filter
            df = df[df['time'] <= '2099-12-31']
            
            df['year'] = df['time'].dt.year
            yearly_means = df.groupby('year')[variable].mean()

            x = yearly_means.index.values
            y = yearly_means.values
            if len(x) > 1:  # Need at least 2 points for regression
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                trend = intercept + slope * x

                ax2.plot(x, y, 'o', color=colors.get(scenario, '#888888'),
                         markersize=5, alpha=0.7, label=f'{labels.get(scenario, scenario)} - yearly')
                ax2.plot(x, trend, '--', color=colors.get(scenario, '#888888'), linewidth=2,
                          label=f'{labels.get(scenario, scenario)} trend (slope={slope:.4f} m/year)')

        # Set x-axis limit to 2099
        ax2.set_xlim([2040, 2100])
        
        ax2.set_xlabel('Year')
        ax2.set_ylabel(f'Mean {ylabel}')
        ax2.set_title(f'Long-term Trend Analysis - {title_var} - {scenario_text}')
        ax2.legend(facecolor='#f0f0f0', edgecolor='#999999', labelcolor='#222222', fontsize=9)
        ax2.grid(True, alpha=0.4, color='#dddddd')
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

        st.text(info_text)

        csv_info = st.session_state.get('_last_proj_csv')
        if csv_info:
            out_name, csv_data = csv_info
            st.download_button("⬇️ Download combined projections CSV", data=csv_data,
                                file_name=out_name, mime="text/csv")
    else:
        st.info("Generate projections to see results here.")

    show_footer()
# ============================================================================
# TAB 4 — Results / Export
# ============================================================================

with tab_results:
    st.markdown("### 💾 Export Results")
    st.caption("Download any generated data and figures as a single ZIP file.")

    if st.button("Prepare export package", type="primary"):
        import zipfile

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        buf = io.BytesIO()
        exported_files = []

        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Baseline data
            for var, df in st.session_state.baseline_data.items():
                if df is not None:
                    name = f"baseline_{var}_{timestamp}.csv"
                    zf.writestr(name, df.to_csv(index=False))
                    exported_files.append(name)

            # Projection data
            for var in ['swh', 'tm']:
                for scenario in ['rcp45', 'rcp85']:
                    df = st.session_state.projection_data[var][scenario]
                    if df is not None:
                        name = f"projection_{var}_{scenario}_{timestamp}.csv"
                        zf.writestr(name, df.to_csv(index=False))
                        exported_files.append(name)

            # Test data
            if st.session_state.test_data is not None:
                name = f"test_data_{timestamp}.csv"
                zf.writestr(name, st.session_state.test_data.to_csv(index=False))
                exported_files.append(name)

            # GPD results
            for var in ['swh', 'tm']:
                for scenario in ['rcp45', 'rcp85']:
                    result = st.session_state.gpd_results[var][scenario]
                    if result is not None:
                        gpd_df = pd.DataFrame([{
                            'variable': var,
                            'scenario': scenario,
                            'threshold': result.get('threshold', np.nan),
                            'xi': result.get('xi', np.nan),
                            'beta': result.get('beta', np.nan),
                            'n_peaks': result.get('n_peaks', 0)
                        }])
                        name = f"gpd_{var}_{scenario}_{timestamp}.csv"
                        zf.writestr(name, gpd_df.to_csv(index=False))
                        exported_files.append(name)

            # Projected series
            for var in ['swh', 'tm']:
                for scenario in ['rcp45', 'rcp85']:
                    df = st.session_state.projected_series[var][scenario]
                    if df is not None:
                        name = f"projected_{var}_{scenario}_{timestamp}.csv"
                        zf.writestr(name, df.to_csv(index=False))
                        exported_files.append(name)

        if exported_files:
            st.session_state['_export_zip'] = (f"wave_climate_export_{timestamp}.zip", buf.getvalue())
            summary_lines = [f"Export prepared: {len(exported_files)} file(s)", ""]
            for f in exported_files[:10]:
                summary_lines.append(f"  • {f}")
            if len(exported_files) > 10:
                summary_lines.append(f"  • ... and {len(exported_files) - 10} more")
            st.session_state['_export_summary'] = "\n".join(summary_lines)
        else:
            st.warning("No results available to export yet — load data and run some analysis first.")

    export_zip = st.session_state.get('_export_zip')
    if export_zip:
        name, data = export_zip
        st.download_button("⬇️ Download export ZIP", data=data, file_name=name, mime="application/zip")

    st.markdown("### 📊 Analysis Summary")
    summary = st.session_state.get('_export_summary')
    if summary:
        st.code(summary, language=None)
    else:
        st.info("No export prepared yet.")

    show_footer()
