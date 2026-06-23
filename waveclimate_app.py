"""
Wave Climate Projection Tool — Streamlit Edition (Optimized)
Professional web app for Wave Climate Analysis under RCP Scenarios
Optimized for faster file uploads while maintaining original structure
"""

import os
import io
from datetime import datetime
import time

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import genpareto, linregress

# Import your custom analysis module
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

DARK_PLOT_BG = '#333333'

# ============================================================================
# CACHE DECORATORS - Add these to speed up file loading
# ============================================================================

@st.cache_data(ttl=3600, max_entries=20)
def load_csv_with_cache(file_bytes, variable_type):
    """
    Cached version of CSV loading - much faster on subsequent loads
    """
    try:
        # Read CSV with optimized parameters (removed deprecated infer_datetime_format)
        df = pd.read_csv(
            io.BytesIO(file_bytes),
            parse_dates=['time'],
            low_memory=False
        )
        
        # Ensure time is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])
        
        # Sort and deduplicate
        df = df.drop_duplicates(subset=['time'])
        df = df.sort_values('time').reset_index(drop=True)
        
        return df, None
    except Exception as e:
        return None, f"Error loading file: {str(e)}"

@st.cache_data(ttl=3600, max_entries=10)
def merge_csv_files_cached(file_data_list, variable_type):
    """
    Cached version of CSV merging - much faster
    """
    if not file_data_list:
        return None
    
    all_dfs = []
    for file_name, file_bytes in file_data_list:
        df, _ = load_csv_with_cache(file_bytes, variable_type)
        if df is not None:
            all_dfs.append(df)
    
    if not all_dfs:
        return None
    
    # Concatenate all dataframes
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=['time'])
    combined = combined.sort_values('time').reset_index(drop=True)
    
    return combined

@st.cache_data(ttl=3600, max_entries=10)
def run_gpd_analysis_cached(data_series, decluster_hours, threshold_percentile):
    """
    Cached version of GPD analysis
    """
    try:
        # Use the original analysis function if available
        if hasattr(analysis, 'run_gpd_analysis'):
            return analysis.run_gpd_analysis(data_series, decluster_hours, threshold_percentile)
        else:
            # Fallback implementation if analysis module doesn't have the function
            return run_gpd_analysis_fallback(data_series, decluster_hours, threshold_percentile)
    except Exception as e:
        st.error(f"GPD analysis error: {str(e)}")
        return None

def run_gpd_analysis_fallback(data_series, decluster_hours, threshold_percentile):
    """
    Fallback GPD analysis if analysis module doesn't have the function
    """
    try:
        # Remove NaN values
        data_series = data_series.dropna()
        if len(data_series) == 0:
            return None
        
        # Calculate threshold
        threshold = np.percentile(data_series, threshold_percentile)
        
        # Find peaks above threshold
        peaks = data_series[data_series > threshold]
        
        # Declustering
        if decluster_hours > 0 and len(peaks) > 0:
            peaks_df = pd.DataFrame({
                'value': peaks.values,
                'time': peaks.index
            })
            peaks_df = peaks_df.sort_values('time')
            
            declustered_peaks = []
            current_peak = None
            last_time = None
            
            for idx, row in peaks_df.iterrows():
                if current_peak is None:
                    current_peak = row['value']
                    last_time = row['time']
                else:
                    time_diff = (row['time'] - last_time).total_seconds() / 3600.0
                    if time_diff >= decluster_hours:
                        declustered_peaks.append(current_peak)
                        current_peak = row['value']
                        last_time = row['time']
                    else:
                        current_peak = max(current_peak, row['value'])
                        last_time = max(last_time, row['time'])
            
            if current_peak is not None:
                declustered_peaks.append(current_peak)
            
            peaks_series = pd.Series(declustered_peaks)
        else:
            peaks_series = peaks
        
        if len(peaks_series) == 0:
            return None
        
        # Fit GPD
        exceedances = peaks_series - threshold
        exceedances = exceedances[exceedances > 0]
        
        if len(exceedances) < 5:
            return None
        
        params = genpareto.fit(exceedances)
        xi, loc, beta = params
        
        result = {
            'threshold': threshold,
            'n_peaks': len(peaks_series),
            'xi': xi,
            'beta': beta,
            'loc': loc,
            'exceedances': exceedances,
            'declustered': peaks_series,
            'peaks': peaks
        }
        
        return result
    except Exception as e:
        print(f"GPD analysis error: {str(e)}")
        return None

@st.cache_data(ttl=3600, max_entries=10)
def generate_projection_cached(baseline_df, projection_df, scenario, variable):
    """
    Cached version of projection generation
    """
    try:
        # Use the original analysis function if available
        if hasattr(analysis, 'generate_projection'):
            # We need to handle the progress callback differently for cached version
            return analysis.generate_projection(baseline_df, projection_df, scenario, variable, progress_callback=None)
        else:
            # Fallback implementation
            return generate_projection_fallback(baseline_df, projection_df, scenario, variable)
    except Exception as e:
        return None, str(e)

def generate_projection_fallback(baseline_df, projection_df, scenario, variable):
    """
    Fallback projection generation if analysis module doesn't have the function
    """
    try:
        if baseline_df is None or projection_df is None:
            return None, "Missing data"
            
        if variable not in baseline_df.columns or variable not in projection_df.columns:
            return None, f"Column '{variable}' not found in data"
        
        # Ensure time columns are datetime
        baseline_df = baseline_df.copy()
        projection_df = projection_df.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(baseline_df['time']):
            baseline_df['time'] = pd.to_datetime(baseline_df['time'])
        if not pd.api.types.is_datetime64_any_dtype(projection_df['time']):
            projection_df['time'] = pd.to_datetime(projection_df['time'])
        
        # Calculate baseline statistics
        baseline_mean = baseline_df[variable].mean()
        baseline_std = baseline_df[variable].std()
        
        # Calculate projection statistics
        projection_mean = projection_df[variable].mean()
        projection_std = projection_df[variable].std()
        
        # Calculate scaling factors
        std_factor = projection_std / baseline_std if baseline_std != 0 else 1.0
        
        # Apply scaling to create projected series
        projected = baseline_df.copy()
        projected[variable] = (projected[variable] - baseline_mean) * std_factor + projection_mean
        
        # Add scenario information
        projected['scenario'] = scenario
        
        # Add some randomness
        noise = np.random.normal(0, projection_std * 0.1, len(projected))
        projected[variable] = projected[variable] + noise
        
        # Ensure values are positive
        projected[variable] = np.maximum(projected[variable], 0)
        
        # Add time shift
        projection_years = projection_df['time'].dt.year
        if len(projection_years) > 0:
            min_year = projection_years.min()
            base_times = projected['time']
            projected['time'] = base_times + pd.Timedelta(days=(min_year - 2005) * 365.25)
        
        return projected, None
        
    except Exception as e:
        return None, str(e)

# ============================================================================
# Session state initialization
# ============================================================================

def init_state():
    defaults = {
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
        '_last_gpd': None,
        '_last_proj': None,
        '_last_proj_csv': None,
        '_export_zip': None,
        '_export_summary': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()

# ============================================================================
# Helper: data preview
# ============================================================================

def style_dark_axes(ax, fig):
    """Apply the same dark styling used throughout the original desktop app."""
    ax.set_facecolor(DARK_PLOT_BG)
    fig.patch.set_facecolor(DARK_PLOT_BG)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_color('#666666')

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
        try:
            c3.metric("Date range", f"{pd.to_datetime(df['time']).min().date()} → {pd.to_datetime(df['time']).max().date()}")
        except:
            pass

    if 'swh' in df.columns:
        s = df['swh']
        st.caption(f"SWH — mean: {s.mean():.4f}  |  std: {s.std():.4f}  |  min: {s.min():.4f}  |  max: {s.max():.4f}")
    elif 'tm' in df.columns:
        s = df['tm']
        st.caption(f"Tm — mean: {s.mean():.4f}  |  std: {s.std():.4f}  |  min: {s.min():.4f}  |  max: {s.max():.4f}")

    st.dataframe(df.head(100), use_container_width=True, height=280)
    st.caption(f"Showing first 100 of {len(df):,} rows")

# ============================================================================
# Optimized file loading functions (replace analysis module calls)
# ============================================================================

def load_baseline_csv_optimized(file, variable_type):
    """
    Optimized version of analysis.load_baseline_csv
    """
    file_bytes = file.getvalue()
    df, warning = load_csv_with_cache(file_bytes, variable_type)
    return df, warning

def load_projection_csv_optimized(file, variable_type):
    """
    Optimized version of analysis.load_projection_csv
    """
    file_bytes = file.getvalue()
    df, warning = load_csv_with_cache(file_bytes, variable_type)
    return df, warning

def load_test_data_optimized(file):
    """
    Optimized version of analysis.load_test_data
    """
    try:
        file_bytes = file.getvalue()
        df, warning = load_csv_with_cache(file_bytes, 'swh')
        if warning:
            return None, warning
        return df, None
    except Exception as e:
        return None, str(e)

def merge_csv_files_optimized(files, variable_type):
    """
    Optimized version of analysis.merge_csv_files
    """
    if not files:
        return None
    
    # Prepare file list for caching
    file_data_list = [(f.name, f.getvalue()) for f in files]
    
    # Use cached merge
    merged = merge_csv_files_cached(tuple(file_data_list), variable_type)
    
    return merged

# ============================================================================
# Header
# ============================================================================

st.markdown(
    """
    <style>
    .stApp { background-color: #1e1e1e; }
    section[data-testid="stSidebar"] { background-color: #252525; }
    .stButton button { width: 100%; }
    </style>
    """,
    unsafe_allow_html=True,
)

col_title, col_version = st.columns([4, 1])
with col_title:
    st.markdown("## 🌊 Wave Climate Projection Tool")
with col_version:
    st.markdown(
        "<div style='text-align:right; color:#888; padding-top:14px;'>"
        "v2.0 | Optimized | RCP4.5/RCP8.5</div>",
        unsafe_allow_html=True,
    )

tab_data, tab_gpd, tab_proj, tab_results = st.tabs(
    ["📁 Data Loading", "🎯 GPD Analysis", "🔄 Projections", "📊 Results"]
)

# ============================================================================
# TAB 1 — Data Loading (Optimized)
# ============================================================================

with tab_data:
    st.markdown(
        """
Load your CSV data in order:
1. **Baseline Data (2005)** — SWH and Tm
2. **RCP 4.5 Projections (2041–2100)** — SWH and Tm
3. **RCP 8.5 Projections (2041–2100)** — SWH and Tm

Optionally load a **test dataset** (e.g. Brisbane format) for validation.
"""
    )

    # ---------------- Baseline ----------------
    with st.expander("📊 Baseline Data (2005)", expanded=True):
        bc1, bc2 = st.columns(2)
        with bc1:
            st.markdown("**Significant Wave Height (SWH)**")
            f = st.file_uploader("Baseline SWH CSV", type="csv", key="up_baseline_swh")
            if f is not None and st.button("Load Baseline SWH", key="btn_baseline_swh", type="primary"):
                with st.spinner("Loading file..."):
                    start_time = time.time()
                    df, warning = load_baseline_csv_optimized(f, 'swh')
                    st.session_state.baseline_data['swh'] = df
                    if warning:
                        st.warning(warning)
                    else:
                        st.success(f"✅ Baseline SWH loaded: {len(df):,} records in {time.time()-start_time:.1f}s")
        with bc2:
            st.markdown("**Mean Wave Period (Tm)**")
            f = st.file_uploader("Baseline Tm CSV", type="csv", key="up_baseline_tm")
            if f is not None and st.button("Load Baseline Tm", key="btn_baseline_tm", type="primary"):
                with st.spinner("Loading file..."):
                    start_time = time.time()
                    df, warning = load_baseline_csv_optimized(f, 'tm')
                    st.session_state.baseline_data['tm'] = df
                    if warning:
                        st.warning(warning)
                    else:
                        st.success(f"✅ Baseline Tm loaded: {len(df):,} records in {time.time()-start_time:.1f}s")

    # ---------------- RCP 4.5 ----------------
    with st.expander("🔵 RCP 4.5 Projection Data (2041–2100)"):
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown("**Significant Wave Height (SWH)**")
            f = st.file_uploader("RCP 4.5 SWH CSV (single file)", type="csv", key="up_rcp45_swh")
            if f is not None and st.button("Load RCP 4.5 SWH", key="btn_rcp45_swh", type="primary"):
                with st.spinner("Loading file..."):
                    start_time = time.time()
                    df, warning = load_projection_csv_optimized(f, 'swh')
                    st.session_state.projection_data['swh']['rcp45'] = df
                    if warning:
                        st.warning(warning)
                    else:
                        st.success(f"✅ RCP 4.5 SWH loaded: {len(df):,} records in {time.time()-start_time:.1f}s")
            files = st.file_uploader("...or merge multiple CSVs (one folder's worth)", type="csv",
                                      accept_multiple_files=True, key="up_rcp45_swh_multi")
            if files and st.button("Merge CSVs", key="btn_merge_rcp45_swh"):
                with st.spinner(f"Merging {len(files)} files..."):
                    start_time = time.time()
                    merged = merge_csv_files_optimized(files, 'swh')
                    if merged is not None:
                        st.session_state.projection_data['swh']['rcp45'] = merged
                        st.success(f"✅ Merged {len(files)} files → {len(merged):,} records in {time.time()-start_time:.1f}s")
                    else:
                        st.error("No data files found")

        with rc2:
            st.markdown("**Mean Wave Period (Tm)**")
            f = st.file_uploader("RCP 4.5 Tm CSV (single file)", type="csv", key="up_rcp45_tm")
            if f is not None and st.button("Load RCP 4.5 Tm", key="btn_rcp45_tm", type="primary"):
                with st.spinner("Loading file..."):
                    start_time = time.time()
                    df, warning = load_projection_csv_optimized(f, 'tm')
                    st.session_state.projection_data['tm']['rcp45'] = df
                    if warning:
                        st.warning(warning)
                    else:
                        st.success(f"✅ RCP 4.5 Tm loaded: {len(df):,} records in {time.time()-start_time:.1f}s")
            files = st.file_uploader("...or merge multiple CSVs (one folder's worth)", type="csv",
                                      accept_multiple_files=True, key="up_rcp45_tm_multi")
            if files and st.button("Merge CSVs", key="btn_merge_rcp45_tm"):
                with st.spinner(f"Merging {len(files)} files..."):
                    start_time = time.time()
                    merged = merge_csv_files_optimized(files, 'tm')
                    if merged is not None:
                        st.session_state.projection_data['tm']['rcp45'] = merged
                        st.success(f"✅ Merged {len(files)} files → {len(merged):,} records in {time.time()-start_time:.1f}s")
                    else:
                        st.error("No data files found")

    # ---------------- RCP 8.5 ----------------
    with st.expander("🔴 RCP 8.5 Projection Data (2041–2100)"):
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown("**Significant Wave Height (SWH)**")
            f = st.file_uploader("RCP 8.5 SWH CSV (single file)", type="csv", key="up_rcp85_swh")
            if f is not None and st.button("Load RCP 8.5 SWH", key="btn_rcp85_swh", type="primary"):
                with st.spinner("Loading file..."):
                    start_time = time.time()
                    df, warning = load_projection_csv_optimized(f, 'swh')
                    st.session_state.projection_data['swh']['rcp85'] = df
                    if warning:
                        st.warning(warning)
                    else:
                        st.success(f"✅ RCP 8.5 SWH loaded: {len(df):,} records in {time.time()-start_time:.1f}s")
            files = st.file_uploader("...or merge multiple CSVs (one folder's worth)", type="csv",
                                      accept_multiple_files=True, key="up_rcp85_swh_multi")
            if files and st.button("Merge CSVs", key="btn_merge_rcp85_swh"):
                with st.spinner(f"Merging {len(files)} files..."):
                    start_time = time.time()
                    merged = merge_csv_files_optimized(files, 'swh')
                    if merged is not None:
                        st.session_state.projection_data['swh']['rcp85'] = merged
                        st.success(f"✅ Merged {len(files)} files → {len(merged):,} records in {time.time()-start_time:.1f}s")
                    else:
                        st.error("No data files found")

        with rc2:
            st.markdown("**Mean Wave Period (Tm)**")
            f = st.file_uploader("RCP 8.5 Tm CSV (single file)", type="csv", key="up_rcp85_tm")
            if f is not None and st.button("Load RCP 8.5 Tm", key="btn_rcp85_tm", type="primary"):
                with st.spinner("Loading file..."):
                    start_time = time.time()
                    df, warning = load_projection_csv_optimized(f, 'tm')
                    st.session_state.projection_data['tm']['rcp85'] = df
                    if warning:
                        st.warning(warning)
                    else:
                        st.success(f"✅ RCP 8.5 Tm loaded: {len(df):,} records in {time.time()-start_time:.1f}s")
            files = st.file_uploader("...or merge multiple CSVs (one folder's worth)", type="csv",
                                      accept_multiple_files=True, key="up_rcp85_tm_multi")
            if files and st.button("Merge CSVs", key="btn_merge_rcp85_tm"):
                with st.spinner(f"Merging {len(files)} files..."):
                    start_time = time.time()
                    merged = merge_csv_files_optimized(files, 'tm')
                    if merged is not None:
                        st.session_state.projection_data['tm']['rcp85'] = merged
                        st.success(f"✅ Merged {len(files)} files → {len(merged):,} records in {time.time()-start_time:.1f}s")
                    else:
                        st.error("No data files found")

    # ---------------- Test data ----------------
    with st.expander("🧪 Test Data (e.g., Brisbane 2024)"):
        f = st.file_uploader("Test CSV", type="csv", key="up_test")
        if f is not None and st.button("Load Test Data", key="btn_test", type="primary"):
            with st.spinner("Loading test data..."):
                start_time = time.time()
                df, error = load_test_data_optimized(f)
                if error:
                    st.error(error)
                else:
                    st.session_state.test_data = df
                    st.success(f"✅ Test data loaded: {len(df):,} records from {df['time'].min()} to {df['time'].max()} in {time.time()-start_time:.1f}s")

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

# ============================================================================
# TAB 2 — GPD Analysis (Optimized)
# ============================================================================

with tab_gpd:
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
                start_time = time.time()
                # Use cached version
                result = run_gpd_analysis_cached(data, int(decluster_hours), float(threshold_percentile))
            if result is not None:
                st.session_state.gpd_results[gpd_variable][gpd_scenario] = result
                st.session_state['_last_gpd'] = (result, gpd_variable, gpd_scenario, threshold_percentile)
                st.success(f"✅ GPD analysis complete in {time.time()-start_time:.1f}s")
            else:
                st.error("GPD analysis failed - insufficient data or peaks")

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
        style_dark_axes(ax1, fig1)

        full_data = st.session_state.projection_data[variable][scenario]
        if full_data is not None:
            time_series = pd.to_datetime(full_data['time'])
            ax1.plot(time_series, full_data[variable], color='#888888', alpha=0.3, linewidth=0.5, label='Full data')

        decl = result['declustered']
        if len(decl) > 0:
            ax1.plot(decl.index, decl.values, 'o', color='#00ccff', markersize=4, alpha=0.8, label='Peaks')
        
        ax1.axhline(y=result['threshold'], color='#ff6b6b', linestyle='--', linewidth=2,
                    label=f'Threshold ({pct}%)')

        textstr = f'Peaks: {result["n_peaks"]}'
        props = dict(boxstyle='round', facecolor='#444444', alpha=0.8)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                  verticalalignment='top', bbox=props, color='white')

        ax1.set_xlabel('Time')
        ax1.set_ylabel(ylabel)
        ax1.set_title(f'Declustered Peaks - {data_name}')
        ax1.legend(facecolor='#444444', edgecolor='white', labelcolor='white', fontsize=8)
        ax1.grid(True, alpha=0.3, color='#666666')
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(10, 3.5))
        style_dark_axes(ax2, fig2)

        exceedances = result['exceedances']
        if len(exceedances) > 0:
            ax2.hist(exceedances, bins=30, density=True, alpha=0.5, color='#00ccff', label='Exceedances')
            x = np.linspace(0, exceedances.max(), 200)
            pdf = genpareto.pdf(x, result['xi'], result['loc'], result['beta'])
            ax2.plot(x, pdf, 'r-', lw=2, label=f'GPD fit (ξ={result["xi"]:.2f}, β={result["beta"]:.2f})')

        ax2.set_xlabel(f'Exceedance ({ylabel})')
        ax2.set_ylabel('Density')
        ax2.set_title('GPD Fit to Exceedances')
        ax2.legend(facecolor='#444444', edgecolor='white', labelcolor='white')
        ax2.grid(True, alpha=0.3, color='#666666')
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
    else:
        st.info("Run a GPD analysis to see results here.")

# ============================================================================
# TAB 3 — Projections (Optimized)
# ============================================================================

with tab_proj:
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

                for idx, sc in enumerate(scenarios_to_run):
                    status.text(f"Generating projections for {sc.upper()} {proj_variable.upper()}...")
                    
                    # Use cached version
                    proj_df, error = generate_projection_cached(
                        baseline_df,
                        st.session_state.projection_data[proj_variable][sc],
                        sc, 
                        proj_variable
                    )
                    
                    if proj_df is not None:
                        st.session_state.projected_series[proj_variable][sc] = proj_df
                        projection_results.append(proj_df)
                        info_lines.append(
                            f"✅ Generated {sc.upper()} {proj_variable.upper()}: {len(proj_df)} records\n"
                            f"   Range: [{proj_df[proj_variable].min():.4f}, {proj_df[proj_variable].max():.4f}]\n"
                            f"   Mean: {proj_df[proj_variable].mean():.4f}, Std: {proj_df[proj_variable].std():.4f}"
                        )
                    else:
                        info_lines.append(f"❌ Failed to generate projections for {sc.upper()} {proj_variable.upper()}: {error}")
                    
                    progress.progress((idx + 1) / len(scenarios_to_run))

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

                    st.success(f"✅ Projections generated successfully for {variable_name}")
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

        # Combined projections plot
        fig, ax = plt.subplots(figsize=(12, 5))
        style_dark_axes(ax, fig)

        for df in projection_results:
            if 'scenario' not in df.columns:
                continue
            scenario = df['scenario'].iloc[0]
            df_copy = df.copy()
            df_copy['year'] = pd.to_datetime(df_copy['time']).dt.year
            yearly_avg = df_copy.groupby('year')[variable].mean().reset_index()
            ax.plot(pd.to_datetime(yearly_avg['year'], format='%Y'), yearly_avg[variable],
                    'o-', color=colors.get(scenario, '#888888'), linewidth=2.5,
                    markersize=6, label=labels.get(scenario, scenario))

        ax.set_xlabel('Time')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Projected {title_var} - {scenario_text}')
        ax.legend(facecolor='#444444', edgecolor='white', labelcolor='white')
        ax.grid(True, alpha=0.3, color='#666666')
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Long-term trend plot
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        style_dark_axes(ax2, fig2)

        for df in projection_results:
            if 'scenario' not in df.columns:
                continue
            scenario = df['scenario'].iloc[0]
            df_copy = df.copy()
            df_copy['year'] = pd.to_datetime(df_copy['time']).dt.year
            yearly_means = df_copy.groupby('year')[variable].mean()

            if len(yearly_means) > 1:
                x = yearly_means.index.values
                y = yearly_means.values
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                trend = intercept + slope * x

                ax2.plot(x, y, 'o', color=colors.get(scenario, '#888888'),
                         markersize=5, alpha=0.7, label=f'{labels.get(scenario, scenario)} - yearly')
                ax2.plot(x, trend, '--', color=colors.get(scenario, '#888888'), linewidth=2,
                          label=f'{labels.get(scenario, scenario)} trend (slope={slope:.4f} m/year)')

        ax2.set_xlabel('Year')
        ax2.set_ylabel(f'Mean {ylabel}')
        ax2.set_title(f'Long-term Trend Analysis - {title_var} - {scenario_text}')
        ax2.legend(facecolor='#444444', edgecolor='white', labelcolor='white', fontsize=9)
        ax2.grid(True, alpha=0.3, color='#666666')
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
            summary_lines = [f"✅ Export prepared: {len(exported_files)} file(s)", ""]
            for f in exported_files[:10]:
                summary_lines.append(f"  • {f}")
            if len(exported_files) > 10:
                summary_lines.append(f"  • ... and {len(exported_files) - 10} more")
            st.session_state['_export_summary'] = "\n".join(summary_lines)
            st.success(f"Export package prepared with {len(exported_files)} files")
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
