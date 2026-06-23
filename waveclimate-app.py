"""
Wave Climate Projection Tool - Streamlit Version
Web-based application for Wave Climate Analysis under RCP Scenarios
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto, linregress
from datetime import datetime
import warnings
import os
from io import StringIO
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Wave Climate Projection Tool",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #0066cc;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #888888;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ffffff;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding: 0.5rem;
        background-color: #333333;
        border-radius: 5px;
    }
    .info-box {
        background-color: #2b2b2b;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #0066cc;
        margin: 1rem 0;
    }
    .stButton > button {
        background-color: #0066cc;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton > button:hover {
        background-color: #1a75d2;
        color: white;
    }
    .stButton > button:active {
        background-color: #0052a3;
    }
    .success-btn > button {
        background-color: #28a745;
    }
    .success-btn > button:hover {
        background-color: #34b750;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for data storage
if 'baseline_data' not in st.session_state:
    st.session_state.baseline_data = {'swh': None, 'tm': None}

if 'projection_data' not in st.session_state:
    st.session_state.projection_data = {
        'swh': {'rcp45': None, 'rcp85': None},
        'tm': {'rcp45': None, 'rcp85': None}
    }

if 'gpd_results' not in st.session_state:
    st.session_state.gpd_results = {
        'swh': {'rcp45': None, 'rcp85': None},
        'tm': {'rcp45': None, 'rcp85': None}
    }

if 'projected_series' not in st.session_state:
    st.session_state.projected_series = {
        'swh': {'rcp45': None, 'rcp85': None},
        'tm': {'rcp45': None, 'rcp85': None}
    }

if 'test_data' not in st.session_state:
    st.session_state.test_data = None

# Helper functions
def standardize_column_names(df, data_type):
    """Standardize column names to 'swh' or 'tm'"""
    df = df.copy()
    
    if data_type == 'swh':
        possible_names = ['swh', 'SWH', 'hs', 'Hs', 'hs (m)', 'Hs (m)', 
                         'significant_wave_height', 'significant wave height']
        for col in df.columns:
            col_lower = col.lower()
            if any(name.lower() in col_lower for name in possible_names):
                df.rename(columns={col: 'swh'}, inplace=True)
                break
        if 'swh' not in df.columns and 'mp1' in df.columns:
            df.rename(columns={'mp1': 'swh'}, inplace=True)
    
    elif data_type == 'tm':
        possible_names = ['tm', 'Tm', 'TM', 'tp', 'Tp', 'tp (s)', 'Tp (s)', 
                         'mean_wave_period', 'mean wave period']
        for col in df.columns:
            col_lower = col.lower()
            if any(name.lower() in col_lower for name in possible_names):
                df.rename(columns={col: 'tm'}, inplace=True)
                break
        if 'tm' not in df.columns and 'mp1' in df.columns:
            df.rename(columns={'mp1': 'tm'}, inplace=True)
    
    return df

def load_csv_with_date_parsing(file):
    """Load CSV with flexible date parsing"""
    try:
        df = pd.read_csv(file, parse_dates=['time'], date_format='%Y-%m-%d %H:%M:%S')
    except:
        try:
            df = pd.read_csv(file, parse_dates=['time'])
        except:
            try:
                df = pd.read_csv(file)
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'], format='mixed')
            except:
                df = pd.read_csv(file)
    return df

def run_gpd_analysis(data, decluster_hours, percentile):
    """Run GPD analysis"""
    threshold = np.percentile(data, percentile)
    
    if decluster_hours == 0:
        exceedances_raw = data[data > threshold]
        declustered = exceedances_raw
    else:
        exceedances_raw = data[data > threshold]
        declustered = exceedances_raw.resample(f"{decluster_hours}H").max().dropna()
    
    exceedances = declustered - threshold
    
    if len(exceedances) > 0:
        xi, loc, beta = genpareto.fit(exceedances)
    else:
        xi, loc, beta = np.nan, np.nan, np.nan
    
    result = {
        'threshold': threshold,
        'declustered': declustered,
        'exceedances': exceedances,
        'xi': xi,
        'beta': beta,
        'loc': loc,
        'n_peaks': len(declustered)
    }
    return result

def generate_projection(df_obs, df_proj, variable):
    """Generate projected time series"""
    if variable not in df_obs.columns or variable not in df_proj.columns:
        return None
    
    df_proj['time'] = pd.to_datetime(df_proj['time'])
    years = df_proj['time'].dt.year.unique()
    years.sort()
    
    projected_data = []
    
    for year in years:
        df_year = df_proj[df_proj['time'].dt.year == year].copy()
        
        if len(df_year) < 2:
            continue
        
        time_index = pd.date_range(
            start=f"{year}-01-01 00:00:00",
            periods=len(df_year),
            freq='H'
        )[:len(df_year)]
        
        year_df = pd.DataFrame({
            'time': time_index,
            variable: df_year[variable].values[:len(time_index)]
        })
        
        projected_data.append(year_df)
    
    if projected_data:
        return pd.concat(projected_data, ignore_index=True)
    return None

# Main title
st.markdown('<div class="main-header">🌊 Wave Climate Projection Tool</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">A Unified Framework for Wave Climate Analysis Under RCP4.5 and RCP8.5 Scenarios</div>', unsafe_allow_html=True)

# Sidebar for data loading
with st.sidebar:
    st.markdown("## 📁 Data Loading")
    
    # Baseline Data
    st.markdown("### 📊 Baseline Data (2005)")
    
    baseline_swh_file = st.file_uploader("Baseline SWH", type=['csv'], key="baseline_swh")
    if baseline_swh_file is not None:
        df = load_csv_with_date_parsing(baseline_swh_file)
        df = standardize_column_names(df, 'swh')
        if 'swh' in df.columns:
            st.session_state.baseline_data['swh'] = df
            st.success(f"✅ Loaded SWH: {len(df)} records")
        else:
            st.error("Could not find SWH column. Available columns: " + ", ".join(df.columns))
    
    baseline_tm_file = st.file_uploader("Baseline Tm", type=['csv'], key="baseline_tm")
    if baseline_tm_file is not None:
        df = load_csv_with_date_parsing(baseline_tm_file)
        df = standardize_column_names(df, 'tm')
        if 'tm' in df.columns:
            st.session_state.baseline_data['tm'] = df
            st.success(f"✅ Loaded Tm: {len(df)} records")
        else:
            st.error("Could not find Tm column. Available columns: " + ", ".join(df.columns))
    
    st.markdown("---")
    
    # RCP 4.5 Data
    st.markdown("### 🔵 RCP 4.5 Projections (2041-2100)")
    
    rcp45_swh_file = st.file_uploader("RCP 4.5 SWH", type=['csv'], key="rcp45_swh")
    if rcp45_swh_file is not None:
        df = load_csv_with_date_parsing(rcp45_swh_file)
        df = standardize_column_names(df, 'swh')
        if 'swh' in df.columns:
            st.session_state.projection_data['swh']['rcp45'] = df
            st.success(f"✅ Loaded RCP 4.5 SWH: {len(df)} records")
        else:
            st.error("Could not find SWH column. Available columns: " + ", ".join(df.columns))
    
    rcp45_tm_file = st.file_uploader("RCP 4.5 Tm", type=['csv'], key="rcp45_tm")
    if rcp45_tm_file is not None:
        df = load_csv_with_date_parsing(rcp45_tm_file)
        df = standardize_column_names(df, 'tm')
        if 'tm' in df.columns:
            st.session_state.projection_data['tm']['rcp45'] = df
            st.success(f"✅ Loaded RCP 4.5 Tm: {len(df)} records")
        else:
            st.error("Could not find Tm column. Available columns: " + ", ".join(df.columns))
    
    st.markdown("---")
    
    # RCP 8.5 Data
    st.markdown("### 🔴 RCP 8.5 Projections (2041-2100)")
    
    rcp85_swh_file = st.file_uploader("RCP 8.5 SWH", type=['csv'], key="rcp85_swh")
    if rcp85_swh_file is not None:
        df = load_csv_with_date_parsing(rcp85_swh_file)
        df = standardize_column_names(df, 'swh')
        if 'swh' in df.columns:
            st.session_state.projection_data['swh']['rcp85'] = df
            st.success(f"✅ Loaded RCP 8.5 SWH: {len(df)} records")
        else:
            st.error("Could not find SWH column. Available columns: " + ", ".join(df.columns))
    
    rcp85_tm_file = st.file_uploader("RCP 8.5 Tm", type=['csv'], key="rcp85_tm")
    if rcp85_tm_file is not None:
        df = load_csv_with_date_parsing(rcp85_tm_file)
        df = standardize_column_names(df, 'tm')
        if 'tm' in df.columns:
            st.session_state.projection_data['tm']['rcp85'] = df
            st.success(f"✅ Loaded RCP 8.5 Tm: {len(df)} records")
        else:
            st.error("Could not find Tm column. Available columns: " + ", ".join(df.columns))
    
    st.markdown("---")
    
    # Test Data
    st.markdown("### 🧪 Test Data")
    test_file = st.file_uploader("Test CSV (e.g., Brisbane 2024)", type=['csv'], key="test_data")
    if test_file is not None:
        try:
            df = pd.read_csv(test_file)
            
            # Handle Brisbane format
            if 'Date/Time (AEST)' in df.columns:
                try:
                    df['time'] = pd.to_datetime(df['Date/Time (AEST)'], format='%Y-%m-%dT%H:%M')
                except:
                    try:
                        df['time'] = pd.to_datetime(df['Date/Time (AEST)'], format='mixed')
                    except:
                        df['time'] = pd.to_datetime(df['Date/Time (AEST)'], infer_datetime_format=True)
                
                df.replace(-99.90, np.nan, inplace=True)
                df.replace(-99.9, np.nan, inplace=True)
                df.replace(-99, np.nan, inplace=True)
                
                if 'Hs (m)' in df.columns:
                    df.rename(columns={'Hs (m)': 'swh'}, inplace=True)
                elif 'Hs' in df.columns:
                    df.rename(columns={'Hs': 'swh'}, inplace=True)
                
                if 'Tp (s)' in df.columns:
                    df.rename(columns={'Tp (s)': 'tm'}, inplace=True)
                
                cols_to_drop = []
                for col in ['Hmax (m)', 'Peak Direction (degrees)', 'SST (degrees C)', 
                           'Tz (s)', 'Date/Time (AEST)']:
                    if col in df.columns:
                        cols_to_drop.append(col)
                if cols_to_drop:
                    df.drop(cols_to_drop, axis=1, inplace=True)
                
                df.dropna(inplace=True)
                df.sort_values('time', inplace=True)
                df.reset_index(drop=True, inplace=True)
                
                st.session_state.test_data = df
                st.success(f"✅ Loaded Test Data: {len(df)} records")
            else:
                st.error("Could not find 'Date/Time (AEST)' column")
        except Exception as e:
            st.error(f"Error loading test data: {str(e)}")

# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Preview", "🎯 GPD Analysis", "🔄 Projections", "📊 Results"])

# Tab 1: Data Preview
with tab1:
    st.markdown("## 📊 Data Preview")
    
    # Show summary of loaded data
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Baseline Data")
        if st.session_state.baseline_data['swh'] is not None:
            df = st.session_state.baseline_data['swh']
            st.write(f"**SWH**: {len(df)} records")
            if 'time' in df.columns:
                st.write(f"Range: {df['time'].min()} to {df['time'].max()}")
            st.write(f"Mean: {df['swh'].mean():.4f}, Std: {df['swh'].std():.4f}")
        else:
            st.write("No baseline data loaded")
        
        if st.session_state.baseline_data['tm'] is not None:
            df = st.session_state.baseline_data['tm']
            st.write(f"**Tm**: {len(df)} records")
            if 'time' in df.columns:
                st.write(f"Range: {df['time'].min()} to {df['time'].max()}")
            st.write(f"Mean: {df['tm'].mean():.4f}, Std: {df['tm'].std():.4f}")
        else:
            st.write("No baseline Tm loaded")
    
    with col2:
        st.markdown("### Projection Data")
        for scenario in ['rcp45', 'rcp85']:
            label = "RCP 4.5" if scenario == 'rcp45' else "RCP 8.5"
            st.write(f"**{label}**")
            for var in ['swh', 'tm']:
                if st.session_state.projection_data[var][scenario] is not None:
                    df = st.session_state.projection_data[var][scenario]
                    status = f"✅ {var.upper()}: {len(df)} records"
                else:
                    status = f"❌ {var.upper()}: Not loaded"
                st.write(f"  {status}")
    
    # Preview selected data
    st.markdown("### Data Preview Table")
    data_options = ["None"]
    if st.session_state.baseline_data['swh'] is not None:
        data_options.append("Baseline SWH")
    if st.session_state.baseline_data['tm'] is not None:
        data_options.append("Baseline Tm")
    if st.session_state.projection_data['swh']['rcp45'] is not None:
        data_options.append("RCP 4.5 SWH")
    if st.session_state.projection_data['tm']['rcp45'] is not None:
        data_options.append("RCP 4.5 Tm")
    if st.session_state.projection_data['swh']['rcp85'] is not None:
        data_options.append("RCP 8.5 SWH")
    if st.session_state.projection_data['tm']['rcp85'] is not None:
        data_options.append("RCP 8.5 Tm")
    if st.session_state.test_data is not None:
        data_options.append("Test Data")
    
    selected_data = st.selectbox("Select dataset to preview:", data_options)
    
    if selected_data != "None":
        # Get the selected data
        if selected_data == "Baseline SWH" and st.session_state.baseline_data['swh'] is not None:
            df = st.session_state.baseline_data['swh']
        elif selected_data == "Baseline Tm" and st.session_state.baseline_data['tm'] is not None:
            df = st.session_state.baseline_data['tm']
        elif selected_data == "RCP 4.5 SWH" and st.session_state.projection_data['swh']['rcp45'] is not None:
            df = st.session_state.projection_data['swh']['rcp45']
        elif selected_data == "RCP 4.5 Tm" and st.session_state.projection_data['tm']['rcp45'] is not None:
            df = st.session_state.projection_data['tm']['rcp45']
        elif selected_data == "RCP 8.5 SWH" and st.session_state.projection_data['swh']['rcp85'] is not None:
            df = st.session_state.projection_data['swh']['rcp85']
        elif selected_data == "RCP 8.5 Tm" and st.session_state.projection_data['tm']['rcp85'] is not None:
            df = st.session_state.projection_data['tm']['rcp85']
        elif selected_data == "Test Data" and st.session_state.test_data is not None:
            df = st.session_state.test_data
        else:
            df = None
        
        if df is not None:
            st.dataframe(df.head(100), use_container_width=True)
            st.caption(f"Showing first 100 rows of {len(df)} total records")

# Tab 2: GPD Analysis
with tab2:
    st.markdown("## 🎯 GPD Analysis")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        variable = st.selectbox("Variable:", ["Significant Wave Height (SWH)", "Mean Wave Period (Tm)"])
        var_key = 'swh' if 'SWH' in variable else 'tm'
    
    with col2:
        scenario = st.selectbox("Scenario:", ["RCP 4.5", "RCP 8.5"])
        scenario_key = 'rcp45' if '4.5' in scenario else 'rcp85'
    
    with col3:
        decluster_hours = st.number_input("Decluster Hours:", min_value=0, max_value=72, value=12)
        threshold_percentile = st.slider("Threshold Percentile:", min_value=90.0, max_value=99.9, value=99.5, step=0.1)
    
    # Check if data is available
    data_available = st.session_state.projection_data[var_key][scenario_key] is not None
    
    if not data_available:
        st.warning(f"Please load {scenario} {variable} data first")
    else:
        if st.button("🎯 Run GPD Analysis", type="primary"):
            with st.spinner(f"Running GPD analysis for {scenario} {variable}..."):
                df = st.session_state.projection_data[var_key][scenario_key]
                
                if var_key not in df.columns:
                    st.error(f"Column '{var_key}' not found in data")
                else:
                    data = pd.Series(
                        df[var_key].values,
                        index=pd.to_datetime(df['time'])
                    )
                    
                    result = run_gpd_analysis(data, decluster_hours, threshold_percentile)
                    st.session_state.gpd_results[var_key][scenario_key] = result
                    
                    # Display results
                    st.markdown("### GPD Analysis Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Threshold", f"{result['threshold']:.4f}")
                    with col2:
                        st.metric("Number of Peaks", result['n_peaks'])
                    with col3:
                        st.metric("ξ (Shape)", f"{result['xi']:.4f}")
                    with col4:
                        st.metric("β (Scale)", f"{result['beta']:.4f}")
                    
                    # Plot results
                    st.markdown("### 📈 Visualizations")
                    
                    # Plot 1: Time series with peaks
                    fig1, ax1 = plt.subplots(figsize=(12, 5))
                    ax1.set_facecolor('#333333')
                    fig1.patch.set_facecolor('#333333')
                    ax1.tick_params(colors='white')
                    ax1.xaxis.label.set_color('white')
                    ax1.yaxis.label.set_color('white')
                    ax1.title.set_color('white')
                    
                    # Full time series
                    ax1.plot(data.index, data.values, color='#888888', alpha=0.3, linewidth=0.5, label='Full data')
                    
                    # Peaks
                    ylabel = 'Hs [m]' if var_key == 'swh' else 'Tm [s]'
                    ax1.plot(result['declustered'].index, result['declustered'].values, 
                            'o', color='#00ccff', markersize=4, alpha=0.8, label='Peaks')
                    ax1.axhline(y=result['threshold'], color='#ff6b6b', 
                               linestyle='--', linewidth=2, label=f'Threshold ({threshold_percentile}%)')
                    
                    ax1.set_xlabel('Time')
                    ax1.set_ylabel(ylabel)
                    ax1.set_title(f'Declustered Peaks - {scenario} {variable}')
                    ax1.legend(facecolor='#444444', edgecolor='white', labelcolor='white', fontsize=8)
                    ax1.grid(True, alpha=0.3, color='#666666')
                    
                    st.pyplot(fig1)
                    
                    # Plot 2: GPD fit
                    fig2, ax2 = plt.subplots(figsize=(12, 5))
                    ax2.set_facecolor('#333333')
                    fig2.patch.set_facecolor('#333333')
                    ax2.tick_params(colors='white')
                    ax2.xaxis.label.set_color('white')
                    ax2.yaxis.label.set_color('white')
                    ax2.title.set_color('white')
                    
                    exceedances = result['exceedances']
                    if len(exceedances) > 0:
                        ax2.hist(exceedances, bins=30, density=True, alpha=0.5, 
                                color='#00ccff', label='Exceedances')
                        
                        x = np.linspace(0, exceedances.max(), 200)
                        pdf = genpareto.pdf(x, result['xi'], result['loc'], result['beta'])
                        ax2.plot(x, pdf, 'r-', lw=2, 
                                label=f'GPD fit (ξ={result["xi"]:.2f}, β={result["beta"]:.2f})')
                    
                    ax2.set_xlabel(f'Exceedance ({ylabel})')
                    ax2.set_ylabel('Density')
                    ax2.set_title('GPD Fit to Exceedances')
                    ax2.legend(facecolor='#444444', edgecolor='white', labelcolor='white')
                    ax2.grid(True, alpha=0.3, color='#666666')
                    
                    st.pyplot(fig2)
                    
                    st.success(f"GPD analysis complete for {scenario} {variable}!")

# Tab 3: Projections
with tab3:
    st.markdown("## 🔄 Generate Projections")
    
    col1, col2 = st.columns(2)
    
    with col1:
        proj_variable = st.selectbox("Variable:", ["Significant Wave Height (SWH)", "Mean Wave Period (Tm)"], key="proj_var")
        proj_var_key = 'swh' if 'SWH' in proj_variable else 'tm'
    
    with col2:
        proj_scenario = st.selectbox("Scenario:", ["RCP 4.5", "RCP 8.5", "Both"], key="proj_scenario")
    
    # Check if required data is available
    baseline_available = st.session_state.baseline_data[proj_var_key] is not None
    proj_data_available = False
    
    if proj_scenario == "Both":
        proj_data_available = (st.session_state.projection_data[proj_var_key]['rcp45'] is not None and 
                              st.session_state.projection_data[proj_var_key]['rcp85'] is not None)
    elif proj_scenario == "RCP 4.5":
        proj_data_available = st.session_state.projection_data[proj_var_key]['rcp45'] is not None
    else:
        proj_data_available = st.session_state.projection_data[proj_var_key]['rcp85'] is not None
    
    if not baseline_available:
        st.warning(f"Please load baseline {proj_variable} data first")
    elif not proj_data_available:
        st.warning(f"Please load {proj_scenario} {proj_variable} data first")
    else:
        if st.button("🔄 Generate Projections", type="primary"):
            with st.spinner(f"Generating projections for {proj_variable}..."):
                # Determine scenarios to run
                scenarios_to_run = []
                if proj_scenario == "Both":
                    scenarios_to_run = ['rcp45', 'rcp85']
                elif proj_scenario == "RCP 4.5":
                    scenarios_to_run = ['rcp45']
                else:
                    scenarios_to_run = ['rcp85']
                
                projection_dfs = []
                variable_name = "SWH" if proj_var_key == 'swh' else "Tm"
                
                for sc in scenarios_to_run:
                    proj_df = generate_projection(
                        st.session_state.baseline_data[proj_var_key],
                        st.session_state.projection_data[proj_var_key][sc],
                        proj_var_key
                    )
                    
                    if proj_df is not None:
                        proj_df['scenario'] = sc
                        st.session_state.projected_series[proj_var_key][sc] = proj_df
                        projection_dfs.append(proj_df)
                
                if projection_dfs:
                    # Plot projections
                    st.markdown("### 📈 Projection Results")
                    
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.set_facecolor('#333333')
                    fig.patch.set_facecolor('#333333')
                    ax.tick_params(colors='white')
                    ax.xaxis.label.set_color('white')
                    ax.yaxis.label.set_color('white')
                    ax.title.set_color('white')
                    
                    colors = {'rcp45': '#0066cc', 'rcp85': '#ff4444'}
                    labels = {'rcp45': 'RCP 4.5', 'rcp85': 'RCP 8.5'}
                    
                    for df in projection_dfs:
                        scenario_key = df['scenario'].iloc[0]
                        df['year'] = pd.to_datetime(df['time']).dt.year
                        yearly_avg = df.groupby('year')[proj_var_key].mean().reset_index()
                        
                        ax.plot(pd.to_datetime(yearly_avg['year'], format='%Y'), 
                               yearly_avg[proj_var_key],
                               'o-', color=colors.get(scenario_key, '#888888'), 
                               linewidth=2.5, markersize=6, 
                               label=labels.get(scenario_key, scenario_key))
                    
                    ylabel = 'Hs [m]' if proj_var_key == 'swh' else 'Tm [s]'
                    ax.set_xlabel('Time')
                    ax.set_ylabel(ylabel)
                    ax.set_title(f'Projected {proj_variable} - {proj_scenario}')
                    ax.legend(facecolor='#444444', edgecolor='white', labelcolor='white')
                    ax.grid(True, alpha=0.3, color='#666666')
                    
                    st.pyplot(fig)
                    
                    # Trend analysis
                    st.markdown("### 📉 Long-term Trend Analysis")
                    
                    fig2, ax2 = plt.subplots(figsize=(12, 5))
                    ax2.set_facecolor('#333333')
                    fig2.patch.set_facecolor('#333333')
                    ax2.tick_params(colors='white')
                    ax2.xaxis.label.set_color('white')
                    ax2.yaxis.label.set_color('white')
                    ax2.title.set_color('white')
                    
                    for df in projection_dfs:
                        scenario_key = df['scenario'].iloc[0]
                        df['year'] = pd.to_datetime(df['time']).dt.year
                        yearly_means = df.groupby('year')[proj_var_key].mean()
                        
                        x = yearly_means.index.values
                        y = yearly_means.values
                        slope, intercept, r_value, p_value, std_err = linregress(x, y)
                        trend = intercept + slope * x
                        
                        ax2.plot(x, y, 'o', color=colors.get(scenario_key, '#888888'), 
                                markersize=5, alpha=0.7, 
                                label=f'{labels.get(scenario_key, scenario_key)} - yearly')
                        ax2.plot(x, trend, '--', color=colors.get(scenario_key, '#888888'), linewidth=2,
                                label=f'{labels.get(scenario_key, scenario_key)} trend (slope={slope:.4f} m/year)')
                    
                    ax2.set_xlabel('Year')
                    ax2.set_ylabel(f'Mean {ylabel}')
                    ax2.set_title(f'Long-term Trend Analysis - {proj_variable}')
                    ax2.legend(facecolor='#444444', edgecolor='white', labelcolor='white', fontsize=9)
                    ax2.grid(True, alpha=0.3, color='#666666')
                    
                    st.pyplot(fig2)
                    
                    # Display statistics
                    st.markdown("### 📊 Statistics")
                    stats_data = []
                    for df in projection_dfs:
                        scenario_key = df['scenario'].iloc[0]
                        stats_data.append({
                            'Scenario': labels.get(scenario_key, scenario_key),
                            'Records': len(df),
                            'Mean': df[proj_var_key].mean(),
                            'Std': df[proj_var_key].std(),
                            'Min': df[proj_var_key].min(),
                            'Max': df[proj_var_key].max()
                        })
                    
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df, use_container_width=True)
                    
                    # Download combined projections
                    combined_df = pd.concat(projection_dfs, ignore_index=True)
                    csv = combined_df.to_csv(index=False)
                    
                    st.download_button(
                        label="💾 Download Projections as CSV",
                        data=csv,
                        file_name=f"projections_{variable_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    st.success("Projections generated successfully!")
                else:
                    st.error("Failed to generate projections")

# Tab 4: Results
with tab4:
    st.markdown("## 📊 Results Summary")
    
    # Summary statistics
    st.markdown("### Data Loading Status")
    
    status_data = {
        'Data Type': [],
        'Status': [],
        'Records': []
    }
    
    # Baseline
    for var, label in [('swh', 'Baseline SWH'), ('tm', 'Baseline Tm')]:
        if st.session_state.baseline_data[var] is not None:
            status_data['Data Type'].append(label)
            status_data['Status'].append('✅ Loaded')
            status_data['Records'].append(len(st.session_state.baseline_data[var]))
        else:
            status_data['Data Type'].append(label)
            status_data['Status'].append('❌ Not loaded')
            status_data['Records'].append(0)
    
    # Projections
    for scenario, scenario_label in [('rcp45', 'RCP 4.5'), ('rcp85', 'RCP 8.5')]:
        for var, var_label in [('swh', 'SWH'), ('tm', 'Tm')]:
            key = f"{scenario_label} {var_label}"
            if st.session_state.projection_data[var][scenario] is not None:
                status_data['Data Type'].append(key)
                status_data['Status'].append('✅ Loaded')
                status_data['Records'].append(len(st.session_state.projection_data[var][scenario]))
            else:
                status_data['Data Type'].append(key)
                status_data['Status'].append('❌ Not loaded')
                status_data['Records'].append(0)
    
    # Test data
    if st.session_state.test_data is not None:
        status_data['Data Type'].append('Test Data')
        status_data['Status'].append('✅ Loaded')
        status_data['Records'].append(len(st.session_state.test_data))
    else:
        status_data['Data Type'].append('Test Data')
        status_data['Status'].append('❌ Not loaded')
        status_data['Records'].append(0)
    
    status_df = pd.DataFrame(status_data)
    st.dataframe(status_df, use_container_width=True)
    
    # GPD Results Summary
    st.markdown("### GPD Analysis Results")
    
    gpd_data = []
    for var, var_label in [('swh', 'SWH'), ('tm', 'Tm')]:
        for scenario, scenario_label in [('rcp45', 'RCP 4.5'), ('rcp85', 'RCP 8.5')]:
            result = st.session_state.gpd_results[var][scenario]
            if result is not None:
                gpd_data.append({
                    'Variable': var_label,
                    'Scenario': scenario_label,
                    'Threshold': f"{result['threshold']:.4f}",
                    'Peaks': result['n_peaks'],
                    'ξ': f"{result['xi']:.4f}",
                    'β': f"{result['beta']:.4f}"
                })
    
    if gpd_data:
        gpd_df = pd.DataFrame(gpd_data)
        st.dataframe(gpd_df, use_container_width=True)
    else:
        st.info("No GPD analysis results available. Run GPD analysis first.")
    
    # Export functionality
    st.markdown("### 💾 Export All Results")
    
    if st.button("📦 Export Complete Dataset", type="primary"):
        # Create a zip file with all data
        import io
        import zipfile
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Export all dataframes
            # Baseline
            for var in ['swh', 'tm']:
                if st.session_state.baseline_data[var] is not None:
                    csv = st.session_state.baseline_data[var].to_csv(index=False)
                    zip_file.writestr(f"baseline_{var}_{timestamp}.csv", csv)
            
            # Projections
            for var in ['swh', 'tm']:
                for scenario in ['rcp45', 'rcp85']:
                    if st.session_state.projection_data[var][scenario] is not None:
                        csv = st.session_state.projection_data[var][scenario].to_csv(index=False)
                        zip_file.writestr(f"projection_{var}_{scenario}_{timestamp}.csv", csv)
            
            # Test data
            if st.session_state.test_data is not None:
                csv = st.session_state.test_data.to_csv(index=False)
                zip_file.writestr(f"test_data_{timestamp}.csv", csv)
            
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
                        csv = gpd_df.to_csv(index=False)
                        zip_file.writestr(f"gpd_{var}_{scenario}_{timestamp}.csv", csv)
            
            # Projected series
            for var in ['swh', 'tm']:
                for scenario in ['rcp45', 'rcp85']:
                    if st.session_state.projected_series[var][scenario] is not None:
                        csv = st.session_state.projected_series[var][scenario].to_csv(index=False)
                        zip_file.writestr(f"projected_{var}_{scenario}_{timestamp}.csv", csv)
        
        zip_buffer.seek(0)
        
        st.download_button(
            label="💾 Download All Data (ZIP)",
            data=zip_buffer,
            file_name=f"wave_climate_results_{timestamp}.zip",
            mime="application/zip"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888888; padding: 1rem;">
    🌊 Wave Climate Projection Tool v1.0 | Built with Streamlit | RCP4.5/RCP8.5 Scenarios
</div>
""", unsafe_allow_html=True)