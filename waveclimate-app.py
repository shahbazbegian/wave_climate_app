"""
Wave Climate Projection Tool
Professional GUI Application for Wave Climate Analysis under RCP Scenarios
CSV-based version - All data in CSV format
Author: Based on research paper "Wave Climate Projection under Climate Change Scenarios"
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto, linregress
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import os
from io import StringIO

st.set_page_config(page_title="Wave Climate Projection Tool", layout="wide")

# ============================================================================
# Helper Functions
# ============================================================================

def standardize_column_names(df, data_type):
    """Standardize column names to 'swh' or 'tm'"""
    df = df.copy()
    
    if data_type == 'swh':
        possible_names = ['swh', 'SWH', 'hs', 'Hs', 'hs (m)', 'Hs (m)', 'significant_wave_height', 'significant wave height']
        for col in df.columns:
            col_lower = col.lower()
            if any(name.lower() in col_lower for name in possible_names):
                df.rename(columns={col: 'swh'}, inplace=True)
                break
        if 'swh' not in df.columns and 'mp1' in df.columns:
            df.rename(columns={'mp1': 'swh'}, inplace=True)
    
    elif data_type == 'tm':
        possible_names = ['tm', 'Tm', 'TM', 'tp', 'Tp', 'tp (s)', 'Tp (s)', 'mean_wave_period', 'mean wave period', 'mp1']
        for col in df.columns:
            col_lower = col.lower()
            if any(name.lower() in col_lower for name in possible_names):
                df.rename(columns={col: 'tm'}, inplace=True)
                break
        if 'tm' not in df.columns and 'mp1' in df.columns:
            df.rename(columns={'mp1': 'tm'}, inplace=True)
    
    return df

def load_csv_with_date_parsing(file_path):
    """Load CSV with flexible date parsing"""
    try:
        df = pd.read_csv(file_path, parse_dates=['time'], date_format='%Y-%m-%d %H:%M:%S')
    except:
        try:
            df = pd.read_csv(file_path, parse_dates=['time'])
        except:
            try:
                df = pd.read_csv(file_path)
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'], format='mixed')
            except:
                df = pd.read_csv(file_path)
    return df

def run_gpd_analysis(data, decluster_hours, percentile, data_name):
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

def generate_projection(df_obs, df_proj, scenario, variable):
    """Generate projected time series"""
    if variable not in df_obs.columns:
        st.error(f"Column '{variable}' not found in baseline data")
        return None
    
    if variable not in df_proj.columns:
        st.error(f"Column '{variable}' not found in projection data")
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
            variable: df_year[variable].values[:len(time_index)],
            'scenario': scenario,
            'variable': variable
        })
        
        projected_data.append(year_df)
    
    if projected_data:
        result_df = pd.concat(projected_data, ignore_index=True)
        return result_df
    else:
        return None

# ============================================================================
# Streamlit UI
# ============================================================================

st.title("🌊 Wave Climate Projection Tool")
st.markdown("### CSV Edition - RCP4.5/RCP8.5 Scenarios")
st.markdown("---")

# Initialize session state
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

# ============================================================================
# Sidebar - Data Loading
# ============================================================================

st.sidebar.header("📁 Data Loading")

# Baseline Data
st.sidebar.subheader("Baseline Data (2005)")
baseline_swh_file = st.sidebar.file_uploader("Baseline SWH", type=['csv'], key="baseline_swh")
baseline_tm_file = st.sidebar.file_uploader("Baseline Tm", type=['csv'], key="baseline_tm")

if baseline_swh_file is not None:
    try:
        df = pd.read_csv(baseline_swh_file)
        df = standardize_column_names(df, 'swh')
        if 'swh' in df.columns:
            df['time'] = pd.to_datetime(df['time']) if 'time' in df.columns else pd.date_range(start='2005-01-01', periods=len(df), freq='H')
            st.session_state.baseline_data['swh'] = df
            st.sidebar.success(f"✅ SWH loaded: {len(df)} records")
        else:
            st.sidebar.error("SWH column not found")
    except Exception as e:
        st.sidebar.error(f"Error loading SWH: {str(e)}")

if baseline_tm_file is not None:
    try:
        df = pd.read_csv(baseline_tm_file)
        df = standardize_column_names(df, 'tm')
        if 'tm' in df.columns:
            df['time'] = pd.to_datetime(df['time']) if 'time' in df.columns else pd.date_range(start='2005-01-01', periods=len(df), freq='H')
            st.session_state.baseline_data['tm'] = df
            st.sidebar.success(f"✅ Tm loaded: {len(df)} records")
        else:
            st.sidebar.error("Tm column not found")
    except Exception as e:
        st.sidebar.error(f"Error loading Tm: {str(e)}")

st.sidebar.markdown("---")

# RCP 4.5 Data
st.sidebar.subheader("RCP 4.5 Projections (2041-2100)")
rcp45_swh_file = st.sidebar.file_uploader("RCP 4.5 SWH", type=['csv'], key="rcp45_swh")
rcp45_tm_file = st.sidebar.file_uploader("RCP 4.5 Tm", type=['csv'], key="rcp45_tm")

if rcp45_swh_file is not None:
    try:
        df = pd.read_csv(rcp45_swh_file)
        df = standardize_column_names(df, 'swh')
        if 'swh' in df.columns:
            df['time'] = pd.to_datetime(df['time']) if 'time' in df.columns else pd.date_range(start='2041-01-01', periods=len(df), freq='H')
            st.session_state.projection_data['swh']['rcp45'] = df
            st.sidebar.success(f"✅ RCP 4.5 SWH loaded: {len(df)} records")
        else:
            st.sidebar.error("SWH column not found")
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")

if rcp45_tm_file is not None:
    try:
        df = pd.read_csv(rcp45_tm_file)
        df = standardize_column_names(df, 'tm')
        if 'tm' in df.columns:
            df['time'] = pd.to_datetime(df['time']) if 'time' in df.columns else pd.date_range(start='2041-01-01', periods=len(df), freq='H')
            st.session_state.projection_data['tm']['rcp45'] = df
            st.sidebar.success(f"✅ RCP 4.5 Tm loaded: {len(df)} records")
        else:
            st.sidebar.error("Tm column not found")
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")

st.sidebar.markdown("---")

# RCP 8.5 Data
st.sidebar.subheader("RCP 8.5 Projections (2041-2100)")
rcp85_swh_file = st.sidebar.file_uploader("RCP 8.5 SWH", type=['csv'], key="rcp85_swh")
rcp85_tm_file = st.sidebar.file_uploader("RCP 8.5 Tm", type=['csv'], key="rcp85_tm")

if rcp85_swh_file is not None:
    try:
        df = pd.read_csv(rcp85_swh_file)
        df = standardize_column_names(df, 'swh')
        if 'swh' in df.columns:
            df['time'] = pd.to_datetime(df['time']) if 'time' in df.columns else pd.date_range(start='2041-01-01', periods=len(df), freq='H')
            st.session_state.projection_data['swh']['rcp85'] = df
            st.sidebar.success(f"✅ RCP 8.5 SWH loaded: {len(df)} records")
        else:
            st.sidebar.error("SWH column not found")
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")

if rcp85_tm_file is not None:
    try:
        df = pd.read_csv(rcp85_tm_file)
        df = standardize_column_names(df, 'tm')
        if 'tm' in df.columns:
            df['time'] = pd.to_datetime(df['time']) if 'time' in df.columns else pd.date_range(start='2041-01-01', periods=len(df), freq='H')
            st.session_state.projection_data['tm']['rcp85'] = df
            st.sidebar.success(f"✅ RCP 8.5 Tm loaded: {len(df)} records")
        else:
            st.sidebar.error("Tm column not found")
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")

# ============================================================================
# Main Content Tabs
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Preview", "🎯 GPD Analysis", "🔄 Projections", "📈 Results"])

# ============================================================================
# Tab 1: Data Preview
# ============================================================================

with tab1:
    st.header("Data Preview")
    
    data_type = st.selectbox("Select data to preview", 
                           ["Baseline SWH", "Baseline Tm", "RCP 4.5 SWH", "RCP 4.5 Tm", "RCP 8.5 SWH", "RCP 8.5 Tm"])
    
    data_map = {
        "Baseline SWH": st.session_state.baseline_data['swh'],
        "Baseline Tm": st.session_state.baseline_data['tm'],
        "RCP 4.5 SWH": st.session_state.projection_data['swh']['rcp45'],
        "RCP 4.5 Tm": st.session_state.projection_data['tm']['rcp45'],
        "RCP 8.5 SWH": st.session_state.projection_data['swh']['rcp85'],
        "RCP 8.5 Tm": st.session_state.projection_data['tm']['rcp85']
    }
    
    df = data_map.get(data_type)
    
    if df is not None:
        st.write(f"**Shape:** {df.shape}")
        st.write(f"**Columns:** {list(df.columns)}")
        if 'time' in df.columns:
            st.write(f"**Date range:** {df['time'].min()} to {df['time'].max()}")
        
        variable = 'swh' if 'SWH' in data_type else 'tm'
        if variable in df.columns:
            st.write(f"**{variable.upper()} Statistics:**")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean", f"{df[variable].mean():.4f}")
            col2.metric("Std", f"{df[variable].std():.4f}")
            col3.metric("Min", f"{df[variable].min():.4f}")
            col4.metric("Max", f"{df[variable].max():.4f}")
        
        st.dataframe(df.head(100))
    else:
        st.info("No data loaded. Please upload data files in the sidebar.")

# ============================================================================
# Tab 2: GPD Analysis
# ============================================================================

with tab2:
    st.header("🎯 GPD Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gpd_variable = st.selectbox("Variable", ["Significant Wave Height (SWH)", "Mean Wave Period (Tm)"])
        variable = 'swh' if 'SWH' in gpd_variable else 'tm'
    
    with col2:
        gpd_scenario = st.selectbox("Scenario", ["RCP 4.5", "RCP 8.5"])
        scenario = 'rcp45' if 'RCP 4.5' in gpd_scenario else 'rcp85'
    
    with col3:
        decluster_hours = st.number_input("Decluster Hours", min_value=0, max_value=72, value=12, step=1)
        threshold_percentile = st.number_input("Threshold Percentile", min_value=90.0, max_value=99.9, value=99.5, step=0.1)
    
    if st.button("🎯 Run GPD Analysis", type="primary"):
        df = st.session_state.projection_data[variable][scenario]
        
        if df is None:
            st.error(f"Please load {gpd_scenario} {gpd_variable} data first")
        elif variable not in df.columns:
            st.error(f"Column '{variable}' not found in data")
        else:
            with st.spinner(f"Running GPD analysis for {gpd_scenario} {gpd_variable}..."):
                data = pd.Series(df[variable].values, index=pd.to_datetime(df['time']))
                result = run_gpd_analysis(data, decluster_hours, threshold_percentile, f"{gpd_scenario} {gpd_variable}")
                
                if result is not None:
                    st.session_state.gpd_results[variable][scenario] = result
                    
                    # Display results
                    st.success(f"GPD Analysis Complete!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Threshold", f"{result['threshold']:.4f}")
                    col2.metric("Number of Peaks", result['n_peaks'])
                    col3.metric("ξ (Shape)", f"{result['xi']:.4f}")
                    col4.metric("β (Scale)", f"{result['beta']:.4f}")
                    
                    # Plot results
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    fig.patch.set_facecolor('#f0f0f0')
                    
                    # Time series with peaks
                    ax1.plot(data.index, data.values, color='gray', alpha=0.3, linewidth=0.5, label='Full data')
                    ax1.plot(result['declustered'].index, result['declustered'].values, 'o', color='blue', markersize=4, label='Peaks')
                    ax1.axhline(y=result['threshold'], color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold_percentile}%)')
                    ax1.set_xlabel('Time')
                    ylabel = 'Hs [m]' if variable == 'swh' else 'Tm [s]'
                    ax1.set_ylabel(ylabel)
                    ax1.set_title(f'Declustered Peaks - {gpd_scenario} {gpd_variable}')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # GPD fit
                    exceedances = result['exceedances']
                    if len(exceedances) > 0:
                        ax2.hist(exceedances, bins=30, density=True, alpha=0.5, color='blue', label='Exceedances')
                        x = np.linspace(0, exceedances.max(), 200)
                        pdf = genpareto.pdf(x, result['xi'], result['loc'], result['beta'])
                        ax2.plot(x, pdf, 'r-', lw=2, label=f'GPD fit (ξ={result["xi"]:.2f}, β={result["beta"]:.2f})')
                    ax2.set_xlabel(f'Exceedance ({ylabel})')
                    ax2.set_ylabel('Density')
                    ax2.set_title('GPD Fit to Exceedances')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)

# ============================================================================
# Tab 3: Projections
# ============================================================================

with tab3:
    st.header("🔄 Generate Projections")
    
    col1, col2 = st.columns(2)
    
    with col1:
        proj_variable = st.selectbox("Variable for Projection", ["Significant Wave Height (SWH)", "Mean Wave Period (Tm)"], key="proj_var")
        variable = 'swh' if 'SWH' in proj_variable else 'tm'
    
    with col2:
        proj_scenario = st.selectbox("Scenario", ["RCP 4.5", "RCP 8.5", "Both"], key="proj_scen")
    
    if st.button("🔄 Generate Projections", type="primary"):
        if st.session_state.baseline_data[variable] is None:
            st.error(f"Please load baseline {variable.upper()} data first")
        else:
            scenarios_to_run = []
            if proj_scenario == "Both":
                scenarios_to_run = ['rcp45', 'rcp85']
            elif proj_scenario == "RCP 4.5":
                scenarios_to_run = ['rcp45']
            else:
                scenarios_to_run = ['rcp85']
            
            missing = False
            for sc in scenarios_to_run:
                if st.session_state.projection_data[variable][sc] is None:
                    st.error(f"Please load {sc.upper()} {variable.upper()} data first")
                    missing = True
                    break
            
            if not missing:
                with st.spinner(f"Generating projections for {proj_variable}..."):
                    projection_results = []
                    
                    for sc in scenarios_to_run:
                        proj_df = generate_projection(
                            st.session_state.baseline_data[variable],
                            st.session_state.projection_data[variable][sc],
                            sc,
                            variable
                        )
                        if proj_df is not None:
                            st.session_state.projected_series[variable][sc] = proj_df
                            projection_results.append(proj_df)
                            st.success(f"✅ {sc.upper()} {proj_variable}: {len(proj_df)} records")
                    
                    if projection_results:
                        # Plot combined projections
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                        fig.patch.set_facecolor('#f0f0f0')
                        
                        colors = {'rcp45': 'blue', 'rcp85': 'red'}
                        labels = {'rcp45': 'RCP 4.5', 'rcp85': 'RCP 8.5'}
                        
                        # Yearly averages
                        for df in projection_results:
                            scenario = df['scenario'].iloc[0]
                            df['year'] = pd.to_datetime(df['time']).dt.year
                            yearly_avg = df.groupby('year')[variable].mean().reset_index()
                            ax1.plot(pd.to_datetime(yearly_avg['year'], format='%Y'), yearly_avg[variable],
                                   'o-', color=colors.get(scenario, 'gray'), linewidth=2.5, 
                                   markersize=6, label=labels.get(scenario, scenario))
                        
                        ylabel = 'Hs [m]' if variable == 'swh' else 'Tm [s]'
                        title_var = 'Significant Wave Height' if variable == 'swh' else 'Mean Wave Period'
                        ax1.set_xlabel('Time')
                        ax1.set_ylabel(ylabel)
                        ax1.set_title(f'Projected {title_var} - {proj_scenario}')
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)
                        
                        # Long-term trends
                        for df in projection_results:
                            scenario = df['scenario'].iloc[0]
                            df['year'] = pd.to_datetime(df['time']).dt.year
                            yearly_means = df.groupby('year')[variable].mean()
                            
                            x = yearly_means.index.values
                            y = yearly_means.values
                            slope, intercept, r_value, p_value, std_err = linregress(x, y)
                            trend = intercept + slope * x
                            
                            ax2.plot(x, y, 'o', color=colors.get(scenario, 'gray'), 
                                   markersize=5, alpha=0.7, label=f'{labels.get(scenario, scenario)} - yearly')
                            ax2.plot(x, trend, '--', color=colors.get(scenario, 'gray'), linewidth=2,
                                   label=f'{labels.get(scenario, scenario)} trend (slope={slope:.4f} {ylabel}/year)')
                        
                        ax2.set_xlabel('Year')
                        ax2.set_ylabel(f'Mean {ylabel}')
                        ax2.set_title(f'Long-term Trend Analysis - {title_var}')
                        ax2.legend()
                        ax2.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Combined data download
                        combined_df = pd.concat(projection_results, ignore_index=True)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        csv = combined_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Projections CSV",
                            data=csv,
                            file_name=f"projections_{proj_scenario.replace(' ', '_')}_{timestamp}.csv",
                            mime="text/csv"
                        )

# ============================================================================
# Tab 4: Results
# ============================================================================

with tab4:
    st.header("📈 Results Summary")
    
    # Show data status
    st.subheader("Data Status")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Baseline Data**")
        for var in ['swh', 'tm']:
            status = "✅" if st.session_state.baseline_data[var] is not None else "❌"
            st.write(f"{status} Baseline {var.upper()}")
    
    with col2:
        st.markdown("**Projection Data**")
        for scenario in ['rcp45', 'rcp85']:
            for var in ['swh', 'tm']:
                status = "✅" if st.session_state.projection_data[var][scenario] is not None else "❌"
                st.write(f"{status} {scenario.upper()} {var.upper()}")
    
    st.markdown("---")
    
    # GPD Results Summary
    st.subheader("GPD Analysis Results")
    gpd_data = []
    for var in ['swh', 'tm']:
        for scenario in ['rcp45', 'rcp85']:
            result = st.session_state.gpd_results[var][scenario]
            if result is not None:
                gpd_data.append({
                    'Variable': var.upper(),
                    'Scenario': scenario.upper(),
                    'Threshold': f"{result['threshold']:.4f}",
                    'Peaks': result['n_peaks'],
                    'ξ': f"{result['xi']:.4f}",
                    'β': f"{result['beta']:.4f}"
                })
    
    if gpd_data:
        st.dataframe(pd.DataFrame(gpd_data))
    else:
        st.info("No GPD analysis results yet. Run GPD analysis in the GPD tab.")
    
    st.markdown("---")
    
    # Projected Series Status
    st.subheader("Projected Series")
    for var in ['swh', 'tm']:
        for scenario in ['rcp45', 'rcp85']:
            df = st.session_state.projected_series[var][scenario]
            if df is not None:
                st.write(f"✅ {scenario.upper()} {var.upper()}: {len(df)} records")
                st.write(f"   Mean: {df[var].mean():.4f}, Min: {df[var].min():.4f}, Max: {df[var].max():.4f}")

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown("### 📋 Instructions")
st.markdown("""
1. **Load Data**: Upload CSV files in the sidebar
2. **Analyze**: Run GPD analysis on projection data
3. **Project**: Generate projections for selected scenarios
4. **Export**: Download results as CSV files
""")
st.markdown("---")
st.caption("Wave Climate Projection Tool v1.0 | CSV Edition | RCP4.5/RCP8.5")
