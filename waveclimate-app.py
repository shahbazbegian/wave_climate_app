"""
Wave Climate Projection Tool
Professional GUI Application for Wave Climate Analysis under RCP Scenarios
CSV-based version - Using Plotly for interactive visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import genpareto, linregress
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page config must be the first Streamlit command
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
# Plotting Functions with Plotly
# ============================================================================

def create_gpd_plot(data, result, variable, scenario, threshold_percentile):
    """Create GPD analysis plots using Plotly"""
    ylabel = 'Hs [m]' if variable == 'swh' else 'Tm [s]'
    title_var = 'Significant Wave Height' if variable == 'swh' else 'Mean Wave Period'
    
    # Create subplots
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=(f'Declustered Peaks - {scenario} {title_var}',
                                      'GPD Fit to Exceedances'))
    
    # Plot 1: Time series with peaks
    fig.add_trace(
        go.Scatter(x=data.index, y=data.values,
                   mode='lines',
                   name='Full data',
                   line=dict(color='gray', width=0.5),
                   opacity=0.3),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=result['declustered'].index, y=result['declustered'].values,
                   mode='markers',
                   name='Peaks',
                   marker=dict(color='blue', size=6)),
        row=1, col=1
    )
    
    # Add threshold line
    fig.add_hline(y=result['threshold'], line_dash="dash", line_color="red",
                  annotation_text=f'Threshold ({threshold_percentile}%)',
                  row=1, col=1)
    
    fig.update_xaxes(title_text='Time', row=1, col=1)
    fig.update_yaxes(title_text=ylabel, row=1, col=1)
    
    # Plot 2: GPD fit
    exceedances = result['exceedances']
    if len(exceedances) > 0:
        # Histogram
        fig.add_trace(
            go.Histogram(x=exceedances.values,
                        nbinsx=30,
                        name='Exceedances',
                        opacity=0.5,
                        marker_color='blue',
                        histnorm='probability density'),
            row=1, col=2
        )
        
        # GPD PDF
        x = np.linspace(0, exceedances.max(), 200)
        pdf = genpareto.pdf(x, result['xi'], result['loc'], result['beta'])
        fig.add_trace(
            go.Scatter(x=x, y=pdf,
                       mode='lines',
                       name=f'GPD fit (ξ={result["xi"]:.2f}, β={result["beta"]:.2f})',
                       line=dict(color='red', width=2)),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text=f'Exceedance ({ylabel})', row=1, col=2)
    fig.update_yaxes(title_text='Density', row=1, col=2)
    
    fig.update_layout(height=500, showlegend=True,
                     template='plotly_white',
                     hovermode='x unified')
    
    return fig

def create_projection_plot(projection_results, variable, scenario_label):
    """Create projection plots using Plotly"""
    ylabel = 'Hs [m]' if variable == 'swh' else 'Tm [s]'
    title_var = 'Significant Wave Height' if variable == 'swh' else 'Mean Wave Period'
    
    colors = {'rcp45': 'blue', 'rcp85': 'red'}
    labels = {'rcp45': 'RCP 4.5', 'rcp85': 'RCP 8.5'}
    
    # Create subplots
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=(f'Projected {title_var} - {scenario_label}',
                                      'Long-term Trend Analysis'))
    
    # Plot 1: Yearly averages
    for df in projection_results:
        scenario = df['scenario'].iloc[0]
        df_copy = df.copy()
        df_copy['year'] = pd.to_datetime(df_copy['time']).dt.year
        yearly_avg = df_copy.groupby('year')[variable].mean().reset_index()
        
        fig.add_trace(
            go.Scatter(x=pd.to_datetime(yearly_avg['year'], format='%Y'),
                      y=yearly_avg[variable],
                      mode='lines+markers',
                      name=labels.get(scenario, scenario),
                      line=dict(color=colors.get(scenario, 'gray'), width=2.5),
                      marker=dict(size=8)),
            row=1, col=1
        )
    
    fig.update_xaxes(title_text='Time', row=1, col=1)
    fig.update_yaxes(title_text=ylabel, row=1, col=1)
    
    # Plot 2: Long-term trends
    for df in projection_results:
        scenario = df['scenario'].iloc[0]
        df_copy = df.copy()
        df_copy['year'] = pd.to_datetime(df_copy['time']).dt.year
        yearly_means = df_copy.groupby('year')[variable].mean()
        
        x = yearly_means.index.values
        y = yearly_means.values
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        trend = intercept + slope * x
        
        fig.add_trace(
            go.Scatter(x=x, y=y,
                       mode='markers',
                       name=f'{labels.get(scenario, scenario)} - yearly',
                       marker=dict(color=colors.get(scenario, 'gray'), size=6),
                       opacity=0.7),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=x, y=trend,
                       mode='lines',
                       name=f'{labels.get(scenario, scenario)} trend (slope={slope:.4f} {ylabel}/year)',
                       line=dict(color=colors.get(scenario, 'gray'), width=2, dash='dash')),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text='Year', row=1, col=2)
    fig.update_yaxes(title_text=f'Mean {ylabel}', row=1, col=2)
    
    fig.update_layout(height=500, showlegend=True,
                     template='plotly_white',
                     hovermode='x unified')
    
    return fig

# ============================================================================
# Initialize session state
# ============================================================================

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
# Streamlit UI
# ============================================================================

st.title("🌊 Wave Climate Projection Tool")
st.markdown("### CSV Edition - RCP4.5/RCP8.5 Scenarios")
st.markdown("---")

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
            if 'time' not in df.columns:
                df['time'] = pd.date_range(start='2005-01-01', periods=len(df), freq='H')
            else:
                df['time'] = pd.to_datetime(df['time'])
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
            if 'time' not in df.columns:
                df['time'] = pd.date_range(start='2005-01-01', periods=len(df), freq='H')
            else:
                df['time'] = pd.to_datetime(df['time'])
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
            if 'time' not in df.columns:
                df['time'] = pd.date_range(start='2041-01-01', periods=len(df), freq='H')
            else:
                df['time'] = pd.to_datetime(df['time'])
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
            if 'time' not in df.columns:
                df['time'] = pd.date_range(start='2041-01-01', periods=len(df), freq='H')
            else:
                df['time'] = pd.to_datetime(df['time'])
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
            if 'time' not in df.columns:
                df['time'] = pd.date_range(start='2041-01-01', periods=len(df), freq='H')
            else:
                df['time'] = pd.to_datetime(df['time'])
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
            if 'time' not in df.columns:
                df['time'] = pd.date_range(start='2041-01-01', periods=len(df), freq='H')
            else:
                df['time'] = pd.to_datetime(df['time'])
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
                    st.success("✅ GPD Analysis Complete!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Threshold", f"{result['threshold']:.4f}")
                    col2.metric("Number of Peaks", result['n_peaks'])
                    col3.metric("ξ (Shape)", f"{result['xi']:.4f}")
                    col4.metric("β (Scale)", f"{result['beta']:.4f}")
                    
                    # Create and display plot
                    fig = create_gpd_plot(data, result, variable, gpd_scenario, threshold_percentile)
                    st.plotly_chart(fig, use_container_width=True)

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
                        # Create and display plot
                        fig = create_projection_plot(projection_results, variable, proj_scenario)
                        st.plotly_chart(fig, use_container_width=True)
                        
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
st.caption("Wave Climate Projection Tool v2.0 | Plotly Edition | RCP4.5/RCP8.5")
