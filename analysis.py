"""
Core data loading and analysis logic for the Wave Climate Projection Tool.
Ported from the original PyQt5 AnalysisWorker class - pure functions, no GUI dependency.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import genpareto, linregress


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
                           'mean_wave_period', 'mean wave period', 'mp1']
        for col in df.columns:
            col_lower = col.lower()
            if any(name.lower() in col_lower for name in possible_names):
                df.rename(columns={col: 'tm'}, inplace=True)
                break
        if 'tm' not in df.columns and 'mp1' in df.columns:
            df.rename(columns={'mp1': 'tm'}, inplace=True)

    return df


def _flexible_read_csv(file_path_or_buffer):
    """Try multiple date parsing strategies, same fallback chain as the original app."""
    try:
        df = pd.read_csv(file_path_or_buffer, parse_dates=['time'], date_format='%Y-%m-%d %H:%M:%S')
        return df
    except Exception:
        pass
    try:
        if hasattr(file_path_or_buffer, 'seek'):
            file_path_or_buffer.seek(0)
        df = pd.read_csv(file_path_or_buffer, parse_dates=['time'])
        return df
    except Exception:
        pass
    try:
        if hasattr(file_path_or_buffer, 'seek'):
            file_path_or_buffer.seek(0)
        df = pd.read_csv(file_path_or_buffer)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], format='mixed')
        return df
    except Exception:
        pass
    if hasattr(file_path_or_buffer, 'seek'):
        file_path_or_buffer.seek(0)
    df = pd.read_csv(file_path_or_buffer)
    return df


def load_baseline_csv(file_obj, variable):
    """Load baseline CSV (file-like or path) with flexible date parsing."""
    df = _flexible_read_csv(file_obj)
    df = standardize_column_names(df, variable)
    warning = None
    if variable not in df.columns:
        warning = f"Could not find {variable.upper()} column. Available columns: {list(df.columns)}"
    return df, warning


def load_projection_csv(file_obj, variable):
    """Load projection CSV (file-like or path) with flexible date parsing."""
    df = _flexible_read_csv(file_obj)
    df = standardize_column_names(df, variable)
    warning = None
    if variable not in df.columns:
        warning = f"Could not find {variable.upper()} column. Available columns: {list(df.columns)}"
    return df, warning


def merge_csv_files(file_objs, variable, progress_callback=None):
    """Merge multiple uploaded CSV files (list of (name, file-like))."""
    all_data = []
    n = len(file_objs)
    for i, (name, f) in enumerate(file_objs):
        try:
            df = pd.read_csv(f, parse_dates=['time'])
        except Exception:
            try:
                f.seek(0)
                df = pd.read_csv(f)
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'], format='mixed')
            except Exception:
                f.seek(0)
                df = pd.read_csv(f)

        df = standardize_column_names(df, variable)
        all_data.append(df)

        if progress_callback:
            progress_callback(int((i + 1) / n * 100))

    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        return merged_df
    return None


def run_gpd_analysis(data, decluster_hours, percentile):
    """Run GPD (Generalized Pareto Distribution) peaks-over-threshold analysis.

    data: pandas Series indexed by datetime.
    """
    threshold = np.percentile(data, percentile)

    if decluster_hours == 0:
        exceedances_raw = data[data > threshold]
        declustered = exceedances_raw
    else:
        exceedances_raw = data[data > threshold]
        declustered = exceedances_raw.resample(f"{decluster_hours}h").max().dropna()

    exceedances = declustered - threshold

    if len(exceedances) > 0:
        xi, loc, beta = genpareto.fit(exceedances)
    else:
        xi, loc, beta = np.nan, np.nan, np.nan

    return {
        'threshold': threshold,
        'declustered': declustered,
        'exceedances': exceedances,
        'xi': xi,
        'beta': beta,
        'loc': loc,
        'n_peaks': len(declustered)
    }


def generate_projection(df_obs, df_proj, scenario, variable, progress_callback=None):
    """Generate a projected time series using slope-based scaling onto each
    projection year, exactly as in the original app."""

    if variable not in df_obs.columns:
        return None, f"Column '{variable}' not found in baseline data"

    if variable not in df_proj.columns:
        return None, f"Column '{variable}' not found in projection data"

    df_proj = df_proj.copy()
    df_proj['time'] = pd.to_datetime(df_proj['time'])
    years = df_proj['time'].dt.year.unique()
    years.sort()

    projected_data = []
    total_years = len(years)

    for i, year in enumerate(years):
        df_year = df_proj[df_proj['time'].dt.year == year].copy()

        if len(df_year) < 2:
            continue

        time_index = pd.date_range(
            start=f"{year}-01-01 00:00:00",
            periods=len(df_year),
            freq='h'
        )[:len(df_year)]

        year_df = pd.DataFrame({
            'time': time_index,
            variable: df_year[variable].values[:len(time_index)],
            'scenario': scenario,
            'variable': variable
        })

        projected_data.append(year_df)
        if progress_callback:
            progress_callback(int((i + 1) / total_years * 100))

    if projected_data:
        result_df = pd.concat(projected_data, ignore_index=True)
        return result_df, None
    return None, f"No projection data generated for {scenario} - {variable}"


def load_test_data(file_obj):
    """Load test data (e.g. Brisbane format) with flexible date parsing and cleanup."""
    df = pd.read_csv(file_obj)

    if 'Date/Time (AEST)' not in df.columns:
        return None, "Could not find 'Date/Time (AEST)' column in the CSV file"

    date_column = df['Date/Time (AEST)']

    try:
        df['time'] = pd.to_datetime(date_column, format='%Y-%m-%dT%H:%M')
    except (ValueError, TypeError):
        try:
            df['time'] = pd.to_datetime(date_column, format='%Y-%m-%dT%H:%M:%S')
        except (ValueError, TypeError):
            try:
                df['time'] = pd.to_datetime(date_column, format='mixed')
            except (ValueError, TypeError):
                try:
                    df['time'] = pd.to_datetime(date_column, infer_datetime_format=True)
                except Exception:
                    df['time'] = pd.to_datetime(date_column)

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

    return df, None
