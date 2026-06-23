"""
Wave Climate Projection Tool
Professional GUI Application for Wave Climate Analysis under RCP Scenarios
CSV-based version - All data in CSV format
Author: Based on research paper "Wave Climate Projection under Climate Change Scenarios"
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from scipy.stats import genpareto, linregress
from sklearn.linear_model import LinearRegression
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QLabel, QLineEdit, QComboBox,
    QFileDialog, QMessageBox, QGroupBox, QGridLayout, QSpinBox,
    QDoubleSpinBox, QProgressBar, QSplitter, QTextEdit, QCheckBox,
    QScrollArea, QFrame, QSlider, QTableWidget, QTableWidgetItem,
    QHeaderView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor, QPixmap

# ============================================================================
# Style Sheet for Modern UI
# ============================================================================

DARK_STYLE = """
QMainWindow {
    background-color: #2b2b2b;
}
QWidget {
    background-color: #2b2b2b;
    color: #ffffff;
    font-family: 'Segoe UI', Arial, sans-serif;
}
QGroupBox {
    font-weight: bold;
    border: 2px solid #555555;
    border-radius: 8px;
    margin-top: 1ex;
    padding-top: 10px;
    background-color: #333333;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px 0 5px;
    color: #ffffff;
}
QPushButton {
    background-color: #4a4a4a;
    border: 1px solid #555555;
    border-radius: 5px;
    padding: 8px;
    font-weight: bold;
    color: #ffffff;
}
QPushButton:hover {
    background-color: #5a5a5a;
    border-color: #777777;
}
QPushButton:pressed {
    background-color: #3a3a3a;
}
QPushButton#primary {
    background-color: #0066cc;
    border-color: #0052a3;
}
QPushButton#primary:hover {
    background-color: #1a75d2;
}
QPushButton#success {
    background-color: #28a745;
    border-color: #1e7e34;
}
QPushButton#success:hover {
    background-color: #34b750;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QTableWidget {
    background-color: #3a3a3a;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 5px;
    color: #ffffff;
}
QTableWidget::item:selected {
    background-color: #0066cc;
}
QHeaderView::section {
    background-color: #4a4a4a;
    color: #ffffff;
    padding: 5px;
    border: 1px solid #555555;
}
QComboBox::drop-down {
    border: none;
}
QComboBox::down-arrow {
    image: none;
    border-left: 5px solid none;
    border-right: 5px solid none;
    border-top: 5px solid #ffffff;
    width: 0;
    height: 0;
}
QTabWidget::pane {
    border: 2px solid #555555;
    border-radius: 8px;
    background-color: #333333;
}
QTabBar::tab {
    background-color: #3a3a3a;
    border: 1px solid #555555;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 8px 12px;
    margin-right: 2px;
    color: #ffffff;
}
QTabBar::tab:selected {
    background-color: #4a4a4a;
    border-bottom: 2px solid #0066cc;
}
QTabBar::tab:hover {
    background-color: #4a4a4a;
}
QProgressBar {
    border: 2px solid #555555;
    border-radius: 5px;
    text-align: center;
    color: #ffffff;
    background-color: #3a3a3a;
}
QProgressBar::chunk {
    background-color: #0066cc;
    border-radius: 3px;
}
QScrollBar:vertical {
    background-color: #3a3a3a;
    width: 12px;
    border-radius: 6px;
}
QScrollBar::handle:vertical {
    background-color: #666666;
    border-radius: 6px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background-color: #888888;
}
QScrollBar:horizontal {
    background-color: #3a3a3a;
    height: 12px;
    border-radius: 6px;
}
QScrollBar::handle:horizontal {
    background-color: #666666;
    border-radius: 6px;
    min-width: 20px;
}
QScrollBar::handle:horizontal:hover {
    background-color: #888888;
}
QCheckBox {
    spacing: 8px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 3px;
    border: 1px solid #555555;
    background-color: #3a3a3a;
}
QCheckBox::indicator:checked {
    background-color: #0066cc;
    border-color: #0052a3;
}
QSlider::groove:horizontal {
    height: 8px;
    background-color: #3a3a3a;
    border-radius: 4px;
}
QSlider::handle:horizontal {
    background-color: #0066cc;
    width: 18px;
    height: 18px;
    margin: -5px 0;
    border-radius: 9px;
}
QSlider::handle:horizontal:hover {
    background-color: #1a75d2;
}
"""

# ============================================================================
# Worker Thread for Long Operations
# ============================================================================

class AnalysisWorker(QThread):
    """Worker thread for running analysis without freezing UI"""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, analysis_type, params):
        super().__init__()
        self.analysis_type = analysis_type
        self.params = params
        
    def run(self):
        try:
            if self.analysis_type == 'load_baseline_csv':
                result = self.load_baseline_csv()
            elif self.analysis_type == 'load_projection_csv':
                result = self.load_projection_csv()
            elif self.analysis_type == 'run_gpd':
                result = self.run_gpd_analysis()
            elif self.analysis_type == 'generate_projection':
                result = self.generate_projection()
            elif self.analysis_type == 'merge_csv_files':
                result = self.merge_csv_files()
            else:
                result = None
            
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
    
    def standardize_column_names(self, df, data_type):
        """Standardize column names to 'swh' or 'tm'"""
        df = df.copy()
        
        if data_type == 'swh':
            # Look for SWH columns
            possible_names = ['swh', 'SWH', 'hs', 'Hs', 'hs (m)', 'Hs (m)', 'significant_wave_height', 'significant wave height']
            for col in df.columns:
                col_lower = col.lower()
                if any(name.lower() in col_lower for name in possible_names):
                    df.rename(columns={col: 'swh'}, inplace=True)
                    break
            # If still not found, try mp1 (from your code)
            if 'swh' not in df.columns and 'mp1' in df.columns:
                df.rename(columns={'mp1': 'swh'}, inplace=True)
        
        elif data_type == 'tm':
            # Look for Tm columns
            possible_names = ['tm', 'Tm', 'TM', 'tp', 'Tp', 'tp (s)', 'Tp (s)', 'mean_wave_period', 'mean wave period', 'mp1']
            for col in df.columns:
                col_lower = col.lower()
                if any(name.lower() in col_lower for name in possible_names):
                    df.rename(columns={col: 'tm'}, inplace=True)
                    break
            # If still not found, try mp1 (from your code)
            if 'tm' not in df.columns and 'mp1' in df.columns:
                df.rename(columns={'mp1': 'tm'}, inplace=True)
        
        return df
    
    def load_baseline_csv(self):
        """Load baseline CSV file with flexible date parsing"""
        file_path = self.params['file_path']
        data_type = self.params.get('data_type', 'baseline')
        variable = self.params.get('variable', 'swh')
        
        self.status.emit(f"Loading {data_type} data...")
        
        # Try multiple date parsing strategies
        try:
            # First try: with specified format
            df = pd.read_csv(file_path, parse_dates=['time'], date_format='%Y-%m-%d %H:%M:%S')
        except:
            try:
                # Second try: let pandas infer
                df = pd.read_csv(file_path, parse_dates=['time'])
            except:
                try:
                    # Third try: read without parsing, then convert
                    df = pd.read_csv(file_path)
                    if 'time' in df.columns:
                        df['time'] = pd.to_datetime(df['time'], format='mixed')
                except:
                    # Fourth try: just read and let user handle
                    df = pd.read_csv(file_path)
        
        # Standardize column names
        df = self.standardize_column_names(df, variable)
        
        # Check if required column exists
        if variable not in df.columns:
            self.status.emit(f"Warning: Could not find {variable.upper()} column. Available columns: {list(df.columns)}")
        
        self.status.emit(f"Loaded {data_type} data: {len(df)} records")
        return df
    
    def load_projection_csv(self):
        """Load projection CSV file with flexible date parsing"""
        file_path = self.params['file_path']
        data_type = self.params.get('data_type', 'projection')
        variable = self.params.get('variable', 'swh')
        
        self.status.emit(f"Loading {data_type} data...")
        
        # Try multiple date parsing strategies
        try:
            # First try: with specified format
            df = pd.read_csv(file_path, parse_dates=['time'], date_format='%Y-%m-%d %H:%M:%S')
        except:
            try:
                # Second try: let pandas infer
                df = pd.read_csv(file_path, parse_dates=['time'])
            except:
                try:
                    # Third try: read without parsing, then convert
                    df = pd.read_csv(file_path)
                    if 'time' in df.columns:
                        df['time'] = pd.to_datetime(df['time'], format='mixed')
                except:
                    # Fourth try: just read and let user handle
                    df = pd.read_csv(file_path)
        
        # Standardize column names
        df = self.standardize_column_names(df, variable)
        
        # Check if required column exists
        if variable not in df.columns:
            self.status.emit(f"Warning: Could not find {variable.upper()} column. Available columns: {list(df.columns)}")
        
        self.status.emit(f"Loaded {data_type} data: {len(df)} records")
        return df
    
    def merge_csv_files(self):
        """Merge multiple CSV files from a folder"""
        folder_path = self.params['folder_path']
        data_type = self.params.get('data_type', 'merged')
        variable = self.params.get('variable', 'swh')
        
        self.status.emit(f"Merging CSV files for {data_type} from {folder_path}...")
        
        csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
        all_data = []
        
        for i, file in enumerate(csv_files):
            file_path = os.path.join(folder_path, file)
            try:
                # Try with date parsing
                df = pd.read_csv(file_path, parse_dates=['time'])
            except:
                try:
                    # If that fails, read without parsing
                    df = pd.read_csv(file_path)
                    if 'time' in df.columns:
                        df['time'] = pd.to_datetime(df['time'], format='mixed')
                except:
                    df = pd.read_csv(file_path)
            
            # Standardize column names
            df = self.standardize_column_names(df, variable)
            all_data.append(df)
            
            self.progress.emit(int((i + 1) / len(csv_files) * 100))
        
        if all_data:
            merged_df = pd.concat(all_data, ignore_index=True)
            self.status.emit(f"Merged {len(csv_files)} files, total {len(merged_df)} records")
            return merged_df
        else:
            self.status.emit("No data files found")
            return None
    
    def run_gpd_analysis(self):
        """Run GPD analysis"""
        data = self.params['data']
        decluster_hours = self.params['decluster_hours']
        percentile = self.params['percentile']
        data_name = self.params.get('data_name', 'Unknown')
        
        self.status.emit(f"Running GPD analysis for {data_name}...")
        
        threshold = np.percentile(data, percentile)
        
        if decluster_hours == 0:
            exceedances_raw = data[data > threshold]
            declustered = exceedances_raw
        else:
            exceedances_raw = data[data > threshold]
            # Resample to decluster
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
        
        self.status.emit(f"GPD analysis complete. Found {len(declustered)} peaks")
        return result
    
    def generate_projection(self):
        """Generate projected time series using slope-based scaling"""
        df_obs = self.params['df_obs']
        df_proj = self.params['df_proj']
        scenario = self.params.get('scenario', 'rcp45')
        variable = self.params.get('variable', 'swh')
        
        self.status.emit(f"Generating projections for {scenario} - {variable}...")
        
        # Check if required columns exist
        if variable not in df_obs.columns:
            self.status.emit(f"Error: Column '{variable}' not found in baseline data")
            return None
        
        if variable not in df_proj.columns:
            self.status.emit(f"Error: Column '{variable}' not found in projection data")
            return None
        
        # Get projection years
        df_proj['time'] = pd.to_datetime(df_proj['time'])
        years = df_proj['time'].dt.year.unique()
        years.sort()
        
        projected_data = []
        total_years = len(years)
        
        for i, year in enumerate(years):
            df_year = df_proj[df_proj['time'].dt.year == year].copy()
            
            if len(df_year) < 2:
                continue
            
            # Create time index for projected data
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
            self.progress.emit(int((i + 1) / total_years * 100))
        
        if projected_data:
            result_df = pd.concat(projected_data, ignore_index=True)
            self.status.emit(f"Generated {len(result_df)} projected records for {scenario} - {variable}")
            return result_df
        else:
            self.status.emit(f"No projection data generated for {scenario} - {variable}")
            return None


# ============================================================================
# Main Application Window
# ============================================================================

class WaveClimateApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wave Climate Projection Tool v1.0 - CSV Edition")
        self.setGeometry(100, 100, 1400, 900)
        
        # Apply dark theme
        self.setStyleSheet(DARK_STYLE)
        
        # Data storage - Structured by type
        self.baseline_data = {
            'swh': None,  # Baseline SWH data
            'tm': None    # Baseline Tm data
        }
        
        self.projection_data = {
            'swh': {
                'rcp45': None,  # RCP 4.5 SWH projections
                'rcp85': None   # RCP 8.5 SWH projections
            },
            'tm': {
                'rcp45': None,  # RCP 4.5 Tm projections
                'rcp85': None   # RCP 8.5 Tm projections
            }
        }
        
        self.gpd_results = {
            'swh': {'rcp45': None, 'rcp85': None},
            'tm': {'rcp45': None, 'rcp85': None}
        }
        
        self.projected_series = {
            'swh': {'rcp45': None, 'rcp85': None},
            'tm': {'rcp45': None, 'rcp85': None}
        }
        
        self.test_data = None
        
        # For sequential projections
        self.projection_results = []
        self.remaining_scenarios = []
        self.current_variable = None
        
        # File paths storage
        self.file_paths = {
            'baseline_swh': '',
            'baseline_tm': '',
            'rcp45_swh': '',
            'rcp45_tm': '',
            'rcp85_swh': '',
            'rcp85_tm': ''
        }
        
        # Folder paths for merging
        self.folder_paths = {
            'rcp45_swh': '',
            'rcp45_tm': '',
            'rcp85_swh': '',
            'rcp85_tm': ''
        }
        
        # Initialize UI
        self.init_ui()
        
        # Show welcome message
        self.show_welcome()
    
    def init_ui(self):
        """Initialize the user interface"""
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Create tabs
        self.create_data_tab()
        self.create_gpd_tab()
        self.create_projection_tab()
        self.create_results_tab()
        
        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.hide()
        self.status_bar.addPermanentWidget(self.progress_bar)
    
    def create_header(self):
        """Create application header"""
        header = QFrame()
        header.setFrameStyle(QFrame.StyledPanel)
        header.setMaximumHeight(80)
        
        layout = QHBoxLayout(header)
        
        # Logo/title
        title = QLabel("🌊 Wave Climate Projection Tool")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setStyleSheet("color: #0066cc;")
        layout.addWidget(title)
        
        layout.addStretch()
        
        # Version
        version = QLabel("v1.0 | CSV Edition | RCP4.5/RCP8.5")
        version.setFont(QFont("Segoe UI", 10))
        version.setStyleSheet("color: #888888;")
        layout.addWidget(version)
        
        return header
    
    def create_data_tab(self):
        """Create data loading tab with structured data types"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Scroll area for all content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        layout.addWidget(scroll)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll.setWidget(scroll_content)
        
        # ==================== Baseline Data Section ====================
        baseline_group = QGroupBox("📊 Baseline Data (2005)")
        baseline_layout = QGridLayout()
        
        # Baseline SWH
        baseline_layout.addWidget(QLabel("Significant Wave Height (SWH):"), 0, 0)
        self.baseline_swh_path = QLineEdit()
        self.baseline_swh_path.setReadOnly(True)
        baseline_layout.addWidget(self.baseline_swh_path, 0, 1, 1, 3)
        
        self.baseline_swh_browse = QPushButton("Browse CSV...")
        self.baseline_swh_browse.clicked.connect(lambda: self.browse_file('baseline_swh'))
        baseline_layout.addWidget(self.baseline_swh_browse, 0, 4)
        
        self.load_baseline_swh_btn = QPushButton("Load Baseline SWH")
        self.load_baseline_swh_btn.setObjectName("primary")
        self.load_baseline_swh_btn.clicked.connect(lambda: self.load_baseline_data('swh'))
        baseline_layout.addWidget(self.load_baseline_swh_btn, 0, 5)
        
        # Baseline Tm
        baseline_layout.addWidget(QLabel("Mean Wave Period (Tm):"), 1, 0)
        self.baseline_tm_path = QLineEdit()
        self.baseline_tm_path.setReadOnly(True)
        baseline_layout.addWidget(self.baseline_tm_path, 1, 1, 1, 3)
        
        self.baseline_tm_browse = QPushButton("Browse CSV...")
        self.baseline_tm_browse.clicked.connect(lambda: self.browse_file('baseline_tm'))
        baseline_layout.addWidget(self.baseline_tm_browse, 1, 4)
        
        self.load_baseline_tm_btn = QPushButton("Load Baseline Tm")
        self.load_baseline_tm_btn.setObjectName("primary")
        self.load_baseline_tm_btn.clicked.connect(lambda: self.load_baseline_data('tm'))
        baseline_layout.addWidget(self.load_baseline_tm_btn, 1, 5)
        
        baseline_group.setLayout(baseline_layout)
        scroll_layout.addWidget(baseline_group)
        
        # ==================== RCP 4.5 Data Section ====================
        rcp45_group = QGroupBox("🔵 RCP 4.5 Projection Data (2041-2100)")
        rcp45_layout = QGridLayout()
        
        # RCP 4.5 SWH
        rcp45_layout.addWidget(QLabel("Significant Wave Height (SWH):"), 0, 0)
        
        self.rcp45_swh_path = QLineEdit()
        self.rcp45_swh_path.setReadOnly(True)
        rcp45_layout.addWidget(self.rcp45_swh_path, 0, 1, 1, 2)
        
        self.rcp45_swh_browse_file = QPushButton("Browse CSV")
        self.rcp45_swh_browse_file.clicked.connect(lambda: self.browse_file('rcp45_swh'))
        rcp45_layout.addWidget(self.rcp45_swh_browse_file, 0, 3)
        
        self.rcp45_swh_browse_folder = QPushButton("Browse Folder")
        self.rcp45_swh_browse_folder.clicked.connect(lambda: self.browse_folder('rcp45_swh'))
        rcp45_layout.addWidget(self.rcp45_swh_browse_folder, 0, 4)
        
        self.rcp45_swh_merge_btn = QPushButton("Merge CSVs")
        self.rcp45_swh_merge_btn.clicked.connect(lambda: self.merge_projection_files('rcp45_swh', 'swh'))
        rcp45_layout.addWidget(self.rcp45_swh_merge_btn, 0, 5)
        
        self.load_rcp45_swh_btn = QPushButton("Load RCP 4.5 SWH")
        self.load_rcp45_swh_btn.setObjectName("primary")
        self.load_rcp45_swh_btn.clicked.connect(lambda: self.load_projection_data('swh', 'rcp45'))
        rcp45_layout.addWidget(self.load_rcp45_swh_btn, 0, 6)
        
        # RCP 4.5 Tm
        rcp45_layout.addWidget(QLabel("Mean Wave Period (Tm):"), 1, 0)
        
        self.rcp45_tm_path = QLineEdit()
        self.rcp45_tm_path.setReadOnly(True)
        rcp45_layout.addWidget(self.rcp45_tm_path, 1, 1, 1, 2)
        
        self.rcp45_tm_browse_file = QPushButton("Browse CSV")
        self.rcp45_tm_browse_file.clicked.connect(lambda: self.browse_file('rcp45_tm'))
        rcp45_layout.addWidget(self.rcp45_tm_browse_file, 1, 3)
        
        self.rcp45_tm_browse_folder = QPushButton("Browse Folder")
        self.rcp45_tm_browse_folder.clicked.connect(lambda: self.browse_folder('rcp45_tm'))
        rcp45_layout.addWidget(self.rcp45_tm_browse_folder, 1, 4)
        
        self.rcp45_tm_merge_btn = QPushButton("Merge CSVs")
        self.rcp45_tm_merge_btn.clicked.connect(lambda: self.merge_projection_files('rcp45_tm', 'tm'))
        rcp45_layout.addWidget(self.rcp45_tm_merge_btn, 1, 5)
        
        self.load_rcp45_tm_btn = QPushButton("Load RCP 4.5 Tm")
        self.load_rcp45_tm_btn.setObjectName("primary")
        self.load_rcp45_tm_btn.clicked.connect(lambda: self.load_projection_data('tm', 'rcp45'))
        rcp45_layout.addWidget(self.load_rcp45_tm_btn, 1, 6)
        
        rcp45_group.setLayout(rcp45_layout)
        scroll_layout.addWidget(rcp45_group)
        
        # ==================== RCP 8.5 Data Section ====================
        rcp85_group = QGroupBox("🔴 RCP 8.5 Projection Data (2041-2100)")
        rcp85_layout = QGridLayout()
        
        # RCP 8.5 SWH
        rcp85_layout.addWidget(QLabel("Significant Wave Height (SWH):"), 0, 0)
        
        self.rcp85_swh_path = QLineEdit()
        self.rcp85_swh_path.setReadOnly(True)
        rcp85_layout.addWidget(self.rcp85_swh_path, 0, 1, 1, 2)
        
        self.rcp85_swh_browse_file = QPushButton("Browse CSV")
        self.rcp85_swh_browse_file.clicked.connect(lambda: self.browse_file('rcp85_swh'))
        rcp85_layout.addWidget(self.rcp85_swh_browse_file, 0, 3)
        
        self.rcp85_swh_browse_folder = QPushButton("Browse Folder")
        self.rcp85_swh_browse_folder.clicked.connect(lambda: self.browse_folder('rcp85_swh'))
        rcp85_layout.addWidget(self.rcp85_swh_browse_folder, 0, 4)
        
        self.rcp85_swh_merge_btn = QPushButton("Merge CSVs")
        self.rcp85_swh_merge_btn.clicked.connect(lambda: self.merge_projection_files('rcp85_swh', 'swh'))
        rcp85_layout.addWidget(self.rcp85_swh_merge_btn, 0, 5)
        
        self.load_rcp85_swh_btn = QPushButton("Load RCP 8.5 SWH")
        self.load_rcp85_swh_btn.setObjectName("primary")
        self.load_rcp85_swh_btn.clicked.connect(lambda: self.load_projection_data('swh', 'rcp85'))
        rcp85_layout.addWidget(self.load_rcp85_swh_btn, 0, 6)
        
        # RCP 8.5 Tm
        rcp85_layout.addWidget(QLabel("Mean Wave Period (Tm):"), 1, 0)
        
        self.rcp85_tm_path = QLineEdit()
        self.rcp85_tm_path.setReadOnly(True)
        rcp85_layout.addWidget(self.rcp85_tm_path, 1, 1, 1, 2)
        
        self.rcp85_tm_browse_file = QPushButton("Browse CSV")
        self.rcp85_tm_browse_file.clicked.connect(lambda: self.browse_file('rcp85_tm'))
        rcp85_layout.addWidget(self.rcp85_tm_browse_file, 1, 3)
        
        self.rcp85_tm_browse_folder = QPushButton("Browse Folder")
        self.rcp85_tm_browse_folder.clicked.connect(lambda: self.browse_folder('rcp85_tm'))
        rcp85_layout.addWidget(self.rcp85_tm_browse_folder, 1, 4)
        
        self.rcp85_tm_merge_btn = QPushButton("Merge CSVs")
        self.rcp85_tm_merge_btn.clicked.connect(lambda: self.merge_projection_files('rcp85_tm', 'tm'))
        rcp85_layout.addWidget(self.rcp85_tm_merge_btn, 1, 5)
        
        self.load_rcp85_tm_btn = QPushButton("Load RCP 8.5 Tm")
        self.load_rcp85_tm_btn.setObjectName("primary")
        self.load_rcp85_tm_btn.clicked.connect(lambda: self.load_projection_data('tm', 'rcp85'))
        rcp85_layout.addWidget(self.load_rcp85_tm_btn, 1, 6)
        
        rcp85_group.setLayout(rcp85_layout)
        scroll_layout.addWidget(rcp85_group)
        
        # ==================== Test Data Section ====================
        test_group = QGroupBox("🧪 Test Data (e.g., Brisbane 2024)")
        test_layout = QGridLayout()
        
        test_layout.addWidget(QLabel("CSV File:"), 0, 0)
        self.test_path = QLineEdit()
        self.test_path.setReadOnly(True)
        test_layout.addWidget(self.test_path, 0, 1, 1, 4)
        
        self.test_browse = QPushButton("Browse...")
        self.test_browse.clicked.connect(lambda: self.browse_file('test'))
        test_layout.addWidget(self.test_browse, 0, 5)
        
        self.load_test_btn = QPushButton("Load Test Data")
        self.load_test_btn.setObjectName("primary")
        self.load_test_btn.clicked.connect(self.load_test_data)
        test_layout.addWidget(self.load_test_btn, 0, 6)
        
        test_group.setLayout(test_layout)
        scroll_layout.addWidget(test_group)
        
        # ==================== Data Preview ====================
        preview_group = QGroupBox("📋 Data Preview")
        preview_layout = QVBoxLayout()
        
        # Current data info
        self.current_data_info = QLabel("No data loaded")
        self.current_data_info.setStyleSheet("color: #888888; padding: 5px;")
        preview_layout.addWidget(self.current_data_info)
        
        # Table for data preview
        self.preview_table = QTableWidget()
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.horizontalHeader().setStretchLastSection(True)
        self.preview_table.setEditTriggers(QTableWidget.NoEditTriggers)
        preview_layout.addWidget(self.preview_table)
        
        preview_group.setLayout(preview_layout)
        scroll_layout.addWidget(preview_group)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "📁 Data Loading")
    
    def create_gpd_tab(self):
        """Create GPD analysis tab - Fixed SWH plot"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Controls
        controls_group = QGroupBox("⚙️ GPD Analysis Settings")
        controls_layout = QGridLayout()
        
        controls_layout.addWidget(QLabel("Variable:"), 0, 0)
        self.gpd_variable = QComboBox()
        self.gpd_variable.addItems(["Significant Wave Height (SWH)", "Mean Wave Period (Tm)"])
        controls_layout.addWidget(self.gpd_variable, 0, 1)
        
        controls_layout.addWidget(QLabel("Scenario:"), 0, 2)
        self.gpd_scenario = QComboBox()
        self.gpd_scenario.addItems(["RCP 4.5", "RCP 8.5"])
        controls_layout.addWidget(self.gpd_scenario, 0, 3)
        
        controls_layout.addWidget(QLabel("Decluster Hours:"), 1, 0)
        self.decluster_hours = QSpinBox()
        self.decluster_hours.setRange(0, 72)
        self.decluster_hours.setValue(12)
        self.decluster_hours.setSuffix(" h")
        controls_layout.addWidget(self.decluster_hours, 1, 1)
        
        controls_layout.addWidget(QLabel("Threshold Percentile:"), 1, 2)
        self.threshold_percentile = QDoubleSpinBox()
        self.threshold_percentile.setRange(90, 99.9)
        self.threshold_percentile.setValue(99.5)
        self.threshold_percentile.setDecimals(1)
        self.threshold_percentile.setSuffix("%")
        controls_layout.addWidget(self.threshold_percentile, 1, 3)
        
        self.run_gpd_btn = QPushButton("🎯 Run GPD Analysis")
        self.run_gpd_btn.setObjectName("success")
        self.run_gpd_btn.clicked.connect(self.run_gpd_analysis)
        controls_layout.addWidget(self.run_gpd_btn, 2, 0, 1, 4)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Results display with two figures
        splitter = QSplitter(Qt.Vertical)
        
        # Top figure - Time series with peaks (Fixed SWH plot)
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        
        self.gpd_figure1 = Figure(figsize=(10, 3.5), facecolor='#333333')
        self.gpd_canvas1 = FigureCanvas(self.gpd_figure1)
        self.gpd_toolbar1 = NavigationToolbar(self.gpd_canvas1, self)
        top_layout.addWidget(self.gpd_toolbar1)
        top_layout.addWidget(self.gpd_canvas1)
        
        splitter.addWidget(top_widget)
        
        # Bottom figure - GPD fit
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        
        self.gpd_figure2 = Figure(figsize=(10, 3.5), facecolor='#333333')
        self.gpd_canvas2 = FigureCanvas(self.gpd_figure2)
        self.gpd_toolbar2 = NavigationToolbar(self.gpd_canvas2, self)
        bottom_layout.addWidget(self.gpd_toolbar2)
        bottom_layout.addWidget(self.gpd_canvas2)
        
        splitter.addWidget(bottom_widget)
        
        layout.addWidget(splitter)
        
        # Results text
        self.gpd_results_text = QTextEdit()
        self.gpd_results_text.setReadOnly(True)
        self.gpd_results_text.setMaximumHeight(100)
        self.gpd_results_text.setFont(QFont("Courier New", 10))
        layout.addWidget(self.gpd_results_text)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "🎯 GPD Analysis")
    
    def create_projection_tab(self):
        """Create projection generation tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Controls
        controls_group = QGroupBox("⚙️ Generate Projections")
        controls_layout = QGridLayout()
        
        controls_layout.addWidget(QLabel("Variable:"), 0, 0)
        self.proj_variable = QComboBox()
        self.proj_variable.addItems(["Significant Wave Height (SWH)", "Mean Wave Period (Tm)"])
        controls_layout.addWidget(self.proj_variable, 0, 1)
        
        controls_layout.addWidget(QLabel("Scenario:"), 0, 2)
        self.proj_scenario = QComboBox()
        self.proj_scenario.addItems(["RCP 4.5", "RCP 8.5", "Both"])
        controls_layout.addWidget(self.proj_scenario, 0, 3)
        
        self.generate_proj_btn = QPushButton("🔄 Generate Projections")
        self.generate_proj_btn.setObjectName("success")
        self.generate_proj_btn.clicked.connect(self.generate_projections)
        controls_layout.addWidget(self.generate_proj_btn, 1, 0, 1, 4)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Figure for projections
        self.proj_figure = Figure(figsize=(12, 5), facecolor='#333333')
        self.proj_canvas = FigureCanvas(self.proj_figure)
        self.proj_toolbar = NavigationToolbar(self.proj_canvas, self)
        layout.addWidget(self.proj_toolbar)
        layout.addWidget(self.proj_canvas)
        
        # Long-term trend figure
        self.trend_figure = Figure(figsize=(12, 4), facecolor='#333333')
        self.trend_canvas = FigureCanvas(self.trend_figure)
        layout.addWidget(self.trend_canvas)
        
        # Projection info text
        self.proj_info_text = QTextEdit()
        self.proj_info_text.setReadOnly(True)
        self.proj_info_text.setMaximumHeight(100)
        self.proj_info_text.setFont(QFont("Courier New", 10))
        self.proj_info_text.setPlaceholderText("Projection information will appear here...")
        layout.addWidget(self.proj_info_text)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "🔄 Projections")
    
    def create_results_tab(self):
        """Create results summary tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Export controls
        export_group = QGroupBox("💾 Export Results")
        export_layout = QHBoxLayout()
        
        self.export_dir = QLineEdit()
        self.export_dir.setReadOnly(True)
        export_layout.addWidget(self.export_dir)
        
        self.export_browse = QPushButton("Browse...")
        self.export_browse.clicked.connect(self.browse_export_folder)
        export_layout.addWidget(self.export_browse)
        
        self.export_btn = QPushButton("Export All Results")
        self.export_btn.setObjectName("success")
        self.export_btn.clicked.connect(self.export_results)
        export_layout.addWidget(self.export_btn)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        # Results summary
        summary_group = QGroupBox("📊 Analysis Summary")
        summary_layout = QVBoxLayout()
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setFont(QFont("Courier New", 11))
        summary_layout.addWidget(self.summary_text)
        
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "📊 Results")
    
    # ============================================================================
    # Helper Methods
    # ============================================================================
    
    def show_welcome(self):
        """Show welcome message in preview"""
        welcome_text = """
╔══════════════════════════════════════════════════════════════╗
║         WAVE CLIMATE PROJECTION TOOL v1.0                    ║
║         CSV Edition - Structured Data Loading                ║
║         A Unified Framework for Wave Climate Analysis        ║
║         Under RCP4.5 and RCP8.5 Scenarios                    ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║   Data Loading Order:                                        ║
║   1. Baseline Data (2005):                                   ║
║      • Significant Wave Height (SWH)                        ║
║      • Mean Wave Period (Tm)                                 ║
║                                                              ║
║   2. RCP 4.5 Projections (2041-2100):                       ║
║      • SWH - RCP 4.5                                        ║
║      • Tm - RCP 4.5                                         ║
║                                                              ║
║   3. RCP 8.5 Projections (2041-2100):                       ║
║      • SWH - RCP 8.5                                        ║
║      • Tm - RCP 8.5                                         ║
║                                                              ║
║   Ready to begin! Please load your CSV data in order.        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """
        self.current_data_info.setText(welcome_text)
        self.preview_table.setRowCount(0)
        self.preview_table.setColumnCount(0)
    
    def browse_file(self, file_type):
        """Browse for a CSV file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Select CSV file", "",
            "CSV files (*.csv);;All files (*.*)"
        )
        
        if file_path:
            self.file_paths[file_type] = file_path
            
            # Update the corresponding line edit
            if file_type == 'baseline_swh':
                self.baseline_swh_path.setText(file_path)
            elif file_type == 'baseline_tm':
                self.baseline_tm_path.setText(file_path)
            elif file_type == 'rcp45_swh':
                self.rcp45_swh_path.setText(file_path)
            elif file_type == 'rcp45_tm':
                self.rcp45_tm_path.setText(file_path)
            elif file_type == 'rcp85_swh':
                self.rcp85_swh_path.setText(file_path)
            elif file_type == 'rcp85_tm':
                self.rcp85_tm_path.setText(file_path)
            elif file_type == 'test':
                self.test_path.setText(file_path)
    
    def browse_folder(self, folder_type):
        """Browse for a folder containing CSV files"""
        folder_path = QFileDialog.getExistingDirectory(
            self, f"Select folder containing CSV files"
        )
        
        if folder_path:
            self.folder_paths[folder_type] = folder_path
            
            # Update the corresponding line edit
            if folder_type == 'rcp45_swh':
                self.rcp45_swh_path.setText(folder_path)
            elif folder_type == 'rcp45_tm':
                self.rcp45_tm_path.setText(folder_path)
            elif folder_type == 'rcp85_swh':
                self.rcp85_swh_path.setText(folder_path)
            elif folder_type == 'rcp85_tm':
                self.rcp85_tm_path.setText(folder_path)
    
    def browse_export_folder(self):
        """Browse for export folder"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select export folder"
        )
        
        if folder_path:
            self.export_dir.setText(folder_path)
    
    def show_progress(self, show=True):
        """Show or hide progress bar"""
        if show:
            self.progress_bar.show()
        else:
            self.progress_bar.hide()
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def update_status(self, message):
        """Update status bar"""
        self.status_bar.showMessage(message)
    
    def show_error(self, message):
        """Show error message"""
        QMessageBox.critical(self, "Error", message)
    
    def show_success(self, message):
        """Show success message"""
        QMessageBox.information(self, "Success", message)
    
    def update_data_preview(self, df, title):
        """Update data preview table"""
        if df is None or len(df) == 0:
            return
        
        # Show first 100 rows
        display_df = df.head(100)
        
        self.preview_table.setRowCount(len(display_df))
        self.preview_table.setColumnCount(len(display_df.columns))
        self.preview_table.setHorizontalHeaderLabels(display_df.columns)
        
        for i, row in display_df.iterrows():
            for j, col in enumerate(display_df.columns):
                item = QTableWidgetItem(str(row[col]))
                self.preview_table.setItem(i, j, item)
        
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Update info
        info = f"📊 {title}\n"
        info += f"Shape: {df.shape}\n"
        info += f"Date range: {df['time'].min()} to {df['time'].max()}\n"
        
        # Show statistics for the appropriate column
        if 'swh' in df.columns:
            info += f"SWH Statistics:\n"
            info += f"  Mean: {df['swh'].mean():.4f}\n"
            info += f"  Std: {df['swh'].std():.4f}\n"
            info += f"  Min: {df['swh'].min():.4f}\n"
            info += f"  Max: {df['swh'].max():.4f}\n"
        elif 'tm' in df.columns:
            info += f"Tm Statistics:\n"
            info += f"  Mean: {df['tm'].mean():.4f}\n"
            info += f"  Std: {df['tm'].std():.4f}\n"
            info += f"  Min: {df['tm'].min():.4f}\n"
            info += f"  Max: {df['tm'].max():.4f}\n"
        
        info += f"Showing first 100 of {len(df)} rows"
        
        self.current_data_info.setText(info)
    
    # ============================================================================
    # Data Loading Methods
    # ============================================================================
    
    def load_baseline_data(self, variable):
        """Load baseline data for specific variable"""
        if variable == 'swh':
            file_path = self.baseline_swh_path.text()
            data_name = "Baseline SWH"
        else:
            file_path = self.baseline_tm_path.text()
            data_name = "Baseline Tm"
        
        if not file_path:
            self.show_error(f"Please select a {data_name} CSV file first")
            return
        
        self.show_progress(True)
        self.update_status(f"Loading {data_name}...")
        
        self.worker = AnalysisWorker('load_baseline_csv', {
            'file_path': file_path,
            'data_type': data_name,
            'variable': variable
        })
        self.worker.progress.connect(self.update_progress)
        self.worker.status.connect(self.update_status)
        self.worker.finished.connect(lambda df: self.on_baseline_loaded(df, variable))
        self.worker.error.connect(self.on_worker_error)
        self.worker.start()
    
    def on_baseline_loaded(self, df, variable):
        """Handle loaded baseline data"""
        if df is not None:
            self.baseline_data[variable] = df
            data_name = "SWH" if variable == 'swh' else "Tm"
            self.update_data_preview(df, f"Baseline {data_name} (2005)")
            
            self.show_progress(False)
            self.update_status(f"Baseline {data_name} data loaded successfully")
            self.show_success(f"Baseline {data_name} data loaded successfully!")
        else:
            self.show_progress(False)
            self.show_error(f"Failed to load baseline {variable} data")
    
    def load_projection_data(self, variable, scenario):
        """Load projection data for specific variable and scenario"""
        # Get file path from the appropriate line edit
        if variable == 'swh' and scenario == 'rcp45':
            file_path = self.rcp45_swh_path.text()
            data_name = "RCP 4.5 SWH"
        elif variable == 'tm' and scenario == 'rcp45':
            file_path = self.rcp45_tm_path.text()
            data_name = "RCP 4.5 Tm"
        elif variable == 'swh' and scenario == 'rcp85':
            file_path = self.rcp85_swh_path.text()
            data_name = "RCP 8.5 SWH"
        elif variable == 'tm' and scenario == 'rcp85':
            file_path = self.rcp85_tm_path.text()
            data_name = "RCP 8.5 Tm"
        else:
            self.show_error("Invalid data selection")
            return
        
        if not file_path or not os.path.exists(file_path):
            self.show_error(f"Please select a {data_name} CSV file first")
            return
        
        # Check if it's a file or folder
        if os.path.isfile(file_path):
            # It's a file, load directly
            self.show_progress(True)
            self.update_status(f"Loading {data_name}...")
            
            self.worker = AnalysisWorker('load_projection_csv', {
                'file_path': file_path,
                'data_type': data_name,
                'variable': variable
            })
            self.worker.progress.connect(self.update_progress)
            self.worker.status.connect(self.update_status)
            self.worker.finished.connect(lambda df: self.on_projection_loaded(df, variable, scenario))
            self.worker.error.connect(self.on_worker_error)
            self.worker.start()
        else:
            self.show_error(f"Please select a valid CSV file, not a folder. Use Merge button for folders.")
    
    def on_projection_loaded(self, df, variable, scenario):
        """Handle loaded projection data"""
        if df is not None:
            self.projection_data[variable][scenario] = df
            data_name = f"{scenario.upper()} {variable.upper()}"
            self.update_data_preview(df, f"{data_name} Projection (2041-2100)")
            
            self.show_progress(False)
            self.update_status(f"{data_name} data loaded successfully")
            self.show_success(f"{data_name} data loaded successfully!")
        else:
            self.show_progress(False)
            self.show_error(f"Failed to load {scenario} {variable} data")
    
    def merge_projection_files(self, data_key, variable):
        """Merge multiple CSV files for a specific data type"""
        folder_path = self.folder_paths.get(data_key, '')
        
        if not folder_path or not os.path.isdir(folder_path):
            self.show_error(f"Please select a valid folder for {data_key}")
            return
        
        self.show_progress(True)
        self.update_status(f"Merging CSV files for {data_key}...")
        
        self.worker = AnalysisWorker('merge_csv_files', {
            'folder_path': folder_path,
            'data_type': data_key,
            'variable': variable
        })
        self.worker.progress.connect(self.update_progress)
        self.worker.status.connect(self.update_status)
        self.worker.finished.connect(lambda df: self.on_files_merged(df, data_key, variable))
        self.worker.error.connect(self.on_worker_error)
        self.worker.start()
    
    def on_files_merged(self, df, data_key, variable):
        """Handle merged files"""
        if df is not None:
            # Parse data_key to get variable and scenario
            if 'rcp45' in data_key:
                scenario = 'rcp45'
            elif 'rcp85' in data_key:
                scenario = 'rcp85'
            else:
                scenario = 'unknown'
            
            self.projection_data[variable][scenario] = df
            
            # Save merged file
            output_path = os.path.join(os.path.dirname(self.folder_paths[data_key]), 
                                      f"merged_{data_key}_data.csv")
            df.to_csv(output_path, index=False)
            
            # Update the corresponding line edit with the saved file path
            if data_key == 'rcp45_swh':
                self.rcp45_swh_path.setText(output_path)
                self.file_paths['rcp45_swh'] = output_path
            elif data_key == 'rcp45_tm':
                self.rcp45_tm_path.setText(output_path)
                self.file_paths['rcp45_tm'] = output_path
            elif data_key == 'rcp85_swh':
                self.rcp85_swh_path.setText(output_path)
                self.file_paths['rcp85_swh'] = output_path
            elif data_key == 'rcp85_tm':
                self.rcp85_tm_path.setText(output_path)
                self.file_paths['rcp85_tm'] = output_path
            
            self.update_data_preview(df, f"Merged {data_key.upper()} Data")
            
            self.show_progress(False)
            self.update_status(f"Files merged successfully for {data_key}")
            self.show_success(f"Merged {len(df)} records saved to:\n{output_path}")
        else:
            self.show_progress(False)
            self.show_error(f"Failed to merge files for {data_key}")
    
    def load_test_data(self):
        """Load test data (like Brisbane example) with flexible date parsing"""
        file_path = self.test_path.text()
        if not file_path:
            self.show_error("Please select a test CSV file first")
            return
        
        try:
            # Read CSV without parsing dates initially
            df = pd.read_csv(file_path)
            
            # Handle Brisbane format with ISO8601 dates
            if 'Date/Time (AEST)' in df.columns:
                # Try multiple date parsing strategies
                date_column = df['Date/Time (AEST)']
                
                # Method 1: Try ISO8601 format with T separator
                try:
                    df['time'] = pd.to_datetime(date_column, format='%Y-%m-%dT%H:%M')
                except (ValueError, TypeError):
                    try:
                        # Method 2: Try ISO8601 with seconds
                        df['time'] = pd.to_datetime(date_column, format='%Y-%m-%dT%H:%M:%S')
                    except (ValueError, TypeError):
                        try:
                            # Method 3: Let pandas infer with mixed format
                            df['time'] = pd.to_datetime(date_column, format='mixed')
                        except (ValueError, TypeError):
                            try:
                                # Method 4: Use infer_datetime_format
                                df['time'] = pd.to_datetime(date_column, infer_datetime_format=True)
                            except:
                                # Method 5: Last resort - use to_datetime with default settings
                                df['time'] = pd.to_datetime(date_column)
                
                # Clean data - replace -99.90 and other invalid values with NaN
                df.replace(-99.90, np.nan, inplace=True)
                df.replace(-99.9, np.nan, inplace=True)
                df.replace(-99, np.nan, inplace=True)
                
                # Rename columns
                if 'Hs (m)' in df.columns:
                    df.rename(columns={'Hs (m)': 'swh'}, inplace=True)
                elif 'Hs' in df.columns:
                    df.rename(columns={'Hs': 'swh'}, inplace=True)
                
                # Keep Tp for future use if available
                if 'Tp (s)' in df.columns:
                    df.rename(columns={'Tp (s)': 'tm'}, inplace=True)
                
                # Drop unwanted columns
                cols_to_drop = []
                for col in ['Hmax (m)', 'Peak Direction (degrees)', 'SST (degrees C)', 
                           'Tz (s)', 'Date/Time (AEST)']:
                    if col in df.columns:
                        cols_to_drop.append(col)
                
                if cols_to_drop:
                    df.drop(cols_to_drop, axis=1, inplace=True)
                
                # Remove rows with NaN values
                initial_len = len(df)
                df.dropna(inplace=True)
                
                # Sort by time
                df.sort_values('time', inplace=True)
                df.reset_index(drop=True, inplace=True)
                
                self.test_data = df
                self.update_data_preview(df, "Test Data (Brisbane 2024)")
                
                self.update_status(f"Test data loaded successfully: {len(df)} records")
                self.show_success(f"Test data loaded successfully!\n{len(df)} records from {df['time'].min()} to {df['time'].max()}")
                
            else:
                self.show_error("Could not find 'Date/Time (AEST)' column in the CSV file")
            
        except Exception as e:
            self.show_error(f"Error loading test data: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def on_worker_error(self, error_message):
        """Handle worker error"""
        self.show_progress(False)
        self.show_error(f"Error: {error_message}")
        self.update_status("Error occurred")
    
    # ============================================================================
    # Analysis Methods
    # ============================================================================
    
    def run_gpd_analysis(self):
        """Run GPD analysis - Fixed SWH plot"""
        variable_idx = self.gpd_variable.currentIndex()
        variable = 'swh' if variable_idx == 0 else 'tm'
        
        scenario_idx = self.gpd_scenario.currentIndex()
        scenario = 'rcp45' if scenario_idx == 0 else 'rcp85'
        
        # Get appropriate dataframe
        df = self.projection_data[variable][scenario]
        
        if df is None:
            self.show_error(f"Please load {scenario.upper()} {variable.upper()} data first")
            return
        
        if variable not in df.columns:
            self.show_error(f"Column '{variable}' not found in the data. Available columns: {list(df.columns)}")
            return
        
        data_name = f"{scenario.upper()} {variable.upper()}"
        
        # Prepare time series
        data = pd.Series(
            df[variable].values,
            index=pd.to_datetime(df['time'])
        )
        
        self.show_progress(True)
        self.update_status(f"Running GPD analysis for {data_name}...")
        
        self.worker = AnalysisWorker('run_gpd', {
            'data': data,
            'decluster_hours': self.decluster_hours.value(),
            'percentile': self.threshold_percentile.value(),
            'data_name': data_name
        })
        self.worker.progress.connect(self.update_progress)
        self.worker.status.connect(self.update_status)
        self.worker.finished.connect(lambda result: self.on_gpd_complete(result, variable, scenario))
        self.worker.error.connect(self.on_worker_error)
        self.worker.start()
    
    def on_gpd_complete(self, result, variable, scenario):
        """Handle GPD analysis complete"""
        if result is not None:
            data_name = f"{scenario.upper()} {variable.upper()}"
            self.gpd_results[variable][scenario] = result
            
            # Display results
            ylabel = 'Hs [m]' if variable == 'swh' else 'Tm [s]'
            text = "🎯 GPD ANALYSIS RESULTS\n"
            text += "=" * 60 + "\n\n"
            text += f"Dataset: {data_name}\n"
            text += f"Threshold: {result['threshold']:.4f} {ylabel} ({self.threshold_percentile.value()}%)\n"
            text += f"Decluster window: {self.decluster_hours.value()} hours\n"
            text += f"Number of peaks: {result['n_peaks']}\n"
            text += f"GPD Parameters:\n"
            text += f"  ξ (shape): {result['xi']:.4f}\n"
            text += f"  β (scale): {result['beta']:.4f}\n"
            
            self.gpd_results_text.setText(text)
            
            # Plot results
            self.plot_gpd_results(result, data_name, variable)
            
            self.show_progress(False)
            self.update_status(f"GPD analysis complete for {data_name}")
            self.show_success(f"GPD analysis complete for {data_name}!")
        else:
            self.show_progress(False)
            self.show_error(f"Failed to run GPD analysis for {scenario} {variable}")
    
    def plot_gpd_results(self, result, data_name, variable):
        """Plot GPD analysis results - Fixed SWH plot"""
        # First plot - Time series with peaks (Fixed)
        self.gpd_figure1.clear()
        ax1 = self.gpd_figure1.add_subplot(111)
        
        ax1.set_facecolor('#333333')
        self.gpd_figure1.patch.set_facecolor('#333333')
        ax1.tick_params(colors='white')
        ax1.xaxis.label.set_color('white')
        ax1.yaxis.label.set_color('white')
        ax1.title.set_color('white')
        for spine in ax1.spines.values():
            spine.set_color('#666666')
        
        # Get the full data from the original source (not just peaks)
        variable_idx = self.gpd_variable.currentIndex()
        var = 'swh' if variable_idx == 0 else 'tm'
        scenario_idx = self.gpd_scenario.currentIndex()
        scen = 'rcp45' if scenario_idx == 0 else 'rcp85'
        
        full_data = self.projection_data[var][scen]
        if full_data is not None:
            # Plot the full time series
            time_series = pd.to_datetime(full_data['time'])
            ax1.plot(time_series, full_data[var], color='#888888', alpha=0.3, linewidth=0.5, label='Full data')
        
        # Plot peaks
        data = result['declustered']
        ylabel = 'Hs [m]' if variable == 'swh' else 'Tm [s]'
        ax1.plot(data.index, data.values, 'o', color='#00ccff', 
                markersize=4, alpha=0.8, label='Peaks')
        ax1.axhline(y=result['threshold'], color='#ff6b6b', 
                   linestyle='--', linewidth=2, label=f'Threshold ({self.threshold_percentile.value()}%)')
        
        # Add text box with peak count
        textstr = f'Peaks: {result["n_peaks"]}'
        props = dict(boxstyle='round', facecolor='#444444', alpha=0.8)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, color='white')
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel(ylabel)
        ax1.set_title(f'Declustered Peaks - {data_name}')
        ax1.legend(facecolor='#444444', edgecolor='white', labelcolor='white', fontsize=8)
        ax1.grid(True, alpha=0.3, color='#666666')
        
        self.gpd_canvas1.draw()
        
        # Second plot - GPD fit
        self.gpd_figure2.clear()
        ax2 = self.gpd_figure2.add_subplot(111)
        
        ax2.set_facecolor('#333333')
        self.gpd_figure2.patch.set_facecolor('#333333')
        ax2.tick_params(colors='white')
        ax2.xaxis.label.set_color('white')
        ax2.yaxis.label.set_color('white')
        ax2.title.set_color('white')
        for spine in ax2.spines.values():
            spine.set_color('#666666')
        
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
        
        self.gpd_canvas2.draw()
    
    def generate_projections(self):
        """Generate projected time series for selected scenarios"""
        variable_idx = self.proj_variable.currentIndex()
        variable = 'swh' if variable_idx == 0 else 'tm'
        
        scenario_text = self.proj_scenario.currentText()
        
        if self.baseline_data[variable] is None:
            self.show_error(f"Please load baseline {variable.upper()} data first")
            return
        
        # Check if baseline data has the required column
        if variable not in self.baseline_data[variable].columns:
            self.show_error(f"Column '{variable}' not found in baseline data. Available columns: {list(self.baseline_data[variable].columns)}")
            return
        
        # Determine scenarios to run
        scenarios_to_run = []
        if scenario_text == "Both":
            scenarios_to_run = ['rcp45', 'rcp85']
        elif scenario_text == "RCP 4.5":
            scenarios_to_run = ['rcp45']
        else:  # RCP 8.5
            scenarios_to_run = ['rcp85']
        
        # Check if all required data is loaded and has correct columns
        for sc in scenarios_to_run:
            if self.projection_data[variable][sc] is None:
                self.show_error(f"Please load {sc.upper()} {variable.upper()} data first")
                return
            if variable not in self.projection_data[variable][sc].columns:
                self.show_error(f"Column '{variable}' not found in {sc.upper()} {variable.upper()} data. Available columns: {list(self.projection_data[variable][sc].columns)}")
                return
        
        if not scenarios_to_run:
            return
        
        self.show_progress(True)
        self.update_status(f"Generating projections for {variable.upper()}...")
        
        # Clear previous results
        self.projection_results = []
        self.projected_series[variable] = {}
        self.remaining_scenarios = scenarios_to_run.copy()
        self.current_variable = variable
        
        # Clear projection info
        self.proj_info_text.clear()
        
        # Start first projection
        self.run_projection_sequential()
    
    def run_projection_sequential(self):
        """Run projections one after another"""
        if not self.remaining_scenarios:
            # All projections completed
            self.on_all_projections_complete()
            return
        
        scenario = self.remaining_scenarios.pop(0)
        variable = self.current_variable
        
        self.update_status(f"Generating projections for {scenario.upper()} {variable.upper()}...")
        
        # Create worker for this scenario
        self.proj_worker = AnalysisWorker('generate_projection', {
            'df_obs': self.baseline_data[variable],
            'df_proj': self.projection_data[variable][scenario],
            'scenario': scenario,
            'variable': variable
        })
        
        # Connect signals
        self.proj_worker.finished.connect(
            lambda df, s=scenario, v=variable: self.on_projection_generated(df, v, s)
        )
        self.proj_worker.progress.connect(self.update_progress)
        self.proj_worker.status.connect(self.update_status)
        self.proj_worker.error.connect(self.on_worker_error)
        
        self.proj_worker.start()
    
    def on_projection_generated(self, proj_df, variable, scenario):
        """Handle individual projection generation"""
        # Disconnect signals to avoid multiple connections
        if hasattr(self, 'proj_worker'):
            try:
                self.proj_worker.finished.disconnect()
            except:
                pass
        
        if proj_df is not None:
            # Store the result
            self.projected_series[variable][scenario] = proj_df
            self.projection_results.append(proj_df)
            
            # Update info text
            info = f"✅ Generated {scenario.upper()} {variable.upper()}: {len(proj_df)} records\n"
            info += f"   Range: [{proj_df[variable].min():.4f}, {proj_df[variable].max():.4f}]\n"
            info += f"   Mean: {proj_df[variable].mean():.4f}, Std: {proj_df[variable].std():.4f}\n"
            self.proj_info_text.append(info)
            
            self.update_status(f"Completed {scenario.upper()} {variable.upper()} - {len(proj_df)} records")
        else:
            self.update_status(f"Failed to generate projections for {scenario.upper()} {variable.upper()}")
        
        # Run next projection
        self.run_projection_sequential()
    
    def on_all_projections_complete(self):
        """Handle completion of all projections"""
        if not self.projection_results:
            self.show_progress(False)
            self.show_error("No projections were generated successfully")
            return
        
        variable = self.current_variable
        variable_name = "SWH" if variable == 'swh' else "Tm"
        scenario_text = self.proj_scenario.currentText()
        
        # Plot all projections together
        self.plot_combined_projections(self.projection_results, variable_name, variable, scenario_text)
        
        # Save combined projections
        combined_df = pd.concat(self.projection_results, ignore_index=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"projection_{variable_name}_{scenario_text.replace(' ', '_')}_{timestamp}.csv"
        combined_df.to_csv(output_path, index=False)
        
        self.proj_info_text.append(f"\n✅ All projections completed and saved to:\n{output_path}")
        
        self.show_progress(False)
        self.update_status(f"Projections generated successfully for {variable_name}")
        self.show_success(f"Projections generated and saved to:\n{output_path}")
        
        # Clean up
        self.remaining_scenarios = []
        self.current_variable = None
    
    def plot_combined_projections(self, projection_dfs, variable_name, variable, scenario_text):
        """Plot combined projections with RCP 4.5 in blue and RCP 8.5 in red"""
        self.proj_figure.clear()
        
        # Create single plot for yearly averages
        ax = self.proj_figure.add_subplot(111)
        
        ax.set_facecolor('#333333')
        self.proj_figure.patch.set_facecolor('#333333')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#666666')
        
        colors = {'rcp45': '#0066cc', 'rcp85': '#ff4444'}
        labels = {'rcp45': 'RCP 4.5', 'rcp85': 'RCP 8.5'}
        
        for df in projection_dfs:
            if 'scenario' in df.columns:
                scenario = df['scenario'].iloc[0]
            else:
                continue
            
            # Calculate yearly averages
            df['year'] = pd.to_datetime(df['time']).dt.year
            yearly_avg = df.groupby('year')[variable].mean().reset_index()
            
            # Plot with appropriate color
            ax.plot(pd.to_datetime(yearly_avg['year'], format='%Y'), yearly_avg[variable],
                   'o-', color=colors.get(scenario, '#888888'), linewidth=2.5, 
                   markersize=6, label=labels.get(scenario, scenario))
        
        ylabel = 'Hs [m]' if variable_name == 'SWH' else 'Tm [s]'
        title_var = 'Significant Wave Height' if variable_name == 'SWH' else 'Mean Wave Period'
        
        ax.set_xlabel('Time')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Projected {title_var} - {scenario_text}')
        ax.legend(facecolor='#444444', edgecolor='white', labelcolor='white')
        ax.grid(True, alpha=0.3, color='#666666')
        
        self.proj_canvas.draw()
        
        # Plot long-term trends for all scenarios
        self.plot_combined_trends(projection_dfs, variable_name, variable, scenario_text)
    
    def plot_combined_trends(self, projection_dfs, variable_name, variable, scenario_text):
        """Plot long-term trends for all scenarios"""
        self.trend_figure.clear()
        ax = self.trend_figure.add_subplot(111)
        
        ax.set_facecolor('#333333')
        self.trend_figure.patch.set_facecolor('#333333')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#666666')
        
        colors = {'rcp45': '#0066cc', 'rcp85': '#ff4444'}
        labels = {'rcp45': 'RCP 4.5', 'rcp85': 'RCP 8.5'}
        
        for df in projection_dfs:
            if 'scenario' in df.columns:
                scenario = df['scenario'].iloc[0]
            else:
                continue
            
            # Calculate yearly means
            df['year'] = pd.to_datetime(df['time']).dt.year
            yearly_means = df.groupby('year')[variable].mean()
            
            # Linear regression
            x = yearly_means.index.values
            y = yearly_means.values
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            trend = intercept + slope * x
            
            # Plot yearly means and trend
            ax.plot(x, y, 'o', color=colors.get(scenario, '#888888'), 
                   markersize=5, alpha=0.7, label=f'{labels.get(scenario, scenario)} - yearly')
            ax.plot(x, trend, '--', color=colors.get(scenario, '#888888'), linewidth=2,
                   label=f'{labels.get(scenario, scenario)} trend (slope={slope:.4f} m/year)')
        
        ylabel = 'Hs [m]' if variable_name == 'SWH' else 'Tm [s]'
        title_var = 'Significant Wave Height' if variable_name == 'SWH' else 'Mean Wave Period'
        
        ax.set_xlabel('Year')
        ax.set_ylabel(f'Mean {ylabel}')
        ax.set_title(f'Long-term Trend Analysis - {title_var} - {scenario_text}')
        ax.legend(facecolor='#444444', edgecolor='white', labelcolor='white', fontsize=9)
        ax.grid(True, alpha=0.3, color='#666666')
        
        self.trend_canvas.draw()
    
    def export_results(self):
        """Export all results to files"""
        export_dir = self.export_dir.text()
        if not export_dir:
            self.show_error("Please select an export folder first")
            return
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            exported_files = []
            
            # Export baseline data
            for var, df in self.baseline_data.items():
                if df is not None:
                    path = os.path.join(export_dir, f"baseline_{var}_{timestamp}.csv")
                    df.to_csv(path, index=False)
                    exported_files.append(path)
            
            # Export projection data
            for var in ['swh', 'tm']:
                for scenario in ['rcp45', 'rcp85']:
                    df = self.projection_data[var][scenario]
                    if df is not None:
                        path = os.path.join(export_dir, f"projection_{var}_{scenario}_{timestamp}.csv")
                        df.to_csv(path, index=False)
                        exported_files.append(path)
            
            # Export test data
            if self.test_data is not None:
                path = os.path.join(export_dir, f"test_data_{timestamp}.csv")
                self.test_data.to_csv(path, index=False)
                exported_files.append(path)
            
            # Export GPD results
            for var in ['swh', 'tm']:
                for scenario in ['rcp45', 'rcp85']:
                    result = self.gpd_results[var][scenario]
                    if result is not None:
                        gpd_df = pd.DataFrame([{
                            'variable': var,
                            'scenario': scenario,
                            'threshold': result.get('threshold', np.nan),
                            'xi': result.get('xi', np.nan),
                            'beta': result.get('beta', np.nan),
                            'n_peaks': result.get('n_peaks', 0)
                        }])
                        path = os.path.join(export_dir, f"gpd_{var}_{scenario}_{timestamp}.csv")
                        gpd_df.to_csv(path, index=False)
                        exported_files.append(path)
            
            # Export projected series
            for var in ['swh', 'tm']:
                for scenario in ['rcp45', 'rcp85']:
                    df = self.projected_series[var][scenario]
                    if df is not None:
                        path = os.path.join(export_dir, f"projected_{var}_{scenario}_{timestamp}.csv")
                        df.to_csv(path, index=False)
                        exported_files.append(path)
            
            # Save figures
            figures = [
                (self.gpd_figure1, "gpd_timeseries"),
                (self.gpd_figure2, "gpd_fit"),
                (self.proj_figure, "projections"),
                (self.trend_figure, "long_term_trend")
            ]
            
            for fig, name in figures:
                if fig.axes:
                    path = os.path.join(export_dir, f"{name}_{timestamp}.png")
                    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='#333333')
                    exported_files.append(path)
            
            # Update summary
            summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                    EXPORT COMPLETE                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Export Directory: {export_dir}
║  Timestamp: {timestamp}
║  Files Exported: {len(exported_files)}
║                                                              ║
║  Exported Files:                                            ║"""
            
            for f in exported_files[:10]:  # Show first 10
                summary += f"\n  • {os.path.basename(f)}"
            
            if len(exported_files) > 10:
                summary += f"\n  • ... and {len(exported_files) - 10} more"
            
            summary += "\n║                                                              \n"
            summary += "╚══════════════════════════════════════════════════════════════╝"
            
            self.summary_text.setText(summary)
            
            self.show_success(f"Results exported successfully to {export_dir}")
            
        except Exception as e:
            self.show_error(f"Error exporting results: {str(e)}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = WaveClimateApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()