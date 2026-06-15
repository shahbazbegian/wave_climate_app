Wave Climate Projection Tool

A professional GUI application for wave climate analysis under climate change scenarios (RCP 4.5 and RCP 8.5). This tool enables researchers and coastal engineers to analyze wave height and period data, perform extreme value analysis using Generalized Pareto Distribution (GPD), and generate climate projections.
Features
📊 Data Management

    Structured data loading for Baseline (2005), RCP 4.5, and RCP 8.5 scenarios

    Support for two wave parameters:

        Significant Wave Height (SWH)

        Mean Wave Period (Tm)

    Flexible CSV import with automatic column name detection

    Batch processing - merge multiple CSV files from a folder

    Test data support (e.g., Brisbane 2024 wave data)

🎯 Statistical Analysis

    Generalized Pareto Distribution (GPD) for extreme value analysis

    Configurable declustering window (0-72 hours)

    Adjustable threshold percentile (90-99.9%)

    Automated peak detection and threshold-based exceedance analysis

🔄 Projection Generation

    Climate scenario projections for 2041-2100

    Year-by-year projection generation

    Long-term trend analysis with linear regression

    Visual comparison between RCP 4.5 and RCP 8.5 scenarios

📈 Visualization

    Interactive matplotlib figures with zoom/pan tools

    Time series plots with declustered peaks

    GPD fit visualization

    Yearly average projections

    Long-term trend plots with statistical fits

💾 Export Capabilities

    Export all data (baseline, projections, results) to CSV

    Save figures as high-resolution PNG files

    Comprehensive analysis summary

Installation
Prerequisites

    Python 3.8 or higher

    pip package manager
