# Chemical-concentraion-anomalies
This project demonstrates advanced sensor data processing and anomaly detection capabilities using real-world air quality data. It's designed to identify unusual patterns, sensor malfunctions, or environmental events in continuous monitoring systems.
## Contact & Support

GitHub Issues: Report bugs or request features

Email: bekim_hyseni@hotmail.fr

LinkedIn: https://ch.linkedin.com/in/bekim-hyseni

 Dataset

**Source**: Air Quality Data from UCI Machine Learning Repository

**Sensors**: CO, NO₂, O₃, Temperature, Humidity, Particulate Matter (PM2.5, PM10)

**Time Series**: Continuous environmental monitoring data

**Use Case**: Industrial IoT, Smart Cities, Environmental Monitoring

## Key Features

 Advanced Data Processing

Smart Type Detection: Automatically identifies numeric, date, and categorical columns\
Missing Value Handling: Multiple imputation strategies (median, mean, forward-fill)\
Outlier Management: Statistical outlier detection and removal using IQR method\
Data Smoothing: Rolling average filters for noisy sensor data\
Feature Scaling: StandardScaler and RobustScaler options\
Validation Pipeline: Comprehensive data quality checks\

 Multiple Detection Algorithms

Isolation Forest: Tree-based ensemble method for anomaly detection\
Elliptic Envelope: Robust covariance estimation for outlier detection\
DBSCAN: Density-based clustering for anomaly identification\
Statistical Methods: Z-score based threshold detection

📈 Interactive Visualizations

Time Series Plots: Interactive Plotly charts with anomaly highlighting\
Score Distributions: Histogram analysis of anomaly scores\
Feature Importance: Cohen's d effect size analysis\
Correlation Heatmaps: Feature relationship analysis\
Method Comparisons: Side-by-side algorithm performance

 Professional Dashboard

Real-time Processing: Live data analysis and visualization\
Parameter Tuning: Interactive algorithm parameter adjustment\
Export Functionality: Download results and anomalous data points\
Performance Monitoring: Processing time and memory usage tracking\
Multi-language Support: English interface for international clients

##  Quick Setup
CLONE THE REPOSITORY
```bash
git clone https://github.com/yourusername/environmental-anomaly-detection.git
cd environmental-anomaly-detection
```

CREATE VIRTUAL ENVIRONMENT (RECOMMENDED)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
INSTALL DEPENDECIES
```python
pip install -r requirements.txt
```
RUN THE APPLICATION
```bash
streamlit run app.py
```

The dashboard will open in your browser at http://localhost:8501

##  Dependencies
streamlit>=1.28.0\
pandas>=1.5.0\
numpy>=1.21.0\
plotly>=5.15.0\
scikit-learn>=1.3.0\
scipy>=1.9.0

##  User Guide
1. Load Your Data

Upload CSV files using the sidebar file uploader\
Supported formats: CSV with headers\
Sample data is provided for testing

2. Configure Data Processing

Select missing value handling strategy\
Choose whether to remove outliers\
Enable data smoothing if needed\
Select scaling method for features

3. Run Anomaly Detection

Choose detection algorithm\
Adjust algorithm-specific parameters\
Run detection and analyze results

4. Analyze Results

Overview Tab: Summary statistics and key metrics\
Visualizations Tab: Interactive charts and plots\
Detailed Analysis Tab: Feature importance and correlations\
Comparison Tab: Multi-algorithm performance comparison

## Algorithm Details
**Isolation Forest**

Principle: Isolates anomalies by random feature selection\
Best for: High-dimensional data, mixed normal/anomalous patterns\
Parameters: contamination, n_estimators, max_samples

**Elliptic Envelope**

Principle: Fits robust covariance estimate to data\
Best for: Gaussian-distributed data, statistical outliers\
Parameters: contamination, support_fraction

**DBSCAN**

Principle: Density-based clustering approach\
Best for: Non-spherical anomalies, varying density patterns\
Parameters: eps, min_samples

**Statistical Methods**

Principle: Z-score threshold-based detection\
Best for: Simple baseline, interpretable results\
Parameters: threshold (typically 2-3 standard deviations)

## Troubleshooting

**Common Issues**

*"Cannot use median strategy with non-numeric data"*

**Solution**: The system automatically handles this by separating column types

Ensure your data has proper numeric columns for analysis

*"No valid data remaining after cleaning"*

**Solution**: Check data quality and missing value percentage

Reduce outlier removal sensitivity or change missing value strategy

*Memory Issues with Large Datasets*

**Solution**: Process data in chunks or use sampling
```bash
df_sample = df.sample(n=10000)  # Process subset for initial analysis
```

Visualizations Tab: Interactive charts and plots\
Detailed Analysis Tab: Feature importance and correlations\
Comparison Tab: Multi-algorithm performance comparison

## Acknowledgments

UCI Machine Learning Repository for providing the air quality dataset

Streamlit Community for the excellent framework

Plotly Team for interactive visualization capabilities

Scikit-learn Contributors for machine learning algorithms

