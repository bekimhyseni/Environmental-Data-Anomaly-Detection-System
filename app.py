"""
On the anaconda prompt, run the analysis with:
cd "C:\the\path\where\the\scripts\are"
streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_processor import DataProcessor
from anomaly_detector import AnomalyDetector

# Page configuration
st.set_page_config(
    page_title="Environmental Anomaly Detection",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        border-radius: 10px;
        background: linear-gradient(90deg, #f0f8f0, #e8f5e8);
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    .stAlert > div {
        border-radius: 8px;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa, #e9ecef);
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'processor' not in st.session_state:
        st.session_state.processor = DataProcessor()
    if 'detector' not in st.session_state:
        st.session_state.detector = AnomalyDetector()

def main():
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🌍 Environmental Data Anomaly Detection System</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x120/2E8B57/FFFFFF?text=Environmental+Analytics", 
                caption="Environmental Data Analytics")
        
        st.markdown("## 🚀 Project Overview")
        st.markdown("""
        **Objective**: Demonstrate sensor data processing and anomaly detection capabilities
        
        **Dataset**: Air Quality Data from UCI Repository
        - CO, NO₂, O₃ sensors
        - Temperature & Humidity
        - Particulate Matter (PM2.5, PM10)
        
        **Methods**:
        - Statistical Analysis
        - Machine Learning Detection
        - Interactive Visualizations
        """)
        
        st.markdown("## 📊 Data Processing")
        
        # Data loading section
        if st.button("🔄 Load Air Quality Dataset", type="primary"):
            load_dataset()
        
        if st.session_state.data_loaded:
            st.success("✅ Dataset loaded successfully!")
            
            # Processing options
            st.markdown("### ⚙️ Processing Options")
            
            missing_strategy = st.selectbox(
                "Missing Data Strategy:",
                ["median", "mean", "forward_fill"],
                help="Strategy for handling missing values in numeric data"
            )
            
            outlier_removal = st.checkbox(
                "Remove Statistical Outliers",
                value=True,
                help="Remove data points beyond 3 standard deviations"
            )
            
            apply_smoothing = st.checkbox(
                "Apply Data Smoothing",
                value=False,
                help="Apply rolling average to reduce noise"
            )
            
            if st.button("🔧 Process Data"):
                process_data(missing_strategy, outlier_removal, apply_smoothing)
    
    # Main content
    if not st.session_state.data_loaded:
        show_welcome_page()
    else:
        show_main_dashboard()

def load_dataset():
    """Load the Air Quality dataset"""
    try:
        with st.spinner('Loading Air Quality dataset from UCI repository...'):
            # Try to load from ucimlrepo first
            try:
                from ucimlrepo import fetch_ucirepo
                air_quality = fetch_ucirepo(id=360)
                df = air_quality.data.features
                st.session_state.raw_data = df
                st.session_state.dataset_info = {
                    'source': 'UCI ML Repository',
                    'name': 'Air Quality',
                    'id': 360
                }
            except:
                # Fallback: create sample air quality data
                df = create_sample_air_quality_data()
                st.session_state.raw_data = df
                st.session_state.dataset_info = {
                    'source': 'Generated Sample',
                    'name': 'Air Quality Sample',
                    'id': 'sample'
                }
            
            st.session_state.data_loaded = True
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")

def create_sample_air_quality_data():
    """Create sample air quality data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate date range
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='H')
    
    # Generate correlated environmental data
    base_temp = 20 + 10 * np.sin(np.arange(n_samples) * 2 * np.pi / (24 * 30)) + np.random.normal(0, 2, n_samples)
    humidity = 60 + 20 * np.sin(np.arange(n_samples) * 2 * np.pi / (24 * 7)) + np.random.normal(0, 5, n_samples)
    
    # Air quality parameters
    co = 2.5 + 1.5 * np.random.exponential(1, n_samples) + 0.1 * base_temp
    no2 = 40 + 30 * np.random.gamma(2, 1, n_samples) + 0.5 * humidity
    o3 = 80 + 40 * np.random.beta(2, 3, n_samples)
    pm25 = 15 + 25 * np.random.lognormal(0, 0.5, n_samples)
    pm10 = pm25 * (1.5 + 0.3 * np.random.normal(0, 0.1, n_samples))
    
    # Add some anomalies
    anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    co[anomaly_indices] *= np.random.uniform(3, 8, len(anomaly_indices))
    no2[anomaly_indices] *= np.random.uniform(2, 5, len(anomaly_indices))
    
    df = pd.DataFrame({
        'DateTime': dates,
        'CO_sensor': co,
        'NO2_sensor': no2,
        'O3_sensor': o3,
        'Temperature': base_temp,
        'Humidity': np.clip(humidity, 0, 100),
        'PM2_5': pm25,
        'PM10': pm10,
        'Pressure': 1013.25 + np.random.normal(0, 10, n_samples)
    })
    
    return df

def process_data(missing_strategy, outlier_removal, apply_smoothing):
    """Process the loaded data"""
    try:
        with st.spinner('Processing data...'):
            # Process data using DataProcessor
            processed_data, processing_summary = st.session_state.processor.process_data(
                st.session_state.raw_data,
                missing_strategy=missing_strategy,
                remove_outliers=outlier_removal,
                apply_smoothing=apply_smoothing
            )
            
            st.session_state.processed_data = processed_data
            st.session_state.processing_summary = processing_summary
            st.session_state.data_processed = True
            
            st.success("✅ Data processed successfully!")
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")

def show_welcome_page():
    """Show welcome page when no data is loaded"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ## 🎯 Welcome to Environmental Anomaly Detection
        
        This advanced dashboard analyzes environmental sensor data to detect anomalies and unusual patterns.
        
        ### 🔧 Key Features:
        
        🔄 **Smart Data Processing**
        - Automatic data type detection
        - Multiple missing value strategies  
        - Statistical outlier removal
        - Data smoothing options
        
        🤖 **Advanced Anomaly Detection**
        - Isolation Forest Algorithm
        - Elliptic Envelope Method
        - DBSCAN Clustering
        - Statistical Threshold Detection
        
        📊 **Interactive Visualizations**
        - Real-time sensor data plots
        - Anomaly score distributions
        - Feature correlation analysis
        - Time series trend analysis
        
        🎯 **Comprehensive Analysis**
        - Feature importance ranking
        - Detailed anomaly reports
        - Statistical comparisons
        - Export capabilities
        
        ### 📋 Getting Started:
        
        1. **Load Dataset** 📥 - Click "Load Air Quality Dataset" in the sidebar
        2. **Configure Processing** ⚙️ - Choose your data processing options
        3. **Process Data** 🔧 - Clean and prepare your data
        4. **Detect Anomalies** 🎯 - Run anomaly detection algorithms
        5. **Analyze Results** 📈 - Explore interactive visualizations
        
        ---
        
        **👈 Ready to begin? Load the dataset from the sidebar!**
        """)

def show_main_dashboard():
    """Show main dashboard when data is loaded"""
    
    # Data overview section
    if hasattr(st.session_state, 'raw_data'):
        show_data_overview()
    
    # Data processing section
    if hasattr(st.session_state, 'data_processed') and st.session_state.data_processed:
        show_processing_results()
        show_anomaly_detection()

def show_data_overview():
    """Show data overview section"""
    st.markdown("## 📊 Dataset Overview")
    
    data = st.session_state.raw_data
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(data):,}")
    with col2:
        st.metric("Features", len(data.columns))
    with col3:
        missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        st.metric("Missing Data", f"{missing_pct:.1f}%")
    with col4:
        memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Memory Usage", f"{memory_mb:.1f} MB")
    
    # Data preview
    with st.expander("🔍 Data Preview", expanded=False):
        tab1, tab2, tab3 = st.tabs(["Sample Data", "Data Types", "Missing Values"])
        
        with tab1:
            st.dataframe(data.head(20), use_container_width=True)
        
        with tab2:
            dtype_df = pd.DataFrame({
                'Column': data.columns,
                'Data Type': data.dtypes.astype(str),
                'Non-Null Count': data.count(),
                'Null Count': data.isnull().sum()
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        with tab3:
            missing_data = data.isnull().sum()
            if missing_data.sum() > 0:
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Count': missing_data.values,
                    'Missing Percentage': (missing_data.values / len(data)) * 100
                }).sort_values('Missing Count', ascending=False)
                
                fig = px.bar(
                    missing_df[missing_df['Missing Count'] > 0], 
                    x='Column', 
                    y='Missing Percentage',
                    title='Missing Data by Column'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("✅ No missing data found!")

def show_processing_results():
    """Show data processing results"""
    st.markdown("## 🔧 Data Processing Results")
    
    summary = st.session_state.processing_summary
    
    # Processing summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Processed Records", f"{summary['rows_processed']:,}")
    with col2:
        st.metric("Numeric Features", summary['numeric_features'])
    with col3:
        if summary.get('outliers_removed', 0) > 0:
            st.metric("Outliers Removed", summary['outliers_removed'])
        else:
            st.metric("Outliers Removed", "0")
    with col4:
        st.metric("Processing Time", f"{summary.get('processing_time', 0):.2f}s")
    
    # Show processed data preview
    with st.expander("📋 Processed Data Preview", expanded=False):
        st.dataframe(st.session_state.processed_data.head(10), use_container_width=True)

def show_anomaly_detection():
    """Show anomaly detection section"""
    st.markdown("## 🎯 Anomaly Detection")
    
    processed_data = st.session_state.processed_data
    
    # Detection parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        detection_method = st.selectbox(
            "Detection Method:",
            ["isolation_forest", "elliptic_envelope", "dbscan", "statistical"],
            format_func=lambda x: {
                "isolation_forest": "🌲 Isolation Forest",
                "elliptic_envelope": "🎯 Elliptic Envelope", 
                "dbscan": "🔗 DBSCAN Clustering",
                "statistical": "📊 Statistical Threshold"
            }[x],
            help="Choose anomaly detection algorithm"
        )
    
    with col2:
        if detection_method in ['isolation_forest', 'elliptic_envelope']:
            contamination = st.slider(
                "Expected Contamination:",
                min_value=0.01,
                max_value=0.3,
                value=0.1,
                step=0.01,
                help="Expected proportion of anomalies"
            )
        elif detection_method == 'statistical':
            threshold = st.slider(
                "Threshold (Std Dev):",
                min_value=1.0,
                max_value=5.0,
                value=3.0,
                step=0.1,
                help="Number of standard deviations for threshold"
            )
        else:  # DBSCAN
            eps = st.slider(
                "EPS (Neighborhood):",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.1,
                help="Maximum distance between samples"
            )
    
    with col3:
        if detection_method == 'dbscan':
            min_samples = st.slider(
                "Min Samples:",
                min_value=2,
                max_value=20,
                value=5,
                help="Minimum samples in neighborhood"
            )
    
    # Feature selection
    st.markdown("### 📊 Feature Selection")
    numeric_columns = processed_data.select_dtypes(include=[np.number]).columns.tolist()
    
    selected_features = st.multiselect(
        "Select features for anomaly detection:",
        options=numeric_columns,
        default=numeric_columns[:min(6, len(numeric_columns))],
        help="Choose which features to analyze"
    )
    
    if not selected_features:
        st.warning("⚠️ Please select at least one feature.")
        return
    
    # Run detection
    if st.button("🚀 Run Anomaly Detection", type="primary"):
        run_anomaly_detection(detection_method, selected_features, processed_data, locals())

def run_anomaly_detection(method, features, data, params):
    """Run anomaly detection with selected parameters"""
    try:
        with st.spinner(f'Running {method} anomaly detection...'):
            # Prepare parameters
            method_params = {}
            if method in ['isolation_forest', 'elliptic_envelope']:
                method_params['contamination'] = params['contamination']
            elif method == 'statistical':
                method_params['threshold'] = params['threshold']
            elif method == 'dbscan':
                method_params['eps'] = params['eps']
                method_params['min_samples'] = params['min_samples']
            
            # Run detection
            analysis_data = data[features]
            result = st.session_state.detector.detect_anomalies(
                analysis_data, 
                method=method,
                **method_params
            )
            
            # Store results
            st.session_state.anomaly_result = result
            st.session_state.selected_features = features
            st.session_state.detection_method = method
            st.session_state.analysis_data = analysis_data
            
            # Show results
            show_anomaly_results(result, features, analysis_data, method)
            
    except Exception as e:
        st.error(f"❌ Error in anomaly detection: {str(e)}")

def show_anomaly_results(result, features, data, method):
    """Display anomaly detection results"""
    anomaly_count = np.sum(result['labels'] == 1)
    total_points = len(result['labels'])
    anomaly_rate = (anomaly_count / total_points) * 100
    
    # Results summary
    st.success(f"✅ Anomaly detection completed using {method}!")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Data Points", f"{total_points:,}")
    with col2:
        st.metric("Anomalies Detected", f"{anomaly_count:,}")
    with col3:
        st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
    with col4:
        if anomaly_count > 0:
            avg_score = np.mean(result['scores'][result['labels'] == 1])
            st.metric("Avg Anomaly Score", f"{avg_score:.4f}")
        else:
            st.metric("Avg Anomaly Score", "N/A")
    
    # Detailed results tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Time Series", "📊 Score Distribution", 
        "🔍 Feature Analysis", "⚠️ Anomaly Details", "🆚 Model Comparison"
    ])
    
    with tab1:
        show_time_series_results(data, result, features)
    
    with tab2:
        show_score_distribution(result)
    
    with tab3:
        show_feature_analysis(data, result, features)
    
    with tab4:
        show_anomaly_details(data, result)
    
    with tab5:
        show_model_comparison(data, features)

def show_time_series_results(data, result, features):
    """Show time series visualization"""
    st.markdown("### 📈 Time Series Analysis")
    
    fig = st.session_state.detector.create_time_series_plot(data, result, features)
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional time series insights
    if np.sum(result['labels'] == 1) > 0:
        st.markdown("#### 🔍 Time Series Insights")
        anomaly_indices = np.where(result['labels'] == 1)[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("First Anomaly at Index", anomaly_indices[0])
        with col2:
            st.metric("Last Anomaly at Index", anomaly_indices[-1])

def show_score_distribution(result):
    """Show anomaly score distribution"""
    st.markdown("### 📊 Anomaly Score Distribution")
    
    fig = st.session_state.detector.create_score_distribution_plot(result)
    st.plotly_chart(fig, use_container_width=True)
    
    # Score statistics
    scores = result['scores']
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Score", f"{np.mean(scores):.4f}")
    with col2:
        st.metric("Std Deviation", f"{np.std(scores):.4f}")
    with col3:
        st.metric("Min Score", f"{np.min(scores):.4f}")
    with col4:
        st.metric("Max Score", f"{np.max(scores):.4f}")

def show_feature_analysis(data, result, features):
    """Show feature analysis"""
    st.markdown("### 🔍 Feature Analysis")
    
    # Feature importance plot
    fig_importance = st.session_state.detector.create_feature_importance_plot(
        data, result, features
    )
    if fig_importance:
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Feature statistics comparison
    if np.sum(result['labels'] == 1) > 0:
        st.markdown("#### 📋 Statistical Comparison: Normal vs Anomalous")
        
        normal_mask = result['labels'] == 0
        anomaly_mask = result['labels'] == 1
        
        comparison_stats = []
        for feature in features:
            normal_data = data[feature][normal_mask]
            anomaly_data = data[feature][anomaly_mask]
            
            comparison_stats.append({
                'Feature': feature,
                'Normal Mean': f"{np.mean(normal_data):.3f}",
                'Anomaly Mean': f"{np.mean(anomaly_data):.3f}",
                'Normal Std': f"{np.std(normal_data):.3f}",
                'Anomaly Std': f"{np.std(anomaly_data):.3f}",
                'Mean Difference': f"{abs(np.mean(anomaly_data) - np.mean(normal_data)):.3f}"
            })
        
        comparison_df = pd.DataFrame(comparison_stats)
        st.dataframe(comparison_df, use_container_width=True)

def show_anomaly_details(data, result):
    """Show detailed anomaly information"""
    st.markdown("### ⚠️ Anomaly Details")
    
    anomaly_count = np.sum(result['labels'] == 1)
    
    if anomaly_count > 0:
        # Get anomalous data points
        anomaly_indices = np.where(result['labels'] == 1)[0]
        anomaly_data = data.iloc[anomaly_indices].copy()
        anomaly_data['Anomaly_Score'] = result['scores'][result['labels'] == 1]
        anomaly_data = anomaly_data.sort_values('Anomaly_Score', ascending=False)
        
        st.dataframe(anomaly_data, use_container_width=True)
        
        # Download button
        csv = anomaly_data.to_csv(index=True)
        st.download_button(
            label="📥 Download Anomalous Data Points",
            data=csv,
            file_name="environmental_anomalies.csv",
            mime="text/csv"
        )
        
        # Top anomalies
        st.markdown("#### 🔝 Top 5 Most Anomalous Points")
        st.dataframe(anomaly_data.head(), use_container_width=True)
        
    else:
        st.info("ℹ️ No anomalies detected with current parameters.")
        st.markdown("""
        **💡 Suggestions:**
        - Try increasing the contamination rate
        - Use a different detection method
        - Adjust algorithm-specific parameters
        - Review feature selection
        """)

def show_model_comparison(data, features):
    """Show comparison between different models"""
    st.markdown("### 🆚 Model Comparison")
    
    if st.button("🔄 Compare All Methods"):
        compare_all_methods(data, features)

def compare_all_methods(data, features):
    """Compare all anomaly detection methods"""
    try:
        with st.spinner('Comparing all detection methods...'):
            methods = ['isolation_forest', 'elliptic_envelope', 'dbscan', 'statistical']
            comparison_results = {}
            
            for method in methods:
                # Set default parameters for each method
                if method == 'isolation_forest':
                    params = {'contamination': 0.1}
                elif method == 'elliptic_envelope':
                    params = {'contamination': 0.1}
                elif method == 'dbscan':
                    params = {'eps': 0.5, 'min_samples': 5}
                else:  # statistical
                    params = {'threshold': 3.0}
                
                result = st.session_state.detector.detect_anomalies(data, method, **params)
                comparison_results[method] = result
            
            # Create comparison table
            comparison_data = []
            for method_name, result in comparison_results.items():
                anomaly_count = np.sum(result['labels'] == 1)
                anomaly_rate = (anomaly_count / len(result['labels'])) * 100
                
                comparison_data.append({
                    'Method': method_name.replace('_', ' ').title(),
                    'Anomalies Detected': anomaly_count,
                    'Anomaly Rate (%)': f"{anomaly_rate:.2f}%",
                    'Avg Score': f"{np.mean(result['scores']):.4f}",
                    'Max Score': f"{np.max(result['scores']):.4f}"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Visualization of comparison
            fig = px.bar(
                comparison_df, 
                x='Method', 
                y='Anomalies Detected',
                title='Anomalies Detected by Each Method',
                color='Method'
            )
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"❌ Error in model comparison: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <h4>🌍 Environmental Data Anomaly Detection System</h4>
    <p>Built with Streamlit • Powered by Advanced Machine Learning</p>  
    <p>Supporting environmental monitoring and data-driven insights</p>
    <p><strong>Methods:</strong> Isolation Forest | Elliptic Envelope | DBSCAN | Statistical Analysis</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
