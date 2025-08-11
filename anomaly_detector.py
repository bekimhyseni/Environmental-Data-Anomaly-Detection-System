import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    """
    Advanced anomaly detection system for environmental sensor data
    Supports multiple detection algorithms with comprehensive analysis
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.detection_history = []
        
    def detect_anomalies(self, data, method='isolation_forest', **kwargs):
        """
        Main anomaly detection method
        
        Parameters:
        - data: DataFrame with numeric features
        - method: Detection method ('isolation_forest', 'elliptic_envelope', 'dbscan', 'statistical')
        - **kwargs: Method-specific parameters
        
        Returns:
        - Dictionary with labels, scores, and metadata
        """
        try:
            # Validate and prepare data
            data_clean = self._prepare_data(data)
            
            if len(data_clean) == 0:
                raise ValueError("No valid data remaining after cleaning.")
            
            # Apply detection method
            if method == 'isolation_forest':
                result = self._isolation_forest_detection(data_clean, **kwargs)
            elif method == 'elliptic_envelope':
                result = self._elliptic_envelope_detection(data_clean, **kwargs)
            elif method == 'dbscan':
                result = self._dbscan_detection(data_clean, **kwargs)
            elif method == 'statistical':
                result = self._statistical_detection(data_clean, **kwargs)
            else:
                raise ValueError(f"Unknown detection method: {method}")
            
            # Add metadata
            result['method'] = method
            result['data_shape'] = data_clean.shape
            result['feature_names'] = data_clean.columns.tolist()
            
            # Store in history
            self.detection_history.append({
                'method': method,
                'timestamp': pd.Timestamp.now(),
                'anomaly_count': np.sum(result['labels'] == 1),
                'total_points': len(result['labels'])
            })
            
            return result
            
        except Exception as e:
            raise Exception(f"Anomaly detection error with {method}: {str(e)}")
    
    def _prepare_data(self, data):
        """Prepare and validate data for anomaly detection"""
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            raise ValueError("No numeric columns found in data.")
        
        # Remove rows with any NaN values
        clean_data = numeric_data.dropna()
        
        if len(clean_data) == 0:
            raise ValueError("No valid rows remaining after removing NaN values.")
        
        # Remove constant columns
        constant_columns = [col for col in clean_data.columns if clean_data[col].nunique() <= 1]
        if constant_columns:
            print(f"Removing constant columns: {constant_columns}")
            clean_data = clean_data.drop(columns=constant_columns)
        
        return clean_data
    
    def _isolation_forest_detection(self, data, contamination=0.1, random_state=42, **kwargs):
        """Isolation Forest anomaly detection"""
        model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
            **kwargs
        )
        
        # Fit and predict
        labels = model.fit_predict(data)
        scores = model.decision_function(data)
        
        # Convert labels: -1 (anomaly) -> 1, 1 (normal) -> 0
        binary_labels = np.where(labels == -1, 1, 0)
        
        # Store model
        self.models['isolation_forest'] = model
        
        return {
            'labels': binary_labels,
            'scores': scores,
            'model': model
        }
    
    def _elliptic_envelope_detection(self, data, contamination=0.1, random_state=42, **kwargs):
        """Elliptic Envelope anomaly detection"""
        model = EllipticEnvelope(
            contamination=contamination,
            random_state=random_state,
            **kwargs
        )
        
        # Fit and predict
        labels = model.fit_predict(data)
        scores = model.decision_function(data)
        
        # Convert labels: -1 (anomaly) -> 1, 1 (normal) -> 0
        binary_labels = np.where(labels == -1, 1, 0)
        
        # Store model
        self.models['elliptic_envelope'] = model
        
        return {
            'labels': binary_labels,
            'scores': scores,
            'model': model
        }
    
    def _dbscan_detection(self, data, eps=0.5, min_samples=5, **kwargs):
        """DBSCAN clustering-based anomaly detection"""
        # Scale data for DBSCAN
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        cluster_labels = model.fit_predict(data_scaled)
        
        # Points labeled as -1 are considered anomalies (noise)
        binary_labels = np.where(cluster_labels == -1, 1, 0)
        
        # Calculate distance-based anomaly scores
        scores = self._calculate_dbscan_scores(data_scaled, cluster_labels, model)
        
        # Store model and scaler
        self.models['dbscan'] = model
        self.scalers['dbscan'] = scaler
        
        return {
            'labels': binary_labels,
            'scores': scores,
            'model': model,
            'cluster_labels': cluster_labels
        }
    
    def _statistical_detection(self, data, threshold=3.0, **kwargs):
        """Statistical threshold-based anomaly detection"""
        # Calculate Z-scores for each feature
        z_scores = np.abs(stats.zscore(data))
        
        # A point is anomalous if any feature exceeds the threshold
        anomaly_mask = (z_scores > threshold).any(axis=1)
        binary_labels = anomaly_mask.astype(int)
        
        # Use maximum Z-score as anomaly score
        scores = np.max(z_scores, axis=1)
        
        return {
            'labels': binary_labels,
            'scores': scores,
            'z_scores': z_scores,
            'threshold': threshold
        }
    
    def _calculate_dbscan_scores(self, data, cluster_labels, model):
        """Calculate anomaly scores for DBSCAN results"""
        scores = np.zeros(len(data))
        
        # For each point, calculate distance to nearest cluster center
        unique_clusters = np.unique(cluster_labels)
        cluster_centers = {}
        
        # Calculate cluster centers (excluding noise points)
        for cluster in unique_clusters:
            if cluster != -1:  # -1 represents noise/anomalies
                cluster_points = data[cluster_labels == cluster]
                cluster_centers[cluster] = np.mean(cluster_points, axis=0)
        
        # Calculate scores
        for i, point in enumerate(data):
            if cluster_labels[i] == -1:  # Noise point (anomaly)
                # Distance to nearest cluster center
                if cluster_centers:
                    distances = [np.linalg.norm(point - center) for center in cluster_centers.values()]
                    scores[i] = min(distances)
                else:
                    scores[i] = 1.0  # Default high score if no clusters
            else:
                # Distance to own cluster center
                center = cluster_centers[cluster_labels[i]]
                scores[i] = np.linalg.norm(point - center)
        
        return scores
    
    def create_time_series_plot(self, data, result, features):
        """Create interactive time series plot with anomalies highlighted"""
        try:
            n_features = len(features)
            
            # Create subplots
            fig = make_subplots(
                rows=min(n_features, 4),
                cols=1,
                subplot_titles=features[:4],  # Limit to 4 features for readability
                vertical_spacing=0.05
            )
            
            colors = px.colors.qualitative.Set1
            
            for i, feature in enumerate(features[:4]):
                row = i + 1
                
                # Normal points
                normal_mask = result['labels'] == 0
                fig.add_trace(
                    go.Scatter(
                        x=data.index[normal_mask],
                        y=data[feature][normal_mask],
                        mode='markers',
                        name=f'{feature} (Normal)',
                        marker=dict(color=colors[i % len(colors)], size=4, opacity=0.6),
                        showlegend=(i == 0)
                    ),
                    row=row, col=1
                )
                
                # Anomalous points
                anomaly_mask = result['labels'] == 1
                if np.any(anomaly_mask):
                    fig.add_trace(
                        go.Scatter(
                            x=data.index[anomaly_mask],
                            y=data[feature][anomaly_mask],
                            mode='markers',
                            name=f'{feature} (Anomaly)',
                            marker=dict(color='red', size=8, symbol='x'),
                            showlegend=(i == 0)
                        ),
                        row=row, col=1
                    )
            
            fig.update_layout(
                title="🕐 Time Series Analysis with Anomaly Detection",
                height=200 * min(n_features, 4),
                hovermode='x unified',
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating time series plot: {e}")
            return self._create_simple_scatter_plot(data, result, features)
    
    def create_score_distribution_plot(self, result):
        """Create anomaly score distribution plot"""
        try:
            scores = result['scores']
            labels = result['labels']
            
            # Create histogram
            fig = go.Figure()
            
            # Normal points
            normal_scores = scores[labels == 0]
            fig.add_trace(go.Histogram(
                x=normal_scores,
                name='Normal Points',
                opacity=0.7,
                marker_color='lightblue',
                nbinsx=30
            ))
            
            # Anomalous points
            anomaly_scores = scores[labels == 1]
            if len(anomaly_scores) > 0:
                fig.add_trace(go.Histogram(
                    x=anomaly_scores,
                    name='Anomalous Points',
                    opacity=0.7,
                    marker_color='red',
                    nbinsx=20
                ))
            
            fig.update_layout(
                title="📊 Anomaly Score Distribution",
                xaxis_title="Anomaly Score",
                yaxis_title="Frequency",
                barmode='overlay',
                template='plotly_white',
                height=400
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating score distribution plot: {e}")
            return go.Figure()
    
    def create_feature_importance_plot(self, data, result, features):
        """Create feature importance plot based on variance analysis"""
        try:
            if np.sum(result['labels'] == 1) == 0:
                return None
            
            normal_mask = result['labels'] == 0
            anomaly_mask = result['labels'] == 1
            
            importance_scores = []
            for feature in features:
                normal_data = data[feature][normal_mask]
                anomaly_data = data[feature][anomaly_mask]
                
                # Calculate importance based on mean difference (normalized)
                normal_mean = np.mean(normal_data)
                anomaly_mean = np.mean(anomaly_data)
                normal_std = np.std(normal_data)
                
                # Normalized difference (Cohen's d effect size)
                if normal_std > 0:
                    importance = abs(anomaly_mean - normal_mean) / normal_std
                else:
                    importance = 0
                
                importance_scores.append(importance)
            
            # Sort features by importance
            feature_importance = list(zip(features, importance_scores))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            sorted_features, sorted_scores = zip(*feature_importance) if feature_importance else ([], [])
            
            # Create bar plot
            fig = go.Figure(data=[
                go.Bar(
                    y=list(sorted_features),
                    x=list(sorted_scores),
                    orientation='h',
                    marker=dict(
                        color=list(sorted_scores),
                        colorscale='Reds',
                        colorbar=dict(title="Importance Score")
                    ),
                    text=[f'{score:.3f}' for score in sorted_scores],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="🎯 Feature Importance for Anomaly Detection",
                xaxis_title="Importance Score (Cohen's d)",
                yaxis_title="Features",
                height=max(400, len(features) * 30),
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating feature importance plot: {e}")
            return None
    
    def _create_simple_scatter_plot(self, data, result, features):
        """Create simple scatter plot as fallback"""
        try:
            if len(features) < 2:
                return go.Figure()
            
            feature_x, feature_y = features[0], features[1]
            
            fig = go.Figure()
            
            # Normal points
            normal_mask = result['labels'] == 0
            fig.add_trace(go.Scatter(
                x=data[feature_x][normal_mask],
                y=data[feature_y][normal_mask],
                mode='markers',
                name='Normal',
                marker=dict(color='lightblue', size=6, opacity=0.6)
            ))
            
            # Anomalous points
            anomaly_mask = result['labels'] == 1
            if np.any(anomaly_mask):
                fig.add_trace(go.Scatter(
                    x=data[feature_x][anomaly_mask],
                    y=data[feature_y][anomaly_mask],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(color='red', size=10, symbol='x')
                ))
            
            fig.update_layout(
                title=f"📊 {feature_x} vs {feature_y}",
                xaxis_title=feature_x,
                yaxis_title=feature_y,
                template='plotly_white',
                height=500
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating scatter plot: {e}")
            return go.Figure()
    
    def get_detection_summary(self):
        """Get summary of all detection runs"""
        if not self.detection_history:
            return "No detections performed yet."
        
        summary = pd.DataFrame(self.detection_history)
        return summary.groupby('method').agg({
            'anomaly_count': ['mean', 'std'],
            'total_points': 'mean'
        }).round(2)
    
    def compare_methods(self, data, methods=['isolation_forest', 'elliptic_envelope', 'statistical']):
        """Compare multiple detection methods"""
        results = {}
        
        for method in methods:
            try:
                if method == 'statistical':
                    result = self.detect_anomalies(data, method, threshold=3.0)
                else:
                    result = self.detect_anomalies(data, method, contamination=0.1)
                results[method] = result
            except Exception as e:
                print(f"Error with {method}: {e}")
        
        return results
