import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import warnings
import time
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Advanced data processor for environmental sensor data
    Handles multiple data types, missing values, outliers, and preprocessing
    """
    
    def __init__(self):
        self.scaler = None
        self.imputer = None
        self.column_types = {}
        self.processing_stats = {}
        
    def process_data(self, df, missing_strategy='median', remove_outliers=True, apply_smoothing=False, scaling_method='standard'):
        """
        Main data processing pipeline
        
        Parameters:
        - df: Input DataFrame
        - missing_strategy: Strategy for handling missing values ('median', 'mean', 'forward_fill')
        - remove_outliers: Whether to remove statistical outliers
        - apply_smoothing: Whether to apply rolling average smoothing
        - scaling_method: Scaling method ('standard', 'robust', 'none')
        
        Returns:
        - processed_df: Processed DataFrame
        - processing_summary: Dictionary with processing statistics
        """
        start_time = time.time()
        
        try:
            # Step 1: Analyze data structure
            self.column_types = self._analyze_data_types(df)
            
            # Step 2: Handle different column types
            processed_df = self._separate_and_process_columns(df)
            
            # Step 3: Handle missing values for numeric columns
            processed_df = self._handle_missing_values(processed_df, missing_strategy)
            
            # Step 4: Remove outliers if requested
            outliers_removed = 0
            if remove_outliers:
                processed_df, outliers_removed = self._remove_outliers(processed_df)
            
            # Step 5: Apply smoothing if requested
            if apply_smoothing:
                processed_df = self._apply_smoothing(processed_df)
            
            # Step 6: Scale numeric features if requested
            if scaling_method != 'none':
                processed_df = self._scale_features(processed_df, scaling_method)
            
            # Step 7: Final validation
            processed_df = self._validate_data(processed_df)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create processing summary
            processing_summary = {
                'rows_processed': len(processed_df),
                'original_rows': len(df),
                'numeric_features': len(self.column_types['numeric']),
                'date_features': len(self.column_types['date']),
                'categorical_features': len(self.column_types['categorical']),
                'outliers_removed': outliers_removed,
                'missing_strategy': missing_strategy,
                'scaling_method': scaling_method,
                'smoothing_applied': apply_smoothing,
                'processing_time': processing_time,
                'memory_usage_mb': processed_df.memory_usage(deep=True).sum() / 1024 / 1024
            }
            
            return processed_df, processing_summary
            
        except Exception as e:
            raise Exception(f"Data processing error: {str(e)}")
    
    def _analyze_data_types(self, df):
        """Analyze and categorize column types"""
        column_types = {
            'numeric': [],
            'date': [],
            'categorical': [],
            'text': []
        }
        
        for col in df.columns:
            # Skip if all values are null
            if df[col].isnull().all():
                continue
                
            # Check for date columns
            if self._is_date_column(df[col]):
                column_types['date'].append(col)
            # Check for numeric columns
            elif pd.api.types.is_numeric_dtype(df[col]):
                column_types['numeric'].append(col)
            # Check for categorical/text columns
            elif df[col].dtype == 'object':
                unique_ratio = df[col].nunique() / len(df[col].dropna())
                if unique_ratio < 0.5:  # If less than 50% unique values, treat as categorical
                    column_types['categorical'].append(col)
                else:
                    column_types['text'].append(col)
            else:
                column_types['numeric'].append(col)
        
        return column_types
    
    def _is_date_column(self, series):
        """Check if a series contains date-like data"""
        if series.dtype == 'object':
            # Sample a few non-null values to check for date patterns
            sample = series.dropna().head(10)
            if len(sample) == 0:
                return False
            
            try:
                # Try to parse sample values as dates
                pd.to_datetime(sample, errors='raise')
                return True
            except:
                return False
        
        return pd.api.types.is_datetime64_any_dtype(series)
    
    def _separate_and_process_columns(self, df):
        """Process different column types appropriately"""
        processed_df = df.copy()
        
        # Convert date columns
        for col in self.column_types['date']:
            try:
                processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')
            except Exception as e:
                print(f"Warning: Could not convert {col} to datetime: {e}")
                # Move to categorical if datetime conversion fails
                self.column_types['categorical'].append(col)
                self.column_types['date'].remove(col)
        
        # Ensure numeric columns are properly typed
        for col in self.column_types['numeric']:
            try:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
            except Exception as e:
                print(f"Warning: Could not convert {col} to numeric: {e}")
        
        return processed_df
    
    def _handle_missing_values(self, df, strategy):
        """Handle missing values in numeric columns"""
        numeric_cols = self.column_types['numeric']
        
        if not numeric_cols:
            return df
        
        df_copy = df.copy()
        
        if strategy in ['median', 'mean']:
            # Use sklearn imputer for median/mean
            self.imputer = SimpleImputer(strategy=strategy)
            df_copy[numeric_cols] = self.imputer.fit_transform(df_copy[numeric_cols])
        
        elif strategy == 'forward_fill':
            # Forward fill for time series data
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(method='ffill')
            # Backward fill for any remaining NaNs
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(method='bfill')
        
        return df_copy
    
    def _remove_outliers(self, df):
        """Remove statistical outliers using IQR method"""
        numeric_cols = self.column_types['numeric']
        
        if not numeric_cols:
            return df, 0
        
        df_copy = df.copy()
        initial_rows = len(df_copy)
        
        for col in numeric_cols:
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Remove outliers
            df_copy = df_copy[(df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)]
        
        outliers_removed = initial_rows - len(df_copy)
        return df_copy, outliers_removed
    
    def _apply_smoothing(self, df, window_size=5):
        """Apply rolling average smoothing to numeric columns"""
        numeric_cols = self.column_types['numeric']
        
        if not numeric_cols:
            return df
        
        df_copy = df.copy()
        
        for col in numeric_cols:
            df_copy[f'{col}_smoothed'] = df_copy[col].rolling(window=window_size, center=True).mean()
            # Use smoothed values where available, original values otherwise
            df_copy[col] = df_copy[f'{col}_smoothed'].fillna(df_copy[col])
            # Drop the temporary smoothed column
            df_copy.drop(f'{col}_smoothed', axis=1, inplace=True)
        
        return df_copy
    
    def _scale_features(self, df, method):
        """Scale numeric features"""
        numeric_cols = self.column_types['numeric']
        
        if not numeric_cols:
            return df
        
        df_copy = df.copy()
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        
        df_copy[numeric_cols] = self.scaler.fit_transform(df_copy[numeric_cols])
        
        return df_copy
    
    def _validate_data(self, df):
        """Final data validation and cleaning"""
        # Remove any rows with all NaN values
        df_clean = df.dropna(how='all')
        
        # Remove any columns with all NaN values
        df_clean = df_clean.dropna(axis=1, how='all')
        
        # Replace any infinite values with NaN and then handle them
        numeric_cols = self.column_types['numeric']
        if numeric_cols:
            df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        
        return df_clean
    
    def get_processing_info(self):
        """Get information about the processing pipeline"""
        return {
            'column_types': self.column_types,
            'scaler_type': type(self.scaler).__name__ if self.scaler else None,
            'imputer_strategy': self.imputer.strategy if self.imputer else None
        }
    
    def transform_new_data(self, df):
        """Transform new data using fitted preprocessors"""
        if self.scaler is None and self.imputer is None:
            raise ValueError("No fitted preprocessors found. Call process_data() first.")
        
        df_copy = df.copy()
        numeric_cols = self.column_types['numeric']
        
        # Apply imputation if fitted
        if self.imputer and numeric_cols:
            df_copy[numeric_cols] = self.imputer.transform(df_copy[numeric_cols])
        
        # Apply scaling if fitted
        if self.scaler and numeric_cols:
            df_copy[numeric_cols] = self.scaler.transform(df_copy[numeric_cols])
        
        return df_copy
