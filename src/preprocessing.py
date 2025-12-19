"""
Preprocessing Module for Chronic Disease Risk Prediction System

This module handles:
- Missing value imputation
- Outlier detection and handling
- Feature scaling and normalization
- Categorical encoding
- Train/test splitting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import warnings


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for clinical data.
    
    Features:
    - Multiple imputation strategies
    - Outlier detection and handling
    - Feature scaling (standard, minmax, robust)
    - Categorical encoding (label, one-hot)
    - Stratified train/test splitting
    """
    
    def __init__(
        self,
        numerical_features: List[str] = None,
        categorical_features: List[str] = None,
        binary_features: List[str] = None,
        scaling_method: str = 'standard',
        imputation_strategy: str = 'median',
        handle_outliers: bool = True,
        outlier_method: str = 'iqr',
        outlier_threshold: float = 1.5
    ):
        """
        Initialize the preprocessor.
        
        Args:
            numerical_features: List of numerical column names
            categorical_features: List of categorical column names
            binary_features: List of binary (0/1) column names
            scaling_method: 'standard', 'minmax', or 'robust'
            imputation_strategy: 'mean', 'median', 'mode', or 'knn'
            handle_outliers: Whether to handle outliers
            outlier_method: 'iqr', 'zscore', or 'clip'
            outlier_threshold: Threshold for outlier detection
        """
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.binary_features = binary_features or []
        
        self.scaling_method = scaling_method
        self.imputation_strategy = imputation_strategy
        self.handle_outliers = handle_outliers
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        
        # Fitted components
        self.scaler = None
        self.numerical_imputer = None
        self.categorical_imputer = None
        self.label_encoders = {}
        self.onehot_encoder = None
        self.feature_names = []
        self.outlier_bounds = {}
        
        self._is_fitted = False
    
    def _get_default_features(self) -> Dict[str, List[str]]:
        """Return default feature categorization for clinical data."""
        return {
            'numerical': [
                'age', 'bmi', 'systolic_bp', 'diastolic_bp', 'heart_rate',
                'fasting_glucose', 'hba1c', 'total_cholesterol', 'hdl_cholesterol',
                'ldl_cholesterol', 'triglycerides', 'creatinine', 'egfr', 'diet_quality_score'
            ],
            'categorical': [
                'gender', 'ethnicity', 'smoking_status', 'alcohol_consumption', 'physical_activity_level'
            ],
            'binary': [
                'family_history_diabetes', 'family_history_cvd', 'family_history_kidney_disease',
                'hypertension_diagnosed', 'previous_cardiovascular_event'
            ]
        }
    
    def infer_feature_types(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> None:
        """
        Automatically infer feature types from the dataframe.
        
        Args:
            df: Input DataFrame
            exclude_cols: Columns to exclude (e.g., patient_id, targets)
        """
        exclude_cols = exclude_cols or ['patient_id', 'diabetes_risk', 'cvd_risk', 'kidney_disease_risk']
        
        self.numerical_features = []
        self.categorical_features = []
        self.binary_features = []
        
        for col in df.columns:
            if col in exclude_cols:
                continue
            
            if df[col].dtype in ['int64', 'float64']:
                unique_values = df[col].dropna().unique()
                if len(unique_values) <= 2 and set(unique_values).issubset({0, 1, 0.0, 1.0}):
                    self.binary_features.append(col)
                else:
                    self.numerical_features.append(col)
            else:
                self.categorical_features.append(col)
        
        print(f"Inferred feature types:")
        print(f"  Numerical: {len(self.numerical_features)} features")
        print(f"  Categorical: {len(self.categorical_features)} features")
        print(f"  Binary: {len(self.binary_features)} features")
    
    def analyze_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing value statistics
        """
        missing_stats = pd.DataFrame({
            'column': df.columns,
            'missing_count': df.isnull().sum().values,
            'missing_percentage': (df.isnull().sum().values / len(df) * 100).round(2),
            'dtype': df.dtypes.values
        })
        missing_stats = missing_stats[missing_stats['missing_count'] > 0].sort_values('missing_percentage', ascending=False)
        
        return missing_stats
    
    def detect_outliers(self, df: pd.DataFrame, method: str = None) -> Dict[str, Tuple[float, float]]:
        """
        Detect outliers in numerical features.
        
        Args:
            df: Input DataFrame
            method: 'iqr' or 'zscore'
            
        Returns:
            Dictionary mapping feature names to (lower_bound, upper_bound) tuples
        """
        method = method or self.outlier_method
        bounds = {}
        
        for col in self.numerical_features:
            if col not in df.columns:
                continue
            
            data = df[col].dropna()
            
            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - self.outlier_threshold * IQR
                upper = Q3 + self.outlier_threshold * IQR
            elif method == 'zscore':
                mean = data.mean()
                std = data.std()
                lower = mean - self.outlier_threshold * std
                upper = mean + self.outlier_threshold * std
            else:
                lower = data.min()
                upper = data.max()
            
            bounds[col] = (lower, upper)
        
        return bounds
    
    def handle_outliers_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers by clipping to bounds.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with outliers handled
        """
        df = df.copy()
        
        for col, (lower, upper) in self.outlier_bounds.items():
            if col in df.columns:
                original_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
                df[col] = df[col].clip(lower=lower, upper=upper)
                if original_outliers > 0:
                    print(f"  Clipped {original_outliers} outliers in '{col}'")
        
        return df
    
    def fit(self, df: pd.DataFrame, target_col: str = None) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            df: Training DataFrame
            target_col: Target column name (excluded from preprocessing)
            
        Returns:
            self
        """
        df = df.copy()
        
        # Infer feature types if not already set
        if not self.numerical_features and not self.categorical_features:
            exclude = ['patient_id'] + ([target_col] if target_col else [])
            self.infer_feature_types(df, exclude_cols=exclude)
        
        # Fit outlier bounds
        if self.handle_outliers:
            self.outlier_bounds = self.detect_outliers(df)
        
        # Fit numerical imputer
        if self.numerical_features:
            available_numerical = [f for f in self.numerical_features if f in df.columns]
            if available_numerical:
                if self.imputation_strategy == 'knn':
                    self.numerical_imputer = KNNImputer(n_neighbors=5)
                else:
                    strategy = 'median' if self.imputation_strategy == 'median' else 'mean'
                    self.numerical_imputer = SimpleImputer(strategy=strategy)
                
                self.numerical_imputer.fit(df[available_numerical])
        
        # Fit categorical imputer
        if self.categorical_features:
            available_categorical = [f for f in self.categorical_features if f in df.columns]
            if available_categorical:
                self.categorical_imputer = SimpleImputer(strategy='most_frequent')
                self.categorical_imputer.fit(df[available_categorical])
        
        # Fit scaler
        if self.numerical_features:
            available_numerical = [f for f in self.numerical_features if f in df.columns]
            if available_numerical:
                if self.scaling_method == 'standard':
                    self.scaler = StandardScaler()
                elif self.scaling_method == 'minmax':
                    self.scaler = MinMaxScaler()
                elif self.scaling_method == 'robust':
                    self.scaler = RobustScaler()
                
                # Handle outliers before fitting scaler
                temp_df = df.copy()
                if self.handle_outliers:
                    temp_df = self.handle_outliers_transform(temp_df)
                
                # Impute before fitting scaler
                imputed = self.numerical_imputer.transform(temp_df[available_numerical])
                self.scaler.fit(imputed)
        
        # Fit label encoders for categorical features
        for col in self.categorical_features:
            if col in df.columns:
                self.label_encoders[col] = LabelEncoder()
                # Handle missing values for fitting
                values = df[col].fillna('MISSING')
                self.label_encoders[col].fit(values)
        
        # Build feature names list
        self._build_feature_names()
        
        self._is_fitted = True
        return self
    
    def _build_feature_names(self) -> None:
        """Build the final feature names list after encoding."""
        self.feature_names = []
        
        # Add numerical features
        for f in self.numerical_features:
            self.feature_names.append(f)
        
        # Add encoded categorical features
        for f in self.categorical_features:
            self.feature_names.append(f + '_encoded')
        
        # Add binary features
        for f in self.binary_features:
            self.feature_names.append(f)
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted before transform. Call fit() first.")
        
        df = df.copy()
        result = {}
        
        # Handle outliers
        if self.handle_outliers:
            df = self.handle_outliers_transform(df)
        
        # Transform numerical features
        available_numerical = [f for f in self.numerical_features if f in df.columns]
        if available_numerical and self.numerical_imputer is not None:
            imputed = self.numerical_imputer.transform(df[available_numerical])
            scaled = self.scaler.transform(imputed)
            
            for i, col in enumerate(available_numerical):
                result[col] = scaled[:, i]
        
        # Transform categorical features
        for col in self.categorical_features:
            if col in df.columns and col in self.label_encoders:
                # Handle unseen categories
                values = df[col].fillna('MISSING').astype(str)
                known_classes = set(self.label_encoders[col].classes_)
                values = values.apply(lambda x: x if x in known_classes else 'UNKNOWN')
                
                # Add UNKNOWN to classes if needed
                if 'UNKNOWN' not in known_classes:
                    self.label_encoders[col].classes_ = np.append(
                        self.label_encoders[col].classes_, 'UNKNOWN'
                    )
                
                result[col + '_encoded'] = self.label_encoders[col].transform(values)
        
        # Add binary features
        for col in self.binary_features:
            if col in df.columns:
                result[col] = df[col].fillna(0).astype(int).values
        
        return pd.DataFrame(result)
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: Training DataFrame
            target_col: Target column name
            
        Returns:
            Transformed DataFrame
        """
        self.fit(df, target_col)
        return self.transform(df)
    
    def split_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            test_size: Proportion of data for test set
            val_size: Proportion of training data for validation
            random_state: Random seed
            stratify: Whether to use stratified splitting
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Separate features and target
        feature_cols = [c for c in df.columns if c != target_col and c != 'patient_id']
        X = df[feature_cols]
        y = df[target_col]
        
        # First split: train+val vs test
        stratify_col = y if stratify else None
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_col
        )
        
        # Second split: train vs val
        val_adjusted = val_size / (1 - test_size)
        stratify_col = y_temp if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_adjusted, random_state=random_state, stratify=stratify_col
        )
        
        print(f"Data split sizes:")
        print(f"  Train: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
        print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(df)*100:.1f}%)")
        print(f"  Test: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")
        
        if stratify:
            print(f"\nTarget distribution:")
            print(f"  Train: {y_train.mean():.2%} positive")
            print(f"  Validation: {y_val.mean():.2%} positive")
            print(f"  Test: {y_test.mean():.2%} positive")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_cv_splits(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        random_state: int = 42
    ) -> StratifiedKFold:
        """
        Get cross-validation splits.
        
        Args:
            X: Features DataFrame
            y: Target Series
            n_splits: Number of CV folds
            random_state: Random seed
            
        Returns:
            StratifiedKFold object
        """
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    def save(self, filepath: str) -> None:
        """
        Save the fitted preprocessor to disk.
        
        Args:
            filepath: Path to save the preprocessor
        """
        joblib.dump(self, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'DataPreprocessor':
        """
        Load a fitted preprocessor from disk.
        
        Args:
            filepath: Path to the saved preprocessor
            
        Returns:
            Loaded DataPreprocessor object
        """
        return joblib.load(filepath)
    
    def get_preprocessing_summary(self) -> Dict:
        """
        Get a summary of the preprocessing configuration.
        
        Returns:
            Dictionary with preprocessing settings
        """
        return {
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'binary_features': self.binary_features,
            'scaling_method': self.scaling_method,
            'imputation_strategy': self.imputation_strategy,
            'handle_outliers': self.handle_outliers,
            'outlier_method': self.outlier_method,
            'outlier_threshold': self.outlier_threshold,
            'is_fitted': self._is_fitted,
            'n_features_out': len(self.feature_names) if self.feature_names else 0
        }


if __name__ == "__main__":
    # Demonstration
    from data_ingestion import DataIngestion
    
    # Generate sample data
    ingestion = DataIngestion()
    df = ingestion.generate_synthetic_data(n_samples=500)
    
    print("=" * 50)
    print("Data Preprocessing Demonstration")
    print("=" * 50)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        scaling_method='standard',
        imputation_strategy='median',
        handle_outliers=True
    )
    
    # Analyze missing values
    print("\nMissing Value Analysis:")
    missing_stats = preprocessor.analyze_missing_values(df)
    if len(missing_stats) > 0:
        print(missing_stats)
    else:
        print("  No missing values detected")
    
    # Split data
    print("\nSplitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        df, target_col='diabetes_risk'
    )
    
    # Fit and transform
    print("\nFitting preprocessor on training data...")
    preprocessor.fit(X_train)
    
    print("\nTransforming data...")
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"\nProcessed feature shape: {X_train_processed.shape}")
    print(f"Feature names: {preprocessor.feature_names[:5]}...")
    
    # Summary
    print("\nPreprocessing Summary:")
    summary = preprocessor.get_preprocessing_summary()
    for key, value in summary.items():
        if isinstance(value, list) and len(value) > 3:
            print(f"  {key}: {value[:3]}... ({len(value)} total)")
        else:
            print(f"  {key}: {value}")
