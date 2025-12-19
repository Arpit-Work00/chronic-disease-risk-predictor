"""
Feature Engineering Module for Chronic Disease Risk Prediction System

This module handles:
- Derived feature creation
- Feature interactions
- Clinical risk score calculations
- Feature selection based on importance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import warnings


class FeatureEngineer:
    """
    Feature engineering pipeline for clinical data.
    
    Creates clinically meaningful derived features and performs
    feature selection to identify the most predictive variables.
    """
    
    def __init__(self, create_interactions: bool = True, create_risk_scores: bool = True):
        """
        Initialize the feature engineer.
        
        Args:
            create_interactions: Whether to create feature interactions
            create_risk_scores: Whether to create clinical risk score features
        """
        self.create_interactions = create_interactions
        self.create_risk_scores = create_risk_scores
        self.selected_features = []
        self.feature_importance = {}
        self._is_fitted = False
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from raw clinical data.
        
        Args:
            df: Input DataFrame with raw features
            
        Returns:
            DataFrame with additional derived features
        """
        df = df.copy()
        
        # Age-based features
        if 'age' in df.columns:
            df['age_group'] = pd.cut(
                df['age'],
                bins=[0, 30, 45, 60, 75, 120],
                labels=['young', 'middle', 'senior', 'elderly', 'very_elderly']
            ).astype(str)
            df['age_squared'] = df['age'] ** 2
            df['age_decade'] = (df['age'] // 10) * 10
        
        # BMI categories (WHO classification)
        if 'bmi' in df.columns:
            df['bmi_category'] = pd.cut(
                df['bmi'],
                bins=[0, 18.5, 25, 30, 35, 100],
                labels=['underweight', 'normal', 'overweight', 'obese', 'severely_obese']
            ).astype(str)
            df['bmi_squared'] = df['bmi'] ** 2
            df['is_obese'] = (df['bmi'] >= 30).astype(int)
        
        # Blood pressure features
        if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
            df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
            df['mean_arterial_pressure'] = df['diastolic_bp'] + (df['pulse_pressure'] / 3)
            
            # Hypertension stages
            df['hypertension_stage'] = 0
            df.loc[(df['systolic_bp'] >= 120) | (df['diastolic_bp'] >= 80), 'hypertension_stage'] = 1
            df.loc[(df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 80), 'hypertension_stage'] = 2
            df.loc[(df['systolic_bp'] >= 140) | (df['diastolic_bp'] >= 90), 'hypertension_stage'] = 3
            df.loc[(df['systolic_bp'] >= 180) | (df['diastolic_bp'] >= 120), 'hypertension_stage'] = 4
        
        # Glycemic features
        if 'fasting_glucose' in df.columns:
            df['glucose_category'] = pd.cut(
                df['fasting_glucose'],
                bins=[0, 100, 126, 200, 1000],
                labels=['normal', 'prediabetic', 'diabetic', 'severe_diabetic']
            ).astype(str)
            df['is_prediabetic'] = ((df['fasting_glucose'] >= 100) & (df['fasting_glucose'] < 126)).astype(int)
            df['is_diabetic_range'] = (df['fasting_glucose'] >= 126).astype(int)
        
        if 'hba1c' in df.columns:
            df['hba1c_category'] = pd.cut(
                df['hba1c'],
                bins=[0, 5.7, 6.5, 8.0, 20],
                labels=['normal', 'prediabetic', 'diabetic', 'poorly_controlled']
            ).astype(str)
        
        # Lipid ratios (important cardiovascular risk markers)
        if 'total_cholesterol' in df.columns and 'hdl_cholesterol' in df.columns:
            df['tc_hdl_ratio'] = df['total_cholesterol'] / df['hdl_cholesterol'].clip(lower=1)
            df['non_hdl_cholesterol'] = df['total_cholesterol'] - df['hdl_cholesterol']
        
        if 'ldl_cholesterol' in df.columns and 'hdl_cholesterol' in df.columns:
            df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / df['hdl_cholesterol'].clip(lower=1)
        
        if 'triglycerides' in df.columns and 'hdl_cholesterol' in df.columns:
            df['tg_hdl_ratio'] = df['triglycerides'] / df['hdl_cholesterol'].clip(lower=1)
        
        # Dyslipidemia indicator
        if all(col in df.columns for col in ['total_cholesterol', 'ldl_cholesterol', 'hdl_cholesterol', 'triglycerides']):
            df['dyslipidemia'] = (
                (df['total_cholesterol'] > 200) |
                (df['ldl_cholesterol'] > 130) |
                (df['hdl_cholesterol'] < 40) |
                (df['triglycerides'] > 150)
            ).astype(int)
        
        # Kidney function features
        if 'egfr' in df.columns:
            df['ckd_stage'] = pd.cut(
                df['egfr'],
                bins=[0, 15, 30, 45, 60, 90, 200],
                labels=['stage5', 'stage4', 'stage3b', 'stage3a', 'stage2', 'stage1']
            ).astype(str)
            df['low_egfr'] = (df['egfr'] < 60).astype(int)
        
        if 'creatinine' in df.columns:
            df['elevated_creatinine'] = (df['creatinine'] > 1.2).astype(int)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create clinically meaningful feature interactions.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features added
        """
        df = df.copy()
        
        # Age-BMI interaction (metabolic risk)
        if 'age' in df.columns and 'bmi' in df.columns:
            df['age_bmi_interaction'] = df['age'] * df['bmi'] / 100
        
        # Age-BP interaction (cardiovascular risk)
        if 'age' in df.columns and 'systolic_bp' in df.columns:
            df['age_bp_interaction'] = df['age'] * df['systolic_bp'] / 1000
        
        # Glucose-BMI interaction (diabetes risk)
        if 'fasting_glucose' in df.columns and 'bmi' in df.columns:
            df['glucose_bmi_interaction'] = df['fasting_glucose'] * df['bmi'] / 1000
        
        # HbA1c-Glucose consistency check
        if 'hba1c' in df.columns and 'fasting_glucose' in df.columns:
            # Estimated average glucose from HbA1c
            df['estimated_avg_glucose'] = 28.7 * df['hba1c'] - 46.7
            df['glucose_hba1c_discrepancy'] = abs(df['fasting_glucose'] - df['estimated_avg_glucose'])
        
        # Combined metabolic syndrome indicator
        metabolic_cols = ['is_obese', 'hypertension_stage', 'is_prediabetic', 'dyslipidemia']
        available_cols = [c for c in metabolic_cols if c in df.columns]
        if len(available_cols) >= 3:
            df['metabolic_syndrome_score'] = df[available_cols].sum(axis=1)
            df['has_metabolic_syndrome'] = (df['metabolic_syndrome_score'] >= 3).astype(int)
        
        return df
    
    def create_clinical_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create standardized clinical risk scores.
        
        Implements simplified versions of established clinical risk calculators.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with clinical risk score features
        """
        df = df.copy()
        
        # Simplified Framingham-like CVD risk score
        if all(col in df.columns for col in ['age', 'gender', 'total_cholesterol', 
                                              'hdl_cholesterol', 'systolic_bp', 'smoking_status']):
            cvd_score = np.zeros(len(df))
            
            # Age points
            cvd_score += np.where(df['age'] >= 60, 3, np.where(df['age'] >= 50, 2, np.where(df['age'] >= 40, 1, 0)))
            
            # Cholesterol points
            cvd_score += np.where(df['total_cholesterol'] >= 240, 2, np.where(df['total_cholesterol'] >= 200, 1, 0))
            
            # HDL points (negative for good HDL)
            cvd_score += np.where(df['hdl_cholesterol'] < 40, 2, np.where(df['hdl_cholesterol'] < 50, 1, 0))
            
            # BP points
            cvd_score += np.where(df['systolic_bp'] >= 160, 3, np.where(df['systolic_bp'] >= 140, 2, np.where(df['systolic_bp'] >= 120, 1, 0)))
            
            # Smoking points
            cvd_score += np.where(df['smoking_status'] == 'Current', 2, np.where(df['smoking_status'] == 'Former', 1, 0))
            
            df['framingham_like_score'] = cvd_score
        
        # Simplified Finnish Diabetes Risk Score (FINDRISC-like)
        if all(col in df.columns for col in ['age', 'bmi', 'physical_activity_level', 
                                              'family_history_diabetes']):
            diabetes_score = np.zeros(len(df))
            
            # Age points
            diabetes_score += np.where(df['age'] >= 64, 4, np.where(df['age'] >= 55, 3, np.where(df['age'] >= 45, 2, 0)))
            
            # BMI points
            diabetes_score += np.where(df['bmi'] >= 30, 3, np.where(df['bmi'] >= 25, 1, 0))
            
            # Physical activity
            diabetes_score += np.where(df['physical_activity_level'] == 'Sedentary', 2, 0)
            
            # Family history
            diabetes_score += np.where(df['family_history_diabetes'] == 1, 3, 0)
            
            df['findrisc_like_score'] = diabetes_score
        
        # Kidney disease risk score
        if all(col in df.columns for col in ['age', 'egfr', 'hypertension_diagnosed', 'family_history_diabetes']):
            kidney_score = np.zeros(len(df))
            
            # eGFR (lower is worse)
            kidney_score += np.where(df['egfr'] < 30, 4, np.where(df['egfr'] < 60, 2, np.where(df['egfr'] < 90, 1, 0)))
            
            # Age
            kidney_score += np.where(df['age'] >= 65, 2, np.where(df['age'] >= 50, 1, 0))
            
            # Hypertension
            kidney_score += np.where(df['hypertension_diagnosed'] == 1, 2, 0)
            
            # Diabetes/family history
            kidney_score += np.where(df['family_history_diabetes'] == 1, 1, 0)
            
            df['kidney_risk_score'] = kidney_score
        
        return df
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = None,
        selection_method: str = 'mutual_info'
    ) -> 'FeatureEngineer':
        """
        Fit feature selection on the training data.
        
        Args:
            X: Training features DataFrame
            y: Training target Series
            n_features: Number of features to select (None = all)
            selection_method: 'mutual_info', 'f_classif', or 'rfe'
            
        Returns:
            self
        """
        # Get numerical columns only for feature selection
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numerical = X[numerical_cols].fillna(0)
        
        if n_features is None:
            n_features = len(numerical_cols)
        
        # Compute feature importance
        if selection_method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=min(n_features, len(numerical_cols)))
            selector.fit(X_numerical, y)
            scores = selector.scores_
        elif selection_method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=min(n_features, len(numerical_cols)))
            selector.fit(X_numerical, y)
            scores = selector.scores_
        else:  # RFE
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_numerical, y)
            scores = rf.feature_importances_
        
        # Store feature importance
        self.feature_importance = dict(zip(numerical_cols, scores))
        
        # Sort and select top features
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        self.selected_features = [f[0] for f in sorted_features[:n_features]]
        
        self._is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering transformations.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame with derived features
        """
        # Create derived features
        df = self.create_derived_features(df)
        
        # Create interaction features
        if self.create_interactions:
            df = self.create_interaction_features(df)
        
        # Create clinical risk scores
        if self.create_risk_scores:
            df = self.create_clinical_risk_scores(df)
        
        return df
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = None,
        selection_method: str = 'mutual_info'
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.
        """
        transformed = self.transform(X)
        self.fit(transformed, y, n_features, selection_method)
        return transformed
    
    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get the top N most important features.
        
        Args:
            n: Number of features to return
            
        Returns:
            List of (feature_name, importance_score) tuples
        """
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Get derived features grouped by category.
        
        Returns:
            Dictionary mapping category names to feature lists
        """
        return {
            'age_derived': ['age_group', 'age_squared', 'age_decade'],
            'bmi_derived': ['bmi_category', 'bmi_squared', 'is_obese'],
            'bp_derived': ['pulse_pressure', 'mean_arterial_pressure', 'hypertension_stage'],
            'glucose_derived': ['glucose_category', 'is_prediabetic', 'is_diabetic_range', 'hba1c_category'],
            'lipid_derived': ['tc_hdl_ratio', 'non_hdl_cholesterol', 'ldl_hdl_ratio', 'tg_hdl_ratio', 'dyslipidemia'],
            'kidney_derived': ['ckd_stage', 'low_egfr', 'elevated_creatinine'],
            'interaction': ['age_bmi_interaction', 'age_bp_interaction', 'glucose_bmi_interaction', 
                           'glucose_hba1c_discrepancy', 'metabolic_syndrome_score', 'has_metabolic_syndrome'],
            'risk_scores': ['framingham_like_score', 'findrisc_like_score', 'kidney_risk_score']
        }


if __name__ == "__main__":
    # Demonstration
    from data_ingestion import DataIngestion
    
    # Generate sample data
    ingestion = DataIngestion()
    df = ingestion.generate_synthetic_data(n_samples=500)
    
    print("=" * 50)
    print("Feature Engineering Demonstration")
    print("=" * 50)
    
    print(f"\nOriginal features: {len(df.columns)}")
    print(f"Original columns: {list(df.columns)}")
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(
        create_interactions=True,
        create_risk_scores=True
    )
    
    # Transform data
    df_engineered = feature_engineer.transform(df)
    
    print(f"\nAfter feature engineering: {len(df_engineered.columns)} features")
    
    # Show new features
    new_features = set(df_engineered.columns) - set(df.columns)
    print(f"\nNew derived features ({len(new_features)}):")
    for feature in sorted(new_features):
        print(f"  - {feature}")
    
    # Fit feature selection
    print("\nFitting feature selection...")
    target_col = 'diabetes_risk'
    feature_cols = [c for c in df_engineered.columns if c not in ['patient_id', 'diabetes_risk', 'cvd_risk', 'kidney_disease_risk']]
    X = df_engineered[feature_cols]
    y = df_engineered[target_col]
    
    # Convert categorical to numeric for feature selection
    X_numeric = X.select_dtypes(include=[np.number])
    feature_engineer.fit(X_numeric, y, n_features=15, selection_method='mutual_info')
    
    print("\nTop 15 most important features:")
    for feature, importance in feature_engineer.get_top_features(15):
        print(f"  {feature}: {importance:.4f}")
