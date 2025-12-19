"""
Explainability Module for Chronic Disease Risk Prediction System

This module handles:
- SHAP-based feature explanations
- Global and local feature importance
- Patient-level explanation generation
- Visualization of explanations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import warnings


class ExplainabilityEngine:
    """
    SHAP-based explainability for clinical risk predictions.
    
    Provides:
    - Global feature importance
    - Individual prediction explanations
    - Feature contribution visualization
    - Clinical interpretation of risk factors
    """
    
    # Clinical interpretation templates
    FEATURE_INTERPRETATIONS = {
        'age': {
            'high': 'Advanced age increases disease risk',
            'low': 'Younger age is associated with lower risk'
        },
        'bmi': {
            'high': 'Elevated BMI indicates higher metabolic risk',
            'low': 'BMI within healthy range reduces risk'
        },
        'systolic_bp': {
            'high': 'Elevated blood pressure increases cardiovascular risk',
            'low': 'Blood pressure within normal range is favorable'
        },
        'fasting_glucose': {
            'high': 'Elevated fasting glucose suggests impaired glucose metabolism',
            'low': 'Normal glucose levels reduce diabetes risk'
        },
        'hba1c': {
            'high': 'Elevated HbA1c indicates poor glycemic control',
            'low': 'Normal HbA1c suggests good glycemic control'
        },
        'total_cholesterol': {
            'high': 'High cholesterol increases cardiovascular risk',
            'low': 'Cholesterol levels within normal range'
        },
        'hdl_cholesterol': {
            'high': 'High HDL (good cholesterol) is protective',
            'low': 'Low HDL increases cardiovascular risk'
        },
        'ldl_cholesterol': {
            'high': 'High LDL (bad cholesterol) increases risk',
            'low': 'LDL within target range is favorable'
        },
        'triglycerides': {
            'high': 'Elevated triglycerides indicate metabolic risk',
            'low': 'Normal triglyceride levels are favorable'
        },
        'egfr': {
            'high': 'Good kidney function (high eGFR) is favorable',
            'low': 'Reduced eGFR indicates kidney disease risk'
        },
        'creatinine': {
            'high': 'Elevated creatinine may indicate kidney stress',
            'low': 'Normal creatinine suggests healthy kidney function'
        },
        'smoking_status_encoded': {
            'high': 'Current/former smoking increases risk',
            'low': 'Non-smoking status is protective'
        },
        'family_history_diabetes': {
            'high': 'Family history increases diabetes risk',
            'low': 'No family history is favorable'
        },
        'family_history_cvd': {
            'high': 'Family history increases cardiovascular risk',
            'low': 'No family history is favorable'
        },
        'hypertension_diagnosed': {
            'high': 'Diagnosed hypertension increases risk',
            'low': 'No hypertension diagnosis is favorable'
        }
    }
    
    def __init__(
        self,
        model,
        feature_names: List[str],
        model_type: str = 'tree',
        background_data: pd.DataFrame = None,
        max_samples: int = 100
    ):
        """
        Initialize the explainability engine.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            model_type: 'tree' for tree-based models, 'linear' for linear models, 'kernel' for others
            background_data: Background data for SHAP (optional)
            max_samples: Maximum samples for background data
        """
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.max_samples = max_samples
        
        # Handle calibrated models
        self._underlying_model = self._get_underlying_model(model)
        
        # Initialize SHAP explainer
        self.explainer = None
        self.background_data = background_data
        
        # Cache for computed values
        self._shap_values_cache = {}
        self._global_importance = None
    
    def _get_underlying_model(self, model):
        """Extract the underlying model from calibrated wrapper."""
        if hasattr(model, 'estimator'):
            return model.estimator
        elif hasattr(model, 'base_estimator'):
            return model.base_estimator
        elif hasattr(model, 'calibrated_classifiers_'):
            # For CalibratedClassifierCV
            return model.calibrated_classifiers_[0].estimator
        return model
    
    def initialize_explainer(self, X: pd.DataFrame) -> None:
        """
        Initialize the SHAP explainer with background data.
        
        Args:
            X: Training data for background
        """
        # Sample background data if needed
        if len(X) > self.max_samples:
            background = X.sample(n=self.max_samples, random_state=42)
        else:
            background = X
        
        self.background_data = background
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if self.model_type == 'tree':
                try:
                    self.explainer = shap.TreeExplainer(self._underlying_model)
                except Exception:
                    # Fall back to kernel explainer
                    self.explainer = shap.KernelExplainer(
                        self.model.predict_proba,
                        background.values
                    )
            elif self.model_type == 'linear':
                self.explainer = shap.LinearExplainer(
                    self._underlying_model,
                    background.values
                )
            else:
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba,
                    background.values
                )
    
    def compute_shap_values(
        self,
        X: pd.DataFrame,
        check_additivity: bool = False
    ) -> np.ndarray:
        """
        Compute SHAP values for given samples.
        
        Args:
            X: Samples to explain
            check_additivity: Whether to check SHAP value additivity
            
        Returns:
            SHAP values array
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer() first.")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Only TreeExplainer supports check_additivity
            if self.model_type == 'tree' and hasattr(self.explainer, 'shap_values'):
                try:
                    shap_values = self.explainer.shap_values(
                        X.values if hasattr(X, 'values') else X,
                        check_additivity=check_additivity
                    )
                except TypeError:
                    # Fallback if check_additivity is not supported
                    shap_values = self.explainer.shap_values(
                        X.values if hasattr(X, 'values') else X
                    )
            else:
                shap_values = self.explainer.shap_values(
                    X.values if hasattr(X, 'values') else X
                )
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # For binary classification, take positive class
            shap_values = shap_values[1]
        
        return shap_values
    
    def get_global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Compute global feature importance from SHAP values.
        
        Args:
            X: Data to compute importance from
            
        Returns:
            DataFrame with feature importances
        """
        shap_values = self.compute_shap_values(X)
        
        # Mean absolute SHAP value per feature
        importance = np.abs(shap_values).mean(axis=0)
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Normalize to sum to 1
        df['importance_normalized'] = df['importance'] / df['importance'].sum()
        
        self._global_importance = df
        return df
    
    def explain_prediction(
        self,
        X_single: pd.DataFrame,
        top_n: int = 5
    ) -> Dict:
        """
        Generate explanation for a single prediction.
        
        Args:
            X_single: Single sample to explain (1 row)
            top_n: Number of top contributing features to show
            
        Returns:
            Dictionary with prediction explanation
        """
        if len(X_single) != 1:
            X_single = X_single.iloc[[0]]
        
        # Get SHAP values
        shap_values = self.compute_shap_values(X_single)[0]
        
        # Get prediction probability
        prob = self.model.predict_proba(X_single)[0, 1]
        
        # Sort features by absolute SHAP value
        feature_contributions = []
        for i, (feature, shap_val) in enumerate(zip(self.feature_names, shap_values)):
            feature_value = X_single.iloc[0, i]
            
            # Determine direction
            direction = 'increases risk' if shap_val > 0 else 'decreases risk'
            
            # Get clinical interpretation
            interpretation = self._get_interpretation(feature, shap_val)
            
            feature_contributions.append({
                'feature': feature,
                'value': float(feature_value),
                'shap_value': float(shap_val),
                'contribution': float(abs(shap_val)),
                'direction': direction,
                'interpretation': interpretation
            })
        
        # Sort by absolute contribution
        feature_contributions.sort(key=lambda x: x['contribution'], reverse=True)
        
        # Get expected value (base rate)
        if hasattr(self.explainer, 'expected_value'):
            expected_value = self.explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
        else:
            expected_value = 0.5
        
        return {
            'risk_score': float(prob),
            'base_risk': float(expected_value),
            'total_shap_contribution': float(sum(shap_values)),
            'top_contributing_factors': feature_contributions[:top_n],
            'all_contributions': feature_contributions
        }
    
    def _get_interpretation(self, feature: str, shap_value: float) -> str:
        """
        Get clinical interpretation for a feature contribution.
        """
        # Check for exact match
        if feature in self.FEATURE_INTERPRETATIONS:
            interpretations = self.FEATURE_INTERPRETATIONS[feature]
            return interpretations['high'] if shap_value > 0 else interpretations['low']
        
        # Check for partial match
        for key in self.FEATURE_INTERPRETATIONS:
            if key in feature:
                interpretations = self.FEATURE_INTERPRETATIONS[key]
                return interpretations['high'] if shap_value > 0 else interpretations['low']
        
        # Default interpretation
        if shap_value > 0:
            return f"Higher {feature} increases predicted risk"
        else:
            return f"Lower {feature} decreases predicted risk"
    
    def generate_patient_report(
        self,
        patient_data: pd.DataFrame,
        patient_id: str = None
    ) -> str:
        """
        Generate a human-readable explanation report for a patient.
        
        Args:
            patient_data: Single patient's data
            patient_id: Optional patient identifier
            
        Returns:
            Formatted explanation report
        """
        explanation = self.explain_prediction(patient_data)
        
        report = []
        report.append("=" * 60)
        report.append("PATIENT RISK ASSESSMENT REPORT")
        if patient_id:
            report.append(f"Patient ID: {patient_id}")
        report.append("=" * 60)
        
        # Risk score
        risk_score = explanation['risk_score']
        if risk_score < 0.2:
            risk_category = "LOW"
            color_indicator = "🟢"
        elif risk_score < 0.5:
            risk_category = "MODERATE"
            color_indicator = "🟡"
        elif risk_score < 0.8:
            risk_category = "HIGH"
            color_indicator = "🟠"
        else:
            risk_category = "VERY HIGH"
            color_indicator = "🔴"
        
        report.append(f"\n{color_indicator} RISK SCORE: {risk_score:.1%} ({risk_category})")
        report.append(f"   Population baseline: {explanation['base_risk']:.1%}")
        
        # Contributing factors
        report.append("\n--- TOP CONTRIBUTING FACTORS ---")
        for i, factor in enumerate(explanation['top_contributing_factors'], 1):
            direction_arrow = "↑" if factor['shap_value'] > 0 else "↓"
            report.append(f"\n{i}. {factor['feature']}")
            report.append(f"   Value: {factor['value']:.2f}")
            report.append(f"   Impact: {direction_arrow} {factor['direction']} (contribution: {factor['contribution']:.3f})")
            report.append(f"   → {factor['interpretation']}")
        
        # Summary
        report.append("\n" + "=" * 60)
        report.append("Note: This risk assessment is intended to support clinical")
        report.append("decision-making and should not replace medical judgment.")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def plot_feature_importance(
        self,
        X: pd.DataFrame,
        top_n: int = 15,
        save_path: str = None
    ) -> None:
        """
        Plot global feature importance bar chart.
        """
        if self._global_importance is None:
            self.get_global_importance(X)
        
        plt.figure(figsize=(10, 8))
        
        top_features = self._global_importance.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance_normalized'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Mean |SHAP value|', fontsize=12)
        plt.title('Global Feature Importance (SHAP)', fontsize=14)
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.close()
    
    def plot_waterfall(
        self,
        X_single: pd.DataFrame,
        save_path: str = None
    ) -> None:
        """
        Plot SHAP waterfall chart for a single prediction.
        """
        if len(X_single) != 1:
            X_single = X_single.iloc[[0]]
        
        shap_values = self.compute_shap_values(X_single)
        
        # Get expected value
        if hasattr(self.explainer, 'expected_value'):
            expected_value = self.explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
        else:
            expected_value = 0
        
        plt.figure(figsize=(10, 8))
        
        # Create SHAP Explanation object
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=expected_value,
            data=X_single.values[0],
            feature_names=self.feature_names
        )
        
        shap.plots.waterfall(explanation, show=False)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Waterfall plot saved to {save_path}")
        
        plt.close()
    
    def plot_summary(
        self,
        X: pd.DataFrame,
        plot_type: str = 'dot',
        max_display: int = 15,
        save_path: str = None
    ) -> None:
        """
        Plot SHAP summary plot.
        
        Args:
            X: Data to summarize
            plot_type: 'dot' or 'bar'
            max_display: Maximum features to display
            save_path: Path to save plot
        """
        shap_values = self.compute_shap_values(X)
        
        plt.figure(figsize=(10, 8))
        
        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names,
            plot_type=plot_type,
            max_display=max_display,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Summary plot saved to {save_path}")
        
        plt.close()


if __name__ == "__main__":
    # Demonstration
    from data_ingestion import DataIngestion
    from preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer
    from model_training import ModelTrainer
    
    print("=" * 60)
    print("Explainability Demonstration")
    print("=" * 60)
    
    # Generate and prepare data
    print("\n1. Preparing data...")
    ingestion = DataIngestion()
    df = ingestion.generate_synthetic_data(n_samples=500)
    
    feature_engineer = FeatureEngineer()
    df = feature_engineer.transform(df)
    
    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        df, target_col='diabetes_risk'
    )
    
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Train a model
    print("\n2. Training model...")
    trainer = ModelTrainer(algorithms=['xgboost'], cv_folds=3)
    models = trainer.train_all_models(X_train_processed, y_train, calibrate=False)
    model = models['xgboost']
    
    # Initialize explainer
    print("\n3. Initializing explainability engine...")
    explainer = ExplainabilityEngine(
        model=model,
        feature_names=list(X_train_processed.columns),
        model_type='tree'
    )
    explainer.initialize_explainer(X_train_processed)
    
    # Global importance
    print("\n4. Computing global feature importance...")
    importance = explainer.get_global_importance(X_test_processed)
    print("\nTop 10 most important features:")
    print(importance.head(10).to_string(index=False))
    
    # Individual explanation
    print("\n5. Generating patient report...")
    sample_patient = X_test_processed.iloc[[0]]
    report = explainer.generate_patient_report(sample_patient, patient_id="TEST-001")
    print(report)
