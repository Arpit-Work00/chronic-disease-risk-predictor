"""
Predictor Module for Chronic Disease Risk Prediction System

Main prediction API that integrates all components:
- Data preprocessing
- Model prediction
- Explainability
- Uncertainty estimation
- Risk categorization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
import joblib
import yaml
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_ingestion import DataIngestion
from preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
from explainability import ExplainabilityEngine
from uncertainty import UncertaintyEstimator
from bias_analysis import BiasAnalyzer


class ChronicDiseaseRiskPredictor:
    """
    Main prediction interface for chronic disease risk assessment.
    
    Provides:
    - Risk score prediction (0-1 probability)
    - Confidence intervals
    - Feature explanations (SHAP)
    - Risk category classification
    - Clinical recommendations
    """
    
    # Risk categories based on probability thresholds
    RISK_CATEGORIES = {
        (0.0, 0.2): {'label': 'Low', 'color': '🟢'},
        (0.2, 0.5): {'label': 'Moderate', 'color': '🟡'},
        (0.5, 0.8): {'label': 'High', 'color': '🟠'},
        (0.8, 1.0): {'label': 'Very High', 'color': '🔴'}
    }
    
    # Clinical recommendations based on risk factors
    RECOMMENDATIONS = {
        'high_glucose': [
            "Monitor fasting blood glucose regularly",
            "Consider HbA1c testing every 3-6 months",
            "Consult endocrinologist if levels remain elevated"
        ],
        'high_bmi': [
            "Aim for gradual weight loss (5-10% of body weight)",
            "Increase physical activity to 150+ minutes/week",
            "Consider nutritional counseling"
        ],
        'high_bp': [
            "Monitor blood pressure at home",
            "Reduce sodium intake",
            "Discuss antihypertensive therapy if lifestyle changes insufficient"
        ],
        'high_cholesterol': [
            "Adopt heart-healthy diet (reduce saturated fats)",
            "Consider statin therapy based on overall CV risk",
            "Recheck lipid panel in 3-6 months"
        ],
        'smoking': [
            "Strongly consider smoking cessation",
            "Ask about nicotine replacement or cessation medications",
            "Refer to smoking cessation program"
        ],
        'low_egfr': [
            "Monitor kidney function regularly",
            "Avoid nephrotoxic medications",
            "Consider nephrology referral if eGFR < 45"
        ],
        'general': [
            "Schedule follow-up appointment for risk reassessment",
            "Maintain healthy lifestyle with regular exercise",
            "Ensure adequate sleep (7-9 hours/night)"
        ]
    }
    
    def __init__(
        self,
        model_path: str = None,
        config_path: str = None,
        target_disease: str = 'diabetes'
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to saved model (optional)
            config_path: Path to config file (optional)
            target_disease: Disease to predict ('diabetes', 'cvd', 'kidney')
        """
        self.target_disease = target_disease
        self.target_column = f'{target_disease}_risk'
        
        # Load config if provided
        self.config = self._load_config(config_path) if config_path else {}
        
        # Components (initialized later)
        self.ingestion = DataIngestion()
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.model = None
        self.explainer = None
        self.uncertainty_estimator = None
        
        self._is_fitted = False
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def fit(
        self,
        data: pd.DataFrame = None,
        generate_synthetic: bool = True,
        n_samples: int = 2000,
        algorithm: str = 'xgboost'
    ) -> 'ChronicDiseaseRiskPredictor':
        """
        Train the complete prediction pipeline.
        
        Args:
            data: Training data (optional)
            generate_synthetic: Whether to generate synthetic data if none provided
            n_samples: Number of synthetic samples
            algorithm: Model algorithm to use
            
        Returns:
            self
        """
        print("=" * 60)
        print(f"Training Chronic Disease Risk Predictor")
        print(f"Target: {self.target_disease}")
        print("=" * 60)
        
        # Step 1: Data ingestion
        print("\n[1/5] Data Ingestion...")
        if data is None and generate_synthetic:
            data = self.ingestion.generate_synthetic_data(n_samples=n_samples)
            print(f"Generated {n_samples} synthetic patient records")
        elif data is None:
            raise ValueError("No data provided and generate_synthetic=False")
        
        # Step 2: Feature engineering
        print("\n[2/5] Feature Engineering...")
        data = self.feature_engineer.transform(data)
        print(f"Created {len(data.columns)} features")
        
        # Step 3: Preprocessing
        print("\n[3/5] Preprocessing...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.split_data(
            data, target_col=self.target_column
        )
        
        self.preprocessor.fit(X_train)
        X_train_processed = self.preprocessor.transform(X_train)
        X_val_processed = self.preprocessor.transform(X_val)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Step 4: Model training
        print("\n[4/5] Model Training...")
        trainer = ModelTrainer(
            algorithms=[algorithm],
            cv_folds=5,
            scoring='roc_auc'
        )
        models = trainer.train_all_models(X_train_processed, y_train, calibrate=True)
        self.model = models[algorithm]
        
        # Step 5: Setup explainability and uncertainty
        print("\n[5/5] Setting up Explainability & Uncertainty...")
        
        # Explainability
        model_type = 'tree' if algorithm in ['random_forest', 'xgboost', 'lightgbm'] else 'linear'
        self.explainer = ExplainabilityEngine(
            model=self.model,
            feature_names=list(X_train_processed.columns),
            model_type=model_type
        )
        self.explainer.initialize_explainer(X_train_processed)
        
        # Uncertainty
        self.uncertainty_estimator = UncertaintyEstimator(
            model=self.model,
            confidence_level=0.95,
            n_bootstrap=100
        )
        self.uncertainty_estimator.calibrate_model(X_val_processed, y_val)
        
        # Evaluate
        print("\n--- Model Evaluation ---")
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_model(self.model, X_test_processed, y_test, algorithm)
        print(f"Test AUC-ROC: {results['auc_roc']:.4f}")
        print(f"Test AUC-PR: {results['auc_pr']:.4f}")
        print(f"Brier Score: {results['brier_score']:.4f}")
        
        self._is_fitted = True
        print("\n✓ Training complete!")
        
        return self
    
    def predict(
        self,
        patient_data: Union[Dict, pd.DataFrame],
        include_explanation: bool = True,
        include_confidence: bool = True,
        include_recommendations: bool = True,
        top_n_factors: int = 5
    ) -> Dict:
        """
        Generate risk prediction with explanations.
        
        Args:
            patient_data: Patient features as dict or DataFrame
            include_explanation: Include SHAP explanations
            include_confidence: Include confidence intervals
            include_recommendations: Include clinical recommendations
            top_n_factors: Number of top contributing factors to show
            
        Returns:
            Comprehensive prediction result dictionary
        """
        if not self._is_fitted:
            raise ValueError("Predictor not fitted. Call fit() first or load a trained model.")
        
        # Convert dict to DataFrame if needed
        if isinstance(patient_data, dict):
            patient_data = pd.DataFrame([patient_data])
        
        # Ensure single row
        if len(patient_data) > 1:
            print(f"Warning: Multiple rows provided. Using first row only.")
            patient_data = patient_data.iloc[[0]]
        
        # Apply feature engineering
        patient_engineered = self.feature_engineer.transform(patient_data)
        
        # Preprocess
        patient_processed = self.preprocessor.transform(patient_engineered)
        
        # Get risk score
        risk_score = float(self.model.predict_proba(patient_processed)[0, 1])
        
        # Determine risk category
        risk_category = self._get_risk_category(risk_score)
        
        # Build result
        result = {
            'risk_score': round(risk_score, 4),
            'risk_percentage': f"{risk_score:.1%}",
            'risk_category': risk_category['label'],
            'risk_indicator': risk_category['color'],
            'target_disease': self.target_disease
        }
        
        # Add confidence interval
        if include_confidence:
            summary = self.uncertainty_estimator.get_prediction_summary(patient_processed)
            result['confidence_interval'] = [
                round(summary['confidence_interval'][0], 4),
                round(summary['confidence_interval'][1], 4)
            ]
            result['confidence_level'] = '95%'
            result['prediction_confidence'] = summary['confidence_label']
        
        # Add explanation
        if include_explanation:
            explanation = self.explainer.explain_prediction(patient_processed, top_n=top_n_factors)
            result['top_contributing_factors'] = [
                {
                    'feature': f['feature'],
                    'contribution': round(f['contribution'], 4),
                    'direction': f['direction'],
                    'interpretation': f['interpretation']
                }
                for f in explanation['top_contributing_factors']
            ]
        
        # Add recommendations
        if include_recommendations:
            result['recommendations'] = self._generate_recommendations(
                patient_data.iloc[0],
                result.get('top_contributing_factors', [])
            )
        
        return result
    
    def predict_batch(
        self,
        patients_data: pd.DataFrame,
        include_explanation: bool = False
    ) -> pd.DataFrame:
        """
        Generate predictions for multiple patients.
        
        Args:
            patients_data: DataFrame with patient features
            include_explanation: Include explanations (slower)
            
        Returns:
            DataFrame with predictions for each patient
        """
        results = []
        
        for idx in range(len(patients_data)):
            patient = patients_data.iloc[[idx]]
            
            result = self.predict(
                patient,
                include_explanation=include_explanation,
                include_confidence=True,
                include_recommendations=False
            )
            
            result['patient_index'] = idx
            results.append(result)
        
        return pd.DataFrame(results)
    
    def _get_risk_category(self, risk_score: float) -> Dict:
        """Map risk score to category."""
        for (low, high), category in self.RISK_CATEGORIES.items():
            if low <= risk_score < high:
                return category
        return self.RISK_CATEGORIES[(0.8, 1.0)]  # Default to highest
    
    def _generate_recommendations(
        self,
        patient_data: pd.Series,
        contributing_factors: List[Dict]
    ) -> List[str]:
        """Generate personalized clinical recommendations."""
        recommendations = []
        
        # Check for specific risk factors in input data
        if 'fasting_glucose' in patient_data and patient_data.get('fasting_glucose', 0) > 126:
            recommendations.extend(self.RECOMMENDATIONS['high_glucose'][:2])
        
        if 'bmi' in patient_data and patient_data.get('bmi', 0) > 30:
            recommendations.extend(self.RECOMMENDATIONS['high_bmi'][:2])
        
        if 'systolic_bp' in patient_data and patient_data.get('systolic_bp', 0) > 140:
            recommendations.extend(self.RECOMMENDATIONS['high_bp'][:2])
        
        if 'total_cholesterol' in patient_data and patient_data.get('total_cholesterol', 0) > 240:
            recommendations.extend(self.RECOMMENDATIONS['high_cholesterol'][:1])
        
        if 'smoking_status' in patient_data and patient_data.get('smoking_status') == 'Current':
            recommendations.extend(self.RECOMMENDATIONS['smoking'][:2])
        
        if 'egfr' in patient_data and patient_data.get('egfr', 100) < 60:
            recommendations.extend(self.RECOMMENDATIONS['low_egfr'][:2])
        
        # Add general recommendations if few specific ones
        if len(recommendations) < 3:
            recommendations.extend(self.RECOMMENDATIONS['general'][:2])
        
        # Deduplicate while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:5]  # Top 5 recommendations
    
    def save_model(self, filepath: str) -> None:
        """
        Save the complete predictor to disk.
        
        Args:
            filepath: Path to save the model
        """
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_engineer': self.feature_engineer,
            'target_disease': self.target_disease,
            'config': self.config
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a saved predictor from disk.
        
        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.preprocessor = model_data['preprocessor']
        self.feature_engineer = model_data['feature_engineer']
        self.target_disease = model_data['target_disease']
        self.config = model_data.get('config', {})
        
        # Reinitialize explainer and uncertainty
        # Note: These need training data to fully initialize
        self._is_fitted = True
        print(f"Model loaded from {filepath}")
    
    def generate_report(
        self,
        patient_data: Union[Dict, pd.DataFrame],
        patient_id: str = None
    ) -> str:
        """
        Generate a complete clinical risk report.
        
        Args:
            patient_data: Patient features
            patient_id: Optional patient identifier
            
        Returns:
            Formatted report string
        """
        result = self.predict(patient_data)
        
        report = []
        report.append("=" * 60)
        report.append("CHRONIC DISEASE RISK ASSESSMENT REPORT")
        report.append("=" * 60)
        
        if patient_id:
            report.append(f"Patient ID: {patient_id}")
        report.append(f"Disease: {self.target_disease.upper()}")
        report.append(f"Assessment Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
        
        report.append("\n" + "-" * 40)
        report.append("RISK ASSESSMENT")
        report.append("-" * 40)
        
        report.append(f"\n{result['risk_indicator']} Risk Score: {result['risk_percentage']}")
        report.append(f"   Category: {result['risk_category']}")
        
        if 'confidence_interval' in result:
            ci = result['confidence_interval']
            report.append(f"   95% Confidence Interval: [{ci[0]:.1%}, {ci[1]:.1%}]")
            report.append(f"   Prediction Confidence: {result['prediction_confidence']}")
        
        if 'top_contributing_factors' in result:
            report.append("\n" + "-" * 40)
            report.append("KEY RISK FACTORS")
            report.append("-" * 40)
            
            for i, factor in enumerate(result['top_contributing_factors'], 1):
                arrow = "↑" if 'increases' in factor['direction'] else "↓"
                report.append(f"\n{i}. {factor['feature']}")
                report.append(f"   {arrow} {factor['direction']}")
                report.append(f"   → {factor['interpretation']}")
        
        if 'recommendations' in result:
            report.append("\n" + "-" * 40)
            report.append("CLINICAL RECOMMENDATIONS")
            report.append("-" * 40)
            
            for i, rec in enumerate(result['recommendations'], 1):
                report.append(f"{i}. {rec}")
        
        report.append("\n" + "=" * 60)
        report.append("DISCLAIMER")
        report.append("=" * 60)
        report.append("This risk assessment is intended to support clinical")
        report.append("decision-making and should not replace medical judgment.")
        report.append("Please consult with a healthcare provider for personalized")
        report.append("medical advice and treatment decisions.")
        
        return "\n".join(report)


def run_demo():
    """Run a demonstration of the risk prediction system."""
    print("\n" + "=" * 60)
    print("CHRONIC DISEASE RISK PREDICTION SYSTEM - DEMO")
    print("=" * 60)
    
    # Initialize and train
    predictor = ChronicDiseaseRiskPredictor(target_disease='diabetes')
    predictor.fit(generate_synthetic=True, n_samples=1500, algorithm='xgboost')
    
    # Create sample patient
    sample_patient = {
        'patient_id': 'DEMO-001',
        'age': 55,
        'gender': 'Male',
        'ethnicity': 'Caucasian',
        'bmi': 32.5,
        'systolic_bp': 145,
        'diastolic_bp': 92,
        'heart_rate': 78,
        'fasting_glucose': 118,
        'hba1c': 6.2,
        'total_cholesterol': 225,
        'hdl_cholesterol': 42,
        'ldl_cholesterol': 145,
        'triglycerides': 190,
        'creatinine': 1.1,
        'egfr': 75,
        'smoking_status': 'Former',
        'alcohol_consumption': 'Moderate',
        'physical_activity_level': 'Light',
        'diet_quality_score': 45,
        'family_history_diabetes': 1,
        'family_history_cvd': 1,
        'family_history_kidney_disease': 0,
        'hypertension_diagnosed': 1,
        'previous_cardiovascular_event': 0
    }
    
    # Generate report
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTION")
    print("=" * 60)
    
    report = predictor.generate_report(sample_patient, patient_id='DEMO-001')
    print(report)
    
    # Also show JSON output
    print("\n" + "=" * 60)
    print("JSON OUTPUT FORMAT")
    print("=" * 60)
    
    result = predictor.predict(sample_patient)
    print(json.dumps(result, indent=2))
    
    return predictor


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Chronic Disease Risk Prediction System')
    parser.add_argument('--demo', action='store_true', help='Run demonstration')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--disease', type=str, default='diabetes',
                       choices=['diabetes', 'cvd', 'kidney'],
                       help='Target disease to predict')
    parser.add_argument('--save-path', type=str, default=None,
                       help='Path to save trained model')
    
    args = parser.parse_args()
    
    if args.demo or args.train:
        predictor = run_demo()
        
        if args.save_path:
            predictor.save_model(args.save_path)
    else:
        print("Chronic Disease Risk Prediction System")
        print("Use --demo to run a demonstration")
        print("Use --help for more options")
