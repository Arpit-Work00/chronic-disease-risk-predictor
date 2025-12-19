"""
Test Suite for Chronic Disease Risk Prediction System
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_ingestion import DataIngestion
from preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
from explainability import ExplainabilityEngine
from uncertainty import UncertaintyEstimator
from bias_analysis import BiasAnalyzer


class TestDataIngestion:
    """Tests for data ingestion module."""
    
    def test_synthetic_data_generation(self):
        """Test that synthetic data is generated correctly."""
        ingestion = DataIngestion()
        df = ingestion.generate_synthetic_data(n_samples=100)
        
        assert len(df) == 100
        assert 'patient_id' in df.columns
        assert 'age' in df.columns
        assert 'diabetes_risk' in df.columns
    
    def test_synthetic_data_ranges(self):
        """Test that synthetic data values are within expected ranges."""
        ingestion = DataIngestion()
        df = ingestion.generate_synthetic_data(n_samples=500)
        
        assert df['age'].min() >= 18
        assert df['age'].max() <= 95
        assert df['bmi'].min() >= 12
        assert df['bmi'].max() <= 70
        assert set(df['gender'].unique()).issubset({'Male', 'Female', 'Other'})
    
    def test_data_validation(self):
        """Test data validation functionality."""
        ingestion = DataIngestion()
        df = ingestion.generate_synthetic_data(n_samples=100)
        
        is_valid, errors = ingestion.validate_data(df)
        assert is_valid
        assert len(errors) == 0


class TestPreprocessing:
    """Tests for preprocessing module."""
    
    @pytest.fixture
    def sample_data(self):
        ingestion = DataIngestion()
        return ingestion.generate_synthetic_data(n_samples=200)
    
    def test_preprocessor_fit_transform(self, sample_data):
        """Test preprocessor fit and transform."""
        preprocessor = DataPreprocessor()
        
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
            sample_data, target_col='diabetes_risk'
        )
        
        preprocessor.fit(X_train)
        X_transformed = preprocessor.transform(X_train)
        
        assert len(X_transformed) == len(X_train)
        assert not X_transformed.isnull().any().any()
    
    def test_split_proportions(self, sample_data):
        """Test that data split proportions are correct."""
        preprocessor = DataPreprocessor()
        
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
            sample_data, target_col='diabetes_risk',
            test_size=0.2, val_size=0.1
        )
        
        total = len(sample_data)
        assert abs(len(X_test) / total - 0.2) < 0.05
        assert abs(len(X_val) / total - 0.1) < 0.05


class TestFeatureEngineering:
    """Tests for feature engineering module."""
    
    @pytest.fixture
    def sample_data(self):
        ingestion = DataIngestion()
        return ingestion.generate_synthetic_data(n_samples=100)
    
    def test_derived_features_created(self, sample_data):
        """Test that derived features are created."""
        engineer = FeatureEngineer()
        df_transformed = engineer.transform(sample_data)
        
        # Check for new features
        original_cols = set(sample_data.columns)
        new_cols = set(df_transformed.columns) - original_cols
        
        assert len(new_cols) > 0
        assert 'bmi_category' in df_transformed.columns
        assert 'age_group' in df_transformed.columns
    
    def test_interaction_features(self, sample_data):
        """Test interaction feature creation."""
        engineer = FeatureEngineer(create_interactions=True)
        df_transformed = engineer.transform(sample_data)
        
        assert 'age_bmi_interaction' in df_transformed.columns


class TestModelTraining:
    """Tests for model training module."""
    
    @pytest.fixture
    def prepared_data(self):
        ingestion = DataIngestion()
        df = ingestion.generate_synthetic_data(n_samples=300)
        
        engineer = FeatureEngineer()
        df = engineer.transform(df)
        
        preprocessor = DataPreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
            df, target_col='diabetes_risk'
        )
        
        preprocessor.fit(X_train)
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        return X_train_processed, X_test_processed, y_train, y_test
    
    def test_model_training(self, prepared_data):
        """Test that model trains successfully."""
        X_train, X_test, y_train, y_test = prepared_data
        
        trainer = ModelTrainer(
            algorithms=['logistic_regression'],
            cv_folds=3
        )
        
        models = trainer.train_all_models(X_train, y_train, tune_hyperparameters=False)
        
        assert 'logistic_regression' in models
        assert hasattr(models['logistic_regression'], 'predict_proba')
    
    def test_model_predictions(self, prepared_data):
        """Test that model makes valid predictions."""
        X_train, X_test, y_train, y_test = prepared_data
        
        trainer = ModelTrainer(algorithms=['logistic_regression'], cv_folds=3)
        models = trainer.train_all_models(X_train, y_train, tune_hyperparameters=False)
        
        model = models['logistic_regression']
        predictions = model.predict_proba(X_test)
        
        assert predictions.shape[0] == len(X_test)
        assert predictions.shape[1] == 2
        assert (predictions >= 0).all() and (predictions <= 1).all()


class TestModelEvaluation:
    """Tests for model evaluation module."""
    
    @pytest.fixture
    def trained_model(self):
        ingestion = DataIngestion()
        df = ingestion.generate_synthetic_data(n_samples=300)
        
        engineer = FeatureEngineer()
        df = engineer.transform(df)
        
        preprocessor = DataPreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
            df, target_col='diabetes_risk'
        )
        
        preprocessor.fit(X_train)
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        trainer = ModelTrainer(algorithms=['logistic_regression'], cv_folds=3)
        models = trainer.train_all_models(X_train_processed, y_train, tune_hyperparameters=False)
        
        return models['logistic_regression'], X_test_processed, y_test
    
    def test_evaluation_metrics(self, trained_model):
        """Test that evaluation metrics are computed."""
        model, X_test, y_test = trained_model
        
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_model(model, X_test, y_test)
        
        assert 'auc_roc' in results
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 0 <= results['auc_roc'] <= 1


class TestExplainability:
    """Tests for explainability module."""
    
    @pytest.fixture
    def trained_model_and_data(self):
        ingestion = DataIngestion()
        df = ingestion.generate_synthetic_data(n_samples=200)
        
        engineer = FeatureEngineer()
        df = engineer.transform(df)
        
        preprocessor = DataPreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
            df, target_col='diabetes_risk'
        )
        
        preprocessor.fit(X_train)
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        trainer = ModelTrainer(algorithms=['logistic_regression'], cv_folds=3)
        models = trainer.train_all_models(X_train_processed, y_train, 
                                          tune_hyperparameters=False, calibrate=False)
        
        return models['logistic_regression'], X_train_processed, X_test_processed
    
    def test_explainer_initialization(self, trained_model_and_data):
        """Test explainer initialization."""
        model, X_train, X_test = trained_model_and_data
        
        explainer = ExplainabilityEngine(
            model=model,
            feature_names=list(X_train.columns),
            model_type='linear'
        )
        explainer.initialize_explainer(X_train)
        
        assert explainer.explainer is not None
    
    def test_prediction_explanation(self, trained_model_and_data):
        """Test individual prediction explanation."""
        model, X_train, X_test = trained_model_and_data
        
        explainer = ExplainabilityEngine(
            model=model,
            feature_names=list(X_train.columns),
            model_type='linear'
        )
        explainer.initialize_explainer(X_train)
        
        explanation = explainer.explain_prediction(X_test.iloc[[0]])
        
        assert 'risk_score' in explanation
        assert 'top_contributing_factors' in explanation
        assert len(explanation['top_contributing_factors']) > 0


class TestUncertainty:
    """Tests for uncertainty estimation module."""
    
    @pytest.fixture
    def trained_model(self):
        ingestion = DataIngestion()
        df = ingestion.generate_synthetic_data(n_samples=200)
        
        preprocessor = DataPreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
            df, target_col='diabetes_risk'
        )
        
        preprocessor.fit(X_train)
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        trainer = ModelTrainer(algorithms=['logistic_regression'], cv_folds=3)
        models = trainer.train_all_models(X_train_processed, y_train, 
                                          tune_hyperparameters=False, calibrate=False)
        
        return models['logistic_regression'], X_test_processed
    
    def test_confidence_intervals(self, trained_model):
        """Test confidence interval computation."""
        model, X_test = trained_model
        
        estimator = UncertaintyEstimator(model=model, confidence_level=0.95)
        
        point_est, lower, upper = estimator.compute_confidence_interval(X_test)
        
        assert len(point_est) == len(X_test)
        assert (lower <= point_est).all()
        assert (point_est <= upper).all()


class TestBiasAnalysis:
    """Tests for bias analysis module."""
    
    def test_group_metrics_computation(self):
        """Test demographic group metrics computation."""
        # Create sample data
        np.random.seed(42)
        n = 200
        
        y_true = pd.Series(np.random.binomial(1, 0.3, n))
        y_pred = np.random.binomial(1, 0.35, n)
        y_proba = np.random.beta(2, 5, n)
        groups = pd.Series(np.random.choice(['A', 'B', 'C'], n))
        
        analyzer = BiasAnalyzer()
        metrics = analyzer.compute_group_metrics(y_true, y_pred, y_proba, groups)
        
        assert len(metrics) == 3  # Three groups
        assert 'accuracy' in metrics.columns
        assert 'tpr' in metrics.columns


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_full_pipeline(self):
        """Test the full prediction pipeline."""
        # This is a simplified integration test
        ingestion = DataIngestion()
        df = ingestion.generate_synthetic_data(n_samples=300)
        
        engineer = FeatureEngineer()
        df = engineer.transform(df)
        
        preprocessor = DataPreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
            df, target_col='diabetes_risk'
        )
        
        preprocessor.fit(X_train)
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        trainer = ModelTrainer(algorithms=['logistic_regression'], cv_folds=3)
        models = trainer.train_all_models(X_train_processed, y_train, 
                                          tune_hyperparameters=False)
        
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_model(
            models['logistic_regression'],
            X_test_processed,
            y_test
        )
        
        assert results['auc_roc'] > 0.5  # Better than random


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
