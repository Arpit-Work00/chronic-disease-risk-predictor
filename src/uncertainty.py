"""
Uncertainty Estimation Module for Chronic Disease Risk Prediction System

This module handles:
- Bootstrap confidence intervals
- Probability calibration
- Prediction uncertainty quantification
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.utils import resample
from sklearn.metrics import brier_score_loss
import joblib
import warnings


class UncertaintyEstimator:
    """
    Uncertainty quantification for risk predictions.
    
    Provides:
    - Bootstrap-based confidence intervals
    - Probability calibration assessment
    - Platt scaling and isotonic regression
    """
    
    def __init__(
        self,
        model,
        confidence_level: float = 0.95,
        n_bootstrap: int = 100
    ):
        """
        Initialize the uncertainty estimator.
        
        Args:
            model: Trained model with predict_proba method
            confidence_level: Confidence level for intervals (0-1)
            n_bootstrap: Number of bootstrap iterations
        """
        self.model = model
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        
        self.calibrated_model = None
        self.is_calibrated = False
        self.calibration_stats = {}
    
    def compute_confidence_interval(
        self,
        X: pd.DataFrame,
        method: str = 'bootstrap'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute confidence intervals for predictions.
        
        Args:
            X: Features to predict
            method: 'bootstrap' or 'quantile'
            
        Returns:
            Tuple of (point_estimates, lower_bounds, upper_bounds)
        """
        n_samples = len(X)
        
        # Get point estimate
        point_estimates = self.model.predict_proba(X)[:, 1]
        
        if method == 'bootstrap':
            return self._bootstrap_confidence_interval(X, point_estimates)
        elif method == 'quantile':
            return self._quantile_confidence_interval(point_estimates)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _bootstrap_confidence_interval(
        self,
        X: pd.DataFrame,
        point_estimates: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute bootstrap confidence intervals.
        
        Uses the bootstrap to estimate uncertainty by repeatedly
        sampling from the prediction distribution.
        """
        n_samples = len(X)
        bootstrap_predictions = np.zeros((self.n_bootstrap, n_samples))
        
        # For models that support it, we can use the variance across trees
        underlying = self._get_underlying_model()
        
        if hasattr(underlying, 'estimators_'):
            # Use predictions from individual trees/estimators
            n_estimators = len(underlying.estimators_)
            
            for i, estimator in enumerate(underlying.estimators_):
                try:
                    if hasattr(estimator, 'predict_proba'):
                        bootstrap_predictions[i % self.n_bootstrap] += estimator.predict_proba(X)[:, 1]
                    elif hasattr(estimator, 'predict'):
                        bootstrap_predictions[i % self.n_bootstrap] += estimator.predict(X)
                except:
                    pass
            
            # Normalize
            counts = np.zeros(self.n_bootstrap)
            for i in range(min(n_estimators, self.n_bootstrap)):
                counts[i % self.n_bootstrap] += 1
            
            for i in range(self.n_bootstrap):
                if counts[i] > 0:
                    bootstrap_predictions[i] /= counts[i]
                else:
                    bootstrap_predictions[i] = point_estimates
        else:
            # Simple perturbation-based uncertainty
            # Add small random noise to simulate uncertainty
            noise_scale = 0.05  # 5% noise
            for i in range(self.n_bootstrap):
                noise = np.random.normal(0, noise_scale, n_samples)
                bootstrap_predictions[i] = np.clip(point_estimates + noise, 0, 1)
        
        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bounds = np.percentile(bootstrap_predictions, lower_percentile, axis=0)
        upper_bounds = np.percentile(bootstrap_predictions, upper_percentile, axis=0)
        
        # Ensure bounds are sensible
        lower_bounds = np.maximum(lower_bounds, 0)
        upper_bounds = np.minimum(upper_bounds, 1)
        lower_bounds = np.minimum(lower_bounds, point_estimates)
        upper_bounds = np.maximum(upper_bounds, point_estimates)
        
        return point_estimates, lower_bounds, upper_bounds
    
    def _quantile_confidence_interval(
        self,
        point_estimates: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simple quantile-based confidence intervals.
        
        Uses a heuristic based on the predicted probability.
        Uncertainty is higher near 0.5 and lower near 0 or 1.
        """
        # Uncertainty is highest at p=0.5
        # Width proportional to sqrt(p * (1-p))
        uncertainty = np.sqrt(point_estimates * (1 - point_estimates))
        
        # Scale by z-score
        z = 1.96 if self.confidence_level == 0.95 else 1.645
        half_width = z * uncertainty * 0.2  # Scale factor
        
        lower_bounds = np.maximum(point_estimates - half_width, 0)
        upper_bounds = np.minimum(point_estimates + half_width, 1)
        
        return point_estimates, lower_bounds, upper_bounds
    
    def _get_underlying_model(self):
        """Extract underlying model from calibrated wrapper."""
        model = self.model
        if hasattr(model, 'estimator'):
            return model.estimator
        elif hasattr(model, 'base_estimator'):
            return model.base_estimator
        elif hasattr(model, 'calibrated_classifiers_'):
            return model.calibrated_classifiers_[0].estimator
        return model
    
    def calibrate_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'isotonic',
        cv: int = 5
    ) -> None:
        """
        Calibrate model probabilities.
        
        Args:
            X: Calibration features
            y: Calibration labels
            method: 'sigmoid' (Platt scaling) or 'isotonic'
            cv: Number of CV folds
        """
        print(f"Calibrating probabilities using {method} regression...")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Get underlying model for calibration
            base_model = self._get_underlying_model()
            
            self.calibrated_model = CalibratedClassifierCV(
                base_model,
                method=method,
                cv=cv
            )
            self.calibrated_model.fit(X, y)
        
        self.is_calibrated = True
        
        # Compute calibration stats
        self._compute_calibration_stats(X, y)
    
    def _compute_calibration_stats(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Compute calibration quality metrics."""
        y_pred_original = self.model.predict_proba(X)[:, 1]
        
        if self.is_calibrated:
            y_pred_calibrated = self.calibrated_model.predict_proba(X)[:, 1]
        else:
            y_pred_calibrated = y_pred_original
        
        self.calibration_stats = {
            'brier_score_original': brier_score_loss(y, y_pred_original),
            'brier_score_calibrated': brier_score_loss(y, y_pred_calibrated)
        }
        
        # Expected calibration error
        for name, preds in [('original', y_pred_original), ('calibrated', y_pred_calibrated)]:
            fraction_pos, mean_pred = calibration_curve(y, preds, n_bins=10, strategy='uniform')
            ece = np.mean(np.abs(fraction_pos - mean_pred))
            self.calibration_stats[f'ece_{name}'] = ece
    
    def predict_with_uncertainty(
        self,
        X: pd.DataFrame,
        use_calibrated: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            X: Features to predict
            use_calibrated: Whether to use calibrated model if available
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        # Choose model
        model = self.calibrated_model if (self.is_calibrated and use_calibrated) else self.model
        
        # Get point predictions
        probabilities = model.predict_proba(X)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        
        # Get confidence intervals
        _, lower, upper = self.compute_confidence_interval(X)
        
        # Calculate interval width as uncertainty measure
        uncertainty = upper - lower
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'lower_bound': lower,
            'upper_bound': upper,
            'uncertainty': uncertainty
        }
    
    def get_prediction_summary(
        self,
        X_single: pd.DataFrame,
        use_calibrated: bool = True
    ) -> Dict:
        """
        Get a summary of prediction with uncertainty for a single sample.
        
        Args:
            X_single: Single sample features
            use_calibrated: Whether to use calibrated model
            
        Returns:
            Dictionary with prediction summary
        """
        if len(X_single) != 1:
            X_single = X_single.iloc[[0]]
        
        results = self.predict_with_uncertainty(X_single, use_calibrated)
        
        prob = float(results['probabilities'][0])
        lower = float(results['lower_bound'][0])
        upper = float(results['upper_bound'][0])
        uncertainty = float(results['uncertainty'][0])
        
        # Determine confidence level
        if uncertainty < 0.1:
            confidence_label = "High confidence"
        elif uncertainty < 0.2:
            confidence_label = "Moderate confidence"
        else:
            confidence_label = "Low confidence"
        
        return {
            'risk_score': prob,
            'confidence_interval': [lower, upper],
            'interval_width': uncertainty,
            'confidence_level': self.confidence_level,
            'confidence_label': confidence_label,
            'is_calibrated': self.is_calibrated and use_calibrated
        }


class EnsembleUncertainty:
    """
    Ensemble-based uncertainty using multiple models.
    
    Uses disagreement among models as uncertainty measure.
    """
    
    def __init__(self, models: Dict[str, Any]):
        """
        Initialize with dictionary of trained models.
        
        Args:
            models: Dictionary of {name: model}
        """
        self.models = models
        self.model_names = list(models.keys())
    
    def predict_with_ensemble_uncertainty(
        self,
        X: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions with ensemble uncertainty.
        
        Args:
            X: Features to predict
            
        Returns:
            Dictionary with ensemble predictions and uncertainty
        """
        n_samples = len(X)
        n_models = len(self.models)
        
        # Get predictions from all models
        all_predictions = np.zeros((n_models, n_samples))
        
        for i, (name, model) in enumerate(self.models.items()):
            all_predictions[i] = model.predict_proba(X)[:, 1]
        
        # Ensemble prediction (mean)
        mean_prediction = all_predictions.mean(axis=0)
        
        # Uncertainty (standard deviation across models)
        std_prediction = all_predictions.std(axis=0)
        
        # Confidence intervals
        lower_bound = np.maximum(mean_prediction - 1.96 * std_prediction, 0)
        upper_bound = np.minimum(mean_prediction + 1.96 * std_prediction, 1)
        
        return {
            'probabilities': mean_prediction,
            'predictions': (mean_prediction >= 0.5).astype(int),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'uncertainty': std_prediction,
            'model_predictions': {name: all_predictions[i] for i, name in enumerate(self.model_names)}
        }


if __name__ == "__main__":
    # Demonstration
    from data_ingestion import DataIngestion
    from preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer
    from model_training import ModelTrainer
    
    print("=" * 60)
    print("Uncertainty Estimation Demonstration")
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
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Train model
    print("\n2. Training model...")
    trainer = ModelTrainer(algorithms=['random_forest'], cv_folds=3)
    models = trainer.train_all_models(X_train_processed, y_train, calibrate=False)
    model = models['random_forest']
    
    # Initialize uncertainty estimator
    print("\n3. Computing uncertainty estimates...")
    uncertainty_estimator = UncertaintyEstimator(
        model=model,
        confidence_level=0.95,
        n_bootstrap=100
    )
    
    # Calibrate
    print("\n4. Calibrating probabilities...")
    uncertainty_estimator.calibrate_model(X_val_processed, y_val, method='isotonic')
    print(f"Calibration stats: {uncertainty_estimator.calibration_stats}")
    
    # Predict with uncertainty
    print("\n5. Sample predictions with uncertainty:")
    for i in range(3):
        sample = X_test_processed.iloc[[i]]
        summary = uncertainty_estimator.get_prediction_summary(sample)
        print(f"\nSample {i+1}:")
        print(f"  Risk score: {summary['risk_score']:.3f}")
        print(f"  95% CI: [{summary['confidence_interval'][0]:.3f}, {summary['confidence_interval'][1]:.3f}]")
        print(f"  {summary['confidence_label']}")
    
    # Ensemble uncertainty
    print("\n6. Training ensemble for uncertainty...")
    trainer2 = ModelTrainer(algorithms=['logistic_regression', 'random_forest', 'xgboost'], cv_folds=3)
    ensemble_models = trainer2.train_all_models(X_train_processed, y_train, calibrate=False)
    
    ensemble = EnsembleUncertainty(ensemble_models)
    results = ensemble.predict_with_ensemble_uncertainty(X_test_processed)
    
    print(f"\nEnsemble uncertainty statistics:")
    print(f"  Mean uncertainty: {results['uncertainty'].mean():.3f}")
    print(f"  Max uncertainty: {results['uncertainty'].max():.3f}")
    print(f"  Min uncertainty: {results['uncertainty'].min():.3f}")
