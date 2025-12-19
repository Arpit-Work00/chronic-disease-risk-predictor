"""
Model Training Module for Chronic Disease Risk Prediction System

This module handles:
- Training multiple classification models
- Hyperparameter tuning with cross-validation
- Model serialization and loading
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path
import warnings
import time


class ModelTrainer:
    """
    Multi-model training pipeline with hyperparameter tuning.
    
    Supports:
    - Logistic Regression (interpretable baseline)
    - Random Forest
    - XGBoost
    - LightGBM
    """
    
    # Default hyperparameter grids
    DEFAULT_PARAM_GRIDS = {
        'logistic_regression': {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l2'],
            'max_iter': [1000],
            'class_weight': ['balanced', None]
        },
        'random_forest': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', None]
        },
        'xgboost': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'scale_pos_weight': [1, 3, 5]
        },
        'lightgbm': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'num_leaves': [31, 63],
            'class_weight': ['balanced', None]
        }
    }
    
    def __init__(
        self,
        algorithms: List[str] = None,
        cv_folds: int = 5,
        scoring: str = 'roc_auc',
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize the model trainer.
        
        Args:
            algorithms: List of algorithms to train
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for optimization
            random_state: Random seed
            n_jobs: Number of parallel jobs
        """
        self.algorithms = algorithms or ['logistic_regression', 'random_forest', 'xgboost']
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.models = {}
        self.best_params = {}
        self.cv_results = {}
        self.training_times = {}
        self.feature_names = []
    
    def _get_base_model(self, algorithm: str):
        """
        Get the base model instance for an algorithm.
        
        Args:
            algorithm: Algorithm name
            
        Returns:
            Sklearn-compatible model instance
        """
        if algorithm == 'logistic_regression':
            return LogisticRegression(random_state=self.random_state, solver='lbfgs')
        elif algorithm == 'random_forest':
            return RandomForestClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
        elif algorithm == 'xgboost':
            return xgb.XGBClassifier(
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=self.n_jobs
            )
        elif algorithm == 'lightgbm':
            return lgb.LGBMClassifier(
                random_state=self.random_state,
                verbose=-1,
                n_jobs=self.n_jobs
            )
        elif algorithm == 'gradient_boosting':
            return GradientBoostingClassifier(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def train_single_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        algorithm: str,
        param_grid: Dict = None,
        tune_hyperparameters: bool = True
    ) -> Tuple[Any, Dict, float]:
        """
        Train a single model with optional hyperparameter tuning.
        
        Args:
            X: Training features
            y: Training target
            algorithm: Algorithm name
            param_grid: Custom hyperparameter grid
            tune_hyperparameters: Whether to perform grid search
            
        Returns:
            Tuple of (trained_model, best_params, cv_score)
        """
        print(f"\n{'='*50}")
        print(f"Training {algorithm.upper()}")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        base_model = self._get_base_model(algorithm)
        
        if tune_hyperparameters:
            param_grid = param_grid or self.DEFAULT_PARAM_GRIDS.get(algorithm, {})
            
            if param_grid:
                print(f"Performing grid search with {self.cv_folds}-fold CV...")
                
                # Use RandomizedSearchCV for large grids
                n_combinations = np.prod([len(v) for v in param_grid.values()])
                
                if n_combinations > 50:
                    search = RandomizedSearchCV(
                        base_model,
                        param_grid,
                        n_iter=30,
                        cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                        scoring=self.scoring,
                        n_jobs=self.n_jobs,
                        random_state=self.random_state,
                        verbose=0
                    )
                else:
                    search = GridSearchCV(
                        base_model,
                        param_grid,
                        cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                        scoring=self.scoring,
                        n_jobs=self.n_jobs,
                        verbose=0
                    )
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    search.fit(X, y)
                
                best_model = search.best_estimator_
                best_params = search.best_params_
                cv_score = search.best_score_
                
                print(f"Best parameters: {best_params}")
                print(f"Best CV {self.scoring}: {cv_score:.4f}")
            else:
                # No hyperparameters to tune
                best_model = base_model
                cv_scores = cross_val_score(
                    best_model, X, y,
                    cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                    scoring=self.scoring,
                    n_jobs=self.n_jobs
                )
                best_model.fit(X, y)
                best_params = {}
                cv_score = cv_scores.mean()
                print(f"CV {self.scoring}: {cv_score:.4f} (+/- {cv_scores.std():.4f})")
        else:
            # No tuning, just fit
            best_model = base_model
            best_model.fit(X, y)
            best_params = {}
            cv_score = 0.0
        
        training_time = time.time() - start_time
        print(f"Training time: {training_time:.2f} seconds")
        
        return best_model, best_params, cv_score
    
    def train_all_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        tune_hyperparameters: bool = True,
        calibrate: bool = True
    ) -> Dict[str, Any]:
        """
        Train all specified algorithms.
        
        Args:
            X: Training features
            y: Training target
            tune_hyperparameters: Whether to tune hyperparameters
            calibrate: Whether to calibrate probability estimates
            
        Returns:
            Dictionary of trained models
        """
        self.feature_names = list(X.columns)
        
        print(f"\nTraining {len(self.algorithms)} models on {len(X)} samples with {len(X.columns)} features")
        print(f"Target distribution: {y.mean():.2%} positive class")
        
        for algorithm in self.algorithms:
            try:
                model, params, cv_score = self.train_single_model(
                    X, y, algorithm, tune_hyperparameters=tune_hyperparameters
                )
                
                # Optionally calibrate probabilities
                if calibrate:
                    print(f"Calibrating probability estimates...")
                    calibrated = CalibratedClassifierCV(model, cv=3, method='isotonic')
                    calibrated.fit(X, y)
                    self.models[algorithm] = calibrated
                else:
                    self.models[algorithm] = model
                
                self.best_params[algorithm] = params
                self.cv_results[algorithm] = cv_score
                
            except Exception as e:
                print(f"Error training {algorithm}: {str(e)}")
                continue
        
        return self.models
    
    def get_feature_importance(self, algorithm: str = None) -> pd.DataFrame:
        """
        Get feature importance from trained models.
        
        Args:
            algorithm: Specific algorithm, or None for all
            
        Returns:
            DataFrame with feature importance scores
        """
        importance_data = []
        
        algorithms_to_check = [algorithm] if algorithm else self.algorithms
        
        for algo in algorithms_to_check:
            if algo not in self.models:
                continue
            
            model = self.models[algo]
            
            # Handle calibrated models
            if hasattr(model, 'estimator'):
                model = model.estimator
            elif hasattr(model, 'base_estimator'):
                model = model.base_estimator
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                continue
            
            for feature, importance in zip(self.feature_names, importances):
                importance_data.append({
                    'algorithm': algo,
                    'feature': feature,
                    'importance': importance
                })
        
        df = pd.DataFrame(importance_data)
        
        if len(df) > 0:
            # Normalize importance within each algorithm
            for algo in df['algorithm'].unique():
                mask = df['algorithm'] == algo
                total = df.loc[mask, 'importance'].sum()
                if total > 0:
                    df.loc[mask, 'importance'] = df.loc[mask, 'importance'] / total
        
        return df
    
    def get_best_model(self) -> Tuple[str, Any]:
        """
        Get the best performing model based on CV score.
        
        Returns:
            Tuple of (algorithm_name, trained_model)
        """
        if not self.cv_results:
            raise ValueError("No models have been trained yet")
        
        best_algo = max(self.cv_results, key=self.cv_results.get)
        return best_algo, self.models[best_algo]
    
    def get_training_summary(self) -> pd.DataFrame:
        """
        Get a summary of all trained models.
        
        Returns:
            DataFrame with model training summary
        """
        summary_data = []
        
        for algo in self.algorithms:
            if algo in self.models:
                summary_data.append({
                    'algorithm': algo,
                    'cv_score': self.cv_results.get(algo, 0),
                    'n_params_tuned': len(self.best_params.get(algo, {})),
                    'training_time': self.training_times.get(algo, 0)
                })
        
        return pd.DataFrame(summary_data).sort_values('cv_score', ascending=False)
    
    def save_model(self, algorithm: str, filepath: str) -> None:
        """
        Save a trained model to disk.
        
        Args:
            algorithm: Algorithm name
            filepath: Path to save the model
        """
        if algorithm not in self.models:
            raise ValueError(f"Model {algorithm} not found")
        
        model_data = {
            'model': self.models[algorithm],
            'params': self.best_params.get(algorithm, {}),
            'cv_score': self.cv_results.get(algorithm, 0),
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def save_all_models(self, directory: str) -> None:
        """
        Save all trained models to a directory.
        
        Args:
            directory: Directory path
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        for algo in self.models:
            filepath = path / f"{algo}_model.joblib"
            self.save_model(algo, str(filepath))
    
    @staticmethod
    def load_model(filepath: str) -> Tuple[Any, Dict]:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Tuple of (model, metadata)
        """
        model_data = joblib.load(filepath)
        return model_data['model'], model_data


if __name__ == "__main__":
    # Demonstration
    from data_ingestion import DataIngestion
    from preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer
    
    print("=" * 60)
    print("Model Training Demonstration")
    print("=" * 60)
    
    # Generate and preprocess data
    print("\n1. Generating synthetic data...")
    ingestion = DataIngestion()
    df = ingestion.generate_synthetic_data(n_samples=1000)
    
    print("\n2. Feature engineering...")
    feature_engineer = FeatureEngineer()
    df = feature_engineer.transform(df)
    
    print("\n3. Preprocessing...")
    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        df, target_col='diabetes_risk'
    )
    
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    
    print("\n4. Training models...")
    trainer = ModelTrainer(
        algorithms=['logistic_regression', 'random_forest', 'xgboost'],
        cv_folds=5,
        scoring='roc_auc'
    )
    
    models = trainer.train_all_models(
        X_train_processed, y_train,
        tune_hyperparameters=True,
        calibrate=True
    )
    
    print("\n5. Training Summary:")
    summary = trainer.get_training_summary()
    print(summary)
    
    print("\n6. Feature Importance (Top 10):")
    importance = trainer.get_feature_importance()
    top_features = importance.groupby('feature')['importance'].mean().sort_values(ascending=False).head(10)
    for feature, imp in top_features.items():
        print(f"  {feature}: {imp:.4f}")
    
    print("\n7. Best Model:")
    best_algo, best_model = trainer.get_best_model()
    print(f"  {best_algo} with CV score: {trainer.cv_results[best_algo]:.4f}")
