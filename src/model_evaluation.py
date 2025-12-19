"""
Model Evaluation Module for Chronic Disease Risk Prediction System

This module handles:
- Classification performance metrics
- Calibration assessment
- Clinical utility evaluation
- Model comparison
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, brier_score_loss,
    log_loss
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ModelEvaluator:
    """
    Comprehensive model evaluation for clinical risk prediction.
    
    Includes:
    - Standard classification metrics
    - Probability calibration assessment
    - Clinical utility metrics (NNT, PPV, NPV)
    - Visualization of results
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the evaluator.
        
        Args:
            output_dir: Directory to save evaluation plots
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.evaluation_results = {}
    
    def evaluate_model(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str = 'model',
        threshold: float = 0.5
    ) -> Dict:
        """
        Comprehensive evaluation of a single model.
        
        Args:
            model: Trained model with predict_proba method
            X: Test features
            y: Test labels
            model_name: Name identifier for the model
            threshold: Decision threshold
            
        Returns:
            Dictionary with all evaluation metrics
        """
        # Get predictions
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        results = {
            'model_name': model_name,
            'n_samples': len(y),
            'n_positive': int(y.sum()),
            'prevalence': float(y.mean()),
            'threshold': threshold,
            
            # Discrimination metrics
            'auc_roc': roc_auc_score(y, y_pred_proba),
            'auc_pr': average_precision_score(y, y_pred_proba),
            
            # Classification metrics
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'specificity': self._calculate_specificity(y, y_pred),
            
            # Calibration metrics
            'brier_score': brier_score_loss(y, y_pred_proba),
            'log_loss': log_loss(y, y_pred_proba),
            
            # Confusion matrix
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            
            # Clinical utility
            'ppv': precision_score(y, y_pred, zero_division=0),  # Positive predictive value
            'npv': self._calculate_npv(y, y_pred),  # Negative predictive value
        }
        
        # Calculate NNT (Number Needed to Treat) if applicable
        if results['ppv'] > 0:
            results['nnt'] = 1 / results['ppv']
        else:
            results['nnt'] = float('inf')
        
        # Store results
        self.evaluation_results[model_name] = results
        
        return results
    
    def _calculate_specificity(self, y_true, y_pred) -> float:
        """Calculate specificity (true negative rate)."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def _calculate_npv(self, y_true, y_pred) -> float:
        """Calculate negative predictive value."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    def evaluate_multiple_models(
        self,
        models: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Evaluate multiple models and compare results.
        
        Args:
            models: Dictionary of {name: model}
            X: Test features
            y: Test labels
            threshold: Decision threshold
            
        Returns:
            DataFrame with comparison metrics
        """
        all_results = []
        
        for name, model in models.items():
            results = self.evaluate_model(model, X, y, name, threshold)
            all_results.append(results)
        
        df = pd.DataFrame(all_results)
        
        # Select key columns for display
        display_cols = [
            'model_name', 'auc_roc', 'auc_pr', 'accuracy', 'precision', 
            'recall', 'f1_score', 'brier_score'
        ]
        
        return df[display_cols].sort_values('auc_roc', ascending=False)
    
    def find_optimal_threshold(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        criterion: str = 'f1'
    ) -> Tuple[float, Dict]:
        """
        Find the optimal decision threshold.
        
        Args:
            model: Trained model
            X: Validation features
            y: Validation labels
            criterion: 'f1', 'youden' (J statistic), or 'precision_recall_balance'
            
        Returns:
            Tuple of (optimal_threshold, metrics_at_threshold)
        """
        y_pred_proba = model.predict_proba(X)[:, 1]
        thresholds = np.linspace(0.1, 0.9, 81)
        
        best_threshold = 0.5
        best_score = 0
        
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            
            if criterion == 'f1':
                score = f1_score(y, y_pred, zero_division=0)
            elif criterion == 'youden':
                # Youden's J statistic = sensitivity + specificity - 1
                sensitivity = recall_score(y, y_pred, zero_division=0)
                specificity = self._calculate_specificity(y, y_pred)
                score = sensitivity + specificity - 1
            elif criterion == 'precision_recall_balance':
                precision = precision_score(y, y_pred, zero_division=0)
                recall = recall_score(y, y_pred, zero_division=0)
                score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            else:
                raise ValueError(f"Unknown criterion: {criterion}")
            
            if score > best_score:
                best_score = score
                best_threshold = thresh
        
        # Calculate metrics at optimal threshold
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        metrics = {
            'threshold': best_threshold,
            'criterion': criterion,
            'criterion_score': best_score,
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'specificity': self._calculate_specificity(y, y_pred)
        }
        
        return best_threshold, metrics
    
    def get_calibration_data(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        n_bins: int = 10
    ) -> Dict:
        """
        Get calibration curve data.
        
        Args:
            model: Trained model
            X: Test features
            y: Test labels
            n_bins: Number of bins for calibration
            
        Returns:
            Dictionary with calibration data
        """
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y, y_pred_proba, n_bins=n_bins, strategy='uniform'
        )
        
        # Calculate calibration error
        calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        
        return {
            'fraction_of_positives': fraction_of_positives.tolist(),
            'mean_predicted_value': mean_predicted_value.tolist(),
            'expected_calibration_error': calibration_error,
            'brier_score': brier_score_loss(y, y_pred_proba)
        }
    
    def plot_roc_curves(
        self,
        models: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        save_path: str = None
    ) -> None:
        """
        Plot ROC curves for multiple models.
        """
        plt.figure(figsize=(10, 8))
        
        for name, model in models.items():
            y_pred_proba = model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_pred_proba)
            auc = roc_auc_score(y, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"ROC curves saved to {save_path}")
        
        plt.close()
    
    def plot_precision_recall_curves(
        self,
        models: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        save_path: str = None
    ) -> None:
        """
        Plot Precision-Recall curves for multiple models.
        """
        plt.figure(figsize=(10, 8))
        
        for name, model in models.items():
            y_pred_proba = model.predict_proba(X)[:, 1]
            precision, recall, _ = precision_recall_curve(y, y_pred_proba)
            ap = average_precision_score(y, y_pred_proba)
            plt.plot(recall, precision, label=f'{name} (AP = {ap:.3f})', linewidth=2)
        
        # Baseline (prevalence)
        baseline = y.mean()
        plt.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.3f})', linewidth=1)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves Comparison', fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"PR curves saved to {save_path}")
        
        plt.close()
    
    def plot_calibration_curves(
        self,
        models: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        n_bins: int = 10,
        save_path: str = None
    ) -> None:
        """
        Plot calibration curves for multiple models.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Calibration curve
        ax1 = axes[0]
        for name, model in models.items():
            calibration_data = self.get_calibration_data(model, X, y, n_bins)
            ax1.plot(
                calibration_data['mean_predicted_value'],
                calibration_data['fraction_of_positives'],
                'o-', label=f"{name} (ECE={calibration_data['expected_calibration_error']:.3f})",
                linewidth=2, markersize=6
            )
        
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=1)
        ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax1.set_ylabel('Fraction of Positives', fontsize=12)
        ax1.set_title('Calibration Curves', fontsize=14)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Histogram of predictions
        ax2 = axes[1]
        for name, model in models.items():
            y_pred_proba = model.predict_proba(X)[:, 1]
            ax2.hist(y_pred_proba, bins=30, alpha=0.5, label=name, density=True)
        
        ax2.set_xlabel('Predicted Probability', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('Prediction Distribution', fontsize=14)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Calibration plots saved to {save_path}")
        
        plt.close()
    
    def plot_confusion_matrix(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str = 'Model',
        threshold: float = 0.5,
        save_path: str = None
    ) -> None:
        """
        Plot confusion matrix heatmap.
        """
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        cm = confusion_matrix(y, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive']
        )
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.title(f'Confusion Matrix - {model_name}\n(threshold={threshold})', fontsize=14)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def generate_evaluation_report(
        self,
        models: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        output_dir: str = None
    ) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            models: Dictionary of trained models
            X: Test features
            y: Test labels
            output_dir: Directory to save plots
            
        Returns:
            Formatted report string
        """
        output_dir = Path(output_dir) if output_dir else self.output_dir
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluate all models
        comparison_df = self.evaluate_multiple_models(models, X, y)
        
        # Generate plots
        if output_dir:
            self.plot_roc_curves(models, X, y, output_dir / 'roc_curves.png')
            self.plot_precision_recall_curves(models, X, y, output_dir / 'pr_curves.png')
            self.plot_calibration_curves(models, X, y, save_path=output_dir / 'calibration_curves.png')
        
        # Build report
        report = []
        report.append("=" * 60)
        report.append("MODEL EVALUATION REPORT")
        report.append("=" * 60)
        report.append(f"\nDataset: {len(y)} samples, {y.mean():.1%} positive class\n")
        
        report.append("\n--- Model Comparison ---")
        report.append(comparison_df.to_string(index=False))
        
        report.append("\n\n--- Detailed Results ---")
        for name, results in self.evaluation_results.items():
            report.append(f"\n{name.upper()}:")
            report.append(f"  AUC-ROC: {results['auc_roc']:.4f}")
            report.append(f"  AUC-PR: {results['auc_pr']:.4f}")
            report.append(f"  Accuracy: {results['accuracy']:.4f}")
            report.append(f"  Precision: {results['precision']:.4f}")
            report.append(f"  Recall: {results['recall']:.4f}")
            report.append(f"  Specificity: {results['specificity']:.4f}")
            report.append(f"  Brier Score: {results['brier_score']:.4f}")
            report.append(f"  PPV: {results['ppv']:.4f}")
            report.append(f"  NPV: {results['npv']:.4f}")
            if results['nnt'] != float('inf'):
                report.append(f"  NNT: {results['nnt']:.1f}")
        
        return "\n".join(report)


if __name__ == "__main__":
    # Demonstration
    from data_ingestion import DataIngestion
    from preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer
    from model_training import ModelTrainer
    
    print("=" * 60)
    print("Model Evaluation Demonstration")
    print("=" * 60)
    
    # Generate and prepare data
    print("\n1. Preparing data...")
    ingestion = DataIngestion()
    df = ingestion.generate_synthetic_data(n_samples=1000)
    
    feature_engineer = FeatureEngineer()
    df = feature_engineer.transform(df)
    
    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        df, target_col='diabetes_risk'
    )
    
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Train models
    print("\n2. Training models...")
    trainer = ModelTrainer(
        algorithms=['logistic_regression', 'random_forest', 'xgboost'],
        cv_folds=3
    )
    models = trainer.train_all_models(X_train_processed, y_train)
    
    # Evaluate
    print("\n3. Evaluating models...")
    evaluator = ModelEvaluator()
    report = evaluator.generate_evaluation_report(models, X_test_processed, y_test)
    print(report)
    
    # Find optimal thresholds
    print("\n\n4. Optimal Thresholds:")
    for name, model in models.items():
        opt_thresh, metrics = evaluator.find_optimal_threshold(model, X_test_processed, y_test)
        print(f"\n{name}:")
        print(f"  Optimal threshold (F1): {opt_thresh:.2f}")
        print(f"  Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}")
