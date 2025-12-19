"""
Bias Analysis Module for Chronic Disease Risk Prediction System

This module handles:
- Fairness metrics computation across demographic groups
- Disparity detection and quantification
- Bias mitigation recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class BiasAnalyzer:
    """
    Fairness and bias analysis for clinical risk prediction.
    
    Evaluates model performance across protected attributes:
    - Age groups
    - Gender
    - Ethnicity
    
    Computes fairness metrics:
    - Demographic parity
    - Equalized odds
    - Predictive parity
    """
    
    # Fairness thresholds (per common guidelines)
    DISPARITY_THRESHOLD = 0.8  # 80% rule
    
    def __init__(
        self,
        protected_attributes: List[str] = None,
        reference_groups: Dict[str, str] = None
    ):
        """
        Initialize the bias analyzer.
        
        Args:
            protected_attributes: List of protected attribute column names
            reference_groups: Dictionary mapping attribute to reference group value
        """
        self.protected_attributes = protected_attributes or ['age_group', 'gender']
        self.reference_groups = reference_groups or {}
        self.analysis_results = {}
    
    def create_demographic_groups(
        self,
        df: pd.DataFrame,
        age_col: str = 'age',
        gender_col: str = 'gender'
    ) -> pd.DataFrame:
        """
        Create demographic group columns for analysis.
        
        Args:
            df: Input DataFrame
            age_col: Age column name
            gender_col: Gender column name
            
        Returns:
            DataFrame with demographic group columns added
        """
        df = df.copy()
        
        # Create age groups if age column exists
        if age_col in df.columns:
            df['age_group'] = pd.cut(
                df[age_col],
                bins=[0, 40, 55, 65, 120],
                labels=['Under 40', '40-54', '55-64', '65+']
            ).astype(str)
        
        return df
    
    def compute_group_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        groups: pd.Series
    ) -> pd.DataFrame:
        """
        Compute metrics for each demographic group.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            groups: Group membership
            
        Returns:
            DataFrame with per-group metrics
        """
        results = []
        
        for group in groups.unique():
            mask = groups == group
            n_samples = mask.sum()
            
            if n_samples < 10:  # Skip groups with too few samples
                continue
            
            y_t = y_true[mask]
            y_p = y_pred[mask]
            y_pp = y_pred_proba[mask]
            
            # Compute metrics
            metrics = {
                'group': group,
                'n_samples': n_samples,
                'prevalence': y_t.mean(),
                'pred_positive_rate': y_p.mean(),
                'accuracy': accuracy_score(y_t, y_p),
                'precision': precision_score(y_t, y_p, zero_division=0),
                'recall': recall_score(y_t, y_p, zero_division=0),
                'f1_score': f1_score(y_t, y_p, zero_division=0),
                'mean_pred_proba': y_pp.mean()
            }
            
            # Try to compute AUC (may fail if only one class)
            try:
                metrics['auc_roc'] = roc_auc_score(y_t, y_pp)
            except:
                metrics['auc_roc'] = None
            
            # Confusion matrix elements for equalized odds
            tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
            metrics['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # True positive rate
            metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0  # False positive rate
            metrics['tnr'] = tn / (tn + fp) if (tn + fp) > 0 else 0  # True negative rate
            metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0  # False negative rate
            
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def compute_fairness_metrics(
        self,
        group_metrics: pd.DataFrame,
        reference_group: str = None
    ) -> Dict:
        """
        Compute fairness metrics comparing groups.
        
        Args:
            group_metrics: DataFrame from compute_group_metrics
            reference_group: Reference group for comparison
            
        Returns:
            Dictionary with fairness metrics
        """
        if reference_group is None:
            # Use group with largest sample size as reference
            reference_group = group_metrics.loc[
                group_metrics['n_samples'].idxmax(), 'group'
            ]
        
        ref_metrics = group_metrics[group_metrics['group'] == reference_group].iloc[0]
        
        fairness_results = {
            'reference_group': reference_group,
            'groups': []
        }
        
        for _, row in group_metrics.iterrows():
            if row['group'] == reference_group:
                continue
            
            group_result = {
                'group': row['group'],
                'n_samples': row['n_samples']
            }
            
            # Demographic Parity
            # Ratio of positive prediction rates
            if ref_metrics['pred_positive_rate'] > 0:
                dp_ratio = row['pred_positive_rate'] / ref_metrics['pred_positive_rate']
            else:
                dp_ratio = 1.0
            group_result['demographic_parity_ratio'] = dp_ratio
            group_result['demographic_parity_diff'] = row['pred_positive_rate'] - ref_metrics['pred_positive_rate']
            group_result['demographic_parity_pass'] = dp_ratio >= self.DISPARITY_THRESHOLD
            
            # Equalized Odds
            # Requires similar TPR and FPR across groups
            tpr_ratio = row['tpr'] / ref_metrics['tpr'] if ref_metrics['tpr'] > 0 else 1.0
            fpr_ratio = row['fpr'] / ref_metrics['fpr'] if ref_metrics['fpr'] > 0 else 1.0
            group_result['tpr_ratio'] = tpr_ratio
            group_result['fpr_ratio'] = fpr_ratio
            group_result['equalized_odds_tpr_pass'] = tpr_ratio >= self.DISPARITY_THRESHOLD
            group_result['equalized_odds_fpr_pass'] = fpr_ratio >= self.DISPARITY_THRESHOLD
            
            # Predictive Parity
            # Requires similar precision across groups
            if ref_metrics['precision'] > 0:
                pp_ratio = row['precision'] / ref_metrics['precision']
            else:
                pp_ratio = 1.0
            group_result['predictive_parity_ratio'] = pp_ratio
            group_result['predictive_parity_pass'] = pp_ratio >= self.DISPARITY_THRESHOLD
            
            # Performance gap
            group_result['accuracy_gap'] = abs(row['accuracy'] - ref_metrics['accuracy'])
            group_result['auc_gap'] = abs((row['auc_roc'] or 0) - (ref_metrics['auc_roc'] or 0))
            
            fairness_results['groups'].append(group_result)
        
        return fairness_results
    
    def analyze_bias(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        demographic_data: pd.DataFrame,
        threshold: float = 0.5
    ) -> Dict:
        """
        Perform comprehensive bias analysis.
        
        Args:
            model: Trained model
            X: Test features
            y: Test labels
            demographic_data: DataFrame with demographic columns
            threshold: Prediction threshold
            
        Returns:
            Dictionary with complete bias analysis
        """
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        analysis = {
            'overall_metrics': {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, zero_division=0),
                'recall': recall_score(y, y_pred, zero_division=0),
                'f1_score': f1_score(y, y_pred, zero_division=0)
            },
            'attribute_analysis': {}
        }
        
        # Analyze each protected attribute
        for attr in self.protected_attributes:
            if attr not in demographic_data.columns:
                continue
            
            groups = demographic_data[attr]
            
            # Compute group-level metrics
            group_metrics = self.compute_group_metrics(y, y_pred, y_pred_proba, groups)
            
            # Compute fairness metrics
            ref_group = self.reference_groups.get(attr)
            fairness = self.compute_fairness_metrics(group_metrics, ref_group)
            
            # Check for disparities
            disparities = []
            for group_result in fairness['groups']:
                issues = []
                if not group_result['demographic_parity_pass']:
                    issues.append('demographic_parity')
                if not group_result['equalized_odds_tpr_pass']:
                    issues.append('equalized_odds_tpr')
                if not group_result['predictive_parity_pass']:
                    issues.append('predictive_parity')
                
                if issues:
                    disparities.append({
                        'group': group_result['group'],
                        'issues': issues
                    })
            
            analysis['attribute_analysis'][attr] = {
                'group_metrics': group_metrics.to_dict('records'),
                'fairness_metrics': fairness,
                'disparities_detected': len(disparities) > 0,
                'disparities': disparities
            }
        
        self.analysis_results = analysis
        return analysis
    
    def generate_bias_report(self) -> str:
        """
        Generate a human-readable bias analysis report.
        
        Returns:
            Formatted report string
        """
        if not self.analysis_results:
            return "No bias analysis results available. Run analyze_bias() first."
        
        report = []
        report.append("=" * 60)
        report.append("BIAS AND FAIRNESS ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Overall metrics
        report.append("\n--- Overall Model Performance ---")
        overall = self.analysis_results['overall_metrics']
        report.append(f"Accuracy: {overall['accuracy']:.3f}")
        report.append(f"Precision: {overall['precision']:.3f}")
        report.append(f"Recall: {overall['recall']:.3f}")
        report.append(f"F1 Score: {overall['f1_score']:.3f}")
        
        # Per-attribute analysis
        for attr, attr_analysis in self.analysis_results['attribute_analysis'].items():
            report.append(f"\n--- Analysis by {attr.upper()} ---")
            
            # Group metrics summary
            report.append("\nGroup Performance:")
            for group_data in attr_analysis['group_metrics']:
                report.append(f"\n  {group_data['group']} (n={group_data['n_samples']}):")
                report.append(f"    Prevalence: {group_data['prevalence']:.1%}")
                report.append(f"    Positive prediction rate: {group_data['pred_positive_rate']:.1%}")
                report.append(f"    Accuracy: {group_data['accuracy']:.3f}")
                report.append(f"    TPR: {group_data['tpr']:.3f}, FPR: {group_data['fpr']:.3f}")
            
            # Fairness metrics
            fairness = attr_analysis['fairness_metrics']
            report.append(f"\nFairness Metrics (reference: {fairness['reference_group']}):")
            
            for group_result in fairness['groups']:
                report.append(f"\n  {group_result['group']}:")
                
                # Demographic parity
                dp_status = "✓" if group_result['demographic_parity_pass'] else "✗"
                report.append(f"    Demographic Parity: {dp_status} (ratio={group_result['demographic_parity_ratio']:.2f})")
                
                # Equalized odds
                eo_tpr_status = "✓" if group_result['equalized_odds_tpr_pass'] else "✗"
                report.append(f"    Equalized Odds (TPR): {eo_tpr_status} (ratio={group_result['tpr_ratio']:.2f})")
                
                # Predictive parity
                pp_status = "✓" if group_result['predictive_parity_pass'] else "✗"
                report.append(f"    Predictive Parity: {pp_status} (ratio={group_result['predictive_parity_ratio']:.2f})")
            
            # Disparities
            if attr_analysis['disparities_detected']:
                report.append("\n⚠️  DISPARITIES DETECTED:")
                for disp in attr_analysis['disparities']:
                    report.append(f"    • {disp['group']}: Issues with {', '.join(disp['issues'])}")
            else:
                report.append("\n✓ No significant disparities detected")
        
        # Recommendations
        report.append("\n" + "=" * 60)
        report.append("RECOMMENDATIONS")
        report.append("=" * 60)
        report.append(self._generate_recommendations())
        
        return "\n".join(report)
    
    def _generate_recommendations(self) -> str:
        """Generate bias mitigation recommendations."""
        recommendations = []
        
        has_disparities = False
        for attr, analysis in self.analysis_results.get('attribute_analysis', {}).items():
            if analysis.get('disparities_detected'):
                has_disparities = True
                
                for disp in analysis['disparities']:
                    if 'demographic_parity' in disp['issues']:
                        recommendations.append(
                            f"• Consider threshold adjustment for {disp['group']} group"
                        )
                    if 'equalized_odds_tpr' in disp['issues']:
                        recommendations.append(
                            f"• {disp['group']} group has different true positive rates - "
                            "consider resampling or cost-sensitive learning"
                        )
                    if 'predictive_parity' in disp['issues']:
                        recommendations.append(
                            f"• Prediction precision differs for {disp['group']} - "
                            "may need group-specific calibration"
                        )
        
        if not has_disparities:
            recommendations.append("• Model shows acceptable fairness across analyzed groups")
            recommendations.append("• Continue monitoring fairness metrics over time")
        else:
            recommendations.append("\nGeneral mitigation strategies:")
            recommendations.append("• Pre-processing: Rebalance training data across groups")
            recommendations.append("• In-processing: Add fairness constraints during training")
            recommendations.append("• Post-processing: Adjust thresholds per group")
        
        return "\n".join(recommendations)
    
    def plot_group_comparison(
        self,
        attribute: str = None,
        metric: str = 'accuracy',
        save_path: str = None
    ) -> None:
        """
        Plot comparison of metrics across demographic groups.
        """
        if not self.analysis_results:
            print("No analysis results. Run analyze_bias() first.")
            return
        
        if attribute is None:
            attribute = list(self.analysis_results['attribute_analysis'].keys())[0]
        
        if attribute not in self.analysis_results['attribute_analysis']:
            print(f"Attribute {attribute} not found in analysis results.")
            return
        
        group_metrics = pd.DataFrame(
            self.analysis_results['attribute_analysis'][attribute]['group_metrics']
        )
        
        plt.figure(figsize=(10, 6))
        
        x = range(len(group_metrics))
        bars = plt.bar(x, group_metrics[metric], color='steelblue', edgecolor='black')
        
        # Add value labels
        for i, bar in enumerate(bars):
            plt.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{group_metrics[metric].iloc[i]:.3f}',
                ha='center', va='bottom', fontsize=10
            )
        
        plt.xticks(x, group_metrics['group'], rotation=45, ha='right')
        plt.xlabel('Demographic Group', fontsize=12)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.title(f'{metric.capitalize()} by {attribute}', fontsize=14)
        plt.tight_layout()
        
        # Add reference line for overall
        overall_val = self.analysis_results['overall_metrics'].get(metric)
        if overall_val:
            plt.axhline(y=overall_val, color='red', linestyle='--', 
                       label=f'Overall: {overall_val:.3f}')
            plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.close()
    
    def plot_fairness_dashboard(
        self,
        save_path: str = None
    ) -> None:
        """
        Create a fairness dashboard with multiple metrics.
        """
        if not self.analysis_results:
            print("No analysis results. Run analyze_bias() first.")
            return
        
        n_attrs = len(self.analysis_results['attribute_analysis'])
        fig, axes = plt.subplots(n_attrs, 2, figsize=(14, 5*n_attrs))
        
        if n_attrs == 1:
            axes = axes.reshape(1, -1)
        
        for i, (attr, analysis) in enumerate(self.analysis_results['attribute_analysis'].items()):
            group_metrics = pd.DataFrame(analysis['group_metrics'])
            
            # Performance comparison
            ax1 = axes[i, 0]
            metrics_to_plot = ['accuracy', 'precision', 'recall']
            x = np.arange(len(group_metrics))
            width = 0.25
            
            for j, metric in enumerate(metrics_to_plot):
                ax1.bar(x + j*width, group_metrics[metric], width, label=metric.capitalize())
            
            ax1.set_xlabel('Group')
            ax1.set_ylabel('Score')
            ax1.set_title(f'Performance by {attr}')
            ax1.set_xticks(x + width)
            ax1.set_xticklabels(group_metrics['group'], rotation=45, ha='right')
            ax1.legend()
            ax1.set_ylim(0, 1.1)
            
            # Prediction rate comparison
            ax2 = axes[i, 1]
            ax2.bar(x, group_metrics['pred_positive_rate'], color='steelblue', 
                   edgecolor='black', label='Positive Prediction Rate')
            ax2.bar(x, group_metrics['prevalence'], alpha=0.5, color='orange',
                   edgecolor='black', label='Actual Prevalence')
            
            ax2.set_xlabel('Group')
            ax2.set_ylabel('Rate')
            ax2.set_title(f'Prediction Rate vs Prevalence by {attr}')
            ax2.set_xticks(x)
            ax2.set_xticklabels(group_metrics['group'], rotation=45, ha='right')
            ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Dashboard saved to {save_path}")
        
        plt.close()


if __name__ == "__main__":
    # Demonstration
    from data_ingestion import DataIngestion
    from preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer
    from model_training import ModelTrainer
    
    print("=" * 60)
    print("Bias Analysis Demonstration")
    print("=" * 60)
    
    # Generate and prepare data
    print("\n1. Preparing data...")
    ingestion = DataIngestion()
    df = ingestion.generate_synthetic_data(n_samples=1000)
    
    # Keep demographic info before preprocessing
    demographic_cols = ['age', 'gender', 'ethnicity']
    demographics = df[demographic_cols].copy()
    
    feature_engineer = FeatureEngineer()
    df_engineered = feature_engineer.transform(df)
    
    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        df_engineered, target_col='diabetes_risk'
    )
    
    # Align demographics with test set
    test_indices = X_test.index
    demographics_test = demographics.loc[test_indices].copy()
    
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Train model
    print("\n2. Training model...")
    trainer = ModelTrainer(algorithms=['xgboost'], cv_folds=3)
    models = trainer.train_all_models(X_train_processed, y_train, calibrate=True)
    model = models['xgboost']
    
    # Bias analysis
    print("\n3. Analyzing bias...")
    analyzer = BiasAnalyzer(protected_attributes=['age_group', 'gender'])
    
    # Add age groups to demographics
    demographics_test = analyzer.create_demographic_groups(
        demographics_test, age_col='age', gender_col='gender'
    )
    
    analysis = analyzer.analyze_bias(
        model=model,
        X=X_test_processed,
        y=y_test,
        demographic_data=demographics_test
    )
    
    # Print report
    print("\n" + analyzer.generate_bias_report())
