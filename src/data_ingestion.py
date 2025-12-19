"""
Data Ingestion Module for Chronic Disease Risk Prediction System

This module handles:
- Loading clinical data from various sources (CSV, JSON)
- Schema validation
- Synthetic data generation for demonstration purposes
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import yaml


class DataIngestion:
    """
    Handles data loading, validation, and synthetic data generation
    for the chronic disease risk prediction system.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the data ingestion module.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.schema = self._define_schema()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _define_schema(self) -> Dict:
        """
        Define the expected schema for clinical data.
        Returns a dictionary with column names, types, and valid ranges.
        """
        return {
            # Demographics
            'patient_id': {'type': 'string', 'required': True},
            'age': {'type': 'numeric', 'min': 18, 'max': 120, 'required': True},
            'gender': {'type': 'categorical', 'values': ['Male', 'Female', 'Other'], 'required': True},
            'ethnicity': {'type': 'categorical', 'values': ['Caucasian', 'African American', 'Hispanic', 'Asian', 'Other'], 'required': False},
            
            # Vitals
            'systolic_bp': {'type': 'numeric', 'min': 70, 'max': 250, 'required': True},
            'diastolic_bp': {'type': 'numeric', 'min': 40, 'max': 150, 'required': True},
            'heart_rate': {'type': 'numeric', 'min': 40, 'max': 200, 'required': False},
            'bmi': {'type': 'numeric', 'min': 12, 'max': 70, 'required': True},
            
            # Laboratory Results
            'fasting_glucose': {'type': 'numeric', 'min': 50, 'max': 500, 'required': True},
            'hba1c': {'type': 'numeric', 'min': 3.0, 'max': 15.0, 'required': True},
            'total_cholesterol': {'type': 'numeric', 'min': 100, 'max': 400, 'required': True},
            'hdl_cholesterol': {'type': 'numeric', 'min': 20, 'max': 120, 'required': True},
            'ldl_cholesterol': {'type': 'numeric', 'min': 40, 'max': 300, 'required': True},
            'triglycerides': {'type': 'numeric', 'min': 30, 'max': 1000, 'required': True},
            'creatinine': {'type': 'numeric', 'min': 0.3, 'max': 15.0, 'required': True},
            'egfr': {'type': 'numeric', 'min': 5, 'max': 150, 'required': False},
            
            # Lifestyle
            'smoking_status': {'type': 'categorical', 'values': ['Never', 'Former', 'Current'], 'required': True},
            'alcohol_consumption': {'type': 'categorical', 'values': ['None', 'Light', 'Moderate', 'Heavy'], 'required': False},
            'physical_activity_level': {'type': 'categorical', 'values': ['Sedentary', 'Light', 'Moderate', 'Active'], 'required': True},
            'diet_quality_score': {'type': 'numeric', 'min': 0, 'max': 100, 'required': False},
            
            # Medical History
            'family_history_diabetes': {'type': 'binary', 'required': True},
            'family_history_cvd': {'type': 'binary', 'required': True},
            'family_history_kidney_disease': {'type': 'binary', 'required': False},
            'hypertension_diagnosed': {'type': 'binary', 'required': True},
            'previous_cardiovascular_event': {'type': 'binary', 'required': False},
            
            # Target Variables
            'diabetes_risk': {'type': 'binary', 'required': False},
            'cvd_risk': {'type': 'binary', 'required': False},
            'kidney_disease_risk': {'type': 'binary', 'required': False}
        }
    
    def load_csv(self, file_path: str, validate: bool = True) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            file_path: Path to CSV file
            validate: Whether to validate the data against schema
            
        Returns:
            Pandas DataFrame with loaded data
        """
        df = pd.read_csv(file_path)
        
        if validate:
            self.validate_data(df)
        
        return df
    
    def load_json(self, file_path: str, validate: bool = True) -> pd.DataFrame:
        """
        Load data from a JSON file.
        
        Args:
            file_path: Path to JSON file
            validate: Whether to validate the data against schema
            
        Returns:
            Pandas DataFrame with loaded data
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        
        if validate:
            self.validate_data(df)
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate data against the defined schema.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list of validation errors)
        """
        errors = []
        
        # Check required columns
        for col, specs in self.schema.items():
            if specs.get('required', False) and col not in df.columns:
                errors.append(f"Required column '{col}' is missing")
        
        # Validate data types and ranges
        for col in df.columns:
            if col not in self.schema:
                continue
                
            specs = self.schema[col]
            
            if specs['type'] == 'numeric':
                # Check numeric range
                if 'min' in specs:
                    invalid_count = (df[col].dropna() < specs['min']).sum()
                    if invalid_count > 0:
                        errors.append(f"Column '{col}' has {invalid_count} values below minimum {specs['min']}")
                
                if 'max' in specs:
                    invalid_count = (df[col].dropna() > specs['max']).sum()
                    if invalid_count > 0:
                        errors.append(f"Column '{col}' has {invalid_count} values above maximum {specs['max']}")
            
            elif specs['type'] == 'categorical':
                # Check categorical values
                if 'values' in specs:
                    invalid_values = set(df[col].dropna().unique()) - set(specs['values'])
                    if invalid_values:
                        errors.append(f"Column '{col}' has invalid values: {invalid_values}")
            
            elif specs['type'] == 'binary':
                # Check binary values (0, 1)
                valid_values = {0, 1, True, False}
                invalid_values = set(df[col].dropna().unique()) - valid_values
                if invalid_values:
                    errors.append(f"Column '{col}' has non-binary values: {invalid_values}")
        
        is_valid = len(errors) == 0
        
        if errors:
            print("Validation Warnings:")
            for error in errors:
                print(f"  - {error}")
        
        return is_valid, errors
    
    def generate_synthetic_data(
        self, 
        n_samples: int = 1000, 
        random_state: int = 42,
        disease_prevalence: Dict[str, float] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic clinical data for demonstration and testing.
        
        The synthetic data is designed to have realistic correlations between
        risk factors and disease outcomes.
        
        Args:
            n_samples: Number of patient records to generate
            random_state: Random seed for reproducibility
            disease_prevalence: Dict of disease prevalence rates
            
        Returns:
            DataFrame with synthetic patient data
        """
        np.random.seed(random_state)
        
        if disease_prevalence is None:
            disease_prevalence = {
                'diabetes': 0.15,
                'cvd': 0.12,
                'kidney_disease': 0.08
            }
        
        # Generate demographics
        data = {
            'patient_id': [f'P{str(i).zfill(6)}' for i in range(n_samples)],
            'age': np.random.normal(55, 15, n_samples).clip(18, 95).astype(int),
            'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.48, 0.52]),
            'ethnicity': np.random.choice(
                ['Caucasian', 'African American', 'Hispanic', 'Asian', 'Other'],
                n_samples,
                p=[0.60, 0.13, 0.18, 0.06, 0.03]
            )
        }
        
        # Age-dependent BMI (older tends to have slightly higher BMI)
        age_factor = (data['age'] - 18) / 77  # Normalize age
        base_bmi = np.random.normal(26, 5, n_samples)
        data['bmi'] = (base_bmi + age_factor * 3).clip(16, 55)
        
        # BMI-dependent blood pressure
        bmi_factor = (data['bmi'] - 18) / 37
        data['systolic_bp'] = (np.random.normal(120, 15, n_samples) + bmi_factor * 25).clip(85, 200).astype(int)
        data['diastolic_bp'] = (np.random.normal(75, 10, n_samples) + bmi_factor * 15).clip(50, 120).astype(int)
        data['heart_rate'] = np.random.normal(72, 12, n_samples).clip(45, 120).astype(int)
        
        # Lifestyle factors
        data['smoking_status'] = np.random.choice(
            ['Never', 'Former', 'Current'], n_samples, p=[0.55, 0.30, 0.15]
        )
        data['alcohol_consumption'] = np.random.choice(
            ['None', 'Light', 'Moderate', 'Heavy'], n_samples, p=[0.30, 0.40, 0.22, 0.08]
        )
        data['physical_activity_level'] = np.random.choice(
            ['Sedentary', 'Light', 'Moderate', 'Active'], n_samples, p=[0.25, 0.35, 0.28, 0.12]
        )
        data['diet_quality_score'] = np.random.normal(55, 18, n_samples).clip(10, 95)
        
        # Medical history (correlated with age)
        age_risk_factor = (np.array(data['age']) - 30) / 65
        data['family_history_diabetes'] = (np.random.random(n_samples) < 0.25).astype(int)
        data['family_history_cvd'] = (np.random.random(n_samples) < 0.20).astype(int)
        data['family_history_kidney_disease'] = (np.random.random(n_samples) < 0.10).astype(int)
        data['hypertension_diagnosed'] = (np.random.random(n_samples) < (0.15 + age_risk_factor * 0.35)).astype(int)
        data['previous_cardiovascular_event'] = (np.random.random(n_samples) < (0.02 + age_risk_factor * 0.12)).astype(int)
        
        # Laboratory values - correlated with risk factors
        diabetes_risk_score = self._calculate_raw_risk_score(data, 'diabetes')
        cvd_risk_score = self._calculate_raw_risk_score(data, 'cvd')
        kidney_risk_score = self._calculate_raw_risk_score(data, 'kidney')
        
        # Fasting glucose (correlated with diabetes risk)
        base_glucose = np.random.normal(95, 15, n_samples)
        data['fasting_glucose'] = (base_glucose + diabetes_risk_score * 60).clip(65, 350)
        
        # HbA1c (correlated with glucose and diabetes)
        data['hba1c'] = (4.5 + (data['fasting_glucose'] - 70) / 50 + np.random.normal(0, 0.5, n_samples)).clip(4.0, 14.0)
        
        # Cholesterol (correlated with CVD risk)
        data['total_cholesterol'] = (np.random.normal(190, 35, n_samples) + cvd_risk_score * 50).clip(120, 350)
        data['hdl_cholesterol'] = (np.random.normal(55, 15, n_samples) - cvd_risk_score * 15).clip(25, 100)
        data['ldl_cholesterol'] = (data['total_cholesterol'] - data['hdl_cholesterol'] - 
                                   np.random.normal(30, 10, n_samples)).clip(50, 250)
        data['triglycerides'] = (np.random.normal(130, 60, n_samples) + bmi_factor * 80).clip(40, 600)
        
        # Kidney function (correlated with age and kidney risk)
        data['creatinine'] = (np.random.normal(1.0, 0.3, n_samples) + kidney_risk_score * 1.5 + age_factor * 0.3).clip(0.5, 8.0)
        data['egfr'] = (120 - age_factor * 30 - kidney_risk_score * 40 + np.random.normal(0, 15, n_samples)).clip(15, 130)
        
        # Generate target variables based on risk scores
        data['diabetes_risk'] = self._generate_outcome(diabetes_risk_score, disease_prevalence['diabetes'])
        data['cvd_risk'] = self._generate_outcome(cvd_risk_score, disease_prevalence['cvd'])
        data['kidney_disease_risk'] = self._generate_outcome(kidney_risk_score, disease_prevalence['kidney_disease'])
        
        df = pd.DataFrame(data)
        
        # Round numerical columns
        for col in ['bmi', 'fasting_glucose', 'hba1c', 'total_cholesterol', 'hdl_cholesterol', 
                    'ldl_cholesterol', 'triglycerides', 'creatinine', 'egfr', 'diet_quality_score']:
            df[col] = df[col].round(1)
        
        return df
    
    def _calculate_raw_risk_score(self, data: Dict, disease: str) -> np.ndarray:
        """
        Calculate a raw risk score based on known risk factors.
        Used internally for generating correlated synthetic data.
        """
        n = len(data['age'])
        score = np.zeros(n)
        
        # Age contribution
        age_contrib = (np.array(data['age']) - 30) / 50
        score += age_contrib * 0.3
        
        # BMI contribution
        bmi_contrib = (np.array(data['bmi']) - 22) / 20
        score += bmi_contrib * 0.2
        
        # Blood pressure contribution
        bp_contrib = (np.array(data['systolic_bp']) - 110) / 80
        score += bp_contrib * 0.15
        
        # Smoking contribution
        smoking_map = {'Never': 0, 'Former': 0.5, 'Current': 1.0}
        smoking_contrib = np.array([smoking_map[s] for s in data['smoking_status']])
        score += smoking_contrib * 0.15
        
        # Family history
        if disease == 'diabetes':
            score += np.array(data['family_history_diabetes']) * 0.2
        elif disease == 'cvd':
            score += np.array(data['family_history_cvd']) * 0.2
        else:
            score += np.array(data['family_history_kidney_disease']) * 0.2
        
        # Add noise
        score += np.random.normal(0, 0.15, n)
        
        return score.clip(0, 1)
    
    def _generate_outcome(self, risk_score: np.ndarray, base_prevalence: float) -> np.ndarray:
        """
        Generate binary outcomes based on risk scores.
        Uses logistic transformation to convert scores to probabilities.
        """
        # Logistic transformation
        odds = np.exp(3 * (risk_score - 0.5))
        probability = odds / (1 + odds)
        
        # Adjust for desired prevalence
        probability = probability * (base_prevalence / probability.mean()) if probability.mean() > 0 else probability
        probability = probability.clip(0.01, 0.99)
        
        return (np.random.random(len(risk_score)) < probability).astype(int)
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Return feature columns grouped by category.
        """
        return {
            'demographic': ['age', 'gender', 'ethnicity'],
            'vitals': ['systolic_bp', 'diastolic_bp', 'heart_rate', 'bmi'],
            'laboratory': ['fasting_glucose', 'hba1c', 'total_cholesterol', 'hdl_cholesterol',
                          'ldl_cholesterol', 'triglycerides', 'creatinine', 'egfr'],
            'lifestyle': ['smoking_status', 'alcohol_consumption', 'physical_activity_level', 'diet_quality_score'],
            'medical_history': ['family_history_diabetes', 'family_history_cvd', 'family_history_kidney_disease',
                               'hypertension_diagnosed', 'previous_cardiovascular_event']
        }
    
    def get_target_columns(self) -> List[str]:
        """Return list of target variable columns."""
        return ['diabetes_risk', 'cvd_risk', 'kidney_disease_risk']


if __name__ == "__main__":
    # Demonstration
    ingestion = DataIngestion()
    
    print("Generating synthetic clinical data...")
    df = ingestion.generate_synthetic_data(n_samples=1000)
    
    print(f"\nGenerated {len(df)} patient records")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample data:")
    print(df.head())
    
    print(f"\nTarget variable distributions:")
    for target in ingestion.get_target_columns():
        print(f"  {target}: {df[target].mean():.2%} positive")
    
    # Validate the generated data
    print("\nValidating synthetic data...")
    is_valid, errors = ingestion.validate_data(df)
    print(f"Validation passed: {is_valid}")
