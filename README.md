# AI-Based Risk Prediction System for Chronic Diseases



### 🚀 [Live Demo](https://chronic-disease-risk-predictor.streamlit.app) — Try the interactive risk prediction app!

A comprehensive, modular AI system for predicting the risk of chronic diseases (diabetes, cardiovascular disease, kidney failure) from structured clinical and lifestyle data. Built with a focus on **interpretability**, **clinical trust**, **uncertainty quantification**, and **bias/fairness analysis**.

---

## 🎯 Features

- **Multi-Disease Prediction**: Predict risk for diabetes, cardiovascular disease, or kidney failure
- **Explainable AI**: SHAP-based explanations showing which factors contribute most to each prediction
- **Confidence Intervals**: Bootstrap-based uncertainty estimates with every prediction
- **Bias Detection**: Fairness metrics across demographic groups (age, gender, ethnicity)
- **Clinical Recommendations**: Personalized recommendations based on risk factors
- **Multiple Models**: Logistic Regression, Random Forest, XGBoost, LightGBM

---

## 📁 Project Structure

```
chronic_disease_risk_predictor/
├── config/
│   └── config.yaml              # Configuration settings
├── data/
│   ├── raw/                     # Raw input data
│   └── processed/               # Preprocessed data
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py        # Data loading and synthetic data generation
│   ├── preprocessing.py         # Data cleaning, normalization, encoding
│   ├── feature_engineering.py   # Derived features, clinical risk scores
│   ├── model_training.py        # Multi-model training with hyperparameter tuning
│   ├── model_evaluation.py      # Comprehensive evaluation metrics
│   ├── explainability.py        # SHAP-based explanations
│   ├── uncertainty.py           # Confidence intervals and calibration
│   ├── bias_analysis.py         # Fairness metrics and disparity detection
│   └── predictor.py             # Main prediction API
├── models/                      # Saved trained models
├── notebooks/
│   └── demo.ipynb               # Interactive demonstration
├── tests/
│   └── test_pipeline.py         # Unit tests
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/chronic_disease_risk_predictor.git
cd chronic_disease_risk_predictor

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

 📌Data Source and Training Strategy

This system was developed as a prototype to demonstrate an end-to-end Al-based chronic disease risk prediction pipeline.

For development and testing purposes, a combination of publicly available clinical datasets and synthetically generated samples was used. Synthetic data was generated to simulate realistic distributions of clinical variables while preserving privacy and avoiding the use of real patient-identifiable information.

Synthetic data was used strictly for demonstration and experimentation. In a real-world deployment, the model would be trained, validated, and calibrated using clinically collected datasets in collaboration with healthcare professionals.

 ### Run Demo

```bash
# Run the full demonstration
python -m src.predictor --demo
```

### Basic Usage

```python
from src.predictor import ChronicDiseaseRiskPredictor

# Initialize and train (uses synthetic data by default)
predictor = ChronicDiseaseRiskPredictor(target_disease='diabetes')
predictor.fit(generate_synthetic=True, n_samples=2000)

# Make a prediction
patient_data = {
    'age': 55,
    'gender': 'Male',
    'bmi': 32.5,
    'systolic_bp': 145,
    'diastolic_bp': 92,
    'fasting_glucose': 118,
    'hba1c': 6.2,
    'total_cholesterol': 225,
    'hdl_cholesterol': 42,
    'ldl_cholesterol': 145,
    'triglycerides': 190,
    'smoking_status': 'Former',
    'physical_activity_level': 'Light',
    'family_history_diabetes': 1,
    'hypertension_diagnosed': 1
    # ... other features
}

# Get prediction with explanation
result = predictor.predict(patient_data)
print(result)
```

### Output Example

```json
{
    "risk_score": 0.6842,
    "risk_percentage": "68.4%",
    "risk_category": "High",
    "risk_indicator": "🟠",
    "confidence_interval": [0.5921, 0.7654],
    "confidence_level": "95%",
    "top_contributing_factors": [
        {
            "feature": "fasting_glucose",
            "contribution": 0.1523,
            "direction": "increases risk",
            "interpretation": "Elevated fasting glucose suggests impaired glucose metabolism"
        },
        {
            "feature": "bmi",
            "contribution": 0.0987,
            "direction": "increases risk",
            "interpretation": "Elevated BMI indicates higher metabolic risk"
        }
    ],
    "recommendations": [
        "Monitor fasting blood glucose regularly",
        "Aim for gradual weight loss (5-10% of body weight)",
        "Monitor blood pressure at home"
    ]
}
```

---

## 📊 Input Features

### Demographics
| Feature | Type | Description |
|---------|------|-------------|
| age | numeric | Patient age (18-120) |
| gender | categorical | Male, Female, Other |
| ethnicity | categorical | Caucasian, African American, Hispanic, Asian, Other |

### Vitals
| Feature | Type | Description |
|---------|------|-------------|
| systolic_bp | numeric | Systolic blood pressure (mmHg) |
| diastolic_bp | numeric | Diastolic blood pressure (mmHg) |
| heart_rate | numeric | Heart rate (bpm) |
| bmi | numeric | Body Mass Index |

### Laboratory Results
| Feature | Type | Description |
|---------|------|-------------|
| fasting_glucose | numeric | Fasting blood glucose (mg/dL) |
| hba1c | numeric | Glycated hemoglobin (%) |
| total_cholesterol | numeric | Total cholesterol (mg/dL) |
| hdl_cholesterol | numeric | HDL cholesterol (mg/dL) |
| ldl_cholesterol | numeric | LDL cholesterol (mg/dL) |
| triglycerides | numeric | Triglycerides (mg/dL) |
| creatinine | numeric | Serum creatinine (mg/dL) |
| egfr | numeric | Estimated GFR (mL/min/1.73m²) |

### Lifestyle
| Feature | Type | Description |
|---------|------|-------------|
| smoking_status | categorical | Never, Former, Current |
| alcohol_consumption | categorical | None, Light, Moderate, Heavy |
| physical_activity_level | categorical | Sedentary, Light, Moderate, Active |
| diet_quality_score | numeric | Diet quality score (0-100) |

### Medical History
| Feature | Type | Description |
|---------|------|-------------|
| family_history_diabetes | binary | Family history of diabetes (0/1) |
| family_history_cvd | binary | Family history of CVD (0/1) |
| hypertension_diagnosed | binary | Diagnosed hypertension (0/1) |

---

## 🔬 Model Details

### Algorithms Supported
- **Logistic Regression**: Interpretable baseline model
- **Random Forest**: Ensemble of decision trees
- **XGBoost**: Gradient boosting (default)
- **LightGBM**: Fast gradient boosting

### Evaluation Metrics
- AUC-ROC and AUC-PR curves
- Calibration plots (Brier score, ECE)
- Confusion matrix analysis
- Clinical utility metrics (PPV, NPV, NNT)

### Feature Engineering
- BMI categories (WHO classification)
- Blood pressure stages
- Lipid ratios (TC/HDL, LDL/HDL, TG/HDL)
- Clinical risk scores (Framingham-like, FINDRISC-like)
- Metabolic syndrome indicators

---

## ⚖️ Fairness & Bias Analysis

The system includes comprehensive bias analysis across demographic groups:

```python
from src.bias_analysis import BiasAnalyzer

analyzer = BiasAnalyzer(protected_attributes=['age_group', 'gender'])
analysis = analyzer.analyze_bias(model, X_test, y_test, demographics)
print(analyzer.generate_bias_report())
```

### Fairness Metrics Computed
- **Demographic Parity**: Equal positive prediction rates across groups
- **Equalized Odds**: Equal TPR and FPR across groups
- **Predictive Parity**: Equal precision across groups

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

---

## 📚 API Reference

### ChronicDiseaseRiskPredictor

```python
# Main class for risk prediction
predictor = ChronicDiseaseRiskPredictor(
    model_path=None,           # Path to saved model
    config_path=None,          # Path to config file
    target_disease='diabetes'  # 'diabetes', 'cvd', or 'kidney'
)

# Train the model
predictor.fit(
    data=None,                 # Training data (optional)
    generate_synthetic=True,   # Generate synthetic data if none provided
    n_samples=2000,            # Number of synthetic samples
    algorithm='xgboost'        # Model algorithm
)

# Make prediction
result = predictor.predict(
    patient_data,              # Dict or DataFrame
    include_explanation=True,  # Include SHAP explanations
    include_confidence=True,   # Include confidence intervals
    include_recommendations=True  # Include clinical recommendations
)

# Generate full report
report = predictor.generate_report(patient_data, patient_id='P001')

# Save/load model
predictor.save_model('models/diabetes_model.joblib')
predictor.load_model('models/diabetes_model.joblib')
```

---

## ⚠️ Disclaimer

This risk prediction system is intended to **support clinical decision-making** and should **not replace medical judgment**. The predictions and recommendations provided are based on statistical models and should be interpreted by qualified healthcare professionals in the context of each patient's complete clinical picture.
This prediction is generated for educational and decision-support purposes only and is not intended for clinical diagnosis.
---

## 📧 Contact

For questions or support, please open an issue on GitHub.
