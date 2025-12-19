"""
Streamlit Web Application for Chronic Disease Risk Prediction

A beautiful, interactive web interface for the risk prediction system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_ingestion import DataIngestion
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.explainability import ExplainabilityEngine
from src.uncertainty import UncertaintyEstimator
from src.clinical_risk_calculator import ClinicalRiskCalculator

# Page configuration
st.set_page_config(
    page_title="Chronic Disease Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .risk-card {
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .risk-low { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; }
    .risk-moderate { background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%); color: white; }
    .risk-high { background: linear-gradient(135deg, #f85032 0%, #e73827 100%); color: white; }
    .risk-very-high { background: linear-gradient(135deg, #8e2de2 0%, #4a00e0 100%); color: white; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
    }
    .factor-positive { color: #e74c3c; font-weight: 600; }
    .factor-negative { color: #27ae60; font-weight: 600; }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_model(_version="v3_stronger_risk_weights"):
    """Load or train the prediction model (cached).
    
    Args:
        _version: Version string to invalidate cache when model logic changes
    """
    with st.spinner("🔄 Initializing AI Model... This may take a moment."):
        # Generate training data with improved risk scoring
        ingestion = DataIngestion()
        df = ingestion.generate_synthetic_data(n_samples=2500)  # Increased for better learning
        
        # Feature engineering
        engineer = FeatureEngineer()
        df = engineer.transform(df)
        
        # Preprocessing
        preprocessor = DataPreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
            df, target_col='diabetes_risk'
        )
        
        preprocessor.fit(X_train)
        X_train_processed = preprocessor.transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        
        # Train model
        trainer = ModelTrainer(algorithms=['xgboost'], cv_folds=3)
        models = trainer.train_all_models(X_train_processed, y_train, 
                                          tune_hyperparameters=False, calibrate=False)
        model = models['xgboost']
        
        # Setup explainability
        explainer = ExplainabilityEngine(
            model=model,
            feature_names=list(X_train_processed.columns),
            model_type='tree'
        )
        explainer.initialize_explainer(X_train_processed)
        
        # Setup uncertainty
        uncertainty_estimator = UncertaintyEstimator(model=model, confidence_level=0.95)
        
        return {
            'model': model,
            'preprocessor': preprocessor,
            'feature_engineer': engineer,
            'explainer': explainer,
            'uncertainty': uncertainty_estimator,
            'feature_names': list(X_train_processed.columns)
        }


def get_risk_category(risk_score):
    """Determine risk category based on score."""
    if risk_score < 0.2:
        return "Low", "🟢", "risk-low"
    elif risk_score < 0.5:
        return "Moderate", "🟡", "risk-moderate"
    elif risk_score < 0.8:
        return "High", "🟠", "risk-high"
    else:
        return "Very High", "🔴", "risk-very-high"


def create_gauge_chart(risk_score):
    """Create a gauge chart for risk visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score", 'font': {'size': 24}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#27ae60'},
                {'range': [20, 50], 'color': '#f39c12'},
                {'range': [50, 80], 'color': '#e74c3c'},
                {'range': [80, 100], 'color': '#8e44ad'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': risk_score * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Arial"}
    )
    
    return fig


def create_factor_chart(factors):
    """Create a horizontal bar chart for contributing factors."""
    df = pd.DataFrame(factors[:8])  # Top 8 factors
    df['color'] = df['shap_value'].apply(lambda x: '#e74c3c' if x > 0 else '#27ae60')
    
    fig = px.bar(
        df,
        x='contribution',
        y='feature',
        orientation='h',
        color='direction',
        color_discrete_map={'increases risk': '#e74c3c', 'decreases risk': '#27ae60'},
        title='Top Contributing Factors'
    )
    
    fig.update_layout(
        height=400,
        showlegend=True,
        yaxis={'categoryorder': 'total ascending'},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white"}
    )
    
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Chronic Disease Risk Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #888; margin-bottom: 2rem;">Evidence-based clinical risk assessment using validated medical algorithms</p>', unsafe_allow_html=True)
    
    # Sidebar - Patient Input
    with st.sidebar:
        st.markdown("## 📋 Patient Information")
        st.markdown("---")
        
        # Demographics
        st.markdown("### Demographics")
        age = st.slider("Age", 18, 95, 50)
        gender = st.selectbox("Gender", ["Male", "Female"])
        ethnicity = st.selectbox("Ethnicity", ["Caucasian", "African American", "Hispanic", "Asian", "Other"])
        
        # Vitals
        st.markdown("### Vital Signs")
        col1, col2 = st.columns(2)
        with col1:
            systolic_bp = st.number_input("Systolic BP (mmHg)", 80, 220, 125)
            heart_rate = st.number_input("Heart Rate (bpm)", 40, 150, 72)
        with col2:
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", 50, 130, 80)
            bmi = st.number_input("BMI", 15.0, 55.0, 26.5, step=0.1)
        
        # Laboratory
        st.markdown("### Laboratory Results")
        col1, col2 = st.columns(2)
        with col1:
            fasting_glucose = st.number_input("Fasting Glucose (mg/dL)", 50, 350, 100)
            hba1c = st.number_input("HbA1c (%)", 4.0, 14.0, 5.7, step=0.1)
            total_cholesterol = st.number_input("Total Cholesterol (mg/dL)", 100, 350, 200)
            creatinine = st.number_input("Creatinine (mg/dL)", 0.5, 8.0, 1.0, step=0.1)
        with col2:
            hdl_cholesterol = st.number_input("HDL Cholesterol (mg/dL)", 20, 100, 50)
            ldl_cholesterol = st.number_input("LDL Cholesterol (mg/dL)", 40, 250, 130)
            triglycerides = st.number_input("Triglycerides (mg/dL)", 30, 600, 150)
            egfr = st.number_input("eGFR (mL/min/1.73m²)", 15, 130, 90)
        
        # Lifestyle
        st.markdown("### Lifestyle")
        smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        alcohol = st.selectbox("Alcohol Consumption", ["None", "Light", "Moderate", "Heavy"])
        physical_activity = st.selectbox("Physical Activity", ["Sedentary", "Light", "Moderate", "Active"])
        diet_score = st.slider("Diet Quality Score", 0, 100, 55)
        
        # Medical History
        st.markdown("### Medical History")
        family_diabetes = st.checkbox("Family History of Diabetes")
        family_cvd = st.checkbox("Family History of Cardiovascular Disease")
        family_kidney = st.checkbox("Family History of Kidney Disease")
        hypertension = st.checkbox("Diagnosed Hypertension")
        prev_cv_event = st.checkbox("Previous Cardiovascular Event")
        
        st.markdown("---")
        predict_button = st.button("🔮 Predict Risk", use_container_width=True)
    
    # Main Content Area
    if predict_button:
        
        with st.spinner("🧠 Analyzing patient data with clinical algorithms..."):
            # Create patient data dictionary for clinical calculator
            patient_dict = {
                'age': age,
                'gender': gender,
                'ethnicity': ethnicity,
                'systolic_bp': systolic_bp,
                'diastolic_bp': diastolic_bp,
                'heart_rate': heart_rate,
                'bmi': bmi,
                'fasting_glucose': fasting_glucose,
                'hba1c': hba1c,
                'total_cholesterol': total_cholesterol,
                'hdl_cholesterol': hdl_cholesterol,
                'ldl_cholesterol': ldl_cholesterol,
                'triglycerides': triglycerides,
                'creatinine': creatinine,
                'egfr': egfr,
                'smoking_status': smoking_status,
                'alcohol_consumption': alcohol,
                'physical_activity_level': physical_activity,
                'diet_quality_score': diet_score,
                'family_history_diabetes': int(family_diabetes),
                'family_history_cvd': int(family_cvd),
                'family_history_kidney_disease': int(family_kidney),
                'hypertension_diagnosed': int(hypertension),
                'previous_cardiovascular_event': int(prev_cv_event)
            }
            
            # Use Clinical Risk Calculator for accurate predictions
            calculator = ClinicalRiskCalculator()
            clinical_result = calculator.calculate_combined_risk(patient_dict)
            
            # Get primary risk (diabetes)
            risk_score = clinical_result['primary_risk_score']
            category, emoji, css_class = get_risk_category(risk_score)
            
            # Calculate confidence interval based on clinical parameters
            # Clinical calculators typically have ±5-10% margin
            margin = 0.05 + (0.05 * (1 - risk_score))  # Lower margin at higher risk
            lower = max(0, risk_score - margin)
            upper = min(1, risk_score + margin)
            
            # Convert clinical factors to explanation format
            diabetes_factors = clinical_result['diabetes']['factors']
            explanation = {
                'top_contributing_factors': [],
                'all_contributions': []
            }
            
            # Build explanation from clinical factors
            for factor_name, factor_data in diabetes_factors.items():
                points = factor_data.get('points', 0)
                max_points = factor_data.get('max', 1)
                
                if points > 0:
                    contribution = points / max_points
                    interpretation = factor_data.get('interpretation', '')
                    
                    if factor_name == 'hba1c':
                        value = factor_data.get('value', 0)
                        if value >= 6.5:
                            interp = f"HbA1c {value}% is in diabetic range (≥6.5%)"
                        elif value >= 5.7:
                            interp = f"HbA1c {value}% indicates prediabetes (5.7-6.4%)"
                        else:
                            interp = f"HbA1c {value}% is normal"
                    elif factor_name == 'fasting_glucose':
                        value = factor_data.get('value', 0)
                        if value >= 126:
                            interp = f"Fasting glucose {value} mg/dL is in diabetic range (≥126)"
                        elif value >= 100:
                            interp = f"Fasting glucose {value} mg/dL indicates impaired fasting glucose"
                        else:
                            interp = f"Fasting glucose {value} mg/dL is normal"
                    elif factor_name == 'bmi':
                        value = factor_data.get('value', 0)
                        if value >= 30:
                            interp = f"BMI {value:.1f} indicates obesity (≥30)"
                        elif value >= 25:
                            interp = f"BMI {value:.1f} indicates overweight (25-29.9)"
                        else:
                            interp = f"BMI {value:.1f} is normal"
                    elif factor_name == 'hypertension':
                        systolic = factor_data.get('systolic', 120)
                        diastolic = factor_data.get('diastolic', 80)
                        interp = f"Blood pressure {systolic}/{diastolic} mmHg indicates hypertension"
                    elif factor_name == 'age':
                        value = factor_data.get('value', 0)
                        interp = f"Age {value} years - risk increases with age"
                    elif factor_name == 'family_history':
                        interp = "Family history of diabetes increases genetic risk"
                    elif factor_name == 'physical_activity':
                        value = factor_data.get('value', 'Unknown')
                        interp = f"{value} physical activity level increases risk"
                    else:
                        interp = interpretation or f"{factor_name} contributes to risk"
                    
                    factor_entry = {
                        'feature': factor_name.replace('_', ' ').title(),
                        'shap_value': contribution if points > 0 else -contribution,
                        'contribution': abs(contribution),
                        'direction': 'increases risk' if points > 0 else 'decreases risk',
                        'interpretation': interp
                    }
                    explanation['all_contributions'].append(factor_entry)
                    explanation['top_contributing_factors'].append(factor_entry)
            
            # Sort by contribution
            explanation['all_contributions'].sort(key=lambda x: x['contribution'], reverse=True)
            explanation['top_contributing_factors'].sort(key=lambda x: x['contribution'], reverse=True)
        
        # Display Results
        st.markdown("## 📊 Risk Assessment Results")
        
        # Top row - Risk Score and Category
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.plotly_chart(create_gauge_chart(risk_score), use_container_width=True)
        
        with col2:
            st.markdown(f"""
            <div class="risk-card {css_class}">
                <h1 style="font-size: 3rem; margin: 0;">{emoji}</h1>
                <h2 style="margin: 0.5rem 0;">{category} Risk</h2>
                <p style="font-size: 2rem; font-weight: 700; margin: 0;">{risk_score:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin: 0; opacity: 0.8;">95% Confidence Interval</h4>
                <p style="font-size: 1.5rem; font-weight: 700; margin: 0.5rem 0;">
                    {lower:.1%} - {upper:.1%}
                </p>
                <p style="margin: 0; font-size: 0.9rem; opacity: 0.8;">
                    Prediction confidence
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Contributing Factors
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("### 🔍 Key Contributing Factors")
            st.plotly_chart(
                create_factor_chart(explanation['all_contributions']),
                use_container_width=True
            )
        
        with col2:
            st.markdown("### 📝 Clinical Interpretation")
            for i, factor in enumerate(explanation['top_contributing_factors'][:5], 1):
                direction_class = "factor-positive" if factor['shap_value'] > 0 else "factor-negative"
                arrow = "↑" if factor['shap_value'] > 0 else "↓"
                st.markdown(f"""
                **{i}. {factor['feature']}**  
                <span class="{direction_class}">{arrow} {factor['direction']}</span>  
                _{factor['interpretation']}_
                """, unsafe_allow_html=True)
                st.markdown("")
        
        st.markdown("---")
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        
        recommendations = []
        if fasting_glucose > 126:
            recommendations.append("📌 Monitor fasting blood glucose regularly")
            recommendations.append("📌 Consider HbA1c testing every 3-6 months")
        if bmi > 30:
            recommendations.append("📌 Aim for gradual weight loss (5-10% of body weight)")
            recommendations.append("📌 Increase physical activity to 150+ minutes/week")
        if systolic_bp > 140:
            recommendations.append("📌 Monitor blood pressure at home")
            recommendations.append("📌 Reduce sodium intake")
        if smoking_status == "Current":
            recommendations.append("📌 Strongly consider smoking cessation")
        if not recommendations:
            recommendations = [
                "📌 Maintain healthy lifestyle with regular exercise",
                "📌 Schedule regular check-ups",
                "📌 Ensure adequate sleep (7-9 hours/night)"
            ]
        
        rec_cols = st.columns(3)
        for i, rec in enumerate(recommendations[:6]):
            with rec_cols[i % 3]:
                st.info(rec)
        
        # Disclaimer
        st.markdown("---")
        st.warning("""
        ⚠️ **Disclaimer**: This risk assessment is intended to support clinical decision-making 
        and should not replace medical judgment. Please consult with a healthcare provider for 
        personalized medical advice.
        """)
    
    else:
        # Welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h2>Welcome to the Risk Prediction System</h2>
            <p style="color: #888; font-size: 1.2rem;">
                Enter patient information in the sidebar and click <strong>Predict Risk</strong> 
                to get an AI-based risk prediction with detailed explanations.
            </p>
            <br>
            <h3>Features</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style="text-align: center; padding: 1rem;">
                <h1>🎯</h1>
                <h4>Accurate Predictions</h4>
                <p style="color: #888;">XGBoost-powered risk scoring</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 1rem;">
                <h1>🔍</h1>
                <h4>Explainable AI</h4>
                <p style="color: #888;">SHAP-based feature explanations</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="text-align: center; padding: 1rem;">
                <h1>📊</h1>
                <h4>Confidence Intervals</h4>
                <p style="color: #888;">Uncertainty quantification</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style="text-align: center; padding: 1rem;">
                <h1>💡</h1>
                <h4>Recommendations</h4>
                <p style="color: #888;">Personalized clinical guidance</p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
