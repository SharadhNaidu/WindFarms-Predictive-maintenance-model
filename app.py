"""
Wind Turbine Predictive Maintenance Dashboard
Real-time monitoring and alerting system
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import tensorflow as tf
from datetime import datetime, timedelta
import time
import os
import sys

# Import optimization modules
try:
    from wind_farm_optimization import (
        WindFarmOptimizer, 
        PowerOutputForecaster, 
        PerformanceBenchmarking,
        calculate_energy_gains
    )
    from operational_cost_optimizer import (
        DowntimeCostCalculator,
        SmartMaintenanceScheduler,
        CostBenefitAnalyzer
    )
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    OPTIMIZATION_AVAILABLE = False
    print(f"Warning: Optimization modules not available: {e}")

# Page config
st.set_page_config(
    page_title="Wind Turbine Predictive Maintenance",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Custom CSS with improved styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main, .stApp {
        background: #ffffff !important;
    }
    
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 95% !important;
        background: #ffffff;
    }
    
    /* Ensure all text is visible */
    body, p, span, div, label {
        color: #1f2937 !important;
    }
    
    /* Headers */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        color: #1f2937 !important;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .sub-header {
        font-size: 1.3rem;
        text-align: center;
        color: #4b5563 !important;
        margin-bottom: 2.5rem;
        font-weight: 500;
    }
    
    /* Metric Cards */
    [data-testid="stMetric"] {
        background: #ffffff !important;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 2px solid #e5e7eb;
    }
    
    [data-testid="stMetric"] label {
        font-size: 0.875rem !important;
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #000000 !important;
    }
    
    /* Alert Boxes */
    .alert-critical {
        background: #fee2e2;
        color: #991b1b !important;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
        border-left: 5px solid #dc2626;
        margin: 1rem 0;
    }
    
    .alert-warning {
        background: #fef3c7;
        color: #92400e !important;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);
        border-left: 5px solid #f59e0b;
        margin: 1rem 0;
    }
    
    .alert-normal {
        background: #d1fae5;
        color: #065f46 !important;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        border-left: 5px solid #10b981;
        margin: 1rem 0;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1rem;
        margin: 0.5rem;
    }
    
    .status-healthy {
        background: #10b981;
        color: white !important;
    }
    
    .status-issue {
        background: #ef4444;
        color: white !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f3f4f6;
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: #1f2937 !important;
        font-weight: 600;
        padding: 0.8rem 1.5rem;
        transition: all 0.3s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e5e7eb;
        color: #1f2937 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: #2196F3 !important;
        color: white !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #1e40af !important;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] label {
        color: white !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: #2196F3 !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(33, 150, 243, 0.3) !important;
        transition: all 0.2s !important;
    }
    
    .stButton > button:hover {
        background: #1976D2 !important;
        box-shadow: 0 4px 12px rgba(33, 150, 243, 0.4) !important;
        transform: translateY(-1px);
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: #2196F3;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: #e5e7eb;
    }
    
    /* Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid #e5e7eb;
        margin: 1rem 0;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
        border-radius: 10px;
        font-weight: 600;
        color: #1f2937;
    }
    
    /* Info boxes */
    [data-testid="stNotification"][kind="info"],
    .stAlert[kind="info"] {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%) !important;
        border-left: 4px solid #3b82f6 !important;
        border-radius: 10px !important;
    }
    
    [data-testid="stNotification"][kind="success"],
    .stAlert[kind="success"] {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%) !important;
        border-left: 4px solid #10b981 !important;
        border-radius: 10px !important;
    }
    
    [data-testid="stNotification"][kind="warning"],
    .stAlert[kind="warning"] {
        background: linear-gradient(135deg, #fed7aa 0%, #fdba74 100%) !important;
        border-left: 4px solid #f59e0b !important;
        border-radius: 10px !important;
    }
    
    [data-testid="stNotification"][kind="error"],
    .stAlert[kind="error"] {
        background: linear-gradient(135deg, #fecaca 0%, #fca5a5 100%) !important;
        border-left: 4px solid #ef4444 !important;
        border-radius: 10px !important;
    }
    
    /* Section Headers */
    h1, h2, h3, h4 {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    
    h2 {
        border-bottom: 3px solid #2196F3 !important;
        padding-bottom: 0.5rem !important;
        margin-top: 2rem !important;
    }
    
    /* Improve spacing */
    .element-container {
        margin-bottom: 0.5rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1rem !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #667eea !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f3f4f6 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    
    /* Remove default Streamlit padding */
    .main .block-container {
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'failure_state' not in st.session_state:
    st.session_state.failure_state = {
        'gearbox_consecutive_fails': 0,
        'efficiency_consecutive_fails': 0,
        'generator_consecutive_fails': 0,
        'alignment_consecutive_fails': 0,
        'integrity_consecutive_fails': 0
    }
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []
if 'current_sample_index' not in st.session_state:
    st.session_state.current_sample_index = 0
if 'auto_cycle' not in st.session_state:
    st.session_state.auto_cycle = True

@st.cache_resource
def load_models():
    """Load all trained models"""
    base_path = os.path.dirname(__file__)
    models_path = os.path.join(base_path, 'trained_models')
    
    try:
        # Load LSTM models
        gearbox_model = tf.keras.models.load_model(
            os.path.join(models_path, 'gearbox_autoencoder.h5'),
            compile=False
        )
        generator_model = tf.keras.models.load_model(
            os.path.join(models_path, 'generator_autoencoder.h5'),
            compile=False
        )
        
        # Load scalers
        scaler_gb = joblib.load(os.path.join(models_path, 'scaler_gb.joblib'))
        scaler_gen = joblib.load(os.path.join(models_path, 'scaler_gen.joblib'))
        
        # Load other models
        power_curve_model = joblib.load(os.path.join(models_path, 'power_curve_model.joblib'))
        meta_model = joblib.load(os.path.join(models_path, 'meta_model_sensor_fusion_refined.joblib'))
        
        return {
            'gearbox': gearbox_model,
            'generator': generator_model,
            'scaler_gb': scaler_gb,
            'scaler_gen': scaler_gen,
            'power_curve': power_curve_model,
            'meta_model': meta_model
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

@st.cache_data
def load_data():
    """Load SCADA data"""
    try:
        base_path = os.path.dirname(__file__)
        data_path = os.path.join(base_path, 'Kelmarsh_SCADA_2020_3086', 
                                 'Turbine_Data_Kelmarsh_1_2020-01-01_-_2021-01-01_228.csv')
        
        df = pd.read_csv(data_path, skiprows=9, low_memory=False)
        df['# Date and time'] = pd.to_datetime(df['# Date and time'])
        df.set_index('# Date and time', inplace=True)
        df.sort_index(inplace=True)
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def prepare_test_data(df):
    """Prepare a subset of test data for selection"""
    # Select diverse samples from different parts of the dataset
    # Skip first 100 and last 100 rows, select every 100th row
    test_indices = list(range(100, len(df) - 100, 100))
    return test_indices

def calculate_health_score(error, healthy_thresh, failure_thresh):
    """Calculate health score from reconstruction error"""
    if error <= healthy_thresh:
        norm_error = 0
    elif error >= failure_thresh:
        norm_error = 1
    else:
        norm_error = (error - healthy_thresh) / (failure_thresh - healthy_thresh)
    
    health = (1 - norm_error) * 100
    return np.clip(health, 0, 100)

def get_pass_fail_status(score, test_type):
    """Determine pass/fail status based on score and test type"""
    thresholds = {
        'gearbox': {'pass': 60, 'warning': 30},
        'generator': {'pass': 60, 'warning': 30},
        'efficiency': {'pass': 90, 'warning': 50},
        'alignment': {'pass': 90, 'warning': 85},
        'integrity': {'pass': 90, 'warning': 80}
    }
    
    thresh = thresholds.get(test_type, {'pass': 60, 'warning': 30})
    
    if score >= thresh['pass']:
        return "✅ PASS", "green"
    elif score >= thresh['warning']:
        return "⚠️ WARNING", "orange"
    else:
        return "❌ FAIL", "red"

def process_current_data(df, models, index, sequence_length=24):
    """Process current data point and calculate all health scores"""
    
    if index < sequence_length:
        index = sequence_length
    
    # Get current data
    current_row = df.iloc[index]
    
    # Feature columns
    gb_features = [
        'Gear oil temperature (°C)',
        'Gear oil inlet temperature (°C)',
        'Generator RPM (RPM)'
    ]
    
    gen_features = [
        'Stator temperature 1 (°C)',
        'Generator bearing front temperature (°C)',
        'Generator bearing rear temperature (°C)',
        'Power (kW)'
    ]
    
    # Get sequence data
    sequence_data = df.iloc[index-sequence_length:index]
    
    # Gearbox health
    gb_data = sequence_data[gb_features].values
    gb_scaled = models['scaler_gb'].transform(gb_data)
    gb_sequence = gb_scaled.reshape(1, sequence_length, len(gb_features))
    
    gb_pred = models['gearbox'].predict(gb_sequence, verbose=0)
    gb_error = np.mean(np.abs(gb_pred - gb_sequence))
    
    # Use fixed thresholds (you can adjust these based on your training)
    HEALTHY_THRESHOLD_GB = 0.05
    FAILURE_THRESHOLD_GB = 0.25
    gearbox_health = calculate_health_score(gb_error, HEALTHY_THRESHOLD_GB, FAILURE_THRESHOLD_GB)
    
    # Generator health
    gen_data = sequence_data[gen_features].values
    gen_scaled = models['scaler_gen'].transform(gen_data)
    gen_sequence = gen_scaled.reshape(1, sequence_length, len(gen_features))
    
    gen_pred = models['generator'].predict(gen_sequence, verbose=0)
    gen_error = np.mean(np.abs(gen_pred - gen_sequence))
    
    HEALTHY_THRESHOLD_GEN = 0.05
    FAILURE_THRESHOLD_GEN = 0.25
    generator_health = calculate_health_score(gen_error, HEALTHY_THRESHOLD_GEN, FAILURE_THRESHOLD_GEN)
    
    # Blade/Rotor efficiency
    wind_speed = current_row['Wind speed (m/s)']
    actual_power = current_row['Power (kW)']
    predicted_power = models['power_curve'].predict([[wind_speed]])[0]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        efficiency = (predicted_power / actual_power * 100) if actual_power > 0 else 100
    efficiency = np.clip(efficiency, 0, 100)
    
    # Yaw/Pitch alignment
    wind_dir = current_row['Wind direction (°)']
    nacelle_pos = current_row['Nacelle position (°)']
    yaw_error = abs((nacelle_pos - wind_dir + 180) % 360 - 180)
    alignment_score = np.clip(100 - yaw_error, 0, 100)
    
    # Sensor validity (simplified)
    integrity_score = 100  # Default to perfect
    failed_sensors = 0
    
    # Check some basic ranges
    if not (-10 < current_row['Gear oil temperature (°C)'] < 110):
        failed_sensors += 1
    if not (0 <= current_row['Wind speed (m/s)'] < 50):
        failed_sensors += 1
    if not (-10 < current_row['Stator temperature 1 (°C)'] < 180):
        failed_sensors += 1
    
    integrity_score = max(0, 100 - (failed_sensors * 20))
    
    # Calculate overall system health using rule-based logic
    # This is more transparent and responsive than relying solely on the meta-model
    
    # Count critical failures and warnings
    critical_count = 0
    warning_count = 0
    
    if gearbox_health < 30:
        critical_count += 1
    elif gearbox_health < 60:
        warning_count += 1
        
    if generator_health < 30:
        critical_count += 1
    elif generator_health < 60:
        warning_count += 1
        
    if efficiency < 50:
        critical_count += 1
    elif efficiency < 90:
        warning_count += 1
        
    if alignment_score < 85:
        critical_count += 1
    elif alignment_score < 90:
        warning_count += 1
        
    if integrity_score < 80:
        critical_count += 1
    elif integrity_score < 90:
        warning_count += 1
    
    # Determine overall status based on component health
    if critical_count >= 1:
        overall_status_text = 'Potential Issue'
        confidence = 100 - (critical_count * 10 + warning_count * 5)
    elif warning_count >= 2:
        overall_status_text = 'Potential Issue'
        confidence = 100 - (warning_count * 5)
    else:
        overall_status_text = 'Healthy'
        # Calculate confidence based on average health
        avg_health = (gearbox_health + generator_health + efficiency + alignment_score + integrity_score) / 5
        confidence = min(99, avg_health)
    
    confidence = max(50, min(99, confidence))  # Keep between 50-99%
    
    return {
        'gearbox_health': gearbox_health,
        'generator_health': generator_health,
        'efficiency': efficiency,
        'alignment': alignment_score,
        'integrity': integrity_score,
        'overall_status': overall_status_text,
        'overall_confidence': confidence,
        'raw_data': current_row,
        'timestamp': current_row.name,
        'gb_error': gb_error,
        'gen_error': gen_error,
        'yaw_error': yaw_error,
        'critical_count': critical_count,
        'warning_count': warning_count
    }

def check_alerts(scores, failure_state, consecutive_threshold=3):
    """Check for alerts based on health scores"""
    alerts = []
    
    thresholds = {
        'gearbox': {'critical': 30, 'warning': 60},
        'generator': {'critical': 30, 'warning': 60},
        'efficiency': {'critical': 50, 'warning': 90},
        'alignment': {'critical': 85, 'warning': 95},
        'integrity': {'critical': 80, 'warning': 95}
    }
    
    for test, score in scores.items():
        if test in thresholds:
            counter_key = f"{test}_consecutive_fails"
            
            if score < thresholds[test]['critical']:
                failure_state[counter_key] += 1
                if failure_state[counter_key] >= consecutive_threshold:
                    alerts.append({
                        'level': 'CRITICAL',
                        'test': test.upper(),
                        'score': score,
                        'message': f"{test.upper()} health critically low: {score:.1f}%",
                        'consecutive': failure_state[counter_key]
                    })
                else:
                    alerts.append({
                        'level': 'WARNING',
                        'test': test.upper(),
                        'score': score,
                        'message': f"{test.upper()} below critical threshold ({failure_state[counter_key]}/{consecutive_threshold})",
                        'consecutive': failure_state[counter_key]
                    })
            elif score < thresholds[test]['warning']:
                alerts.append({
                    'level': 'WARNING',
                    'test': test.upper(),
                    'score': score,
                    'message': f"{test.upper()} below warning threshold: {score:.1f}%",
                    'consecutive': 0
                })
                failure_state[counter_key] = 0
            else:
                failure_state[counter_key] = 0
    
    return alerts, failure_state

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🌪️ Wind Turbine Predictive Maintenance System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Real-Time Health Monitoring & Alerting</div>', unsafe_allow_html=True)
    
    # Load models and data
    with st.spinner('Loading models and data...'):
        models = load_models()
        df = load_data()
        test_indices = prepare_test_data(df) if df is not None else []
    
    if models is None or df is None:
        st.error("Failed to load models or data. Please check the file paths.")
        return
    
    # Sidebar controls - Test Data Selection
    st.sidebar.header("🔬 Test Data Selection")
    
    # Auto-cycle control
    auto_cycle = st.sidebar.checkbox("Auto-cycle samples", value=st.session_state.auto_cycle, key="auto_cycle_checkbox")
    st.session_state.auto_cycle = auto_cycle
    
    if auto_cycle:
        cycle_interval = st.sidebar.slider("Cycle interval (seconds)", 1.0, 10.0, 3.0, 0.5, key="cycle_interval")
        st.sidebar.info(f"🔄 Auto-cycling through {len(test_indices)} test samples every {cycle_interval}s")
        
        # Display current sample (read-only)
        st.sidebar.metric("Current Sample", f"{st.session_state.current_sample_index + 1}/{len(test_indices)}")
        
        # Manual override buttons
        col1, col2 = st.sidebar.columns(2)
        if col1.button("⏮️ Previous", key="prev_sample"):
            st.session_state.current_sample_index = (st.session_state.current_sample_index - 1) % len(test_indices)
            st.rerun()
        if col2.button("⏭️ Next", key="next_sample"):
            st.session_state.current_sample_index = (st.session_state.current_sample_index + 1) % len(test_indices)
            st.rerun()
    else:
        # Manual selection
        st.session_state.current_sample_index = st.sidebar.selectbox(
            "Select Test Sample",
            options=range(len(test_indices)),
            index=st.session_state.current_sample_index,
            format_func=lambda x: f"Sample {x+1} - {df.index[test_indices[x]].strftime('%Y-%m-%d %H:%M')}"
        )
    
    # Set current index based on selection
    st.session_state.current_index = test_indices[st.session_state.current_sample_index]
    
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Control Panel")
    
    consecutive_threshold = st.sidebar.slider("Alert threshold (consecutive failures)", 1, 10, 3, key="consecutive_threshold_main")
    
    st.sidebar.markdown("---")
    st.sidebar.header("📊 System Info")
    st.sidebar.info(f"""
    **Data Source**: Kelmarsh Wind Farm 2020
    **Total Measurements**: {len(df):,}
    **Test Samples Available**: {len(test_indices)}
    **Current Sample**: {st.session_state.current_sample_index + 1}
    **Models Loaded**: 6 (2 LSTM, 1 RF, 1 DT, 2 Scalers)
    """)
    
    # Process current data
    results = process_current_data(df, models, st.session_state.current_index)
    
    # Check for alerts
    scores_dict = {
        'gearbox': results['gearbox_health'],
        'generator': results['generator_health'],
        'efficiency': results['efficiency'],
        'alignment': results['alignment'],
        'integrity': results['integrity']
    }
    
    alerts, st.session_state.failure_state = check_alerts(
        scores_dict, 
        st.session_state.failure_state,
        consecutive_threshold
    )
    
    # Add alerts to history
    if alerts:
        for alert in alerts:
            alert['timestamp'] = results['timestamp']
            if alert not in st.session_state.alert_history[-5:]:  # Avoid duplicates
                st.session_state.alert_history.append(alert)
    
    # Display alerts
    if alerts:
        for alert in alerts:
            if alert['level'] == 'CRITICAL':
                st.markdown(f'<div class="alert-critical">🔴 {alert["message"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-warning">🟡 {alert["message"]}</div>', unsafe_allow_html=True)
    
    # Overall status
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        status_color = "🟢" if results['overall_status'] == 'Healthy' else "🔴"
        st.markdown(f"### {status_color} Overall Status: **{results['overall_status']}**")
    
    with col2:
        st.metric("Confidence", f"{results['overall_confidence']:.1f}%")
    
    with col3:
        st.metric("Sample", f"{st.session_state.current_sample_index + 1}/{len(test_indices)}")
    
    # Show status breakdown for transparency
    if results['critical_count'] > 0 or results['warning_count'] > 0:
        status_detail = f"⚠️ System Issues: {results['critical_count']} Critical, {results['warning_count']} Warnings"
        st.warning(status_detail)
    
    # Display timestamp
    st.info(f"📅 Timestamp: {results['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Display diagnostic results with pass/fail in modern card layout
    st.markdown("---")
    st.markdown("## 🔍 AI Diagnostic System")
    st.markdown("<p style='color: #6b7280; font-size: 1.1rem; margin-bottom: 1.5rem;'>Multi-Modal Real-Time Health Analysis</p>", unsafe_allow_html=True)
    
    # Create diagnostic results
    diagnostics = [
        {
            "test": "🔧 Gearbox Health",
            "model": "LSTM Autoencoder",
            "score": results['gearbox_health'],
            "details": f"Reconstruction Error: {results['gb_error']:.4f}",
            "type": "gearbox",
            "icon": "🔧"
        },
        {
            "test": "⚡ Generator Health",
            "model": "LSTM Autoencoder",
            "score": results['generator_health'],
            "details": f"Reconstruction Error: {results['gen_error']:.4f}",
            "type": "generator"
        },
        {
            "test": "🌀 Blade/Rotor Efficiency",
            "model": "Random Forest",
            "score": results['efficiency'],
            "details": f"Power Curve Analysis",
            "type": "efficiency"
        },
        {
            "test": "🎯 Yaw/Pitch Alignment",
            "model": "Rule-Based",
            "score": results['alignment'],
            "details": f"Misalignment: {results['yaw_error']:.1f}°",
            "type": "alignment"
        },
        {
            "test": "📊 Sensor Validity",
            "model": "Rule-Based",
            "score": results['integrity'],
            "details": "Data Integrity Check",
            "type": "integrity"
        }
    ]
    
    # Display diagnostics in modern card format
    col_left, col_right = st.columns(2)
    
    for idx, diag in enumerate(diagnostics):
        status, color = get_pass_fail_status(diag['score'], diag['type'])
        
        # Alternate between left and right columns
        col = col_left if idx % 2 == 0 else col_right
        
        with col:
            # Create a card-like container
            score_color = "#10b981" if status == "✅ PASS" else ("#f59e0b" if status == "⚠️ WARNING" else "#ef4444")
            
            st.markdown(f"""
            <div style="
                background: white;
                padding: 1.5rem;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                border-left: 4px solid {score_color};
                margin-bottom: 1rem;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <h4 style="margin: 0; color: #1f2937; font-size: 1.1rem;">{diag['test']}</h4>
                    <span style="font-size: 1.5rem;">{status}</span>
                </div>
                <p style="color: #6b7280; font-size: 0.875rem; margin: 0.25rem 0;">{diag['model']}</p>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.75rem;">
                    <span style="font-size: 1.75rem; font-weight: 700; color: {score_color};">{diag['score']:.1f}%</span>
                    <span style="color: #9ca3af; font-size: 0.875rem;">{diag['details']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Summary counts
    st.markdown("### 📈 Diagnostic Summary")
    col1, col2, col3 = st.columns(3)
    
    passed = sum(1 for d in diagnostics if get_pass_fail_status(d['score'], d['type'])[0] == "✅ PASS")
    warnings = sum(1 for d in diagnostics if get_pass_fail_status(d['score'], d['type'])[0] == "⚠️ WARNING")
    failed = sum(1 for d in diagnostics if get_pass_fail_status(d['score'], d['type'])[0] == "❌ FAIL")
    
    with col1:
        st.metric("✅ Passed", f"{passed}/5")
    with col2:
        st.metric("⚠️ Warnings", f"{warnings}/5")
    with col3:
        st.metric("❌ Failed", f"{failed}/5")
    
    # Detailed metrics in card grid
    st.markdown("---")
    st.markdown("## 📈 Sensor Readings")
    st.markdown("<p style='color: #6b7280; font-size: 1.1rem; margin-bottom: 1.5rem;'>Real-Time SCADA Data</p>", unsafe_allow_html=True)
    
    # Create two rows of 4 metrics each
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🌬️ Wind Speed", f"{results['raw_data']['Wind speed (m/s)']:.2f} m/s")
    with col2:
        st.metric("⚡ Power Output", f"{results['raw_data']['Power (kW)']:.1f} kW")
    with col3:
        st.metric("🌡️ Gearbox Temp", f"{results['raw_data']['Gear oil temperature (°C)']:.1f} °C")
    with col4:
        st.metric("🔄 Generator RPM", f"{results['raw_data']['Generator RPM (RPM)']:.0f}")
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric("🔥 Stator Temp", f"{results['raw_data']['Stator temperature 1 (°C)']:.1f} °C")
    with col6:
        st.metric("⚙️ Bearing Temp", f"{results['raw_data']['Generator bearing front temperature (°C)']:.1f} °C")
    with col7:
        st.metric("🧭 Wind Direction", f"{results['raw_data']['Wind direction (°)']:.0f}°")
    with col8:
        st.metric("🎯 Nacelle Position", f"{results['raw_data']['Nacelle position (°)']:.0f}°")
    
    # Alert History
    if st.session_state.alert_history:
        st.markdown("---")
        st.markdown("### 🚨 Recent Alert History")
        
        for alert in st.session_state.alert_history[-5:]:
            if alert['level'] == 'CRITICAL':
                st.error(f"🔴 {alert['timestamp'].strftime('%Y-%m-%d %H:%M')} - {alert['message']}")
            else:
                st.warning(f"🟡 {alert['timestamp'].strftime('%Y-%m-%d %H:%M')} - {alert['message']}")
    
    # Architecture diagram
    with st.expander("🏗️ System Architecture"):
        st.markdown("""
        ### Multi-Modal AI Diagnostic System
        
        **5 Independent Diagnostic Tests:**
        1. **Gearbox Health** - LSTM Autoencoder (Temperature anomaly detection)
        2. **Blade/Rotor Efficiency** - Random Forest (Power curve analysis)
        3. **Generator Health** - LSTM Autoencoder (Electrical/thermal monitoring)
        4. **Yaw/Pitch Alignment** - Rule-Based (Misalignment detection)
        5. **Sensor Validity** - Rule-Based (Data integrity checks)
        
        **Sensor Fusion:** Decision Tree Meta-Model combines all 5 test results
        
        **Alerting System:**
        - ✅ PASS: Component operating within normal parameters
        - ⚠️ WARNING: Component showing early warning signs
        - ❌ FAIL: Component requires immediate attention
        """)
    
    # NEW: Energy Efficiency & Cost Optimization Section
    if OPTIMIZATION_AVAILABLE:
        st.markdown("---")
        st.markdown("## 🎯 Energy Efficiency & Cost Optimization")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "⚡ Performance Optimization",
            "💰 Cost Analysis",
            "📅 Maintenance Schedule",
            "📊 ROI Insights"
        ])
        
        with tab1:
            st.markdown("### Performance Optimization Insights")
            
            # Performance benchmarking
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Performance Metrics")
                ideal_power = PerformanceBenchmarking.calculate_ideal_power(
                    results['raw_data']['Wind speed (m/s)']
                )
                actual_power = results['raw_data']['Power (kW)']
                performance_ratio = PerformanceBenchmarking.calculate_performance_ratio(
                    actual_power, ideal_power
                )
                
                st.metric("Ideal Power Output", f"{ideal_power:.1f} kW")
                st.metric("Actual Power Output", f"{actual_power:.1f} kW")
                st.metric("Performance Ratio", f"{performance_ratio:.1f}%", 
                         delta=f"{performance_ratio - 100:.1f}%")
                
                power_loss = ideal_power - actual_power
                if power_loss > 0:
                    st.warning(f"⚠️ Power Loss: {power_loss:.1f} kW ({(power_loss/ideal_power*100):.1f}%)")
                else:
                    st.success("✅ Operating at or above expected performance")
            
            with col2:
                st.markdown("#### Power Forecast (Next 6 Hours)")
                forecaster = PowerOutputForecaster()
                current_conditions = {
                    'power': actual_power,
                    'wind_speed': results['raw_data']['Wind speed (m/s)']
                }
                forecast = forecaster.forecast_next_hours(current_conditions, hours=6)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=forecast['hour'],
                    y=forecast['predicted_power_kw'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='#1f77b4', width=3)
                ))
                fig.update_layout(
                    xaxis_title="Hours Ahead",
                    yaxis_title="Power (kW)",
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### Operational Cost Analysis")
            
            cost_calc = DowntimeCostCalculator()
            
            # Calculate potential failure costs
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Potential Failure Costs")
                
                # Get component with lowest health
                components_health = {
                    'Gearbox': results['gearbox_health'],
                    'Generator': results['generator_health'],
                    'Blade/Rotor': results['efficiency']
                }
                
                worst_component = min(components_health, key=components_health.get)
                worst_score = components_health[worst_component]
                
                # Map to failure type
                failure_map = {
                    'Gearbox': 'gearbox',
                    'Generator': 'generator',
                    'Blade/Rotor': 'blade'
                }
                
                failure_cost = cost_calc.calculate_total_operational_cost(
                    failure_type=failure_map[worst_component],
                    downtime_hours=120,
                    avg_power_kw=actual_power
                )
                
                st.metric("At-Risk Component", worst_component)
                st.metric("Health Score", f"{worst_score:.1f}%")
                st.metric("Estimated Failure Cost", f"${failure_cost['total_cost_usd']:,.0f}")
                
                st.markdown("**Cost Breakdown:**")
                st.write(f"• Downtime Loss: ${failure_cost['downtime_cost']['revenue_loss_usd']:,.0f}")
                st.write(f"• Maintenance: ${failure_cost['maintenance_cost']['total_maintenance_cost_usd']:,.0f}")
            
            with col2:
                st.markdown("#### Preventive Maintenance Savings")
                
                # Calculate preventive savings
                preventive_cost = 15000  # Example preventive maintenance cost
                savings = cost_calc.calculate_preventive_savings(
                    reactive_cost=failure_cost['total_cost_usd'],
                    preventive_cost=preventive_cost
                )
                
                st.metric("Reactive Maintenance Cost", f"${savings['reactive_cost_usd']:,.0f}")
                st.metric("Preventive Maintenance Cost", f"${savings['preventive_cost_usd']:,.0f}")
                st.metric("Potential Savings", f"${savings['savings_usd']:,.0f}", 
                         delta=f"{savings['roi_percentage']:.0f}% ROI")
                
                if worst_score < 60:
                    st.error("🔴 Recommend immediate preventive action!")
                elif worst_score < 80:
                    st.warning("🟡 Schedule preventive maintenance soon")
                else:
                    st.success("🟢 Continue monitoring")
        
        with tab3:
            st.markdown("### Smart Maintenance Scheduling")
            
            scheduler = SmartMaintenanceScheduler()
            
            # Prepare component health data
            components_for_schedule = {
                'gearbox': {
                    'health_score': results['gearbox_health'],
                    'failure_probability': max(0, (100 - results['gearbox_health']) / 100 * 0.5)
                },
                'generator': {
                    'health_score': results['generator_health'],
                    'failure_probability': max(0, (100 - results['generator_health']) / 100 * 0.5)
                },
                'blade': {
                    'health_score': results['efficiency'],
                    'failure_probability': max(0, (100 - results['efficiency']) / 100 * 0.3)
                },
                'yaw': {
                    'health_score': results['alignment'],
                    'failure_probability': max(0, (100 - results['alignment']) / 100 * 0.2)
                }
            }
            
            schedule = scheduler.generate_maintenance_schedule(components_for_schedule)
            
            if schedule:
                st.markdown("#### Recommended Maintenance Actions")
                
                for i, task in enumerate(schedule, 1):
                    urgency_colors = {
                        'critical': '🔴',
                        'high': '🟠',
                        'medium': '🟡',
                        'low': '🟢'
                    }
                    
                    with st.container():
                        col1, col2, col3, col4 = st.columns([3, 2, 2, 3])
                        with col1:
                            st.markdown(f"**{urgency_colors[task['urgency']]} {task['component'].upper()}**")
                        with col2:
                            st.write(f"Health: {task['health_score']:.1f}%")
                        with col3:
                            st.write(f"Risk: {task['failure_probability']:.1f}%")
                        with col4:
                            st.write(f"Schedule: {task['recommended_date']}")
                        st.markdown("---")
            else:
                st.success("✅ No urgent maintenance required. All components healthy!")
        
        with tab4:
            st.markdown("### ROI & Value Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Predictive Maintenance ROI")
                
                # Calculate ROI
                roi_analysis = CostBenefitAnalyzer.calculate_predictive_maintenance_roi(
                    baseline_costs=250000,  # Estimated annual reactive costs
                    predictive_costs=180000,  # Estimated annual predictive costs
                    avoided_failures=3  # Estimated failures avoided
                )
                
                st.metric("Annual Savings", f"${roi_analysis['annual_savings_usd']:,.0f}")
                st.metric("ROI", f"{roi_analysis['roi_percentage']:.1f}%")
                st.metric("Payback Period", f"{roi_analysis['payback_period_months']:.1f} months")
                st.metric("Avoided Failures/Year", roi_analysis['avoided_failures'])
            
            with col2:
                st.markdown("#### System Value Estimate")
                
                # Assume 5% uptime improvement from predictive maintenance
                uptime_improvement = 5.0
                annual_energy = 8760 * actual_power / 1000  # MWh
                
                value = CostBenefitAnalyzer.estimate_system_value(
                    annual_energy_output_mwh=annual_energy,
                    uptime_improvement_pct=uptime_improvement
                )
                
                st.metric("Additional Energy Captured", f"{value['additional_energy_mwh']:.0f} MWh/year")
                st.metric("Additional Revenue", f"${value['additional_revenue_usd']:,.0f}/year")
                st.metric("Avoided Emergency Costs", f"${value['avoided_emergency_costs_usd']:,.0f}/year")
                st.metric("Total Annual Value", f"${value['total_annual_value_usd']:,.0f}/year")
            
            # Energy efficiency gains visualization
            st.markdown("#### Energy Efficiency Impact")
            
            baseline_data = {'total_energy_kwh': annual_energy * 1000}
            optimized_data = {'total_energy_kwh': annual_energy * 1000 * 1.05}
            
            gains = calculate_energy_gains(baseline_data, optimized_data)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Energy Gain", f"{gains['energy_gain_kwh']:,.0f} kWh/year")
            with col2:
                st.metric("Efficiency Improvement", f"{gains['energy_gain_percentage']:.1f}%")
            with col3:
                st.metric("CO₂ Reduction", f"{gains['co2_reduction_tons']:.1f} tons/year")
    
    # Auto-cycle logic
    if st.session_state.auto_cycle:
        time.sleep(cycle_interval)
        st.session_state.current_sample_index = (st.session_state.current_sample_index + 1) % len(test_indices)
        st.rerun()

if __name__ == "__main__":
    main()
