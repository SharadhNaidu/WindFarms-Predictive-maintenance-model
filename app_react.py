"""
Wind Turbine Predictive Maintenance Dashboard - React Enhanced
Real-time monitoring with Material-UI components
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

# Import React-based components
from streamlit_elements import elements, mui, html, nivo
from streamlit_card import card

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

# Clean, High-Contrast Design with White Background, Black Text, and Strategic Colors
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* White background throughout */
    .main, .stApp {
        background: #ffffff !important;
    }
    
    .block-container {
        padding-top: 1.5rem !important;
        max-width: 98% !important;
        background: #ffffff;
    }
    
    /* All text is black by default */
    body, p, span, div {
        color: #000000 !important;
    }
    
    /* Metric cards - white with border and colored accents */
    [data-testid="stMetric"] {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e5e5e5;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        border-color: #2196F3;
        box-shadow: 0 4px 12px rgba(33,150,243,0.15);
        transform: translateY(-2px);
    }
    
    [data-testid="stMetric"] label {
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.3px;
    }
    
    [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
    }
    
    /* Blue sidebar with better contrast */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2196F3 0%, #1976D2 100%) !important;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    /* Blue buttons with hover effect */
    .stButton > button {
        background: #2196F3 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 6px rgba(33,150,243,0.3) !important;
    }
    
    .stButton > button:hover {
        background: #1976D2 !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(33,150,243,0.4) !important;
    }
    
    /* Tabs with blue accent */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: white;
        border-bottom: 2px solid #e5e5e5;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #000000;
        font-weight: 600;
        padding: 1rem 2rem;
        border-radius: 8px 8px 0 0;
        border-bottom: 3px solid transparent;
        transition: all 0.2s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #2196F3;
        background: rgba(33,150,243,0.05);
    }
    
    .stTabs [aria-selected="true"] {
        color: #2196F3 !important;
        border-bottom: 3px solid #2196F3 !important;
        background: rgba(33,150,243,0.1) !important;
    }
    
    /* Headers - bold black text */
    h1, h2, h3, h4 {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    
    h1 {
        font-size: 2.5rem !important;
    }
    
    h2 {
        font-size: 2rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }
    
    h3 {
        font-size: 1.5rem !important;
    }
    
    /* Alert boxes with color coding */
    .stAlert {
        border-radius: 10px !important;
        border-left: 5px solid !important;
        font-weight: 500 !important;
        padding: 1rem 1.5rem !important;
        background: white !important;
    }
    
    /* Success/Info - Green */
    [data-baseweb="notification"][kind="info"],
    .stSuccess {
        border-left-color: #4CAF50 !important;
        background: rgba(76,175,80,0.1) !important;
    }
    
    /* Warning - Yellow */
    [data-baseweb="notification"][kind="warning"],
    .stWarning {
        border-left-color: #FFC107 !important;
        background: rgba(255,193,7,0.1) !important;
    }
    
    /* Error - Red */
    [data-baseweb="notification"][kind="error"],
    .stError {
        border-left-color: #F44336 !important;
        background: rgba(244,67,54,0.1) !important;
    }
    
    /* Info - Blue */
    .stInfo {
        border-left-color: #2196F3 !important;
        background: rgba(33,150,243,0.1) !important;
    }
    
    /* Remove default streamlit styling */
    .main .block-container {
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    
    /* Divider lines */
    hr {
        border-color: #e5e5e5 !important;
        margin: 2rem 0 !important;
    }
    
    /* Selectbox and input fields */
    .stSelectbox, .stSlider {
        color: #000000 !important;
    }
    
    /* Checkbox */
    .stCheckbox {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state (same as before)
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
    st.session_state.auto_cycle = False

# Keep all the same helper functions from the original app
@st.cache_resource
def load_models():
    """Load all trained models"""
    base_path = os.path.dirname(__file__)
    models_path = os.path.join(base_path, 'trained_models')
    
    try:
        gearbox_model = tf.keras.models.load_model(
            os.path.join(models_path, 'gearbox_autoencoder.h5'),
            compile=False
        )
        generator_model = tf.keras.models.load_model(
            os.path.join(models_path, 'generator_autoencoder.h5'),
            compile=False
        )
        scaler_gb = joblib.load(os.path.join(models_path, 'scaler_gb.joblib'))
        scaler_gen = joblib.load(os.path.join(models_path, 'scaler_gen.joblib'))
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
        
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def prepare_test_data(df):
    test_indices = list(range(100, len(df) - 100, 100))
    return test_indices

def calculate_health_score(error, healthy_thresh, failure_thresh):
    if error <= healthy_thresh:
        norm_error = 0
    elif error >= failure_thresh:
        norm_error = 1
    else:
        norm_error = (error - healthy_thresh) / (failure_thresh - healthy_thresh)
    health = (1 - norm_error) * 100
    return np.clip(health, 0, 100)

def get_pass_fail_status(score, test_type):
    thresholds = {
        'gearbox': {'pass': 60, 'warning': 30},
        'generator': {'pass': 60, 'warning': 30},
        'efficiency': {'pass': 90, 'warning': 50},
        'alignment': {'pass': 90, 'warning': 85},
        'integrity': {'pass': 90, 'warning': 80}
    }
    thresh = thresholds.get(test_type, {'pass': 60, 'warning': 30})
    if score >= thresh['pass']:
        return "✅ PASS", "success"
    elif score >= thresh['warning']:
        return "⚠️ WARNING", "warning"
    else:
        return "❌ FAIL", "error"

def process_current_data(df, models, index, sequence_length=24):
    """Process current data point and calculate all health scores dynamically"""
    if index < sequence_length:
        index = sequence_length
    
    # Get current data point
    current_row = df.iloc[index]
    
    # Feature columns for gearbox
    gb_features = [
        'Gear oil temperature (°C)',
        'Gear oil inlet temperature (°C)',
        'Generator RPM (RPM)'
    ]
    
    # Feature columns for generator
    gen_features = [
        'Stator temperature 1 (°C)',
        'Generator bearing front temperature (°C)',
        'Generator bearing rear temperature (°C)',
        'Power (kW)'
    ]
    
    # Get sequence data for LSTM models
    sequence_data = df.iloc[index-sequence_length:index]
    
    # Calculate Gearbox health using LSTM autoencoder
    gb_data = sequence_data[gb_features].values
    gb_scaled = models['scaler_gb'].transform(gb_data)
    gb_sequence = gb_scaled.reshape(1, sequence_length, len(gb_features))
    
    # Predict and calculate reconstruction error
    gb_pred = models['gearbox'].predict(gb_sequence, verbose=0)
    gb_error = np.mean(np.abs(gb_pred - gb_sequence))
    
    # Dynamic thresholds based on error distribution
    HEALTHY_THRESHOLD_GB = 0.05
    FAILURE_THRESHOLD_GB = 0.25
    gearbox_health = calculate_health_score(gb_error, HEALTHY_THRESHOLD_GB, FAILURE_THRESHOLD_GB)
    
    # Calculate Generator health using LSTM autoencoder
    gen_data = sequence_data[gen_features].values
    gen_scaled = models['scaler_gen'].transform(gen_data)
    gen_sequence = gen_scaled.reshape(1, sequence_length, len(gen_features))
    
    # Predict and calculate reconstruction error
    gen_pred = models['generator'].predict(gen_sequence, verbose=0)
    gen_error = np.mean(np.abs(gen_pred - gen_sequence))
    
    # Dynamic thresholds
    HEALTHY_THRESHOLD_GEN = 0.05
    FAILURE_THRESHOLD_GEN = 0.25
    generator_health = calculate_health_score(gen_error, HEALTHY_THRESHOLD_GEN, FAILURE_THRESHOLD_GEN)
    
    # Calculate Blade/Rotor Efficiency from power curve
    wind_speed = current_row['Wind speed (m/s)']
    actual_power = current_row['Power (kW)']
    
    # Predict expected power at current wind speed
    predicted_power = models['power_curve'].predict([[wind_speed]])[0]
    
    # Calculate efficiency (actual vs expected)
    with np.errstate(divide='ignore', invalid='ignore'):
        if actual_power > 10:  # Only calculate if turbine is producing power
            efficiency = (actual_power / predicted_power * 100) if predicted_power > 0 else 100
        else:
            efficiency = 100  # Default when not producing
    efficiency = np.clip(efficiency, 0, 150)  # Allow some over-performance
    efficiency = min(efficiency, 100)  # Cap at 100% for health score
    
    # Calculate Yaw/Pitch Alignment score
    wind_dir = current_row['Wind direction (°)']
    nacelle_pos = current_row['Nacelle position (°)']
    
    # Calculate yaw misalignment
    yaw_error = abs((nacelle_pos - wind_dir + 180) % 360 - 180)
    
    # Convert to health score (lower error = higher score)
    alignment_score = max(0, 100 - yaw_error)
    
    # Calculate Sensor Validity / Data Integrity
    integrity_score = 100
    failed_sensors = 0
    
    # Check sensor readings against expected ranges
    if not (-10 < current_row['Gear oil temperature (°C)'] < 110):
        failed_sensors += 1
    if not (0 <= current_row['Wind speed (m/s)'] < 50):
        failed_sensors += 1
    if not (-10 < current_row['Stator temperature 1 (°C)'] < 180):
        failed_sensors += 1
    if not (0 <= current_row['Power (kW)'] < 5000):
        failed_sensors += 1
    
    # Calculate integrity score (each failed sensor reduces score)
    integrity_score = max(0, 100 - (failed_sensors * 25))
    
    # Calculate overall system health using rule-based logic
    # This is more transparent and responsive than relying solely on the meta-model
    
    # Count critical failures (very low health scores)
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
        # Calculate confidence based on how close to thresholds
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

# Main app with React components
def main():
    # Header with clean design
    st.markdown("""
        <div style="text-align: center; padding: 2rem; background: white; border-radius: 12px; 
                    box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin-bottom: 2rem; border: 2px solid #e5e5e5;">
            <h1 style="margin: 0; color: #000000; font-size: 3rem; font-weight: 700;">
                🌪️ Wind Turbine Predictive Maintenance
            </h1>
            <p style="margin: 1rem 0 0 0; color: #000000; font-size: 1.2rem; font-weight: 500;">
                AI-Powered Real-Time Health Monitoring & Analytics
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load models and data
    with st.spinner('🔄 Loading AI models and SCADA data...'):
        models = load_models()
        df = load_data()
        test_indices = prepare_test_data(df) if df is not None else []
    
    if models is None or df is None:
        st.error("❌ Failed to load models or data. Please check the file paths.")
        return
    
    # Sidebar with Material Design
    st.sidebar.markdown("## 🎛️ Control Panel")
    st.sidebar.markdown("---")
    
    auto_cycle = st.sidebar.checkbox("🔄 Auto-cycle samples", value=st.session_state.auto_cycle)
    st.session_state.auto_cycle = auto_cycle
    
    if auto_cycle:
        cycle_interval = st.sidebar.slider("⏱️ Cycle interval (seconds)", 1.0, 10.0, 3.0, 0.5)
        st.sidebar.info(f"Auto-cycling through {len(test_indices)} samples")
        st.sidebar.metric("Current Sample", f"{st.session_state.current_sample_index + 1}/{len(test_indices)}")
        
        col1, col2 = st.sidebar.columns(2)
        if col1.button("⏮️ Previous"):
            st.session_state.current_sample_index = (st.session_state.current_sample_index - 1) % len(test_indices)
            st.rerun()
        if col2.button("⏭️ Next"):
            st.session_state.current_sample_index = (st.session_state.current_sample_index + 1) % len(test_indices)
            st.rerun()
    else:
        st.session_state.current_sample_index = st.sidebar.selectbox(
            "📊 Select Test Sample",
            options=range(len(test_indices)),
            index=st.session_state.current_sample_index,
            format_func=lambda x: f"Sample {x+1} - {df.index[test_indices[x]].strftime('%Y-%m-%d %H:%M')}"
        )
    
    st.session_state.current_index = test_indices[st.session_state.current_sample_index]
    
    st.sidebar.markdown("---")
    consecutive_threshold = st.sidebar.slider("⚠️ Alert threshold", 1, 10, 3)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 System Info")
    st.sidebar.info(f"""
**Total Measurements**: {len(df):,}  
**Test Samples**: {len(test_indices)}  
**Models Loaded**: 6 AI Models  
**Status**: 🟢 Active
    """)
    
    # Process current data
    results = process_current_data(df, models, st.session_state.current_index)
    
    scores_dict = {
        'gearbox': results['gearbox_health'],
        'generator': results['generator_health'],
        'efficiency': results['efficiency'],
        'alignment': results['alignment'],
        'integrity': results['integrity']
    }
    
    alerts, st.session_state.failure_state = check_alerts(scores_dict, st.session_state.failure_state, consecutive_threshold)
    
    # Display alerts with proper color coding
    if alerts:
        for alert in alerts:
            if alert['level'] == 'CRITICAL':
                st.error(f"🔴 **CRITICAL ALERT**: {alert['message']}")
            else:
                st.warning(f"⚠️ **WARNING**: {alert['message']}")
    
    # Status Overview with color-coded status cards
    col1, col2, col3 = st.columns([2, 2, 1])
    
    status_color = "#4CAF50" if results['overall_status'] == 'Healthy' else "#F44336"
    status_icon = "✅" if results['overall_status'] == 'Healthy' else "🔴"
    status_bg = "rgba(76,175,80,0.1)" if results['overall_status'] == 'Healthy' else "rgba(244,67,54,0.1)"
    
    with col1:
        st.markdown(f"""
            <div style="background: {status_bg}; padding: 2rem; border-radius: 12px; 
                        border: 3px solid {status_color}; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                <h2 style="margin: 0; color: {status_color}; font-size: 2rem;">{status_icon} {results['overall_status']}</h2>
                <p style="margin: 0.5rem 0 0 0; color: #000000; font-size: 1.1rem; font-weight: 600;">Overall System Status</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("🎯 Confidence Level", f"{results['overall_confidence']:.1f}%")
    
    with col3:
        st.metric("📊 Sample", f"{st.session_state.current_sample_index + 1}/{len(test_indices)}")
    
    # Show status breakdown
    if results['critical_count'] > 0 or results['warning_count'] > 0:
        status_detail = f"⚠️ {results['critical_count']} Critical, {results['warning_count']} Warnings"
        st.warning(status_detail)
    
    st.info(f"🕐 **Timestamp**: {results['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Diagnostic Cards using streamlit-card
    st.markdown("---")
    st.markdown("## 🔍 AI Diagnostic System")
    
    diagnostics = [
        {'name': '🔧 Gearbox Health', 'score': results['gearbox_health'], 'type': 'gearbox', 'model': 'LSTM Autoencoder'},
        {'name': '⚡ Generator Health', 'score': results['generator_health'], 'type': 'generator', 'model': 'LSTM Autoencoder'},
        {'name': '🌀 Blade/Rotor Efficiency', 'score': results['efficiency'], 'type': 'efficiency', 'model': 'Random Forest'},
        {'name': '🎯 Yaw/Pitch Alignment', 'score': results['alignment'], 'type': 'alignment', 'model': 'Rule-Based'},
        {'name': '📊 Sensor Validity', 'score': results['integrity'], 'type': 'integrity', 'model': 'Rule-Based'},
    ]
    
    # Display diagnostic cards in 2 columns
    for i in range(0, len(diagnostics), 2):
        col1, col2 = st.columns(2)
        
        for idx, col in enumerate([col1, col2]):
            if i + idx < len(diagnostics):
                diag = diagnostics[i + idx]
                status, status_type = get_pass_fail_status(diag['score'], diag['type'])
                
                color_map = {'success': '#4CAF50', 'warning': '#FFC107', 'error': '#F44336'}
                bg_color = color_map.get(status_type, '#4CAF50')
                
                with col:
                    with elements(f"card_{i}_{idx}"):
                        mui.Card(
                            mui.CardContent(
                                html.div(
                                    mui.Typography(diag['name'], variant="h6", sx={"fontWeight": 500}),
                                    mui.Typography(diag['model'], variant="caption", sx={"color": "#666"}),
                                    html.div(
                                        mui.Typography(f"{diag['score']:.1f}%", variant="h3", 
                                                      sx={"color": bg_color, "fontWeight": 700, "marginTop": "1rem"}),
                                        mui.Chip(label=status, sx={"backgroundColor": bg_color, "color": "white", "fontWeight": 500}),
                                        sx={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginTop": "1rem"}
                                    )
                                )
                            ),
                            sx={"minHeight": 180, "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}
                        )
    
    # Sensor Readings
    st.markdown("---")
    st.markdown("## 📡 Sensor Readings")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🌬️ Wind Speed", f"{results['raw_data']['Wind speed (m/s)']:.2f} m/s")
        st.metric("⚡ Power Output", f"{results['raw_data']['Power (kW)']:.1f} kW")
    with col2:
        st.metric("🌡️ Gearbox Temp", f"{results['raw_data']['Gear oil temperature (°C)']:.1f} °C")
        st.metric("🔄 Generator RPM", f"{results['raw_data']['Generator RPM (RPM)']:.0f}")
    with col3:
        st.metric("🔥 Stator Temp", f"{results['raw_data']['Stator temperature 1 (°C)']:.1f} °C")
        st.metric("⚙️ Bearing Temp", f"{results['raw_data']['Generator bearing front temperature (°C)']:.1f} °C")
    with col4:
        st.metric("🧭 Wind Direction", f"{results['raw_data']['Wind direction (°)']:.0f}°")
        st.metric("🎯 Nacelle Position", f"{results['raw_data']['Nacelle position (°)']:.0f}°")
    
    # Optimization Section
    if OPTIMIZATION_AVAILABLE:
        st.markdown("---")
        st.markdown("## 🎯 Energy Efficiency & Cost Optimization")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "⚡ Performance",
            "💰 Cost Analysis", 
            "📅 Maintenance",
            "📊 ROI Insights"
        ])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                ideal_power = PerformanceBenchmarking.calculate_ideal_power(
                    results['raw_data']['Wind speed (m/s)']
                )
                actual_power = results['raw_data']['Power (kW)']
                performance_ratio = PerformanceBenchmarking.calculate_performance_ratio(
                    actual_power, ideal_power
                )
                
                st.metric("🎯 Ideal Power", f"{ideal_power:.1f} kW")
                st.metric("⚡ Actual Power", f"{actual_power:.1f} kW")
                st.metric("📊 Performance Ratio", f"{performance_ratio:.1f}%", 
                         delta=f"{performance_ratio - 100:.1f}%")
            
            with col2:
                forecaster = PowerOutputForecaster()
                forecast = forecaster.forecast_next_hours(
                    {'power': actual_power, 'wind_speed': results['raw_data']['Wind speed (m/s)']}, 
                    hours=6
                )
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=forecast['hour'],
                    y=forecast['predicted_power_kw'],
                    mode='lines+markers',
                    line=dict(color='#2196F3', width=3),
                    marker=dict(size=8, color='#2196F3'),
                    fill='tozeroy',
                    fillcolor='rgba(33, 150, 243, 0.15)'
                ))
                fig.update_layout(
                    title="⏰ Power Forecast (Next 6 Hours)",
                    title_font=dict(size=18, color='#000000', family='Inter'),
                    xaxis_title="Hours Ahead",
                    yaxis_title="Power (kW)",
                    xaxis=dict(title_font=dict(color='#000000'), tickfont=dict(color='#000000')),
                    yaxis=dict(title_font=dict(color='#000000'), tickfont=dict(color='#000000')),
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            cost_calc = DowntimeCostCalculator()
            
            col1, col2 = st.columns(2)
            
            with col1:
                components_health = {
                    'Gearbox': results['gearbox_health'],
                    'Generator': results['generator_health'],
                    'Blade/Rotor': results['efficiency']
                }
                
                worst_component = min(components_health, key=components_health.get)
                worst_score = components_health[worst_component]
                
                failure_map = {'Gearbox': 'gearbox', 'Generator': 'generator', 'Blade/Rotor': 'blade'}
                
                failure_cost = cost_calc.calculate_total_operational_cost(
                    failure_type=failure_map[worst_component],
                    downtime_hours=120,
                    avg_power_kw=actual_power
                )
                
                st.metric("⚠️ At-Risk Component", worst_component)
                st.metric("💔 Health Score", f"{worst_score:.1f}%")
                st.metric("💰 Failure Cost", f"${failure_cost['total_cost_usd']:,.0f}")
            
            with col2:
                preventive_cost = 15000
                savings = cost_calc.calculate_preventive_savings(
                    reactive_cost=failure_cost['total_cost_usd'],
                    preventive_cost=preventive_cost
                )
                
                st.metric("💵 Reactive Cost", f"${savings['reactive_cost_usd']:,.0f}")
                st.metric("💚 Preventive Cost", f"${savings['preventive_cost_usd']:,.0f}")
                st.metric("📈 Potential Savings", f"${savings['savings_usd']:,.0f}", 
                         delta=f"{savings['roi_percentage']:.0f}% ROI")
        
        with tab3:
            scheduler = SmartMaintenanceScheduler()
            
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
            }
            
            schedule = scheduler.generate_maintenance_schedule(components_for_schedule)
            
            if schedule:
                for task in schedule:
                    urgency_colors = {
                        'critical': '#F44336',
                        'high': '#FF9800',
                        'medium': '#FFC107',
                        'low': '#4CAF50'
                    }
                    urgency_bg = {
                        'critical': 'rgba(244,67,54,0.1)',
                        'high': 'rgba(255,152,0,0.1)',
                        'medium': 'rgba(255,193,7,0.1)',
                        'low': 'rgba(76,175,80,0.1)'
                    }
                    color = urgency_colors[task['urgency']]
                    bg = urgency_bg[task['urgency']]
                    
                    st.markdown(f"""
                        <div style="background: {bg}; padding: 1.5rem; border-radius: 10px; 
                                    box-shadow: 0 3px 8px rgba(0,0,0,0.1); margin-bottom: 1rem;
                                    border-left: 5px solid {color}; border: 2px solid {color};">
                            <h4 style="margin: 0; text-transform: uppercase; color: #000000; font-size: 1.2rem;">{task['component']}</h4>
                            <p style="margin: 1rem 0; color: #000000; font-weight: 500;">
                                <strong>Health:</strong> {task['health_score']:.1f}% | 
                                <strong>Risk:</strong> {task['failure_probability']:.1f}% | 
                                <strong>Schedule:</strong> {task['recommended_date']}
                            </p>
                            <span style="background: {color}; color: white; padding: 0.5rem 1rem; 
                                       border-radius: 20px; font-size: 0.9rem; font-weight: 600;">
                                {task['urgency'].upper()}
                            </span>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ All systems healthy! No urgent maintenance required.")
        
        with tab4:
            col1, col2 = st.columns(2)
            
            with col1:
                roi_analysis = CostBenefitAnalyzer.calculate_predictive_maintenance_roi(
                    baseline_costs=250000,
                    predictive_costs=180000,
                    avoided_failures=3
                )
                
                st.metric("💰 Annual Savings", f"${roi_analysis['annual_savings_usd']:,.0f}")
                st.metric("📊 ROI", f"{roi_analysis['roi_percentage']:.1f}%")
                st.metric("⏱️ Payback Period", f"{roi_analysis['payback_period_months']:.1f} months")
                st.metric("🛡️ Avoided Failures", roi_analysis['avoided_failures'])
            
            with col2:
                uptime_improvement = 5.0
                annual_energy = 8760 * actual_power / 1000
                
                value = CostBenefitAnalyzer.estimate_system_value(
                    annual_energy_output_mwh=annual_energy,
                    uptime_improvement_pct=uptime_improvement
                )
                
                st.metric("⚡ Additional Energy", f"{value['additional_energy_mwh']:.0f} MWh/year")
                st.metric("💵 Additional Revenue", f"${value['additional_revenue_usd']:,.0f}/year")
                st.metric("🚨 Avoided Emergency", f"${value['avoided_emergency_costs_usd']:,.0f}/year")
                st.metric("🎯 Total Value", f"${value['total_annual_value_usd']:,.0f}/year")
    
    # Auto-cycle logic
    if st.session_state.auto_cycle:
        time.sleep(cycle_interval)
        st.session_state.current_sample_index = (st.session_state.current_sample_index + 1) % len(test_indices)
        st.rerun()

if __name__ == "__main__":
    main()
