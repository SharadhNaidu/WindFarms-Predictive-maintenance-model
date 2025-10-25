"""
Wind Turbine Efficiency Investigation
Analyze the 0% efficiency issue and provide diagnostic insights
"""

import pandas as pd
import numpy as np
import joblib
import os

print("="*80)
print("INVESTIGATING 0% EFFICIENCY ISSUE")
print("="*80)

# Load the data
print("\n" + "-"*80)
print("Loading SCADA Data...")
print("-"*80)

base_path = r"C:\Users\iamsh\Desktop\Predictive.Ai"
data_path = os.path.join(base_path, 'Kelmarsh_SCADA_2020_3086', 'Turbine_Data_Kelmarsh_1_2020-01-01_-_2021-01-01_228.csv')

turbine_df = pd.read_csv(data_path, skiprows=9)
print(f"✓ Data loaded: {turbine_df.shape[0]} rows × {turbine_df.shape[1]} columns")

# Process timestamp
timestamp_col = '# Date and time'
turbine_df[timestamp_col] = pd.to_datetime(turbine_df[timestamp_col])
turbine_df = turbine_df.sort_values(by=timestamp_col).reset_index(drop=True)

# Split data (80/20)
split_index = int(len(turbine_df) * 0.8)
test_df = turbine_df.iloc[split_index:].reset_index(drop=True)

print(f"\nTest set period: {test_df[timestamp_col].min()} to {test_df[timestamp_col].max()}")
print(f"Test set size: {len(test_df)} rows")

# Get the last data point (where efficiency was 0%)
print("\n" + "="*80)
print("ANALYZING LAST DATA POINT (0% Efficiency)")
print("="*80)

last_idx = len(test_df) - 1
last_timestamp = test_df.loc[last_idx, timestamp_col]

print(f"\nTimestamp: {last_timestamp}")
print("\n" + "-"*80)
print("Key SCADA Parameters:")
print("-"*80)

# Define key columns
wind_speed_col = 'Wind speed (m/s)'
power_col = 'Power (kW)'
rpm_col = 'Generator RPM (RPM)'
wind_dir_col = 'Wind direction (°)'
nacelle_pos_col = 'Nacelle position (°)'
status_col = 'Turbine status'

# Display key parameters
params = {
    'Wind Speed (m/s)': wind_speed_col,
    'Active Power (kW)': power_col,
    'Generator RPM': rpm_col,
    'Wind Direction (°)': wind_dir_col,
    'Nacelle Position (°)': nacelle_pos_col,
    'Turbine Status': status_col
}

for name, col in params.items():
    if col in test_df.columns:
        value = test_df.loc[last_idx, col]
        print(f"  {name:25s}: {value}")
    else:
        print(f"  {name:25s}: Column not found")

# Check additional diagnostic parameters
print("\n" + "-"*80)
print("Temperature Parameters:")
print("-"*80)
temp_cols = [
    'Gearbox bearing temperature (°C)',
    'Gearbox oil temperature (°C)',
    'Generator bearing front temperature (°C)',
    'Generator bearing rear temperature (°C)',
    'Stator temperature 1 (°C)'
]

for col in temp_cols:
    if col in test_df.columns:
        value = test_df.loc[last_idx, col]
        print(f"  {col:45s}: {value:.2f}°C")

# Analyze the efficiency calculation
print("\n" + "="*80)
print("EFFICIENCY CALCULATION ANALYSIS")
print("="*80)

# Load the power curve model
model_path = os.path.join(base_path, 'trained_models', 'power_curve_model.joblib')
if os.path.exists(model_path):
    power_curve_model = joblib.load(model_path)
    print("\n✓ Power curve model loaded")
    
    # Predict expected power
    wind_speed = test_df.loc[last_idx, wind_speed_col]
    actual_power = test_df.loc[last_idx, power_col]
    
    predicted_power = power_curve_model.predict([[wind_speed]])[0]
    
    print(f"\n  Wind Speed         : {wind_speed:.2f} m/s")
    print(f"  Actual Power       : {actual_power:.2f} kW")
    print(f"  Expected Power     : {predicted_power:.2f} kW")
    print(f"  Power Difference   : {actual_power - predicted_power:.2f} kW")
    
    # Calculate efficiency
    if predicted_power > 0:
        efficiency = (actual_power / predicted_power) * 100
        efficiency = max(0, min(100, efficiency))
    else:
        efficiency = 0
    
    print(f"  Calculated Efficiency: {efficiency:.2f}%")
    
    # Explain why it's 0%
    print("\n" + "-"*80)
    print("Root Cause Analysis:")
    print("-"*80)
    
    if actual_power <= 0:
        print("  ❌ Actual power is NEGATIVE or ZERO")
        print(f"     → Turbine is not generating power (actual: {actual_power:.2f} kW)")
        
    if predicted_power <= 0:
        print("  ⚠ Predicted power is ZERO or NEGATIVE")
        print(f"     → Wind speed may be below cut-in speed")
        
    if wind_speed < 3:
        print(f"  ⚠ Wind speed is very low ({wind_speed:.2f} m/s)")
        print("     → Below typical cut-in speed (~3-4 m/s)")
        
    if rpm_col in test_df.columns:
        rpm = test_df.loc[last_idx, rpm_col]
        if rpm < 100:
            print(f"  ⚠ Generator RPM is very low ({rpm:.2f})")
            print("     → Rotor may be stopped or idling")

else:
    print("\n⚠ Power curve model not found")

# Look at last 10 data points to see pattern
print("\n" + "="*80)
print("LAST 10 DATA POINTS - EFFICIENCY TREND")
print("="*80)

last_10 = test_df.tail(10)[[timestamp_col, wind_speed_col, power_col, rpm_col]].copy()

if os.path.exists(model_path):
    # Calculate efficiency for last 10 points
    X_wind = last_10[[wind_speed_col]].values
    predicted_powers = power_curve_model.predict(X_wind)
    
    efficiencies = []
    for i in range(len(last_10)):
        actual = last_10.iloc[i][power_col]
        predicted = predicted_powers[i]
        if predicted > 0:
            eff = (actual / predicted) * 100
            eff = max(0, min(100, eff))
        else:
            eff = 0
        efficiencies.append(eff)
    
    last_10['Predicted_Power'] = predicted_powers
    last_10['Efficiency_%'] = efficiencies
    
    print("\n" + last_10.to_string(index=False))

# Check distribution of low efficiency points in test set
print("\n" + "="*80)
print("LOW EFFICIENCY DISTRIBUTION IN TEST SET")
print("="*80)

if os.path.exists(model_path):
    # Calculate efficiency for entire test set
    X_test_wind = test_df[[wind_speed_col]].values
    predicted_test = power_curve_model.predict(X_test_wind)
    
    test_efficiencies = []
    for i in range(len(test_df)):
        actual = test_df.iloc[i][power_col]
        predicted = predicted_test[i]
        if predicted > 0:
            eff = (actual / predicted) * 100
            eff = max(0, min(100, eff))
        else:
            eff = 0
        test_efficiencies.append(eff)
    
    test_df['efficiency_score'] = test_efficiencies
    
    # Efficiency distribution
    print(f"\nEfficiency Statistics:")
    print(f"  Mean       : {np.mean(test_efficiencies):.2f}%")
    print(f"  Median     : {np.median(test_efficiencies):.2f}%")
    print(f"  Std Dev    : {np.std(test_efficiencies):.2f}%")
    print(f"  Min        : {np.min(test_efficiencies):.2f}%")
    print(f"  Max        : {np.max(test_efficiencies):.2f}%")
    
    # Count points in different efficiency ranges
    critical_count = np.sum(np.array(test_efficiencies) < 50)
    maintenance_count = np.sum((np.array(test_efficiencies) >= 50) & (np.array(test_efficiencies) < 90))
    warning_count = np.sum((np.array(test_efficiencies) >= 90) & (np.array(test_efficiencies) < 95))
    normal_count = np.sum(np.array(test_efficiencies) >= 95)
    
    print(f"\nEfficiency Distribution by Severity:")
    print(f"  Critical (<50%)     : {critical_count:5d} points ({critical_count/len(test_df)*100:.1f}%)")
    print(f"  Maintenance (50-90%): {maintenance_count:5d} points ({maintenance_count/len(test_df)*100:.1f}%)")
    print(f"  Warning (90-95%)    : {warning_count:5d} points ({warning_count/len(test_df)*100:.1f}%)")
    print(f"  Normal (≥95%)       : {normal_count:5d} points ({normal_count/len(test_df)*100:.1f}%)")

# Check turbine status codes
print("\n" + "="*80)
print("TURBINE STATUS CODE ANALYSIS")
print("="*80)

if status_col in test_df.columns:
    print(f"\nLast point status: {test_df.loc[last_idx, status_col]}")
    
    print("\nStatus code distribution in test set:")
    status_counts = test_df[status_col].value_counts()
    print(status_counts.head(10))
    
    # Find what status codes have low efficiency
    if 'efficiency_score' in test_df.columns:
        print("\n" + "-"*80)
        print("Average Efficiency by Status Code (Top 10):")
        print("-"*80)
        status_efficiency = test_df.groupby(status_col)['efficiency_score'].agg(['mean', 'count'])
        status_efficiency = status_efficiency.sort_values('count', ascending=False).head(10)
        print(status_efficiency)
else:
    print("\n⚠ Turbine status column not found")

# Recommendations
print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print("""
1. EFFICIENCY ISSUE ROOT CAUSE:
   • The 0% efficiency is likely due to:
     - Turbine operating below cut-in wind speed
     - Turbine in standby/idle mode
     - Negative power (consuming rather than generating)
   • This is NORMAL BEHAVIOR for wind turbines in low wind conditions

2. META-MODEL IMPROVEMENT:
   • Current meta-model only flags health issues (gearbox/generator < 40%)
   • It does NOT consider efficiency drops in its labels
   • This is why it shows "Healthy" despite 0% efficiency

3. SUGGESTED FIXES:
   
   Option A: Refined Meta-Model Labels (Recommended)
   ────────────────────────────────────────────────
   Change label logic to:
   
   IF (gearbox_health < 40%) OR (generator_health < 40%) OR 
      (efficiency < 50% AND wind_speed > 5) THEN:
       label = 1 (Potential Issue)
   ELSE:
       label = 0 (Healthy)
   
   This will flag efficiency issues ONLY when there's sufficient wind.
   
   Option B: Alerting System (Already Implemented)
   ────────────────────────────────────────────────
   The alerting system ALREADY detects the 0% efficiency as CRITICAL.
   This may be sufficient for real-time monitoring.
   
   Option C: Separate Efficiency Monitor
   ────────────────────────────────────────────────
   Keep meta-model for component health only.
   Use the alerting system for efficiency monitoring.
   This separation of concerns may be clearer.

4. NEXT STEPS:
   • If you want the meta-model to flag efficiency issues:
     → Run the refined version with improved labels
   • If the current alerting system is sufficient:
     → No changes needed (system is working correctly)
""")

print("="*80)
print("INVESTIGATION COMPLETE")
print("="*80)
