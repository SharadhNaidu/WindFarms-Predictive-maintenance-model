"""
REFINED META-MODEL WITH IMPROVED LABELS
This version includes efficiency drops in the label logic
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

print("="*80)
print("REFINED META-MODEL TRAINING")
print("="*80)

# Load data
base_path = r"C:\Users\iamsh\Desktop\Predictive.Ai"
data_path = os.path.join(base_path, 'Kelmarsh_SCADA_2020_3086', 'Turbine_Data_Kelmarsh_1_2020-01-01_-_2021-01-01_228.csv')

print("\nLoading SCADA data...")
turbine_df = pd.read_csv(data_path, skiprows=9)
timestamp_col = '# Date and time'
turbine_df[timestamp_col] = pd.to_datetime(turbine_df[timestamp_col])
turbine_df = turbine_df.sort_values(by=timestamp_col).reset_index(drop=True)

# Split data
split_index = int(len(turbine_df) * 0.8)
test_df = turbine_df.iloc[split_index:].reset_index(drop=True)
print(f"✓ Test set: {len(test_df)} rows")

# Load models and calculate health scores
models_path = os.path.join(base_path, 'trained_models')

print("\n" + "-"*80)
print("Loading Models and Calculating Health Scores...")
print("-"*80)

# 1. Load LSTM models and scalers
from tensorflow import keras
import tensorflow as tf

# Load models without compiling (avoids Keras version issues)
gearbox_model = keras.models.load_model(os.path.join(models_path, 'gearbox_autoencoder.h5'), compile=False)
generator_model = keras.models.load_model(os.path.join(models_path, 'generator_autoencoder.h5'), compile=False)
scaler_gb = joblib.load(os.path.join(models_path, 'scaler_gb.joblib'))
scaler_gen = joblib.load(os.path.join(models_path, 'scaler_gen.joblib'))
power_curve_model = joblib.load(os.path.join(models_path, 'power_curve_model.joblib'))

print("✓ All models loaded")

# Helper function to create sequences
def create_sequences(data, sequence_length=24):
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i+sequence_length])
    return np.array(sequences)

# 2. Calculate Gearbox Health Scores
print("\nCalculating gearbox health scores...")
gearbox_features = [
    'Gear oil temperature (°C)',
    'Gear oil inlet temperature (°C)',
    'Generator RPM (RPM)'
]
X_test_gb = test_df[gearbox_features].values
X_test_gb_scaled = scaler_gb.transform(X_test_gb)
X_test_seq_gb = create_sequences(X_test_gb_scaled, 24)

reconstructions_gb = gearbox_model.predict(X_test_seq_gb, verbose=0)
mae_gb = np.mean(np.abs(X_test_seq_gb - reconstructions_gb), axis=(1, 2))
health_scores_gb = np.clip(100 - (mae_gb * 100), 0, 100)
print(f"✓ Gearbox health: {len(health_scores_gb)} scores calculated")

# 3. Calculate Generator Health Scores
print("Calculating generator health scores...")
generator_features = [
    'Stator temperature 1 (°C)',
    'Generator bearing front temperature (°C)',
    'Generator bearing rear temperature (°C)',
    'Power (kW)'
]
X_test_gen = test_df[generator_features].values
X_test_gen_scaled = scaler_gen.transform(X_test_gen)
X_test_seq_gen = create_sequences(X_test_gen_scaled, 24)

reconstructions_gen = generator_model.predict(X_test_seq_gen, verbose=0)
mae_gen = np.mean(np.abs(X_test_seq_gen - reconstructions_gen), axis=(1, 2))
health_scores_gen = np.clip(100 - (mae_gen * 100), 0, 100)
print(f"✓ Generator health: {len(health_scores_gen)} scores calculated")

# 4. Calculate Efficiency Scores
print("Calculating efficiency scores...")
wind_speed_col = 'Wind speed (m/s)'
power_col = 'Power (kW)'

X_test_wind = test_df[wind_speed_col].values.reshape(-1, 1)
predicted_power = power_curve_model.predict(X_test_wind)
actual_power = test_df[power_col].values

efficiency_scores = []
for i in range(len(actual_power)):
    if predicted_power[i] > 0:
        eff = (actual_power[i] / predicted_power[i]) * 100
        eff = max(0, min(100, eff))
    else:
        eff = 0
    efficiency_scores.append(eff)

efficiency_scores = np.array(efficiency_scores)
print(f"✓ Efficiency: {len(efficiency_scores)} scores calculated")

# 5. Calculate Alignment Scores
print("Calculating alignment scores...")
wind_dir_col = 'Wind direction (°)'
nacelle_pos_col = 'Nacelle position (°)'

wind_dir = test_df[wind_dir_col].values
nacelle_pos = test_df[nacelle_pos_col].values

yaw_errors = np.abs((wind_dir - nacelle_pos + 180) % 360 - 180)
mean_yaw_error = np.mean(yaw_errors)
alignment_score = 100 * (1 - mean_yaw_error / 180)
alignment_scores = np.full(len(test_df), alignment_score)
print(f"✓ Alignment: {alignment_score:.2f}%")

# 6. Simulate Integrity Scores
print("Simulating integrity scores...")
np.random.seed(42)
integrity_scores = np.random.normal(97.5, 1.71, len(test_df))
integrity_scores = np.clip(integrity_scores, 0, 100)
print(f"✓ Integrity: {len(integrity_scores)} scores simulated")

# Align all arrays
min_length = min(len(health_scores_gb), len(health_scores_gen), len(efficiency_scores),
                 len(alignment_scores), len(integrity_scores))

health_scores_gb = health_scores_gb[:min_length]
health_scores_gen = health_scores_gen[:min_length]
efficiency_scores_aligned = efficiency_scores[:min_length]
alignment_scores_aligned = alignment_scores[:min_length]
integrity_scores_aligned = integrity_scores[:min_length]
wind_speeds_aligned = test_df[wind_speed_col].values[:min_length]

print(f"\n✓ All arrays aligned to {min_length} samples")

# Create feature matrix
X_meta = np.column_stack([
    health_scores_gb,
    efficiency_scores_aligned,
    health_scores_gen,
    alignment_scores_aligned,
    integrity_scores_aligned
])

print(f"\n✓ Meta-model feature matrix: {X_meta.shape}")

# REFINED LABEL LOGIC
print("\n" + "="*80)
print("CREATING REFINED LABELS")
print("="*80)

print("\nLabel Logic:")
print("-" * 80)
print("IF (gearbox_health < 40%) OR")
print("   (generator_health < 40%) OR")
print("   (efficiency < 50% AND wind_speed > 5 m/s) THEN:")
print("     label = 1 (Potential Issue)")
print("ELSE:")
print("     label = 0 (Healthy)")
print("-" * 80)

# Create labels with refined logic
y_meta = np.zeros(min_length, dtype=int)

for i in range(min_length):
    # Component health issues
    if health_scores_gb[i] < 40 or health_scores_gen[i] < 40:
        y_meta[i] = 1
    # Efficiency issues (only when sufficient wind)
    elif efficiency_scores_aligned[i] < 50 and wind_speeds_aligned[i] > 5:
        y_meta[i] = 1

# Label distribution
healthy_count = np.sum(y_meta == 0)
issue_count = np.sum(y_meta == 1)

print(f"\nRefined Label Distribution:")
print(f"  Healthy (0)        : {healthy_count:5d} samples ({healthy_count/min_length*100:.1f}%)")
print(f"  Potential Issue (1): {issue_count:5d} samples ({issue_count/min_length*100:.1f}%)")

if issue_count == 0:
    print("\n⚠ WARNING: Still no 'Potential Issue' samples!")
    print("   This means:")
    print("   - No gearbox/generator health < 40%")
    print("   - No efficiency < 50% when wind speed > 5 m/s")
    print("   → All turbine operations are healthy in this dataset")
else:
    print(f"\n✓ Found {issue_count} 'Potential Issue' samples for training")

# Analyze efficiency < 50% cases
low_eff_count = np.sum(efficiency_scores_aligned < 50)
low_eff_with_wind = np.sum((efficiency_scores_aligned < 50) & (wind_speeds_aligned > 5))

print(f"\nEfficiency Analysis:")
print(f"  Total points with efficiency < 50%              : {low_eff_count}")
print(f"  Points with efficiency < 50% AND wind_speed > 5: {low_eff_with_wind}")
print(f"  → These {low_eff_with_wind} points are now flagged as 'Potential Issue'")

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_meta, y_meta, test_size=0.2, random_state=42, stratify=y_meta if issue_count > 0 else None
)

print(f"\n✓ Data split:")
print(f"  Training: {len(X_train)} samples")
print(f"  Testing:  {len(X_test)} samples")

# Train refined meta-model
print("\n" + "="*80)
print("TRAINING REFINED META-MODEL")
print("="*80)

meta_model_refined = DecisionTreeClassifier(max_depth=5, random_state=42)
meta_model_refined.fit(X_train, y_train)

print("\n✓ Refined meta-model training complete")

# Evaluate
y_pred = meta_model_refined.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nMeta-model Accuracy: {accuracy*100:.2f}%")

# Classification report
print("\nClassification Report:")
print("="*80)

unique_test_labels = np.unique(y_test)
if len(unique_test_labels) == 1:
    # Single class in test set
    print(f"⚠ Only one class in test set: {unique_test_labels[0]}")
    target_names = ['Healthy'] if unique_test_labels[0] == 0 else ['Potential Issue']
else:
    target_names = ['Healthy', 'Potential Issue']

print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

# Confusion Matrix
print("\nConfusion Matrix:")
print("="*80)
cm = confusion_matrix(y_test, y_pred)

if cm.shape == (1, 1):
    print(f"Only one class in test set:")
    print(f"  All {len(y_test)} samples classified as: {target_names[0]}")
else:
    print("                 Predicted")
    print("                 Healthy  Issue")
    print(f"Actual Healthy  {cm[0,0]:7d}  {cm[0,1]:6d}")
    print(f"       Issue    {cm[1,0]:7d}  {cm[1,1]:6d}")

# Feature importance
print("\n" + "="*80)
print("FEATURE IMPORTANCE")
print("="*80)
feature_names = ['Gearbox Health', 'Blade Efficiency', 'Generator Health', 
                 'Alignment', 'Integrity']
importances = meta_model_refined.feature_importances_

for name, importance in zip(feature_names, importances):
    print(f"  {name:20s}: {importance:.4f}")

# Test with latest data
print("\n" + "="*80)
print("PREDICTION WITH LATEST DATA")
print("="*80)

latest_features = X_meta[-1].reshape(1, -1)
latest_prediction = meta_model_refined.predict(latest_features)[0]
latest_proba = meta_model_refined.predict_proba(latest_features)[0]

print("\nLatest Health Scores:")
print(f"  Gearbox Health      : {health_scores_gb[-1]:.2f}%")
print(f"  Blade Efficiency    : {efficiency_scores_aligned[-1]:.2f}%")
print(f"  Generator Health    : {health_scores_gen[-1]:.2f}%")
print(f"  Alignment           : {alignment_scores_aligned[-1]:.2f}%")
print(f"  Integrity           : {integrity_scores_aligned[-1]:.2f}%")
print(f"  Wind Speed          : {wind_speeds_aligned[-1]:.2f} m/s")

print("\n" + "="*80)
print("OVERALL TURBINE STATUS (REFINED)")
print("="*80)

if latest_prediction == 0:
    status_icon = "🟢"
    status_text = "Healthy"
    confidence = latest_proba[0] * 100
    recommendation = "Continue normal operation with routine monitoring"
else:
    status_icon = "🔴"
    status_text = "Potential Issue Detected"
    confidence = latest_proba[1] * 100
    recommendation = "Investigate flagged components - check efficiency and component health"

print(f"  {status_icon} STATUS: {status_text}")
print(f"  Confidence: {confidence:.1f}%")
print(f"  Recommendation: {recommendation}")

# Save refined model
print("\n" + "="*80)
print("SAVING REFINED MODEL")
print("="*80)

save_path = os.path.join(models_path, 'meta_model_sensor_fusion_refined.joblib')
joblib.dump(meta_model_refined, save_path)
print(f"✓ Refined meta-model saved to:\n  {save_path}")

# Summary
print("\n" + "="*80)
print("COMPARISON: ORIGINAL VS REFINED")
print("="*80)

print("\nOriginal Meta-Model:")
print("  • Only flagged component health issues (gearbox/generator < 40%)")
print("  • Did NOT consider efficiency drops")
print("  • Result: Showed 'Healthy' despite 0% efficiency")

print("\nRefined Meta-Model:")
print("  • Flags component health issues (gearbox/generator < 40%)")
print("  • ALSO flags efficiency issues (efficiency < 50% when wind > 5 m/s)")
print("  • Result: More comprehensive health assessment")

if low_eff_with_wind > 0:
    print(f"\n✓ Successfully identified {low_eff_with_wind} efficiency issues for training")
else:
    print("\n⚠ No efficiency issues found in this dataset")
    print("  • All low efficiency points occur during low wind conditions")
    print("  • This is normal turbine behavior (not a failure)")
    print("  • Alerting system correctly flagged these as CRITICAL for investigation")

print("\n" + "="*80)
print("REFINED META-MODEL TRAINING COMPLETE")
print("="*80)
