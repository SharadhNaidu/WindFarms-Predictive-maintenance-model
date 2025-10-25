"""
Wind Turbine Predictive Maintenance System
Complete implementation with 5 diagnostic tests, health scoring, and sensor fusion
Platform: Windows 11 with NVIDIA GPU (TensorFlow GPU, Python 3.11)
Dataset: Kelmarsh SCADA 2020
"""

print("=" * 80)
print("STEP 2/14: IMPORTING LIBRARIES & GPU CONFIGURATION")
print("=" * 80)

import pandas as pd
import numpy as np
import os
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_absolute_error, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping

print("\n✓ All libraries imported successfully")

print("\n" + "=" * 80)
print("GPU DETECTION & CONFIGURATION")
print("=" * 80)

gpu_devices = tf.config.list_physical_devices('GPU')

if gpu_devices:
    print(f"✓ GPU(s) detected: {len(gpu_devices)} GPU(s) available")
    print(f"  GPU 0: {gpu_devices[0]}")
    
    for gpu in gpu_devices:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  ✓ Memory growth enabled for {gpu.name}")
        except RuntimeError as e:
            print(f"  ⚠ Warning: Could not set memory growth - {e}")
    
    print("\n✓ TensorFlow will use GPU for LSTM training")
else:
    print("⚠ WARNING: No GPU detected by TensorFlow.")
    print("  Ensure prerequisites are installed correctly:")
    print("    - NVIDIA GPU driver (latest)")
    print("    - CUDA Toolkit 11.8+ or 12.x")
    print("    - cuDNN 8.6+")
    print("    - Python 3.11 environment activated")
    print("  Verify with: nvidia-smi")
    print("  Check TensorFlow docs: https://www.tensorflow.org/install/gpu")
    print("\n⚠ LSTM training will use CPU (may be slow)")

print("\n" + "=" * 80)
print("SETTING RANDOM SEEDS FOR REPRODUCIBILITY")
print("=" * 80)

np.random.seed(42)
tf.random.set_seed(42)

print("✓ Random seeds set (numpy: 42, tensorflow: 42)")

print("\n" + "=" * 80)
print("STEP 2/14 COMPLETE - Ready for Step 3/14: Data Loading")
print("=" * 80)

print("\n" + "=" * 80)
print("STEP 2/13: LOAD DATA (LOCAL PATHS)")
print("=" * 80)

base_data_path = 'C:/Users/iamsh/Desktop/Predictive.Ai/'

print(f"\nBase data path: {base_data_path}")

scada_folder_path = os.path.join(base_data_path, 'Kelmarsh_SCADA_2020_3086')

scada_filename = 'Turbine_Data_Kelmarsh_1_2020-01-01_-_2021-01-01_228.csv'

scada_file_path = os.path.join(scada_folder_path, scada_filename)

print(f"SCADA folder path: {scada_folder_path}")
print(f"SCADA file path: {scada_file_path}")

print("\n" + "-" * 80)
print("Loading SCADA Data...")
print("-" * 80)

try:
    scada_df = pd.read_csv(scada_file_path, skiprows=9, low_memory=False)
    print(f"✓ Successfully loaded SCADA data from: {scada_file_path}")
    print(f"  Dataset shape: {scada_df.shape[0]:,} rows × {scada_df.shape[1]} columns")

except FileNotFoundError:
    print(f"\n⚠ ERROR: SCADA file not found!")
    print(f"  Expected path: {scada_file_path}")
    print(f"\n  TROUBLESHOOTING:")
    print(f"  1. Verify the file exists at the specified location")
    print(f"  2. Check that 'scada_filename' variable matches your actual CSV filename")
    print(f"  3. Verify 'scada_folder_path' is correct")
    print(f"  4. Ensure the Kelmarsh SCADA data has been extracted")
    raise

except Exception as e:
    print(f"\n⚠ ERROR: Failed to load SCADA data")
    print(f"  Error details: {e}")
    print(f"  File path: {scada_file_path}")
    raise

print("\n" + "-" * 80)
print("SCADA Data Preview (First 5 Rows):")
print("-" * 80)
print(scada_df.head())

print("\n" + "-" * 80)
print("SCADA Data Exploration:")
print("-" * 80)

print(f"\nDataset Shape: {scada_df.shape[0]:,} rows × {scada_df.shape[1]} columns")

print("\nDataset Info:")
print("=" * 80)
scada_df.info()

print("\n" + "-" * 80)
print("Missing Values Analysis:")
print("-" * 80)

missing_values = scada_df.isnull().sum()
missing_values_filtered = missing_values[missing_values > 0]

if len(missing_values_filtered) > 0:
    print(f"\n⚠ Found {len(missing_values_filtered)} columns with missing values:\n")
    for col, count in missing_values_filtered.items():
        percentage = (count / len(scada_df)) * 100
        print(f"  {col}: {count:,} missing ({percentage:.2f}%)")
else:
    print("\n✓ No missing values found in the dataset")

print("\n" + "=" * 80)
print("DATA LOADING COMPLETE")
print("=" * 80)
print("Ready for Step 3/13: Data Preprocessing and Feature Engineering\n")

print("\n" + "=" * 80)
print("STEP 3/13: PREPROCESSING - RENAMING, TIMESTAMP, FILTERING, MISSING VALUES")
print("=" * 80)

print("\n" + "-" * 80)
print("Skipping Column Mapping (using original column names)...")
print("-" * 80)

print(f"Column count: {len(scada_df.columns)}")
print(f"\n  Sample columns (first 10):")
for col in list(scada_df.columns)[:10]:
    print(f"    {col}")

print("\n" + "-" * 80)
print("Processing Timestamp Column...")
print("-" * 80)

timestamp_col = '# Date and time'

if timestamp_col in scada_df.columns:
    print(f"Found timestamp column: '{timestamp_col}'")

    try:
        print(f"Converting '{timestamp_col}' to datetime format...")
        scada_df[timestamp_col] = pd.to_datetime(scada_df[timestamp_col])
        print(f"✓ Converted to datetime")

        print(f"Setting '{timestamp_col}' as DataFrame index...")
        scada_df.set_index(timestamp_col, inplace=True)
        print(f"✓ Set as index")

        print(f"Sorting by timestamp...")
        scada_df.sort_index(inplace=True)
        print(f"✓ Sorted by timestamp")

        print(f"\n  Timestamp range:")
        print(f"    Start: {scada_df.index.min()}")
        print(f"    End:   {scada_df.index.max()}")
        print(f"    Duration: {scada_df.index.max() - scada_df.index.min()}")

    except Exception as e:
        print(f"\n⚠ ERROR: Failed to process timestamp column")
        print(f"  Error details: {e}")
        print(f"  Column: '{timestamp_col}'")
        raise

else:
    print(f"⚠ WARNING: Timestamp column '{timestamp_col}' not found in DataFrame")
    print(f"  Available columns: {list(scada_df.columns)[:10]}...")
    print(f"  Please update 'timestamp_col' variable with the correct column name")
    print(f"  Continuing without timestamp processing...")

print("\n" + "-" * 80)
print("Skipping Turbine ID Filtering (already loaded single turbine data)...")
print("-" * 80)

print(f"Using entire dataset (Turbine 1 data only)")
turbine_df = scada_df.copy()
print(f"  Dataset shape: {turbine_df.shape[0]:,} rows × {turbine_df.shape[1]} columns")

print("\n" + "-" * 80)
print("Handling Missing Values...")
print("-" * 80)

print(f"Current DataFrame shape: {turbine_df.shape[0]:,} rows × {turbine_df.shape[1]} columns")

missing_before = turbine_df.isnull().sum().sum()
missing_before_pct = (missing_before / (turbine_df.shape[0] * turbine_df.shape[1])) * 100

print(f"\nMissing values BEFORE imputation:")
print(f"  Total missing: {missing_before:,} ({missing_before_pct:.2f}%)")

numeric_cols = turbine_df.select_dtypes(include=np.number).columns
print(f"\nNumeric columns identified: {len(numeric_cols)}")

print(f"\nApplying forward fill (ffill) to numeric columns...")
turbine_df[numeric_cols] = turbine_df[numeric_cols].ffill()

print(f"Applying backward fill (bfill) to numeric columns...")
turbine_df[numeric_cols] = turbine_df[numeric_cols].bfill()

missing_after = turbine_df.isnull().sum().sum()
missing_after_pct = (missing_after / (turbine_df.shape[0] * turbine_df.shape[1])) * 100
missing_filled = missing_before - missing_after

print(f"\nMissing values AFTER imputation:")
print(f"  Total missing: {missing_after:,} ({missing_after_pct:.2f}%)")
print(f"  Values filled: {missing_filled:,}")

if missing_after > 0:
    print(f"\n⚠ WARNING: {missing_after:,} missing values still remain!")
    remaining_missing = turbine_df.isnull().sum()
    remaining_missing_filtered = remaining_missing[remaining_missing > 0]
    print(f"  Columns with remaining missing values:")
    for col, count in remaining_missing_filtered.items():
        percentage = (count / len(turbine_df)) * 100
        print(f"    {col}: {count:,} missing ({percentage:.2f}%)")
    print(f"  Consider additional imputation strategies or removing these columns")
else:
    print(f"✓ All missing values successfully handled!")

print("\n" + "-" * 80)
print("Processed Turbine Data Preview (First 5 Rows):")
print("-" * 80)
print(turbine_df.head())

print("\n" + "-" * 80)
print("Processed Turbine Data Info:")
print("-" * 80)
turbine_df.info()

print("\n" + "=" * 80)
print("PREPROCESSING COMPLETE")
print("=" * 80)
print("Ready for Step 4/13: Feature Engineering for Diagnostic Tests\n")

print("\n" + "=" * 80)
print("STEP 4/13: DATA SPLITTING")
print("=" * 80)

print("\n" + "-" * 80)
print("Creating Models Directory...")
print("-" * 80)

models_path = os.path.join(base_data_path, 'trained_models')
os.makedirs(models_path, exist_ok=True)
print(f"✓ Models directory created: {models_path}")

print("\n" + "-" * 80)
print("Splitting Data into Training and Testing Sets...")
print("-" * 80)

train_df, test_df = train_test_split(
    turbine_df,
    test_size=0.2,
    random_state=42,
    shuffle=False
)

print(f"\n✓ Data split complete")
print(f"\nOriginal dataset:")
print(f"  Total rows: {turbine_df.shape[0]:,}")
print(f"  Total columns: {turbine_df.shape[1]}")

print(f"\nTraining set (80%):")
print(f"  Shape: {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")
print(f"  Percentage: {(train_df.shape[0] / turbine_df.shape[0]) * 100:.1f}%")

print(f"\nTesting set (20%):")
print(f"  Shape: {test_df.shape[0]:,} rows × {test_df.shape[1]} columns")
print(f"  Percentage: {(test_df.shape[0] / turbine_df.shape[0]) * 100:.1f}%")

if isinstance(train_df.index, pd.DatetimeIndex):
    print(f"\nTemporal split:")
    print(f"  Training period: {train_df.index.min()} to {train_df.index.max()}")
    print(f"  Testing period:  {test_df.index.min()} to {test_df.index.max()}")

print("\n" + "=" * 80)
print("DATA SPLITTING COMPLETE")
print("=" * 80)
print("Ready for Step 5/13: Feature Engineering for Diagnostic Tests\n")

print("\n" + "=" * 80)
print("STEP 5/13: TEST 2 - BLADE/ROTOR (POWER CURVE - RANDOMFORESTREGRESSOR)")
print("=" * 80)

print("\n" + "-" * 80)
print("Defining Feature Columns for Power Curve Model...")
print("-" * 80)

wind_speed_col = 'Wind speed (m/s)'
power_col = 'Power (kW)'

print(f"Wind Speed Column: '{wind_speed_col}'")
print(f"Power Column: '{power_col}'")

required_cols = [wind_speed_col, power_col]
missing_cols = [col for col in required_cols if col not in train_df.columns]

if missing_cols:
    print(f"\n⚠ ERROR: Required columns not found in DataFrame!")
    print(f"  Missing columns: {missing_cols}")
    print(f"  Available columns: {list(train_df.columns)[:10]}...")
    print(f"\n  Please update column names in Step 5")
    raise KeyError(f"Required columns not found: {missing_cols}")

print(f"✓ All required columns found in DataFrame")

print("\n" + "-" * 80)
print("Preparing Training and Testing Data...")
print("-" * 80)

X_train_pc = train_df[[wind_speed_col]]
y_train_pc = train_df[power_col]

X_test_pc = test_df[[wind_speed_col]]
y_test_pc = test_df[power_col]

print(f"Training set:")
print(f"  X_train_pc shape: {X_train_pc.shape}")
print(f"  y_train_pc shape: {y_train_pc.shape}")

print(f"\nTesting set:")
print(f"  X_test_pc shape: {X_test_pc.shape}")
print(f"  y_test_pc shape: {y_test_pc.shape}")

print("\n" + "-" * 80)
print("Training Power Curve Model (Random Forest Regressor)...")
print("-" * 80)

power_curve_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    max_depth=10,
    min_samples_leaf=5
)

print(f"Model parameters:")
print(f"  n_estimators: 100")
print(f"  max_depth: 10")
print(f"  min_samples_leaf: 5")
print(f"  n_jobs: -1 (using all CPU cores)")

print(f"\nTraining model...")
power_curve_model.fit(X_train_pc, y_train_pc)
print(f"✓ Power curve model training complete")

print("\n" + "-" * 80)
print("Evaluating Power Curve Model on Test Set...")
print("-" * 80)

y_pred_pc_test = power_curve_model.predict(X_test_pc)

r2 = r2_score(y_test_pc, y_pred_pc_test)
mae = mean_absolute_error(y_test_pc, y_pred_pc_test)

print(f"\nModel Performance Metrics:")
print(f"  R² Score: {r2:.4f}")
print(f"  Mean Absolute Error (MAE): {mae:.2f}")

if r2 > 0.9:
    print(f"  ✓ Excellent fit (R² > 0.9)")
elif r2 > 0.7:
    print(f"  ✓ Good fit (R² > 0.7)")
elif r2 > 0.5:
    print(f"  ⚠ Moderate fit (R² > 0.5)")
else:
    print(f"  ⚠ Poor fit (R² < 0.5) - consider feature engineering")

print("\n" + "-" * 80)
print("Generating Power Curve Visualization...")
print("-" * 80)

plt.figure(figsize=(12, 6))

plt.scatter(X_test_pc, y_test_pc, alpha=0.3, s=10, label='Actual Data', color='blue')

X_test_sorted = np.sort(X_test_pc.values, axis=0)
y_pred_pc_sorted = power_curve_model.predict(X_test_sorted)

plt.plot(X_test_sorted, y_pred_pc_sorted, color='red', linewidth=2, label='Predicted Power Curve')

plt.title('Blade/Rotor Health - Power Curve Analysis', fontsize=14, fontweight='bold')
plt.xlabel(f'Wind Speed ({wind_speed_col})', fontsize=12)
plt.ylabel(f'Active Power ({power_col})', fontsize=12)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = os.path.join(models_path, 'power_curve_plot.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Power curve plot saved to: {plot_path}")

print("\n" + "-" * 80)
print("Calculating Efficiency Health Score...")
print("-" * 80)

def calculate_efficiency_score(y_true, y_pred):
    """
    Calculate efficiency score as percentage of predicted vs actual power.
    Returns health score from 0-100%.
    
    Parameters:
    - y_true: Actual power output
    - y_pred: Predicted power output
    
    Returns:
    - Health score (0-100%)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        efficiency = np.divide(y_pred, y_true, out=np.zeros_like(y_pred, dtype=float), where=y_true!=0) * 100

    score = np.clip(efficiency, 0, 100)

    return score

test_efficiency_scores = calculate_efficiency_score(y_test_pc.values, y_pred_pc_test)

latest_efficiency_score = test_efficiency_scores[-1]

print(f"Efficiency Health Scores:")
print(f"  Mean: {np.mean(test_efficiency_scores):.2f}%")
print(f"  Median: {np.median(test_efficiency_scores):.2f}%")
print(f"  Std Dev: {np.std(test_efficiency_scores):.2f}%")
print(f"  Min: {np.min(test_efficiency_scores):.2f}%")
print(f"  Max: {np.max(test_efficiency_scores):.2f}%")
print(f"\n  Latest Efficiency Score: {latest_efficiency_score:.2f}%")

print("\n" + "-" * 80)
print("Defining Severity Level Thresholds...")
print("-" * 80)

EFFICIENCY_CRITICAL = 50      # Below 50% = Critical
EFFICIENCY_MAINTENANCE = 90   # 50-90% = Maintenance Required
EFFICIENCY_WARNING = 95       # 90-95% = Warning

print(f"Severity Thresholds:")
print(f"  Critical:    < {EFFICIENCY_CRITICAL}%")
print(f"  Maintenance: {EFFICIENCY_CRITICAL}% - {EFFICIENCY_MAINTENANCE}%")
print(f"  Warning:     {EFFICIENCY_MAINTENANCE}% - {EFFICIENCY_WARNING}%")
print(f"  Normal:      ≥ {EFFICIENCY_WARNING}%")

if latest_efficiency_score < EFFICIENCY_CRITICAL:
    severity_level = "CRITICAL"
    severity_color = "🔴"
elif latest_efficiency_score < EFFICIENCY_MAINTENANCE:
    severity_level = "MAINTENANCE REQUIRED"
    severity_color = "🟠"
elif latest_efficiency_score < EFFICIENCY_WARNING:
    severity_level = "WARNING"
    severity_color = "🟡"
else:
    severity_level = "NORMAL"
    severity_color = "🟢"

print(f"\nCurrent Blade/Rotor Status:")
print(f"  {severity_color} {severity_level} (Efficiency: {latest_efficiency_score:.2f}%)")

print("\n" + "-" * 80)
print("Preparing for Sensor Fusion...")
print("-" * 80)

print(f"✓ Blade/Rotor efficiency score ready for sensor fusion meta-model")
print(f"  Feature name: 'blade_rotor_efficiency_score'")
print(f"  Current value: {latest_efficiency_score:.2f}%")

print("\n" + "-" * 80)
print("Saving Power Curve Model...")
print("-" * 80)

print(f"Models directory: {models_path}")

power_curve_model_path = os.path.join(models_path, 'power_curve_model.joblib')

joblib.dump(power_curve_model, power_curve_model_path)
print(f"✓ Power curve model saved to: {power_curve_model_path}")

print("\n" + "=" * 80)
print("BLADE/ROTOR TEST COMPLETE")
print("=" * 80)
print("Ready for Step 6/13: Test 3 - Gearbox (Temperature Anomaly - LSTM Autoencoder)\n")

print("\n" + "=" * 80)
print("STEP 6/13: TEST 4 - YAW/PITCH ALIGNMENT (RULE-BASED)")
print("=" * 80)

print("\n" + "-" * 80)
print("Defining Feature Columns for Yaw/Pitch Alignment Test...")
print("-" * 80)

wind_dir_col = 'Wind direction (°)'
nacelle_pos_col = 'Nacelle position (°)'

print(f"Wind Direction Column: '{wind_dir_col}'")
print(f"Nacelle Position Column: '{nacelle_pos_col}'")

required_cols_yaw = [wind_dir_col, nacelle_pos_col]
missing_cols_yaw = [col for col in required_cols_yaw if col not in turbine_df.columns]

if missing_cols_yaw:
    print(f"\n⚠ ERROR: Required columns not found in DataFrame!")
    print(f"  Missing columns: {missing_cols_yaw}")
    print(f"  Available columns: {list(turbine_df.columns)[:10]}...")
    print(f"\n  Please update column names in Step 6")
    raise KeyError(f"Required columns not found: {missing_cols_yaw}")

print(f"✓ All required columns found in DataFrame")

print("\n" + "-" * 80)
print("Calculating Yaw Misalignment Error...")
print("-" * 80)

error = (turbine_df[nacelle_pos_col] - turbine_df[wind_dir_col] + 180) % 360 - 180

turbine_df['Yaw_Error'] = np.abs(error)

print(f"✓ Yaw error calculated for full dataset")
print(f"  Mean yaw error: {turbine_df['Yaw_Error'].mean():.2f}°")
print(f"  Max yaw error: {turbine_df['Yaw_Error'].max():.2f}°")

error_test = (test_df[nacelle_pos_col] - test_df[wind_dir_col] + 180) % 360 - 180
test_df['Yaw_Error'] = np.abs(error_test)

print(f"\n✓ Yaw error calculated for test set")
print(f"  Test mean yaw error: {test_df['Yaw_Error'].mean():.2f}°")
print(f"  Test max yaw error: {test_df['Yaw_Error'].max():.2f}°")

print("\n" + "-" * 80)
print("Performing Rule-Based Misalignment Check...")
print("-" * 80)

yaw_error_threshold = 15

print(f"Yaw error threshold: {yaw_error_threshold}°")

misalignment_percentage = (test_df['Yaw_Error'] > yaw_error_threshold).mean() * 100

print(f"\nMisalignment Analysis:")
print(f"  Measurements exceeding threshold: {misalignment_percentage:.2f}%")
print(f"  Total test measurements: {len(test_df):,}")
print(f"  Misaligned measurements: {(test_df['Yaw_Error'] > yaw_error_threshold).sum():,}")

if misalignment_percentage > 20:
    print(f"  ⚠ WARNING: High misalignment rate (>{20}%)")
elif misalignment_percentage > 10:
    print(f"  ⚠ Moderate misalignment rate (>{10}%)")
else:
    print(f"  ✓ Low misalignment rate (<{10}%)")

print("\n" + "-" * 80)
print("Calculating Alignment Health Score...")
print("-" * 80)

def calculate_alignment_score(yaw_error_series):
    """
    Calculate alignment health score based on mean yaw error.
    Lower error = higher score.
    
    Parameters:
    - yaw_error_series: Series of absolute yaw errors
    
    Returns:
    - Health score (0-100%)
    """
    mean_error = yaw_error_series.mean()

    score = 100 - mean_error

    score = np.clip(score, 0, 100)

    return score

test_alignment_score = calculate_alignment_score(test_df['Yaw_Error'])

latest_alignment_score = test_alignment_score

print(f"Alignment Health Score:")
print(f"  Test set score: {test_alignment_score:.2f}%")
print(f"  Mean yaw error: {test_df['Yaw_Error'].mean():.2f}°")
print(f"  Std dev yaw error: {test_df['Yaw_Error'].std():.2f}°")
print(f"\n  Latest Alignment Score: {latest_alignment_score:.2f}%")

print("\n" + "-" * 80)
print("Defining Severity Level Thresholds...")
print("-" * 80)

ALIGNMENT_MAINTENANCE = 90   # Below 90% = Maintenance Required
ALIGNMENT_WARNING = 95       # 90-95% = Warning

print(f"Severity Thresholds:")
print(f"  Maintenance: < {ALIGNMENT_MAINTENANCE}%")
print(f"  Warning:     {ALIGNMENT_MAINTENANCE}% - {ALIGNMENT_WARNING}%")
print(f"  Normal:      ≥ {ALIGNMENT_WARNING}%")

if latest_alignment_score < ALIGNMENT_MAINTENANCE:
    alignment_severity = "MAINTENANCE REQUIRED"
    alignment_color = "🟠"
elif latest_alignment_score < ALIGNMENT_WARNING:
    alignment_severity = "WARNING"
    alignment_color = "🟡"
else:
    alignment_severity = "NORMAL"
    alignment_color = "🟢"

print(f"\nCurrent Yaw/Pitch Alignment Status:")
print(f"  {alignment_color} {alignment_severity} (Score: {latest_alignment_score:.2f}%)")
print(f"  Mean misalignment: {test_df['Yaw_Error'].mean():.2f}°")

print("\n" + "-" * 80)
print("Preparing for Sensor Fusion...")
print("-" * 80)

print(f"✓ Yaw/Pitch alignment score ready for sensor fusion meta-model")
print(f"  Feature name: 'yaw_pitch_alignment_score'")
print(f"  Current value: {latest_alignment_score:.2f}%")

print("\n" + "=" * 80)
print("YAW/PITCH ALIGNMENT TEST COMPLETE")
print("=" * 80)
print("Ready for Step 7/13: Test 1 - Gearbox (Temperature Anomaly - LSTM Autoencoder)\n")

print("\n" + "=" * 80)
print("STEP 7/13: LSTM DATA PREPARATION (SCALING & WINDOWING)")
print("=" * 80)

print("\n" + "-" * 80)
print("Defining Feature Sets for LSTM Models...")
print("-" * 80)

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

print(f"Gearbox Features ({len(gb_features)}):")
for i, feat in enumerate(gb_features, 1):
    print(f"  {i}. {feat}")

print(f"\nGenerator Features ({len(gen_features)}):")
for i, feat in enumerate(gen_features, 1):
    print(f"  {i}. {feat}")

all_lstm_features = gb_features + gen_features
missing_lstm_cols = [col for col in all_lstm_features if col not in train_df.columns]

if missing_lstm_cols:
    print(f"\n⚠ ERROR: Required LSTM feature columns not found in DataFrame!")
    print(f"  Missing columns: {missing_lstm_cols}")
    print(f"  Available columns: {list(train_df.columns)[:15]}...")
    print(f"\n  Please update feature lists in Step 7")
    raise KeyError(f"Required LSTM feature columns not found: {missing_lstm_cols}")

print(f"\n✓ All LSTM feature columns found in DataFrame")

print("\n" + "-" * 80)
print("Initializing StandardScalers...")
print("-" * 80)

scaler_gb = StandardScaler()
scaler_gen = StandardScaler()

print(f"✓ Gearbox scaler initialized (StandardScaler)")
print(f"✓ Generator scaler initialized (StandardScaler)")

print("\n" + "-" * 80)
print("Scaling Gearbox Features...")
print("-" * 80)

scaler_gb.fit(train_df[gb_features])
print(f"✓ Gearbox scaler fitted on training data")

train_scaled_gb = scaler_gb.transform(train_df[gb_features])
print(f"✓ Training data scaled")
print(f"  Shape: {train_scaled_gb.shape}")

test_scaled_gb = scaler_gb.transform(test_df[gb_features])
print(f"✓ Testing data scaled")
print(f"  Shape: {test_scaled_gb.shape}")

print(f"\nGearbox scaling statistics:")
print(f"  Mean values: {scaler_gb.mean_}")
print(f"  Std dev values: {scaler_gb.scale_}")

print("\n" + "-" * 80)
print("Scaling Generator Features...")
print("-" * 80)

scaler_gen.fit(train_df[gen_features])
print(f"✓ Generator scaler fitted on training data")

train_scaled_gen = scaler_gen.transform(train_df[gen_features])
print(f"✓ Training data scaled")
print(f"  Shape: {train_scaled_gen.shape}")

test_scaled_gen = scaler_gen.transform(test_df[gen_features])
print(f"✓ Testing data scaled")
print(f"  Shape: {test_scaled_gen.shape}")

print(f"\nGenerator scaling statistics:")
print(f"  Mean values: {scaler_gen.mean_}")
print(f"  Std dev values: {scaler_gen.scale_}")

print("\n" + "-" * 80)
print("Defining Sequence Creation Function...")
print("-" * 80)

def create_sequences(data, sequence_length):
    """
    Create sliding window sequences from time-series data for LSTM input.
    
    Parameters:
    - data: numpy array of shape (n_samples, n_features)
    - sequence_length: number of timesteps in each sequence
    
    Returns:
    - numpy array of shape (n_sequences, sequence_length, n_features)
    """
    sequences = []

    for i in range(len(data) - sequence_length):
        sequence = data[i:i + sequence_length]
        sequences.append(sequence)

    return np.array(sequences)

print(f"✓ Sequence creation function defined")
print(f"  Function: create_sequences(data, sequence_length)")
print(f"  Input: (n_samples, n_features)")
print(f"  Output: (n_sequences, sequence_length, n_features)")

print("\n" + "-" * 80)
print("Creating Sequences for LSTM Models...")
print("-" * 80)

sequence_length = 24

print(f"Sequence length: {sequence_length} timesteps")
print(f"  (Each sequence contains {sequence_length} consecutive measurements)")

print(f"\nCreating Gearbox sequences...")
X_train_seq_gb = create_sequences(train_scaled_gb, sequence_length)
X_test_seq_gb = create_sequences(test_scaled_gb, sequence_length)

print(f"✓ Gearbox sequences created")
print(f"  X_train_seq_gb shape: {X_train_seq_gb.shape} (n_sequences, timesteps, features)")
print(f"  X_test_seq_gb shape: {X_test_seq_gb.shape} (n_sequences, timesteps, features)")

print(f"\nCreating Generator sequences...")
X_train_seq_gen = create_sequences(train_scaled_gen, sequence_length)
X_test_seq_gen = create_sequences(test_scaled_gen, sequence_length)

print(f"✓ Generator sequences created")
print(f"  X_train_seq_gen shape: {X_train_seq_gen.shape} (n_sequences, timesteps, features)")
print(f"  X_test_seq_gen shape: {X_test_seq_gen.shape} (n_sequences, timesteps, features)")

print(f"\n" + "-" * 80)
print(f"LSTM Data Preparation Summary:")
print(f"-" * 80)
print(f"Gearbox Model:")
print(f"  Features: {len(gb_features)}")
print(f"  Sequence length: {sequence_length}")
print(f"  Training sequences: {X_train_seq_gb.shape[0]:,}")
print(f"  Testing sequences: {X_test_seq_gb.shape[0]:,}")

print(f"\nGenerator Model:")
print(f"  Features: {len(gen_features)}")
print(f"  Sequence length: {sequence_length}")
print(f"  Training sequences: {X_train_seq_gen.shape[0]:,}")
print(f"  Testing sequences: {X_test_seq_gen.shape[0]:,}")

print("\n" + "=" * 80)
print("LSTM DATA PREPARATION COMPLETE")
print("=" * 80)
print("Ready for Step 8/13: Test 1 - Gearbox (LSTM Autoencoder Training)\n")

print("\n" + "=" * 80)
print("STEP 8/13: TEST 1 - GEARBOX FAILURES (LSTM AUTOENCODER)")
print("=" * 80)

print("\n" + "-" * 80)
print("Defining Input Shape for Gearbox LSTM Autoencoder...")
print("-" * 80)

n_features_gb = X_train_seq_gb.shape[2]

input_shape_gb = (sequence_length, n_features_gb)

print(f"Input shape: {input_shape_gb}")
print(f"  Sequence length: {sequence_length} timesteps")
print(f"  Number of features: {n_features_gb}")

print("\n" + "-" * 80)
print("Building Gearbox LSTM Autoencoder Model...")
print("-" * 80)

input_layer_gb = Input(shape=input_shape_gb, name='input_gearbox')

encoded_gb = LSTM(64, activation='relu', name='encoder_lstm')(input_layer_gb)

repeated_gb = RepeatVector(sequence_length, name='repeat_vector')(encoded_gb)

decoded_gb = LSTM(64, activation='relu', return_sequences=True, name='decoder_lstm')(repeated_gb)

output_layer_gb = TimeDistributed(Dense(n_features_gb), name='output_gearbox')(decoded_gb)

model_gb = Model(inputs=input_layer_gb, outputs=output_layer_gb, name='Gearbox_LSTM_Autoencoder')

print(f"✓ Gearbox LSTM Autoencoder model built")
print(f"  Architecture: Input -> LSTM(64) -> RepeatVector -> LSTM(64) -> TimeDistributed(Dense)")

print("\n" + "-" * 80)
print("Compiling Model...")
print("-" * 80)

model_gb.compile(optimizer='adam', loss='mae')

print(f"✓ Model compiled")
print(f"  Optimizer: Adam")
print(f"  Loss: Mean Absolute Error (MAE)")

print(f"\nModel Summary:")
print("=" * 80)
model_gb.summary()
print("=" * 80)

print("\n" + "-" * 80)
print("Defining Early Stopping Callback...")
print("-" * 80)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

print(f"✓ Early stopping configured")
print(f"  Monitor: val_loss")
print(f"  Patience: 5 epochs")
print(f"  Restore best weights: True")

print("\n" + "-" * 80)
print("Training Gearbox LSTM Autoencoder...")
print(f"{'GPU-ACCELERATED' if tf.config.list_physical_devices('GPU') else 'CPU'} Training")
print("-" * 80)

print(f"\nTraining parameters:")
print(f"  Epochs: 20")
print(f"  Batch size: 64")
print(f"  Training sequences: {X_train_seq_gb.shape[0]:,}")
print(f"  Validation sequences: {X_test_seq_gb.shape[0]:,}")

print(f"\nStarting training...\n")

history_gb = model_gb.fit(
    X_train_seq_gb,
    X_train_seq_gb,  # Autoencoder reconstructs its input
    validation_data=(X_test_seq_gb, X_test_seq_gb),
    epochs=20,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)

print(f"\n✓ Gearbox LSTM Autoencoder training complete")
print(f"  Final training loss: {history_gb.history['loss'][-1]:.6f}")
print(f"  Final validation loss: {history_gb.history['val_loss'][-1]:.6f}")
print(f"  Epochs trained: {len(history_gb.history['loss'])}")

print("\n" + "-" * 80)
print("Calculating Training Reconstruction Errors...")
print("-" * 80)

X_train_pred_gb = model_gb.predict(X_train_seq_gb, verbose=0)

train_mae_loss_gb = np.mean(np.abs(X_train_pred_gb - X_train_seq_gb), axis=(1, 2))

print(f"✓ Training reconstruction errors calculated")
print(f"  Total sequences: {len(train_mae_loss_gb):,}")
print(f"  Mean error: {np.mean(train_mae_loss_gb):.6f}")
print(f"  Std dev: {np.std(train_mae_loss_gb):.6f}")
print(f"  Min error: {np.min(train_mae_loss_gb):.6f}")
print(f"  Max error: {np.max(train_mae_loss_gb):.6f}")

print("\n" + "-" * 80)
print("Plotting Reconstruction Error Distribution...")
print("-" * 80)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(train_mae_loss_gb, bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(np.mean(train_mae_loss_gb), color='red', linestyle='--', linewidth=2, label='Mean')
plt.axvline(np.median(train_mae_loss_gb), color='green', linestyle='--', linewidth=2, label='Median')
plt.title('Gearbox Reconstruction Error - Histogram', fontsize=12, fontweight='bold')
plt.xlabel('Reconstruction Error (MAE)', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
sns.kdeplot(train_mae_loss_gb, fill=True, color='blue', alpha=0.6)
plt.axvline(np.mean(train_mae_loss_gb), color='red', linestyle='--', linewidth=2, label='Mean')
plt.axvline(np.median(train_mae_loss_gb), color='green', linestyle='--', linewidth=2, label='Median')
plt.title('Gearbox Reconstruction Error - Density', fontsize=12, fontweight='bold')
plt.xlabel('Reconstruction Error (MAE)', fontsize=10)
plt.ylabel('Density', fontsize=10)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(models_path, 'gearbox_error_distribution.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Error distribution plots saved to: {plot_path}")

print("\n" + "-" * 80)
print("Defining Healthy Threshold...")
print("-" * 80)

HEALTHY_THRESHOLD_GB = np.percentile(train_mae_loss_gb, 99)

print(f"Healthy Threshold (99th percentile): {HEALTHY_THRESHOLD_GB:.6f}")
print(f"  99% of training data has error below this threshold")
print(f"  Sequences exceeding this may indicate anomalies")

print("\n" + "-" * 80)
print("Calculating Gearbox Health Score...")
print("-" * 80)

FAILURE_THRESHOLD_GB = np.max(train_mae_loss_gb) * 5

print(f"Failure Threshold (5x max training error): {FAILURE_THRESHOLD_GB:.6f}")

def calculate_health_score(error, healthy_thresh, failure_thresh):
    """
    Calculate health score from reconstruction error.
    
    Parameters:
    - error: Reconstruction error value
    - healthy_thresh: Threshold for healthy operation
    - failure_thresh: Threshold for failure
    
    Returns:
    - Health score (0-100%)
    """
    if error <= healthy_thresh:
        norm_error = 0
    elif error >= failure_thresh:
        norm_error = 1
    else:
        norm_error = (error - healthy_thresh) / (failure_thresh - healthy_thresh)

    health = (1 - norm_error) * 100

    health = np.clip(health, 0, 100)

    return health

last_seq_pred_gb = model_gb.predict(X_test_seq_gb[-1:], verbose=0)

last_error_gb = np.mean(np.abs(last_seq_pred_gb - X_test_seq_gb[-1:]))

print(f"\nLatest sequence reconstruction error: {last_error_gb:.6f}")

latest_gearbox_health_score = calculate_health_score(
    last_error_gb,
    HEALTHY_THRESHOLD_GB,
    FAILURE_THRESHOLD_GB
)

print(f"\nGearbox Health Score:")
print(f"  Latest Health Score: {latest_gearbox_health_score:.2f}%")
print(f"  Reconstruction Error: {last_error_gb:.6f}")
print(f"  Healthy Threshold: {HEALTHY_THRESHOLD_GB:.6f}")

if last_error_gb <= HEALTHY_THRESHOLD_GB:
    print(f"  Status: ✓ Within normal range")
else:
    print(f"  Status: ⚠ Above normal range - potential anomaly")

print("\n" + "-" * 80)
print("Defining Severity Level Thresholds...")
print("-" * 80)

GB_CRITICAL_TEMP = 100        # Critical temperature threshold (°C)
GB_MAINTENANCE_HEALTH = 30    # Below 30% health = Maintenance Required
GB_WARNING_HEALTH = 60        # 30-60% health = Warning

print(f"Gearbox Severity Thresholds:")
print(f"  Critical Health:    < {GB_MAINTENANCE_HEALTH}%")
print(f"  Maintenance Health: {GB_MAINTENANCE_HEALTH}% - {GB_WARNING_HEALTH}%")
print(f"  Warning Health:     {GB_WARNING_HEALTH}% - 100%")
print(f"  Critical Temperature: > {GB_CRITICAL_TEMP}°C")

if latest_gearbox_health_score < GB_MAINTENANCE_HEALTH:
    gb_severity = "CRITICAL"
    gb_color = "🔴"
elif latest_gearbox_health_score < GB_WARNING_HEALTH:
    gb_severity = "MAINTENANCE REQUIRED"
    gb_color = "🟠"
else:
    gb_severity = "NORMAL"
    gb_color = "🟢"

print(f"\nCurrent Gearbox Status:")
print(f"  {gb_color} {gb_severity} (Health: {latest_gearbox_health_score:.2f}%)")

print("\n" + "-" * 80)
print("Preparing for Sensor Fusion...")
print("-" * 80)

print(f"✓ Gearbox health score ready for sensor fusion meta-model")
print(f"  Feature name: 'gearbox_health_score'")
print(f"  Current value: {latest_gearbox_health_score:.2f}%")

print("\n" + "-" * 80)
print("Saving Gearbox LSTM Model...")
print("-" * 80)

gearbox_model_path = os.path.join(models_path, 'gearbox_lstm_autoencoder.h5')
model_gb.save(gearbox_model_path)
print(f"✓ Gearbox LSTM model saved to: {gearbox_model_path}")

scaler_gb_path = os.path.join(models_path, 'scaler_gearbox.joblib')
joblib.dump(scaler_gb, scaler_gb_path)
print(f"✓ Gearbox scaler saved to: {scaler_gb_path}")

print("\n" + "=" * 80)
print("GEARBOX TEST COMPLETE")
print("=" * 80)
print("Ready for Step 9/13: Test 3 - Generator (LSTM Autoencoder)\n")

print("\n" + "=" * 80)
print("STEP 9/13: TEST 3 - GENERATOR & ELECTRICAL (LSTM AUTOENCODER)")
print("=" * 80)

print("\n" + "-" * 80)
print("Defining Input Shape for Generator LSTM Autoencoder...")
print("-" * 80)

n_features_gen = X_train_seq_gen.shape[2]

input_shape_gen = (sequence_length, n_features_gen)

print(f"Input shape: {input_shape_gen}")
print(f"  Sequence length: {sequence_length} timesteps")
print(f"  Number of features: {n_features_gen}")

print("\n" + "-" * 80)
print("Building Generator LSTM Autoencoder Model...")
print("-" * 80)

input_layer_gen = Input(shape=input_shape_gen, name='input_generator')

encoded_gen = LSTM(64, activation='relu', name='encoder_lstm_gen')(input_layer_gen)

repeated_gen = RepeatVector(sequence_length, name='repeat_vector_gen')(encoded_gen)

decoded_gen = LSTM(64, activation='relu', return_sequences=True, name='decoder_lstm_gen')(repeated_gen)

output_layer_gen = TimeDistributed(Dense(n_features_gen), name='output_generator')(decoded_gen)

model_gen = Model(inputs=input_layer_gen, outputs=output_layer_gen, name='Generator_LSTM_Autoencoder')

print(f"✓ Generator LSTM Autoencoder model built")
print(f"  Architecture: Input -> LSTM(64) -> RepeatVector -> LSTM(64) -> TimeDistributed(Dense)")

print("\n" + "-" * 80)
print("Compiling Model...")
print("-" * 80)

model_gen.compile(optimizer='adam', loss='mae')

print(f"✓ Model compiled")
print(f"  Optimizer: Adam")
print(f"  Loss: Mean Absolute Error (MAE)")

print(f"\nModel Summary:")
print("=" * 80)
model_gen.summary()
print("=" * 80)

print("\n" + "-" * 80)
print("Reusing Early Stopping Callback...")
print("-" * 80)

print(f"✓ Using existing early stopping configuration")
print(f"  Monitor: val_loss")
print(f"  Patience: 5 epochs")
print(f"  Restore best weights: True")

print("\n" + "-" * 80)
print("Training Generator LSTM Autoencoder...")
print(f"{'GPU-ACCELERATED' if tf.config.list_physical_devices('GPU') else 'CPU'} Training")
print("-" * 80)

print(f"\nTraining parameters:")
print(f"  Epochs: 20")
print(f"  Batch size: 64")
print(f"  Training sequences: {X_train_seq_gen.shape[0]:,}")
print(f"  Validation sequences: {X_test_seq_gen.shape[0]:,}")

print(f"\nStarting training...\n")

history_gen = model_gen.fit(
    X_train_seq_gen,
    X_train_seq_gen,  # Autoencoder reconstructs its input
    validation_data=(X_test_seq_gen, X_test_seq_gen),
    epochs=20,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)

print(f"\n✓ Generator LSTM Autoencoder training complete")
print(f"  Final training loss: {history_gen.history['loss'][-1]:.6f}")
print(f"  Final validation loss: {history_gen.history['val_loss'][-1]:.6f}")
print(f"  Epochs trained: {len(history_gen.history['loss'])}")

print("\n" + "-" * 80)
print("Calculating Training Reconstruction Errors...")
print("-" * 80)

X_train_pred_gen = model_gen.predict(X_train_seq_gen, verbose=0)

train_mae_loss_gen = np.mean(np.abs(X_train_pred_gen - X_train_seq_gen), axis=(1, 2))

print(f"✓ Training reconstruction errors calculated")
print(f"  Total sequences: {len(train_mae_loss_gen):,}")
print(f"  Mean error: {np.mean(train_mae_loss_gen):.6f}")
print(f"  Std dev: {np.std(train_mae_loss_gen):.6f}")
print(f"  Min error: {np.min(train_mae_loss_gen):.6f}")
print(f"  Max error: {np.max(train_mae_loss_gen):.6f}")

print("\n" + "-" * 80)
print("Plotting Reconstruction Error Distribution...")
print("-" * 80)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(train_mae_loss_gen, bins=50, alpha=0.7, color='green', edgecolor='black')
plt.axvline(np.mean(train_mae_loss_gen), color='red', linestyle='--', linewidth=2, label='Mean')
plt.axvline(np.median(train_mae_loss_gen), color='orange', linestyle='--', linewidth=2, label='Median')
plt.title('Generator Reconstruction Error - Histogram', fontsize=12, fontweight='bold')
plt.xlabel('Reconstruction Error (MAE)', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
sns.kdeplot(train_mae_loss_gen, fill=True, color='green', alpha=0.6)
plt.axvline(np.mean(train_mae_loss_gen), color='red', linestyle='--', linewidth=2, label='Mean')
plt.axvline(np.median(train_mae_loss_gen), color='orange', linestyle='--', linewidth=2, label='Median')
plt.title('Generator Reconstruction Error - Density', fontsize=12, fontweight='bold')
plt.xlabel('Reconstruction Error (MAE)', fontsize=10)
plt.ylabel('Density', fontsize=10)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(models_path, 'generator_error_distribution.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Error distribution plots saved to: {plot_path}")

print("\n" + "-" * 80)
print("Defining Healthy Threshold...")
print("-" * 80)

HEALTHY_THRESHOLD_GEN = np.percentile(train_mae_loss_gen, 99)

print(f"Healthy Threshold (99th percentile): {HEALTHY_THRESHOLD_GEN:.6f}")
print(f"  99% of training data has error below this threshold")
print(f"  Sequences exceeding this may indicate anomalies")

print("\n" + "-" * 80)
print("Calculating Generator Health Score...")
print("-" * 80)

FAILURE_THRESHOLD_GEN = np.max(train_mae_loss_gen) * 5

print(f"Failure Threshold (5x max training error): {FAILURE_THRESHOLD_GEN:.6f}")

print(f"✓ Reusing calculate_health_score() function")

last_seq_pred_gen = model_gen.predict(X_test_seq_gen[-1:], verbose=0)

last_error_gen = np.mean(np.abs(last_seq_pred_gen - X_test_seq_gen[-1:]))

print(f"\nLatest sequence reconstruction error: {last_error_gen:.6f}")

latest_generator_health_score = calculate_health_score(
    last_error_gen,
    HEALTHY_THRESHOLD_GEN,
    FAILURE_THRESHOLD_GEN
)

print(f"\nGenerator Health Score:")
print(f"  Latest Health Score: {latest_generator_health_score:.2f}%")
print(f"  Reconstruction Error: {last_error_gen:.6f}")
print(f"  Healthy Threshold: {HEALTHY_THRESHOLD_GEN:.6f}")

if last_error_gen <= HEALTHY_THRESHOLD_GEN:
    print(f"  Status: ✓ Within normal range")
else:
    print(f"  Status: ⚠ Above normal range - potential anomaly")

print("\n" + "-" * 80)
print("Defining Severity Level Thresholds...")
print("-" * 80)

GEN_CRITICAL_TEMP = 150        # Critical temperature threshold (°C)
GEN_MAINTENANCE_HEALTH = 30    # Below 30% health = Maintenance Required
GEN_WARNING_HEALTH = 60        # 30-60% health = Warning

print(f"Generator Severity Thresholds:")
print(f"  Critical Health:    < {GEN_MAINTENANCE_HEALTH}%")
print(f"  Maintenance Health: {GEN_MAINTENANCE_HEALTH}% - {GEN_WARNING_HEALTH}%")
print(f"  Warning Health:     {GEN_WARNING_HEALTH}% - 100%")
print(f"  Critical Temperature: > {GEN_CRITICAL_TEMP}°C")

if latest_generator_health_score < GEN_MAINTENANCE_HEALTH:
    gen_severity = "CRITICAL"
    gen_color = "🔴"
elif latest_generator_health_score < GEN_WARNING_HEALTH:
    gen_severity = "MAINTENANCE REQUIRED"
    gen_color = "🟠"
else:
    gen_severity = "NORMAL"
    gen_color = "🟢"

print(f"\nCurrent Generator Status:")
print(f"  {gen_color} {gen_severity} (Health: {latest_generator_health_score:.2f}%)")

print("\n" + "-" * 80)
print("Preparing for Sensor Fusion...")
print("-" * 80)

print(f"✓ Generator health score ready for sensor fusion meta-model")
print(f"  Feature name: 'generator_health_score'")
print(f"  Current value: {latest_generator_health_score:.2f}%")

print("\n" + "-" * 80)
print("Saving Generator LSTM Model...")
print("-" * 80)

generator_model_path = os.path.join(models_path, 'generator_lstm_autoencoder.h5')
model_gen.save(generator_model_path)
print(f"✓ Generator LSTM model saved to: {generator_model_path}")

scaler_gen_path = os.path.join(models_path, 'scaler_generator.joblib')
joblib.dump(scaler_gen, scaler_gen_path)
print(f"✓ Generator scaler saved to: {scaler_gen_path}")

print("\n" + "=" * 80)
print("GENERATOR TEST COMPLETE")
print("=" * 80)
print("Ready for Step 10/13: Test 5 - Sensor Validity (Rule-Based)\n")

print("\n" + "=" * 80)
print("STEP 10/13: TEST 5 - SENSOR AND CONTROL VALIDITY (RULE-BASED)")
print("=" * 80)

print("\n" + "-" * 80)
print("Defining Critical Sensors...")
print("-" * 80)

critical_sensors = [
    'Wind speed (m/s)',
    'Generator RPM (RPM)',
    'Power (kW)',
    'Gear oil temperature (°C)',
    'Stator temperature 1 (°C)'
]

print(f"Critical Sensors Defined ({len(critical_sensors)}):")
for i, sensor in enumerate(critical_sensors, 1):
    print(f"  {i}. {sensor}")

missing_sensors = [sensor for sensor in critical_sensors if sensor not in turbine_df.columns]

if missing_sensors:
    print(f"\n⚠ WARNING: Some critical sensors not found in DataFrame!")
    print(f"  Missing sensors: {missing_sensors}")
    print(f"  These sensors will be excluded from validity checks")
    critical_sensors = [sensor for sensor in critical_sensors if sensor in turbine_df.columns]
    print(f"  Adjusted critical sensors count: {len(critical_sensors)}")
else:
    print(f"\n✓ All critical sensors found in DataFrame")

num_critical_sensors = len(critical_sensors)

if num_critical_sensors == 0:
    print(f"\n⚠ WARNING: No critical sensors available for validity checking!")
    print(f"  Please update the critical_sensors list with valid column names")
    print(f"  Setting num_critical_sensors to 1 to avoid division by zero")
    num_critical_sensors = 1
else:
    print(f"\nTotal critical sensors for monitoring: {num_critical_sensors}")

print("\n" + "-" * 80)
print("Defining Sensor Validity Check Function...")
print("-" * 80)

def check_sensor_validity(current_row):
    """
    Check sensor validity using rule-based logic.
    Detects illogical values, out-of-range measurements, and inconsistencies.
    
    Parameters:
    - current_row: pandas Series representing a single measurement row
    
    Returns:
    - failed_critical_count: Number of critical sensors that failed
    - status_messages: List of failure messages
    """
    failed_sensors_list = []
    status_messages = []

    try:
        if (current_row['Power (kW)'] > 100 and 
            current_row['Generator RPM (RPM)'] < 50):
            status_messages.append('FAIL: Illogical RPM vs Power - High power with low RPM')
            failed_sensors_list.append('Generator RPM (RPM)')
    except (KeyError, TypeError):
        pass  # Skip if columns don't exist or values are invalid

    try:
        if not (-10 < current_row['Gear oil temperature (°C)'] < 110):
            status_messages.append('FAIL: Gearbox Temp Range - Outside normal range (-10°C to 110°C)')
            failed_sensors_list.append('Gear oil temperature (°C)')
    except (KeyError, TypeError):
        pass

    try:
        if not (0 <= current_row['Wind speed (m/s)'] < 50):
            status_messages.append('FAIL: Wind Speed Range - Outside valid range (0-50 m/s)')
            failed_sensors_list.append('Wind speed (m/s)')
    except (KeyError, TypeError):
        pass

    try:
        if not (-10 < current_row['Stator temperature 1 (°C)'] < 180):
            status_messages.append('FAIL: Generator Winding Temp Range - Outside normal range (-10°C to 180°C)')
            failed_sensors_list.append('Stator temperature 1 (°C)')
    except (KeyError, TypeError):
        pass

    try:
        if not (-50 <= current_row['Power (kW)'] <= 5000):
            status_messages.append('FAIL: Active Power Range - Outside expected range (-50 to 5000 kW)')
            failed_sensors_list.append('Power (kW)')
    except (KeyError, TypeError):
        pass

    try:
        if not (0 <= current_row['Generator RPM (RPM)'] <= 2000):
            status_messages.append('FAIL: Generator RPM Range - Outside normal range (0-2000 RPM)')
            failed_sensors_list.append('Generator RPM (RPM)')
    except (KeyError, TypeError):
        pass

    try:
        if (current_row['Power (kW)'] > 500 and 
            current_row['Wind speed (m/s)'] < 3):
            status_messages.append('FAIL: Illogical Wind Speed vs Power - High power with low wind')
            failed_sensors_list.append('Wind speed (m/s)')
    except (KeyError, TypeError):
        pass

    try:
        zero_checks = {
            'Wind speed (m/s)': 'Wind Speed',
            'Generator RPM (RPM)': 'Generator RPM',
        }
        for col, name in zero_checks.items():
            if col in current_row and current_row[col] == 0:
                status_messages.append(f'WARNING: {name} is zero - possible sensor stuck')
    except (KeyError, TypeError):
        pass

    failed_critical_count = sum(1 for sensor in set(failed_sensors_list) if sensor in critical_sensors)

    return failed_critical_count, status_messages

print(f"✓ Sensor validity check function defined")
print(f"  Function: check_sensor_validity(current_row)")
print(f"  Rules implemented:")
print(f"    1. Illogical Power vs RPM")
print(f"    2. Gearbox Temperature Range (-10°C to 110°C)")
print(f"    3. Wind Speed Range (0-50 m/s)")
print(f"    4. Generator Winding Temperature Range (-10°C to 180°C)")
print(f"    5. Active Power Range (-50 to 5000 kW)")
print(f"    6. Generator RPM Range (0-2000 RPM)")
print(f"    7. Illogical Wind Speed vs Power")
print(f"    8. Zero Values Check (stuck sensors)")

print("\n" + "-" * 80)
print("Demonstrating Sensor Validity Check on Latest Data...")
print("-" * 80)

latest_row = test_df.iloc[-1]

print(f"Analyzing latest measurement (most recent test data point)...")
print(f"Timestamp: {latest_row.name if hasattr(latest_row, 'name') else 'N/A'}")

failed_count, messages = check_sensor_validity(latest_row)

print(f"\nValidity Check Results:")
print(f"  Failed critical sensors: {failed_count} out of {num_critical_sensors}")
print(f"  Total issues detected: {len(messages)}")

if messages:
    print(f"\n  Issue Details:")
    for i, msg in enumerate(messages, 1):
        print(f"    {i}. {msg}")
else:
    print(f"\n  ✓ No sensor validity issues detected")

print("\n" + "-" * 80)
print("Calculating Data Integrity Health Score...")
print("-" * 80)

if num_critical_sensors > 0:
    latest_data_integrity_score = ((num_critical_sensors - failed_count) / num_critical_sensors) * 100
else:
    latest_data_integrity_score = 100  # Default to perfect if no sensors defined

print(f"Data Integrity Score:")
print(f"  Total critical sensors: {num_critical_sensors}")
print(f"  Failed sensors: {failed_count}")
print(f"  Healthy sensors: {num_critical_sensors - failed_count}")
print(f"\n  Latest Data Integrity Score: {latest_data_integrity_score:.2f}%")

if latest_data_integrity_score == 100:
    print(f"  Status: ✓ All sensors operating normally")
elif latest_data_integrity_score >= 80:
    print(f"  Status: ⚠ Minor sensor issues detected")
else:
    print(f"  Status: 🔴 Major sensor integrity issues")

print("\n" + "-" * 80)
print("Defining Severity Level Thresholds...")
print("-" * 80)

critical_sensors_for_shutdown = [
    'Wind speed (m/s)',
    'Generator RPM (RPM)'
]

print(f"Critical Sensors for Shutdown:")
for sensor in critical_sensors_for_shutdown:
    print(f"  - {sensor}")

INTEGRITY_MAINTENANCE = 90    # Below 90% = Maintenance Required
INTEGRITY_WARNING = 98        # 90-98% = Warning

print(f"\nSeverity Thresholds:")
print(f"  Maintenance: < {INTEGRITY_MAINTENANCE}%")
print(f"  Warning:     {INTEGRITY_MAINTENANCE}% - {INTEGRITY_WARNING}%")
print(f"  Normal:      ≥ {INTEGRITY_WARNING}%")

shutdown_required = False
try:
    failed_sensors_in_latest = []
    _, temp_messages = check_sensor_validity(latest_row)
    for msg in temp_messages:
        for sensor in critical_sensors_for_shutdown:
            if sensor in msg:
                failed_sensors_in_latest.append(sensor)
                shutdown_required = True
except:
    pass

if shutdown_required:
    integrity_severity = "CRITICAL - SHUTDOWN REQUIRED"
    integrity_color = "🔴🔴"
elif latest_data_integrity_score < INTEGRITY_MAINTENANCE:
    integrity_severity = "MAINTENANCE REQUIRED"
    integrity_color = "🟠"
elif latest_data_integrity_score < INTEGRITY_WARNING:
    integrity_severity = "WARNING"
    integrity_color = "🟡"
else:
    integrity_severity = "NORMAL"
    integrity_color = "🟢"

print(f"\nCurrent Sensor Validity Status:")
print(f"  {integrity_color} {integrity_severity} (Score: {latest_data_integrity_score:.2f}%)")

if shutdown_required:
    print(f"  ⚠⚠ CRITICAL: Shutdown sensors have failed!")
    print(f"  Failed shutdown-critical sensors: {list(set(failed_sensors_in_latest))}")

print("\n" + "-" * 80)
print("Preparing for Sensor Fusion...")
print("-" * 80)

print(f"✓ Data integrity score ready for sensor fusion meta-model")
print(f"  Feature name: 'data_integrity_score'")
print(f"  Current value: {latest_data_integrity_score:.2f}%")

print("\n" + "=" * 80)
print("SENSOR VALIDITY TEST COMPLETE")
print("=" * 80)
print("Ready for Step 11/13: Consolidate Health Scores\n")

print("\n" + "=" * 80)
print("STEP 11/13: ALERTING SYSTEM LOGIC SIMULATION")
print("=" * 80)

print("\n" + "-" * 80)
print("Initializing Failure State Tracking...")
print("-" * 80)

failure_state = {
    'gearbox_consecutive_fails': 0,
    'efficiency_consecutive_fails': 0,
    'generator_consecutive_fails': 0,
    'alignment_consecutive_fails': 0,
    'integrity_consecutive_fails': 0
}

print(f"✓ Failure state initialized")
print(f"  Tracking consecutive failures for 5 diagnostic tests:")
for key, value in failure_state.items():
    print(f"    {key}: {value}")

print("\n" + "-" * 80)
print("Defining Alerting Function...")
print("-" * 80)

def update_failure_state_and_alert(current_scores, current_raw_data, failure_state):
    """
    Update failure state and trigger alerts based on health scores and raw data.
    
    Implements 3-level alerting system:
    - Level 3 (Critical): Immediate alerts based on rule violations
    - Level 2 (Maintenance): Alerts after consecutive failures below maintenance threshold
    - Level 1 (Warning): Alerts when scores drop to warning level
    
    Parameters:
    - current_scores: Dictionary of current health scores
    - current_raw_data: pandas Series of current raw sensor data
    - failure_state: Dictionary tracking consecutive failure counts
    
    Returns:
    - Updated failure_state dictionary
    """
    alert_triggered = False

    print(f"\n" + "=" * 80)
    print(f"ALERTING SYSTEM CHECK")
    print(f"=" * 80)

    print(f"\nLevel 3: Critical Rule Checks...")
    print(f"-" * 80)

    try:
        if current_raw_data['Gear oil temperature (°C)'] > GB_CRITICAL_TEMP:
            print(f"🔴🔴 CRITICAL ALERT: Gearbox Oil Temperature Exceeded!")
            print(f"   Current: {current_raw_data['Gear oil temperature (°C)']:.2f}°C")
            print(f"   Threshold: {GB_CRITICAL_TEMP}°C")
            print(f"   ACTION: Immediate turbine shutdown required!")
            alert_triggered = True
    except (KeyError, TypeError):
        pass

    try:
        if current_raw_data['Stator temperature 1 (°C)'] > GEN_CRITICAL_TEMP:
            print(f"🔴🔴 CRITICAL ALERT: Generator Winding Temperature Exceeded!")
            print(f"   Current: {current_raw_data['Stator temperature 1 (°C)']:.2f}°C")
            print(f"   Threshold: {GEN_CRITICAL_TEMP}°C")
            print(f"   ACTION: Immediate turbine shutdown required!")
            alert_triggered = True
    except (KeyError, TypeError):
        pass

    try:
        if current_scores.get('efficiency', 100) < EFFICIENCY_CRITICAL:
            print(f"🔴🔴 CRITICAL ALERT: Blade/Rotor Efficiency Below Critical Level!")
            print(f"   Current: {current_scores['efficiency']:.2f}%")
            print(f"   Threshold: {EFFICIENCY_CRITICAL}%")
            print(f"   ACTION: Immediate inspection required!")
            alert_triggered = True
    except (KeyError, TypeError):
        pass

    try:
        sensor_failed_count, sensor_messages = check_sensor_validity(current_raw_data)

        for msg in sensor_messages:
            for critical_sensor in critical_sensors_for_shutdown:
                if critical_sensor in msg:
                    print(f"🔴🔴 CRITICAL ALERT: Shutdown-Critical Sensor Failed!")
                    print(f"   Sensor: {critical_sensor}")
                    print(f"   Issue: {msg}")
                    print(f"   ACTION: Immediate turbine shutdown required!")
                    alert_triggered = True
                    break
    except Exception as e:
        pass

    if alert_triggered:
        print(f"\n🔴 CRITICAL ALERT TRIGGERED - Immediate action required!")
        print(f"   Returning current failure state without further checks")
        return failure_state
    else:
        print(f"✓ No critical alerts triggered")

    print(f"\nLevel 1/2: Maintenance & Warning Checks...")
    print(f"-" * 80)

    maint_thresholds = {
        'gearbox': GB_MAINTENANCE_HEALTH,
        'efficiency': EFFICIENCY_MAINTENANCE,
        'generator': GEN_MAINTENANCE_HEALTH,
        'alignment': ALIGNMENT_MAINTENANCE,
        'integrity': INTEGRITY_MAINTENANCE
    }

    warn_thresholds = {
        'gearbox': GB_WARNING_HEALTH,
        'efficiency': EFFICIENCY_WARNING,
        'generator': GEN_WARNING_HEALTH,
        'alignment': ALIGNMENT_WARNING,
        'integrity': INTEGRITY_WARNING
    }

    max_fails = 5

    for test_name, score in current_scores.items():
        counter_key = f"{test_name}_consecutive_fails"

        if counter_key not in failure_state:
            failure_state[counter_key] = 0

        maint_thresh = maint_thresholds.get(test_name, 30)
        warn_thresh = warn_thresholds.get(test_name, 60)

        if score < maint_thresh:
            failure_state[counter_key] += 1

            if failure_state[counter_key] >= max_fails:
                print(f"🟠 MAINTENANCE ALERT: {test_name.upper()}")
                print(f"   Score: {score:.2f}% (Threshold: {maint_thresh}%)")
                print(f"   Consecutive failures: {failure_state[counter_key]}/{max_fails}")
                print(f"   ACTION: Schedule maintenance inspection")
            else:
                print(f"⚠ WARNING: {test_name.upper()} - Below maintenance threshold")
                print(f"   Score: {score:.2f}% (Threshold: {maint_thresh}%)")
                print(f"   Consecutive failures: {failure_state[counter_key]}/{max_fails}")

        elif score < warn_thresh:
            print(f"🟡 WARNING: {test_name.upper()} - Below warning threshold")
            print(f"   Score: {score:.2f}% (Threshold: {warn_thresh}%)")
            failure_state[counter_key] = 0

        else:
            if failure_state[counter_key] > 0:
                print(f"✓ {test_name.upper()}: Recovered to normal")
                print(f"   Score: {score:.2f}%")
            failure_state[counter_key] = 0

    print(f"\n" + "=" * 80)
    return failure_state

print(f"✓ Alerting function defined")
print(f"  Function: update_failure_state_and_alert()")
print(f"  Alert Levels:")
print(f"    Level 3: Critical - Immediate shutdown triggers")
print(f"    Level 2: Maintenance - Consecutive failures (5x)")
print(f"    Level 1: Warning - Score below warning threshold")

print("\n" + "-" * 80)
print("Simulating Alerting System with Latest Data...")
print("-" * 80)

latest_scores = {
    'gearbox': latest_gearbox_health_score,
    'efficiency': latest_efficiency_score,
    'generator': latest_generator_health_score,
    'alignment': latest_alignment_score,
    'integrity': latest_data_integrity_score
}

print(f"\nCurrent Health Scores:")
print(f"-" * 80)
for test_name, score in latest_scores.items():
    if score >= 95:
        emoji = "🟢"
    elif score >= 60:
        emoji = "🟡"
    elif score >= 30:
        emoji = "🟠"
    else:
        emoji = "🔴"

    print(f"  {emoji} {test_name.upper()}: {score:.2f}%")

latest_raw_data = test_df.iloc[-1]

print(f"\nLatest Raw Data Sample:")
print(f"-" * 80)
try:
    print(f"  Gearbox Oil Temp: {latest_raw_data.get('Gear oil temperature (°C)', 'N/A')}")
    print(f"  Generator Winding Temp: {latest_raw_data.get('Stator temperature 1 (°C)', 'N/A')}")
    print(f"  Active Power: {latest_raw_data.get('Power (kW)', 'N/A')}")
    print(f"  Wind Speed: {latest_raw_data.get('Wind speed (m/s)', 'N/A')}")
except:
    print(f"  (Raw data display unavailable)")

print(f"\nExecuting Alerting System...")
updated_state = update_failure_state_and_alert(latest_scores, latest_raw_data, failure_state)

print(f"\nUpdated Failure State:")
print(f"-" * 80)
for key, value in updated_state.items():
    print(f"  {key}: {value}")

print("\n" + "=" * 80)
print("ALERTING SYSTEM SIMULATION COMPLETE")
print("=" * 80)
print("Ready for Step 12/13: Sensor Fusion Meta-Model\n")

print("\n" + "=" * 80)
print("STEP 12/13: SENSOR FUSION META-MODEL (SIMPLIFIED)")
print("=" * 80)

print("\n" + "-" * 80)
print("Generating Historical Health Scores for Meta-Model...")
print("-" * 80)

print(f"Calculating gearbox health scores for all test sequences...")
X_test_pred_gb_all = model_gb.predict(X_test_seq_gb, verbose=0)
test_errors_gb = np.mean(np.abs(X_test_pred_gb_all - X_test_seq_gb), axis=(1, 2))
all_gb_health = np.array([
    calculate_health_score(err, HEALTHY_THRESHOLD_GB, FAILURE_THRESHOLD_GB) 
    for err in test_errors_gb
])
print(f"  ✓ Gearbox health scores: {len(all_gb_health):,} samples")
print(f"    Mean: {np.mean(all_gb_health):.2f}%, Std: {np.std(all_gb_health):.2f}%")

print(f"\nCalculating generator health scores for all test sequences...")
X_test_pred_gen_all = model_gen.predict(X_test_seq_gen, verbose=0)
test_errors_gen = np.mean(np.abs(X_test_pred_gen_all - X_test_seq_gen), axis=(1, 2))
all_gen_health = np.array([
    calculate_health_score(err, HEALTHY_THRESHOLD_GEN, FAILURE_THRESHOLD_GEN) 
    for err in test_errors_gen
])
print(f"  ✓ Generator health scores: {len(all_gen_health):,} samples")
print(f"    Mean: {np.mean(all_gen_health):.2f}%, Std: {np.std(all_gen_health):.2f}%")

print(f"\nUsing blade/rotor efficiency scores from power curve...")
all_efficiency = calculate_efficiency_score(y_test_pc.values, y_pred_pc_test)
print(f"  ✓ Efficiency scores: {len(all_efficiency):,} samples")
print(f"    Mean: {np.mean(all_efficiency):.2f}%, Std: {np.std(all_efficiency):.2f}%")

print(f"\nGenerating alignment scores...")
avg_alignment_score = calculate_alignment_score(test_df['Yaw_Error'])
all_alignment = np.array([avg_alignment_score] * len(y_test_pc))
print(f"  ✓ Alignment scores: {len(all_alignment):,} samples")
print(f"    Mean: {np.mean(all_alignment):.2f}%, Std: {np.std(all_alignment):.2f}%")

print(f"\nSimulating data integrity scores...")
np.random.seed(42)  # For reproducibility
all_integrity = np.random.randint(95, 101, size=len(y_test_pc))
print(f"  ✓ Integrity scores: {len(all_integrity):,} samples")
print(f"    Mean: {np.mean(all_integrity):.2f}%, Std: {np.std(all_integrity):.2f}%")

min_len = min(len(all_gb_health), len(all_gen_health), len(all_efficiency))
print(f"\nAligning all score arrays to minimum length: {min_len:,} samples")

all_gb_health = all_gb_health[:min_len]
all_gen_health = all_gen_health[:min_len]
all_efficiency = all_efficiency[:min_len]
all_alignment = all_alignment[:min_len]
all_integrity = all_integrity[:min_len]

print(f"✓ All arrays aligned")

X_meta = np.vstack([
    all_gb_health,
    all_efficiency,
    all_gen_health,
    all_alignment,
    all_integrity
]).T

print(f"\nMeta-model feature matrix created:")
print(f"  Shape: {X_meta.shape} (samples, features)")
print(f"  Features: [gearbox_health, blade_efficiency, generator_health, alignment, integrity]")

print("\n" + "-" * 80)
print("Creating Labels for Meta-Model Training...")
print("-" * 80)

meta_label_threshold = 40

print(f"Label threshold: {meta_label_threshold}%")
print(f"  Logic: If gearbox_health OR generator_health < {meta_label_threshold}% → Issue (1)")
print(f"         Otherwise → Healthy (0)")

y_meta = np.where(
    (X_meta[:, 0] < meta_label_threshold) | (X_meta[:, 2] < meta_label_threshold),
    1,  # Potential Issue
    0   # Healthy
)

label_counts = np.bincount(y_meta)
print(f"\nLabel distribution:")
print(f"  Healthy (0): {label_counts[0]:,} samples ({(label_counts[0]/len(y_meta)*100):.1f}%)")
if len(label_counts) > 1:
    print(f"  Potential Issue (1): {label_counts[1]:,} samples ({(label_counts[1]/len(y_meta)*100):.1f}%)")
else:
    print(f"  Potential Issue (1): 0 samples (0.0%)")

if len(label_counts) == 1 or label_counts[1] == 0:
    print(f"\n⚠ WARNING: No 'Potential Issue' samples detected!")
    print(f"  This may indicate all systems are healthy in test data")
    print(f"  Meta-model will still train but may not be effective for issue detection")

print("\n" + "-" * 80)
print("Training Sensor Fusion Meta-Model...")
print("-" * 80)

X_meta_train, X_meta_test, y_meta_train, y_meta_test = train_test_split(
    X_meta,
    y_meta,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print(f"Meta-model data split:")
print(f"  Training samples: {X_meta_train.shape[0]:,}")
print(f"  Testing samples: {X_meta_test.shape[0]:,}")

meta_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)

print(f"\nMeta-model configuration:")
print(f"  Algorithm: Decision Tree Classifier")
print(f"  Max depth: 5")
print(f"  Random state: 42")

print(f"\nTraining meta-model...")
meta_model.fit(X_meta_train, y_meta_train)
print(f"✓ Sensor fusion meta-model training complete")

print("\n" + "-" * 80)
print("Evaluating Meta-Model Performance...")
print("-" * 80)

y_meta_pred = meta_model.predict(X_meta_test)

accuracy = np.mean(y_meta_pred == y_meta_test)
print(f"Meta-model Accuracy: {accuracy:.2%}")

print(f"\nClassification Report:")
print("=" * 80)
# Check if we have both classes in test set
unique_test_labels = np.unique(y_meta_test)
if len(unique_test_labels) > 1:
    target_names = ['Healthy', 'Potential Issue']
else:
    target_names = ['Healthy'] if unique_test_labels[0] == 0 else ['Potential Issue']

print(classification_report(
    y_meta_test,
    y_meta_pred,
    target_names=target_names,
    zero_division=0
))

print(f"\nConfusion Matrix:")
print("=" * 80)
cm = confusion_matrix(y_meta_test, y_meta_pred)
if cm.shape[0] == 2:
    print(f"                  Predicted")
    print(f"                  Healthy  Issue")
    print(f"Actual Healthy    {cm[0][0]:7d}  {cm[0][1]:5d}")
    print(f"       Issue      {cm[1][0]:7d}  {cm[1][1]:5d}")
else:
    print(f"Only one class present in test set:")
    print(f"  All samples classified as: {'Healthy' if unique_test_labels[0] == 0 else 'Potential Issue'}")
    print(f"  Total samples: {cm[0][0]:,}")
print("=" * 80)

print(f"\nFeature Importance:")
print(f"-" * 80)
feature_names = ['Gearbox Health', 'Blade Efficiency', 'Generator Health', 'Alignment', 'Integrity']
importances = meta_model.feature_importances_
for name, importance in zip(feature_names, importances):
    print(f"  {name:20s}: {importance:.4f}")

print("\n" + "-" * 80)
print("Predicting Overall Turbine Status...")
print("-" * 80)

latest_scores_vector = np.array([[
    latest_gearbox_health_score,
    latest_efficiency_score,
    latest_generator_health_score,
    latest_alignment_score,
    latest_data_integrity_score
]])

print(f"Latest health scores input to meta-model:")
for name, score in zip(feature_names, latest_scores_vector[0]):
    print(f"  {name:20s}: {score:.2f}%")

overall_status_code = meta_model.predict(latest_scores_vector)[0]

overall_status_proba = meta_model.predict_proba(latest_scores_vector)[0]

status_map = {
    0: 'Healthy',
    1: 'Potential Issue'
}
overall_status = status_map[overall_status_code]

print(f"\n" + "=" * 80)
print(f"OVERALL TURBINE STATUS")
print(f"=" * 80)

if overall_status_code == 0:
    print(f"  🟢 STATUS: {overall_status}")
    print(f"  Confidence: {overall_status_proba[0]:.1%}")
    print(f"  Recommendation: Continue normal operation with routine monitoring")
else:
    print(f"  🔴 STATUS: {overall_status}")
    print(f"  Confidence: {overall_status_proba[1]:.1%}")
    print(f"  Recommendation: Schedule detailed inspection and maintenance")

print("\n" + "-" * 80)
print("Saving Meta-Model...")
print("-" * 80)

meta_model_path = os.path.join(models_path, 'meta_model_sensor_fusion.joblib')
joblib.dump(meta_model, meta_model_path)
print(f"✓ Meta-model saved to: {meta_model_path}")

print("\n" + "=" * 80)
print("SENSOR FUSION META-MODEL COMPLETE")
print("=" * 80)
print("Ready for Step 13/13: Summary Dashboard & Final Report\n")

print("\n" + "=" * 80)
print("STEP 13/13: MODEL SAVING & FINAL NOTES")
print("=" * 80)

print("\n" + "-" * 80)
print("Defining Model Save Paths...")
print("-" * 80)

model_gb_path = os.path.join(models_path, 'gearbox_autoencoder.h5')
model_gen_path = os.path.join(models_path, 'generator_autoencoder.h5')
meta_model_path = os.path.join(models_path, 'meta_model.joblib')
scaler_gb_path = os.path.join(models_path, 'scaler_gb.joblib')
scaler_gen_path = os.path.join(models_path, 'scaler_gen.joblib')

print(f"Model save directory: {models_path}")
print(f"\nModel file paths:")
print(f"  1. Power Curve Model:    {power_curve_model_path}")
print(f"  2. Gearbox LSTM:         {model_gb_path}")
print(f"  3. Generator LSTM:       {model_gen_path}")
print(f"  4. Meta-Model (Fusion):  {meta_model_path}")
print(f"  5. Gearbox Scaler:       {scaler_gb_path}")
print(f"  6. Generator Scaler:     {scaler_gen_path}")

print("\n" + "-" * 80)
print("Saving All Models and Scalers...")
print("-" * 80)

print(f"\nSaving Gearbox LSTM Autoencoder...")
model_gb.save(model_gb_path)
print(f"✓ Gearbox LSTM model saved to: {model_gb_path}")

print(f"\nSaving Generator LSTM Autoencoder...")
model_gen.save(model_gen_path)
print(f"✓ Generator LSTM model saved to: {model_gen_path}")

print(f"\nSaving Meta-Model (Sensor Fusion)...")
joblib.dump(meta_model, meta_model_path)
print(f"✓ Meta-model saved to: {meta_model_path}")

print(f"\nSaving Gearbox Scaler...")
joblib.dump(scaler_gb, scaler_gb_path)
print(f"✓ Gearbox scaler saved to: {scaler_gb_path}")

print(f"\nSaving Generator Scaler...")
joblib.dump(scaler_gen, scaler_gen_path)
print(f"✓ Generator scaler saved to: {scaler_gen_path}")

print(f"\n✓ All models and scalers successfully saved!")

print("\n" + "=" * 80)
print("FINAL NOTES AND RECOMMENDATIONS")
print("=" * 80)

print(f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║           WIND TURBINE PREDICTIVE MAINTENANCE SYSTEM - COMPLETE            ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

SUMMARY OF DIAGNOSTIC TESTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Test 1: GEARBOX FAILURES (LSTM Autoencoder)
   - GPU-Accelerated LSTM Autoencoder
   - Features: Gearbox oil temp, bearing temp, RPM
   - Health Score: {latest_gearbox_health_score:.2f}%
   - Status: {gb_severity}

✓ Test 2: BLADE/ROTOR HEALTH (Power Curve Analysis)
   - Random Forest Regressor
   - Features: Wind speed → Active power
   - Efficiency Score: {latest_efficiency_score:.2f}%
   - Status: {severity_level}

✓ Test 3: GENERATOR & ELECTRICAL (LSTM Autoencoder)
   - GPU-Accelerated LSTM Autoencoder
   - Features: Winding temps, bearing temp, power
   - Health Score: {latest_generator_health_score:.2f}%
   - Status: {gen_severity}

✓ Test 4: YAW/PITCH ALIGNMENT (Rule-Based)
   - Circular difference calculation
   - Yaw error threshold: 15°
   - Alignment Score: {latest_alignment_score:.2f}%
   - Status: {alignment_severity}

✓ Test 5: SENSOR VALIDITY (Rule-Based)
   - 8 validation rules
   - 5 critical sensors monitored
   - Integrity Score: {latest_data_integrity_score:.2f}%
   - Status: {integrity_severity}

✓ SENSOR FUSION META-MODEL (Decision Tree)
   - Combines all 5 diagnostic test scores
   - Binary classification: Healthy / Potential Issue
   - Overall Status: {overall_status}
   - Confidence: {overall_status_proba[overall_status_code]:.1%}

MODELS SAVED:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. {os.path.basename(power_curve_model_path)}
  2. {os.path.basename(model_gb_path)}
  3. {os.path.basename(model_gen_path)}
  4. {os.path.basename(meta_model_path)}
  5. {os.path.basename(scaler_gb_path)}
  6. {os.path.basename(scaler_gen_path)}

ALERTING SYSTEM:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - 3-Level Alert System (Critical / Maintenance / Warning)
  - Consecutive failure tracking
  - Rule-based critical temperature checks
  - Shutdown-critical sensor monitoring

NEXT STEPS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Replace ALL placeholder column names with actual Kelmarsh dataset columns
  2. Verify GPU setup and CUDA/cuDNN installation
  3. Adjust thresholds based on turbine specifications
  4. Test with full Kelmarsh SCADA dataset
  5. Validate predictions against maintenance logs
  6. Deploy with real-time monitoring dashboard
  7. Set up automated alerting (email/SMS)
  8. Schedule regular model retraining

IMPORTANT REMINDERS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⚠ This system is for MONITORING ONLY - not autonomous control
  ⚠ Always verify predictions with qualified technicians
  ⚠ Follow all manufacturer safety protocols
  ⚠ Maintain backup monitoring systems
  ⚠ Review and update thresholds regularly

╔════════════════════════════════════════════════════════════════════════════╗
║  Full script generation complete. Review placeholders and execute steps.   ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

print("\n" + "=" * 80)
print("SCRIPT EXECUTION COMPLETE - ALL 13 STEPS FINISHED")
print("=" * 80)
print(f"\nGenerated: {os.path.abspath(__file__)}")
print(f"Models saved to: {os.path.abspath(models_path)}")
print(f"\n✓ Ready for deployment after placeholder replacement and testing\n")

