import glob
import os
import pickle
import re  # Needed for natural sort
from datetime import timedelta  # For time calculations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
# Import for undersampling
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import \
    auc  # for calculating AUC from precision-recall curve
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from tqdm import tqdm  # for progress bar

# --- IMPORTANT: Adjust the import path for get_model_class if needed ---
try:
    from Final_Clean_v1 import get_model_class
except ImportError:
    print("WARNING: Could not import get_model_class from Final_Clean_v1.py.")
    print("Please ensure Final_Clean_v1.py is in the same directory or accessible.")
    print("Using a dummy model class for demonstration. Actual evaluation requires your model class.")

    # Dummy Model for demonstration if import fails
    class DummyModel(nn.Module):
        def __init__(self, input_channels, seq_len, conv_filters, lstm_units):
            super().__init__()
            # Simplified dummy model to match CNN-LSTM structure
            self.conv_block = nn.Sequential(
                nn.Conv1d(input_channels, conv_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2), # Assume some pooling from your CNN part
                nn.Conv1d(conv_filters[0], conv_filters[1], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2)
            )
            # Adjust LSTM input size based on conv output
            # This is a heuristic. You need to know the exact output shape of your CNN.
            # Assuming two MaxPool1d(2) layers, seq_len becomes seq_len // 4
            self.lstm_input_size = conv_filters[1] * (seq_len // 4) 
            self.lstm = nn.LSTM(self.lstm_input_size, lstm_units, batch_first=True)
            self.fc = nn.Linear(lstm_units, 1)

        def forward(self, x):
            # x shape: (batch_size, channels, sequence_length)
            x = self.conv_block(x)
            # Reshape for LSTM: (batch_size, sequence_length, features)
            x = x.view(x.size(0), 1, -1) # Flatten conv output into a single time step for LSTM
            lstm_out, _ = self.lstm(x)
            out = self.fc(lstm_out[:, -1, :])
            return out
    def get_model_class(model_type):
        if model_type == 'CNN-LSTM':
            return DummyModel
        else:
            raise ValueError(f"Unknown model type: {model_type}")

# --- Configuration (MUST match your training setup and file paths) ---
config = {
    'MODEL_TYPE': 'CNN-LSTM',
    'BASE_SENSORS': ['HR', 'EDA', 'TEMP', 'ACC'],
    'SEGMENT_SECONDS': 30, # Duration of each segment in seconds
    'TARGET_SAMPLING_HZ': 1, # Sampling rate after resampling (as per notebook)
    'PATIENT_ID': 'MSEL_01110', # The patient you're evaluating
    
    # --- YOUR PATHS HERE ---
    # Adjust this path to your saved personalized model bundle for MSEL_01110
    'INFERENCE_BUNDLE_PATH': 'final_v1/training_data/20250609_061959_allModels_persona_0.5/CNN-LSTM/HR_EDA_TEMP_ACC/hp_combo_1/personalized/MSEL_01110/patient_MSEL_01110_inference_bundle.pkl', 
    
    # --- IMPORTANT: This path should point to the ROOT directory of your 20% demo data
    # that mimics the F:\data_9 structure.
    # E.g., if F:\data_9\MSEL_01110 contains your original data,
    # then your demo data might be in F:\Demo_20_Percent_Data\MSEL_01110
    'DEMO_DATA_ROOT_DIR': 'F:\\Demo_Patient\\MSEL_01110', # <--- **ADJUST THIS PATH**
    
    'DEFAULT_ALERT_THRESHOLD': 0.5, # The threshold you used in the Gradio demo

    # --- Preprocessing parameters from your notebook (Crucial for consistency) ---
    'PRE_ICTAL_WINDOW_MINUTES': 30,
    'PRE_ICTAL_GAP_MINUTES': 5,
    'PRE_ICTAL_EXCLUSION_BUFFER_MINUTES': 180,
    'POST_ICTAL_BUFFER_MINUTES': 180,
    'MIN_SEIZURE_DURATION_SEC': 5, # Common filter for annotations
    'MIN_INTER_SEIZURE_INTERVAL_HOURS': 6, # Common filter for annotations
    'SENSOR_MAPPING': {"HR": "HR", "EDA": "EDA", "TEMP": "TEMP", "ACC": "Acc Mag"}, # From data emitter
    
    # --- NEW: Flag to control undersampling for diagnostic purposes ---
    'APPLY_UNDERSAMPLING_TO_DEMO_TEST': False # Set to True for diagnosis, False for realistic evaluation
}

# Calculated parameters (derived from config)
config['SEQ_LEN'] = config['SEGMENT_SECONDS'] * config['TARGET_SAMPLING_HZ'] # e.g., 30 * 1 = 30 samples
config['NUM_FEATURES'] = len(config['BASE_SENSORS']) # e.g., 4 features


print(f"--- Offline Evaluation for Patient {config['PATIENT_ID']} ---")
print(f"Model Type: {config['MODEL_TYPE']}")
print(f"Sensor Combination: {config['BASE_SENSORS']}")
print(f"Segment Length: {config['SEGMENT_SECONDS']}s ({config['SEQ_LEN']} samples)")
print(f"Default Alert Threshold: {config['DEFAULT_ALERT_THRESHOLD']:.2f}")
print(f"Demo Data Root Dir: {config['DEMO_DATA_ROOT_DIR']}")
print(f"Apply Undersampling to Demo Test Data (Diagnostic): {config['APPLY_UNDERSAMPLING_TO_DEMO_TEST']}")


# --- 1. Load Model and Scaler ---
print(f"\n1. Loading personalized model and scaler from: {config['INFERENCE_BUNDLE_PATH']}")
try:
    bundle = pickle.load(open(config['INFERENCE_BUNDLE_PATH'], 'rb'))
    model_hyperparams = bundle['hyperparameters']['model_hyperparameters']
    model_state_dict = bundle['model_state_dict']
    model_type = bundle['model_type']
    scaler = bundle['scaler']

    ModelClass = get_model_class(model_type)
    model = ModelClass(input_channels=config['NUM_FEATURES'], seq_len=config['SEQ_LEN'], **model_hyperparams)
    model.load_state_dict(model_state_dict)
    model.eval() # Set model to evaluation mode
    print("   Model and scaler loaded successfully.")
except FileNotFoundError:
    print(f"FATAL: Model bundle not found at {config['INFERENCE_BUNDLE_PATH']}.")
    print("Please ensure 'INFERENCE_BUNDLE_PATH' is correct.")
    exit()
except Exception as e:
    print(f"FATAL: Error loading bundle: {e}")
    print("Please check the bundle file and your model class definition.")
    exit()


# --- Replicate Preprocessing Functions from your Seizure_Data_Preprocessing_with_gap_minutes.ipynb ---

def natural_sort_key(s):
    """Sorts strings containing numbers in a human-friendly way."""
    match = re.search(r'(\d+)\.parquet$', s)
    return int(match.group(1)) if match else s

def load_raw_sensor_data_for_patient(patient_dir, sensors, sensor_mapping):
    """Loads raw sensor data from parquet files for a given patient."""
    patient_id = os.path.basename(patient_dir)
    sensor_data_dict = {}
    found_sensors = []

    for sensor_name in sensors:
        attr_folder = sensor_name # e.g., 'HR'
        attr_name_part = sensor_mapping[sensor_name] # e.g., 'HR' or 'Acc Mag'

        glob_pattern = os.path.join(
            patient_dir, f"Empatica-{attr_folder}",
            f"{patient_id}_Empatica-{attr_folder}_{attr_name_part}_segment_*.parquet",
        )
        files = sorted(glob.glob(glob_pattern), key=natural_sort_key)

        if not files:
            print(f"   WARNING: No files found for {sensor_name} at {glob_pattern}")
            continue

        all_sensor_df = []
        for file_path in files:
            try:
                chunk_df = pd.read_parquet(file_path)
                if "time" in chunk_df.columns and "data" in chunk_df.columns:
                    # Apply the same scaling as the data emitter
                    chunk_df['data'] = chunk_df['data'] / 1_000_000_000
                    chunk_df["timestamp"] = pd.to_datetime(chunk_df["time"] / 1000, unit="s", utc=True)
                    # Rename 'data' column to sensor_name as done in the notebook
                    chunk_df = chunk_df.rename(columns={"data": sensor_name})
                    all_sensor_df.append(chunk_df[["timestamp", sensor_name]])
            except Exception as e:
                print(f"   Error loading {sensor_name} file {file_path}: {e}")
                continue

        if all_sensor_df:
            sensor_data_dict[sensor_name] = pd.concat(all_sensor_df).set_index("timestamp").sort_index()
            found_sensors.append(sensor_name)
        else:
            print(f"   No valid data loaded for sensor: {sensor_name}")
    return sensor_data_dict, found_sensors

def preprocess_raw_sensor_data(sensor_data_dict, patient_id):
    """Handles zero values, sets timestamp as index, and interpolates missing values."""
    processed_sensor_data_dict = {}
    for sensor_name, df in sensor_data_dict.items():
        # Replace 0 with NaN
        df_processed = df.replace(0, np.nan)
        # Ensure timestamp is datetime and set as index (already done in load_raw_sensor_data_for_patient)
        # Interpolate missing values
        df_processed = df_processed.interpolate(method='time', limit_direction='both')
        processed_sensor_data_dict[sensor_name] = df_processed
    return processed_sensor_data_dict

def load_and_preprocess_seizure_annotations(patient_dir, min_duration_sec, min_inter_seizure_interval_hours):
    """Loads and preprocesses seizure annotations."""
    annotations_path = os.path.join(patient_dir, os.path.basename(patient_dir) + "_SeerAnnotations.csv")
    if not os.path.exists(annotations_path):
        print(f"   WARNING: Seizure annotations file not found at {annotations_path}. No seizure labels will be generated.")
        return pd.DataFrame(columns=['start_time', 'end_time']) # Return empty DataFrame

    annotations_df = pd.read_csv(annotations_path)
    
    # --- FIX: Ensure correct unit for timestamp conversion in annotations ---
    # Based on your previous output, this fixed the 1970 issue.
    # Assuming start_time and end_time in CSV are in milliseconds since epoch
    annotations_df['start_time'] = pd.to_datetime(annotations_df['start_time'], unit='ms', utc=True)
    annotations_df['end_time'] = pd.to_datetime(annotations_df['end_time'], unit='ms', utc=True)
    # --- END FIX ---

    # Filter out short seizures
    annotations_df['duration'] = (annotations_df['end_time'] - annotations_df['start_time']).dt.total_seconds()
    annotations_df = annotations_df[annotations_df['duration'] >= min_duration_sec].copy()

    if annotations_df.empty:
        print("   No seizures found after duration filtering.")
        return annotations_df

    # Identify and keep only 'lead seizures' (based on inter-seizure interval)
    annotations_df = annotations_df.sort_values(by='start_time').reset_index(drop=True)
    lead_seizures = []
    if not annotations_df.empty:
        lead_seizures.append(0) # The first seizure is always a lead seizure
        for i in range(1, len(annotations_df)):
            prev_seizure_end = annotations_df.loc[lead_seizures[-1], 'end_time']
            current_seizure_start = annotations_df.loc[i, 'start_time']
            if (current_seizure_start - prev_seizure_end).total_seconds() / 3600 >= min_inter_seizure_interval_hours:
                lead_seizures.append(i)
    annotations_df = annotations_df.loc[lead_seizures].reset_index(drop=True)
    
    print(f"   Found {len(annotations_df)} lead seizures for {os.path.basename(patient_dir)}.")
    return annotations_df[['start_time', 'end_time']]

def synchronize_and_merge_data(patient_id, sensor_data_dict, target_freq_hz, sensors):
    """Resamples, interpolates, fills NaNs, and merges sensor data."""
    if not sensor_data_dict:
        print(f"   No sensor data to synchronize for {patient_id}.")
        return pd.DataFrame()

    resampled_data = {}
    # Use 's' instead of 'S' for future compatibility (as per FutureWarning)
    rule = f"{int(1/target_freq_hz)}s" # e.g., '1s' for 1Hz

    for sensor_name, df in sensor_data_dict.items():
        # Ensure df is a Series or DataFrame with a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            print(f"   Error: Index for {sensor_name} is not DatetimeIndex. Skipping.")
            continue
        
        # Resample to target frequency
        df_resampled = df.resample(rule).mean() # Take mean for resampling
        df_resampled = df_resampled.interpolate(method='time', limit_direction='both') # Interpolate after resample
        df_resampled = df_resampled.ffill().bfill() # Fill any remaining NaNs

        if df_resampled.empty:
            print(f"   WARNING: Resampled data for {sensor_name} is empty after resampling/interpolation.")
            continue
            
        resampled_data[sensor_name] = df_resampled

    if not resampled_data:
        print(f"   No resampled data available for {patient_id}.")
        return pd.DataFrame()

    # Merge all sensor dataframes
    merged_df = pd.DataFrame(index=pd.DatetimeIndex([])) # Start with an empty DatetimeIndex
    for sensor in sensors: # Merge in specified order
        if sensor in resampled_data:
            if merged_df.empty:
                merged_df = resampled_data[sensor].copy()
            else:
                merged_df = pd.merge(merged_df, resampled_data[sensor], left_index=True, right_index=True, how='outer')
        else:
            merged_df[sensor] = np.nan # Add missing sensors as NaN columns

    # Final fill for any NaNs introduced by outer merge (e.g., if one sensor has gaps not covered by others)
    merged_df = merged_df.interpolate(method='time', limit_direction='both').ffill().bfill()
    merged_df = merged_df.sort_index()

    print(f"   Synchronized and merged data shape for {patient_id}: {merged_df.shape}")
    return merged_df[sensors] # Return with consistent column order

def create_labeled_segments(synced_df, annotations_df, segment_duration_sec, target_freq_hz,
                            pre_ictal_window_min, pre_ictal_gap_min,
                            pre_ictal_exclusion_buffer_min, post_ictal_buffer_min):
    """Creates labeled segments based on synchronized data and annotations."""
    if synced_df.empty:
        print("   No synchronized data to create segments.")
        return [], []

    segment_length_steps = int(segment_duration_sec * target_freq_hz)
    step_size = segment_length_steps # Non-overlapping windows, as per notebook

    all_segments = []
    all_labels = []

    # Define various time windows for labeling
    seizure_windows = []
    pre_ictal_windows = []
    interictal_exclusion_windows = []

    if not annotations_df.empty:
        for _, row in annotations_df.iterrows():
            seizure_start = row['start_time']
            seizure_end = row['end_time']
            seizure_windows.append((seizure_start, seizure_end))

            # Pre-ictal window
            # Pre-ictal window ends (seizure_start - pre_ictal_gap)
            pre_ictal_end = seizure_start - pd.Timedelta(minutes=pre_ictal_gap_min) - pd.Timedelta(seconds=0.5 / target_freq_hz)
            pre_ictal_start = pre_ictal_end - pd.Timedelta(minutes=pre_ictal_window_min)
            pre_ictal_windows.append((pre_ictal_start, pre_ictal_end))

            # Interictal exclusion window (around seizure and pre-ictal)
            exclusion_start = seizure_start - pd.Timedelta(minutes=pre_ictal_exclusion_buffer_min)
            exclusion_end = seizure_end + pd.Timedelta(minutes=post_ictal_buffer_min)
            interictal_exclusion_windows.append((exclusion_start, exclusion_end))

    def check_overlap(segment_start, segment_end, windows):
        for win_start, win_end in windows:
            if max(segment_start, win_start) < min(segment_end, win_end):
                return True
        return False

    # Iterate through the synchronized data to create segments
    # Use tqdm to show progress for segment creation
    for i in tqdm(range(0, len(synced_df) - segment_length_steps + 1, step_size), desc="Creating Segments"):
        segment_df = synced_df.iloc[i : i + segment_length_steps].copy()
        
        # Ensure segment has enough data points
        if len(segment_df) != segment_length_steps:
            continue

        segment_start_time = segment_df.index.min()
        segment_end_time = segment_df.index.max() + pd.Timedelta(seconds=1/target_freq_hz) # End of last sample

        # Labeling logic from the notebook
        label = 0 # Default to interictal

        # Check for pre-ictal label (highest priority for positive labels)
        if check_overlap(segment_start_time, segment_end_time, pre_ictal_windows):
            if not check_overlap(segment_start_time, segment_end_time, seizure_windows): # Must NOT overlap with seizure
                label = 1
            
        # Check for exclusion from interictal (lowest priority, just flags if it's not a clear interictal)
        # If it overlaps with seizure or exclusion buffer, it's not a clear interictal segment.
        # This implies it's discarded if it falls into these zones and isn't pre-ictal.
        is_excluded_from_interictal = check_overlap(segment_start_time, segment_end_time, seizure_windows) or \
                                      check_overlap(segment_start_time, segment_end_time, interictal_exclusion_windows)
        
        if is_excluded_from_interictal and label == 0: # If it's excluded and not pre-ictal, don't include as 0
             continue # Skip this segment as it falls into exclusion zones

        all_segments.append(segment_df)
        all_labels.append(label)

    print(f"   Created {len(all_segments)} raw segments with labels.")
    return all_segments, all_labels

def load_and_segment_demo_data_from_training_pipeline(patient_id, config):
    """
    Loads raw 20% demo data for a patient and segments it according to the full training pipeline.
    Optionally applies undersampling for diagnostic purposes.
    """
    patient_dir = os.path.join(config['DEMO_DATA_ROOT_DIR']) # Path for the specific patient's 20% data
    
    print(f"\n2. Loading and segmenting demo data for {patient_id} (using training pipeline emulation)...")
    
    # 2.1 Load raw sensor data
    sensor_data_dict, found_sensors = load_raw_sensor_data_for_patient(patient_dir, config['BASE_SENSORS'], config['SENSOR_MAPPING'])
    if not sensor_data_dict:
        print(f"   No sensor data loaded for {patient_id} from {patient_dir}. Check paths and data existence.")
        return [], []
        
    # 2.2 Preprocess raw sensor data (replace 0 with NaN, interpolate)
    processed_sensor_data_dict = preprocess_raw_sensor_data(sensor_data_dict, patient_id)

    # 2.3 Load and preprocess seizure annotations
    annotations_df = load_and_preprocess_seizure_annotations(
        patient_dir,
        config['MIN_SEIZURE_DURATION_SEC'],
        config['MIN_INTER_SEIZURE_INTERVAL_HOURS']
    )

    # 2.4 Synchronize and merge sensor data
    synced_df = synchronize_and_merge_data(
        patient_id,
        processed_sensor_data_dict,
        config['TARGET_SAMPLING_HZ'],
        config['BASE_SENSORS']
    )
    if synced_df.empty:
        print(f"   No synchronized data for {patient_id}.")
        return [], []

    # 2.5 Create labeled segments
    segments, labels = create_labeled_segments(
        synced_df,
        annotations_df,
        config['SEGMENT_SECONDS'],
        config['TARGET_SAMPLING_HZ'],
        config['PRE_ICTAL_WINDOW_MINUTES'],
        config['PRE_ICTAL_GAP_MINUTES'],
        config['PRE_ICTAL_EXCLUSION_BUFFER_MINUTES'],
        config['POST_ICTAL_BUFFER_MINUTES']
    )
    
    initial_positive_count = sum(1 for l in labels if l == 1)
    initial_negative_count = sum(1 for l in labels if l == 0)
    print(f"   Initial segments count: {len(segments)} (Pos: {initial_positive_count}, Neg: {initial_negative_count})")

    # --- NEW: Optional Undersampling for Diagnostic Purposes ---
    if config['APPLY_UNDERSAMPLING_TO_DEMO_TEST']:
        print("   Applying RandomUnderSampler to demo test data for diagnostic purposes...")
        
        # Reshape segments for RandomUnderSampler: (num_samples, num_features * seq_len)
        # Store original shapes for reconstruction
        original_segment_shapes = [s.shape for s in segments]
        
        # Flatten each segment DataFrame into a 1D array
        X_flat = np.array([s.values.flatten() for s in segments])
        y = np.array(labels)

        if len(np.unique(y)) < 2:
            print("   Cannot undersample: Only one class present in demo data after segmentation.")
        else:
            rus = RandomUnderSampler(random_state=42) # Using a fixed random_state for reproducibility
            X_resampled_flat, y_resampled = rus.fit_resample(X_flat, y)
            
            # Reconstruct segments from flattened arrays
            resampled_segments = []
            for i, flat_segment in enumerate(X_resampled_flat):
                # Assuming all segments have the same shape: (config['SEQ_LEN'], config['NUM_FEATURES'])
                segment_df = pd.DataFrame(
                    flat_segment.reshape(config['SEQ_LEN'], config['NUM_FEATURES']),
                    columns=config['BASE_SENSORS']
                )
                # Note: Timestamps are lost here. For evaluation, we only need the data and label.
                # If timestamps are critical for some other reason, they'd need to be handled separately.
                resampled_segments.append(segment_df)
            
            segments = resampled_segments
            labels = y_resampled.tolist() # Convert back to list for consistency

            undersampled_positive_count = sum(1 for l in labels if l == 1)
            undersampled_negative_count = sum(1 for l in labels if l == 0)
            print(f"   Undersampled segments count: {len(segments)} (Pos: {undersampled_positive_count}, Neg: {undersampled_negative_count})")
            print("   WARNING: Undersampling on test data is for DIAGNOSIS ONLY. Do NOT use for realistic performance reporting.")
    # --- End NEW: Optional Undersampling ---

    print(f"   Final count of segments: {len(segments)}")
    print(f"   Positive (seizure) segments: {sum(1 for l in labels if l == 1)}")
    print(f"   Negative (non-seizure) segments: {sum(1 for l in labels if l == 0)}")

    return segments, labels


# --- Use the new, comprehensive loading function ---
demo_segments, demo_labels = load_and_segment_demo_data_from_training_pipeline(
    config['PATIENT_ID'], config
)

if not demo_segments: # Check if any segments were loaded
    print("Exiting as no demo data could be loaded/segmented for evaluation.")
    exit()

all_true_labels = []
all_probabilities = []

# --- 3. Make Predictions ---
print("\n3. Making predictions on demo data segments...")
with torch.no_grad(): # Disable gradient calculations for inference
    for i, segment_df in tqdm(enumerate(demo_segments), total=len(demo_segments), desc="Predicting"):
        true_label = demo_labels[i] # Get the label corresponding to the current segment

        # Verify segment shape before processing
        if segment_df.shape[0] != config['SEQ_LEN'] or segment_df.shape[1] != config['NUM_FEATURES']:
            print(f"   WARNING: Skipping segment due to incorrect shape: {segment_df.shape}. Expected: ({config['SEQ_LEN']}, {config['NUM_FEATURES']})")
            continue

        # Apply the loaded scaler (CRITICAL: use the same scaler as training)
        scaled_segment = scaler.transform(segment_df.values)
        
        # Convert to tensor and reshape for model input: (batch_size, channels, sequence_length)
        segment_tensor = torch.tensor(scaled_segment, dtype=torch.float32).permute(1, 0).unsqueeze(0)
        
        # Get raw output from model
        output = model(segment_tensor)
        
        probability = output.item()
        
        # Store results
        all_true_labels.append(true_label)
        all_probabilities.append(probability)

# Convert to numpy arrays for sklearn metrics
all_true_labels = np.array(all_true_labels)
all_probabilities = np.array(all_probabilities)

# --- 4. Analyze Results ---
print("\n--- Evaluation Results on Demo Data ---")

# --- a) Evaluate with the default threshold (0.50) ---
print(f"\nMetrics at default threshold ({config['DEFAULT_ALERT_THRESHOLD']:.2f}):")
all_predictions_at_default_threshold = (all_probabilities >= config['DEFAULT_ALERT_THRESHOLD']).astype(int)

# Handle cases where a class might be missing in true_labels for metrics like precision/recall
# (e.g., if demo data has no positive labels, or no negative labels)
try:
    acc = accuracy_score(all_true_labels, all_predictions_at_default_threshold)
    prec = precision_score(all_true_labels, all_predictions_at_default_threshold, zero_division=0)
    rec = recall_score(all_true_labels, all_predictions_at_default_threshold, zero_division=0)
    f1 = f1_score(all_true_labels, all_predictions_at_default_threshold, zero_division=0)
    cm = confusion_matrix(all_true_labels, all_predictions_at_default_threshold)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
except ValueError as e:
    print(f"Error computing metrics at default threshold: {e}")
    print("This often happens if there's only one class present in the true labels or predictions for the demo data.")


# --- b) Probability Distribution Analysis ---
print("\n--- Probability Distribution ---")
pos_probs = all_probabilities[all_true_labels == 1]
neg_probs = all_probabilities[all_true_labels == 0]

print(f"Number of Positive (Seizure) segments: {len(pos_probs)}")
print(f"Number of Negative (Non-Seizure) segments: {len(neg_probs)}")

if len(pos_probs) > 0:
    print(f"Mean probability for positive (seizure) segments: {np.mean(pos_probs):.4f}")
else:
    print("No positive (seizure) segments in demo data to compute mean probability.")

if len(neg_probs) > 0:
    print(f"Mean probability for negative (non-seizure) segments: {np.mean(neg_probs):.4f}")
else:
    print("No negative (non-seizure) segments in demo data to compute mean probability.")

# Plotting histograms
plt.figure(figsize=(12, 6))
if len(neg_probs) > 0:
    sns.histplot(neg_probs, color='blue', label='Non-Seizure (True Label = 0)', kde=True, stat='density', alpha=0.5, bins=20)
if len(pos_probs) > 0:
    sns.histplot(pos_probs, color='red', label='Seizure (True Label = 1)', kde=True, stat='density', alpha=0.5, bins=20)
plt.axvline(x=config['DEFAULT_ALERT_THRESHOLD'], color='green', linestyle='--', label=f'Default Threshold ({config["DEFAULT_ALERT_THRESHOLD"]:.2f})')
plt.title(f"Predicted Probability Distribution for Patient {config['PATIENT_ID']} Demo Data")
plt.xlabel("Predicted Probability")
plt.ylabel("Density")
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.xlim(0, 1)
plt.show()

# --- c) AUC-ROC and AUC-PR ---
# Check if both classes are present for AUC calculations
if len(np.unique(all_true_labels)) > 1:
    try:
        auc_roc = roc_auc_score(all_true_labels, all_probabilities)
        print(f"\nAUC-ROC: {auc_roc:.4f}")
    except ValueError:
        print("\nAUC-ROC cannot be computed as only one class is present in true labels.")
    
    # Precision-Recall AUC
    precision_val, recall_val, _ = precision_recall_curve(all_true_labels, all_probabilities)
    auc_pr = auc(recall_val, precision_val)
    print(f"AUC-PR: {auc_pr:.4f}")
else:
    print("\nCannot compute AUC-ROC or AUC-PR: Only one class present in true labels for demo data.")


# --- d) Threshold Optimization / Exploration ---
print("\n--- Threshold Exploration ---")
if len(np.unique(all_true_labels)) > 1:
    # Calculate ROC curve for Youden's J
    fpr, tpr, thresholds_roc = roc_curve(all_true_labels, all_probabilities)

    # Find optimal threshold using Youden's J statistic (maximizes TPR - FPR)
    youden_j = tpr - fpr
    optimal_idx_youden = np.argmax(youden_j)
    optimal_threshold_youden = thresholds_roc[optimal_idx_youden]
    print(f"Optimal threshold (Youden's J, maximizes Sensitivity + Specificity): {optimal_threshold_youden:.4f}")

    # Find optimal threshold maximizing F1-score
    best_f1 = 0
    optimal_threshold_f1 = 0
    # Iterate over unique probabilities as potential thresholds to avoid re-calculating for same thresholds
    for t in sorted(np.unique(all_probabilities)): 
        temp_preds = (all_probabilities >= t).astype(int)
        # Ensure both classes are present in predictions for F1 score to be meaningful
        if len(np.unique(temp_preds)) > 1 or (len(np.unique(temp_preds)) == 1 and np.unique(temp_preds)[0] == np.unique(all_true_labels)[0]):
            f1_val = f1_score(all_true_labels, temp_preds, zero_division=0)
            if f1_val > best_f1:
                best_f1 = f1_val
                optimal_threshold_f1 = t
    print(f"Optimal threshold (Max F1-score): {optimal_threshold_f1:.4f} (F1={best_f1:.4f})")
    print("\nIf these optimal thresholds differ significantly from your 0.65 alert threshold,")
    print("it suggests your model might be well-performing but its raw probability outputs")
    print("are not perfectly calibrated to your current alert threshold.")
    print("Consider adjusting your alert threshold for the demo data, or applying probability calibration.")
else:
    print("Cannot perform threshold optimization: Only one class present in true labels for demo data.")


# --- e) Plot ROC Curve ---
if len(np.unique(all_true_labels)) > 1:
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_roc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[optimal_idx_youden], tpr[optimal_idx_youden], marker='o', color='red', s=100,
                label=f'Optimal Threshold (Youden: {optimal_threshold_youden:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {config["PATIENT_ID"]} Demo Data')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # --- f) Plot Precision-Recall Curve ---
    plt.figure(figsize=(8, 6))
    plt.plot(recall_val, precision_val, color='blue', lw=2, label=f'Precision-Recall curve (area = {auc_pr:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {config["PATIENT_ID"]} Demo Data')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.show()