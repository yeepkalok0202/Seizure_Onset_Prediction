import gc  # Garbage collection
import glob
import itertools  # For hyperparameter combinations
import math  # For calculating output length
import os
import random  # For reproducibility
import time
from collections import \
    OrderedDict  # To keep hyperparameter order in filenames

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.utils import class_weight  # Import class_weight
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm  # Import tqdm for progress bars

# --- Configuration ---
DATA_ROOT_DIR = 'F:\\data_9' # Replace with the actual path to your 'data_9' folder
OUTPUT_DIR = 'processed_data_pytorch_adaptive_hyper_pre_post_buffer' # Directory to save results files and models (Changed name for clarity)

# Ensure the base output directory exists early
os.makedirs(OUTPUT_DIR, exist_ok=True) # Moved this up


# --- Feature Flags ---
ENABLE_ADAPTIVE_SENSORS = False # Set to True to iterate through sensor combinations
ENABLE_HYPERPARAMETER_TUNING = False # Set to True to iterate through hyperparameter combinations

# --- Data Processing Parameters (Can be made tunable if needed, but fixed for now) ---
SEGMENT_DURATION_SECONDS = 30

# --- Tunable Hyperparameters (Define lists of values to try) ---
# If ENABLE_HYPERPARAMETER_TUNING is False, only the first value from each list is used.

TUNABLE_SAMPLING_FREQ_HZ = [1] # Example: [1, 10, 128] - Add 128Hz if you want to test raw frequency
TUNABLE_PRE_ICTAL_WINDOW_MINUTES = [30] # Example: [30, 60]
# PRE_ICTAL_EXCLUSION_BUFFER_MINUTES must be >= PRE_ICTAL_WINDOW_MINUTES
TUNABLE_PRE_ICTAL_EXCLUSION_BUFFER_MINUTES = [60] # Example: [30, 60, 90] - Ensure >= corresponding PRE_ICTAL_WINDOW_MINUTES< PRE_ICTAL_EXCLUSION_BUFFER_MINUTES!
TUNABLE_POST_ICTAL_BUFFER_MINUTES = [180] # Example: [60, 120, 180]

TUNABLE_CONV_FILTERS = [[64, 128, 256]] # Example: [[32, 64, 128], [64, 128, 256]]
TUNABLE_CONV_KERNEL_SIZE = [10] # Example: [5, 10, 15]
TUNABLE_POOL_SIZE = [2] # Example: [2, 3]
TUNABLE_LSTM_UNITS = [128] # Example: [64, 128, 256]
TUNABLE_DENSE_UNITS = [64] # Example: [32, 64, 128]

TUNABLE_GENERAL_MODEL_EPOCHS = [50] # Example: [30, 50, 100]
TUNABLE_PERSONALIZATION_EPOCHS = [30] # Example: [20, 30, 50]
TUNABLE_GENERAL_MODEL_LR = [0.001] # Example: [0.001, 0.0005]
TUNABLE_PERSONALIZATION_LR = [0.0001] # Example: [0.0001, 0.00005]
TUNABLE_BATCH_SIZE = [32] # Example: [16, 32]
TUNABLE_PERSONALIZATION_BATCH_SIZE = [16] # Example: [8, 16]

# --- Model Types to Run ---
MODEL_TYPES_TO_RUN = ['CNN-LSTM', 'CNN-BiLSTM'] # Example: ['CNN-LSTM', 'CNN-BiLSTM']

# --- Sensor Combinations (Used if ENABLE_ADAPTIVE_SENSORS is True) ---
# Define base sensors
BASE_SENSORS = ['HR', 'EDA', 'TEMP', 'ACC'] # Sorted for consistent column order if needed

# Generate all combinations of 1 to 4 sensors from BASE_SENSORS
# If you only want specific combinations, define them manually here
if ENABLE_ADAPTIVE_SENSORS:
    SENSOR_COMBINATIONS = []
    for i in range(1, len(BASE_SENSORS) + 1):
        for combo in itertools.combinations(BASE_SENSORS, i):
            SENSOR_COMBINATIONS.append(list(combo))
else:
    # Use the default set if adaptive sensors is disabled
    SENSOR_COMBINATIONS = [BASE_SENSORS]


# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Data Loading and Preprocessing ---

def load_sensor_data_for_patient(patient_dir, sensors):
    """
    Loads Parquet data for specified sensors for a given patient,
    concatenates it, sorts by timestamp, and converts to UTC.
    Does NOT apply scaling yet. Filters to only include specified sensors.

    Args:
        patient_dir (str): The directory for the specific patient.
        sensors (list): List of sensor names (e.g., ['HR', 'EDA']).

    Returns:
        dict: A dictionary where keys are attribute names (e.g., 'HR')
              and values are Pandas DataFrames containing the raw data for the
              specified sensors. Returns an empty dict if no data is found
              for any of the specified sensors.
    """
    attribute_data = {}
    sensor_mapping = {
        'HR': 'HR', 'EDA': 'EDA', 'TEMP': 'TEMP', 'ACC': 'Acc Mag'
    }

    # Ensure sensor names are consistent (lowercase keys in dict)
    target_sensors_lower = [s.lower() for s in sensors]

    for sensor_name in sensors:
        sensor_name_lower = sensor_name.lower()
        if sensor_name_lower not in target_sensors_lower:
             continue # Only process sensors in the target list

        if sensor_name not in sensor_mapping:
            continue

        attr_folder = sensor_name # Empatica folder name often matches sensor name
        attr_name_part = sensor_mapping[sensor_name] # Part of the filename

        # Adjusted glob pattern to be more robust
        parquet_files = sorted(glob.glob(os.path.join(patient_dir, f'Empatica-{attr_folder}', f'*{attr_name_part}*.parquet')))


        if not parquet_files:
            continue

        all_dfs = []
        for file_path in parquet_files:
            try:
                df = pd.read_parquet(file_path)
                if 'time' in df.columns and 'data' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['time'] / 1000, unit='s', utc=True)
                    df = df.rename(columns={'data': sensor_name_lower}) # Rename to lowercase sensor name
                    df = df[['timestamp', sensor_name_lower]]
                    all_dfs.append(df)
                else:
                    print(f"Warning: Parquet file {file_path} does not have expected 'time' and 'data' columns. Skipping.")

            except Exception as e:
                print(f"Error reading Parquet file {file_path}: {e}")

        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df = combined_df.sort_values(by='timestamp').reset_index(drop=True)
            attribute_data[sensor_name_lower] = combined_df # Use lowercase key

    return attribute_data

def load_seizure_annotations(patient_dir):
    """
    Loads and processes the SeerAnnotations CSV for a given patient.
    Converts start_time and end_time to UTC datetime objects.

    Args:
        patient_dir (str): The directory for the specific patient.

    Returns:
        pandas.DataFrame: A DataFrame containing the seizure annotations with
                          'start_time' and 'end_time' as UTC datetime objects.
                          Returns empty df if the file is not found or has incorrect columns.
    """
    annotation_file = os.path.join(patient_dir, f'{os.path.basename(patient_dir)}_SeerAnnotations.csv')
    try:
        annotations_df = pd.read_csv(annotation_file)
        if 'start_time' in annotations_df.columns and 'end_time' in annotations_df.columns:
            annotations_df['start_time'] = pd.to_datetime(annotations_df['start_time'] / 1000, unit='s', utc=True)
            annotations_df['end_time'] = pd.to_datetime(annotations_df['end_time'] / 1000, unit='s', utc=True)
            # Ensure end_time is at least 1 second after start_time to avoid zero-duration seizures
            annotations_df['end_time'] = annotations_df.apply(lambda row: row['end_time'] if row['end_time'] > row['start_time'] else row['start_time'] + pd.Timedelta(seconds=1), axis=1)
            return annotations_df[['start_time', 'end_time']]
        else:
            print(f"Error: Annotation file {annotation_file} does not have expected 'start_time' and 'end_time' columns.")
            return pd.DataFrame(columns=['start_time', 'end_time']) # Return empty df on error
    except FileNotFoundError:
        return pd.DataFrame(columns=['start_time', 'end_time']) # Return empty df if not found
    except Exception as e:
        print(f"Error reading annotation file {annotation_file}: {e}")
        return pd.DataFrame(columns=['start_time', 'end_time']) # Return empty df on other errors


def synchronize_and_merge_data(sensor_data_dict, target_freq_hz):
    """
    Synchronizes sensor data from different sensors to a common time index
    by resampling, merges them, applies Robust Scaling, and handles NaNs.
    Includes only columns that were present in the input dictionary.

    Args:
        sensor_data_dict (dict): Dictionary with sensor names (lowercase) as keys and
                                 DataFrames (with 'timestamp' and data column) as values.
        target_freq_hz (int): The target frequency in Hz for resampling.

    Returns:
        pandas.DataFrame: A single DataFrame with a datetime index containing
                          all synchronized and scaled sensor data. Columns are sorted alphabetically.
                          Returns None if input is empty or no common time found.
    """
    if not sensor_data_dict:
        return None

    resampled_data = {}
    for sensor_name_lower, df in sensor_data_dict.items():
        df = df.set_index('timestamp').sort_index()
        # Ensure target_freq_hz is not zero or negative
        if target_freq_hz <= 0:
            print(f"Error: Target frequency must be positive, got {target_freq_hz}. Cannot resample.")
            return None
        rule = f'{1/target_freq_hz}S'
        try:
            resampled_df = df.asfreq(rule)
            resampled_data[sensor_name_lower] = resampled_df
        except Exception as e:
             print(f"Error during resampling sensor {sensor_name_lower} to {target_freq_hz}Hz: {e}. Skipping sensor.")
             continue


    merged_df = None
    for sensor_name_lower, df in resampled_data.items():
        if merged_df is None:
            merged_df = df
        else:
            # Use 'outer' join for maximum time coverage across all sensors
            merged_df = merged_df.join(df, how='outer')

    if merged_df is None or merged_df.empty:
         return None

    merged_df = merged_df.sort_index()

    # Interpolate missing values
    merged_df = merged_df.interpolate(method='time')
    merged_df = merged_df.fillna(method='ffill')
    merged_df = merged_df.fillna(method='bfill')

    # Drop columns that are still all NaN after interpolation/fill (happens if a sensor had no data)
    merged_df = merged_df.dropna(axis=1, how='all')

    if merged_df.empty: # Check again if it's empty after dropping columns
        # print("Warning: Merged DataFrame is empty after handling NaNs. Skipping scaling.") # Suppress frequent print
        return None # Return None if no usable data


    # Apply Robust Scaling to all remaining data columns
    scaler = RobustScaler()
    data_cols = merged_df.columns # Use all remaining columns
    if not merged_df.empty and len(data_cols) > 0:
        try:
            merged_df[data_cols] = scaler.fit_transform(merged_df[data_cols])
        except Exception as e:
            print(f"Error during scaling: {e}. Skipping scaling.")
            # Decide if you want to return None or unscaled data
            return None # Returning None means this patient/sensor/freq combination is unusable


    elif not merged_df.empty and len(data_cols) == 0:
         # print("Warning: Merged DataFrame is not empty but has no data columns to scale after dropping NaNs.") # Suppress frequent print
         return None # Return None if no columns can be scaled

    # Sort columns alphabetically for consistency
    merged_df = merged_df.sort_index(axis=1)


    return merged_df

def create_labeled_segments(synced_df, annotations_df, segment_duration_sec, pre_ictal_window_min, pre_ictal_exclusion_buffer_min, post_ictal_buffer_min, target_freq_hz):
    """
    Creates segments from synchronized data and labels them
    as pre-ictal (1) or interictal (0) based on seizure annotations. Samples interictal
    segments to attempt class balance. Uses asymmetrical buffers.

    Args:
        synced_df (pandas.DataFrame): DataFrame with synchronized sensor data (datetime index).
                                      Columns must be consistent (e.g., sorted).
        annotations_df (pandas.DataFrame): DataFrame with seizure start/end times.
        segment_duration_sec (int): Duration of each segment in seconds.
        pre_ictal_window_min (int): Time window before seizure onset considered pre-ictal (positive class).
        pre_ictal_exclusion_buffer_min (int): Buffer time *before* seizure onset to exclude for clean interictal. Must be >= pre_ictal_window_min.
        post_ictal_buffer_min (int): Buffer time *after* seizure end to exclude for clean interictal.
        target_freq_hz (int): Frequency data was resampled to (for calculating segment steps).


    Returns:
        tuple: (segments, labels) where segments is a numpy array
               (shape: n_samples, segment_len, num_features) and labels is a numpy array (0 or 1).
               Returns (np.array([]), np.array([])) with appropriate shape if no data or annotations.
    """
    # Need default shape even if inputs are None/empty or invalid
    num_features = synced_df.shape[1] if synced_df is not None and not synced_df.empty else 1 # Fallback to 1 feature
    segment_length_steps = int(segment_duration_sec * target_freq_hz)
    # Ensure segment_length_steps is at least 1 and sensible
    if segment_length_steps <= 0:
        print(f"Error: Calculated segment_length_steps is {segment_length_steps} (Duration {segment_duration_sec}s * Freq {target_freq_hz}Hz). Cannot create segments.")
        segment_length_steps = 1 # Default to 1 step if calculation is bad


    # Validate buffer relationship
    if pre_ictal_exclusion_buffer_min < pre_ictal_window_min:
        print(f"Error: pre_ictal_exclusion_buffer_min ({pre_ictal_exclusion_buffer_min}) must be >= pre_ictal_window_min ({pre_ictal_window_min}). Skipping segmentation.")
        # Return with default shape on error
        return np.array([]).reshape(0, segment_length_steps, num_features), np.array([])


    if synced_df is None or synced_df.empty or len(synced_df.columns) == 0:
        # print("Synced data is empty, has no columns, or annotations are missing. Cannot create segments.") # Suppress frequent print
        # Return with default shape on error
        return np.array([]).reshape(0, segment_length_steps, num_features), np.array([])

    # Ensure annotations are not None, even if empty
    if annotations_df is None:
         annotations_df = pd.DataFrame(columns=['start_time', 'end_time'])


    segments = []
    labels = []
    step_size = segment_length_steps # Use non-overlapping segments


    data_start_time = synced_df.index.min() if not synced_df.empty else None
    data_end_time = synced_df.index.max() if not synced_df.empty else None

    if data_start_time is None or data_end_time is None:
        print("Warning: Synced data has no time index. Cannot create segments.")
        return np.array([]).reshape(0, segment_length_steps, num_features), np.array([]) # Return with default shape


    # Helper to check overlap (inclusive of boundaries)
    def check_overlap(seg_start, seg_end, windows):
        for win_start, win_end in windows:
            # Ensure window is valid before checking overlap
            if win_start is not None and win_end is not None and win_start < win_end:
                # Use half sample period tolerance for boundary check
                if target_freq_hz > 0: # Avoid division by zero
                    overlap_tolerance = pd.Timedelta(seconds=0.5/target_freq_hz)
                else:
                    overlap_tolerance = pd.Timedelta(seconds=0) # No tolerance if freq is zero

                if max(seg_start, win_start) < min(seg_end, win_end) + overlap_tolerance:
                     return True
        return False


    # --- Define Time Windows ---

    # 1. Actual Seizure (Ictal) Windows
    seizure_windows = []
    # print("Defining ictal windows...") # Suppress frequent print
    for i, seizure in annotations_df.iterrows():
         seizure_start = seizure['start_time']
         seizure_end = seizure['end_time']
         if seizure_start is not None and seizure_end is not None and seizure_start < seizure_end:
             seizure_windows.append((seizure_start, seizure_end))
         # else: print(f"Warning: Skipping zero or negative duration seizure annotation: {seizure_start} to {seizure_end}") # Suppress


    # 2. Pre-ictal Windows (Positive Class)
    pre_ictal_windows = []
    # print(f"Defining pre-ictal windows ({pre_ictal_window_min} mins before seizure onset)...") # Suppress frequent print)
    for i, seizure in annotations_df.iterrows():
        seizure_start = seizure['start_time']
        if seizure_start is None: continue # Skip if seizure start is missing

        pre_ictal_start_uncapped = seizure_start - pd.Timedelta(minutes=pre_ictal_window_min)
        # End just before seizure starts (half sample tolerance)
        if target_freq_hz > 0:
            pre_ictal_end = seizure_start - pd.Timedelta(seconds=0.5/target_freq_hz)
        else:
            pre_ictal_end = seizure_start # No tolerance if freq is zero


        # Cap the pre-ictal start at the beginning of the available data
        pre_ictal_start = max(data_start_time, pre_ictal_start_uncapped)


        # Ensure the capped window is valid
        if pre_ictal_start is not None and pre_ictal_end is not None and pre_ictal_start < pre_ictal_end:
             # Ensure pre-ictal window does *not* overlap with the seizure itself
             if not check_overlap(pre_ictal_start, pre_ictal_end, seizure_windows):
                 pre_ictal_windows.append((pre_ictal_start, pre_ictal_end))
                 # print(f" Seizure {i+1}: Pre-ictal window added: {pre_ictal_start} to {pre_ictal_end}") # Suppress frequent print)
             # else: print(f" Seizure {i+1}: Calculated pre-ictal window overlaps with seizure window. Skipped.") # Suppress frequent print)
        # else: print(f" Seizure {i+1}: Calculated pre-ictal window or capped window is invalid. Skipped.") # Suppress frequent print)


    # 3. Interictal Exclusion Windows (Asymmetrical Buffer)
    # These define areas NOT suitable for clean interictal samples
    interictal_exclusion_windows = []
    buffer_before_timedelta = pd.Timedelta(minutes=pre_ictal_exclusion_buffer_min)
    buffer_after_timedelta = pd.Timedelta(minutes=post_ictal_buffer_min)
    # print(f"Defining interictal exclusion windows ({pre_ictal_exclusion_buffer_min} mins before, {post_ictal_buffer_min} mins after)...") # Suppress frequent print)

    for _, seizure in annotations_df.iterrows():
         seizure_start = seizure['start_time']
         seizure_end = seizure['end_time']
         if seizure_start is None or seizure_end is None: continue # Skip if seizure times are missing

         exclusion_start = seizure_start - buffer_before_timedelta
         exclusion_end = seizure_end + buffer_after_timedelta
         # Ensure exclusion window is valid
         if exclusion_start is not None and exclusion_end is not None and exclusion_start < exclusion_end:
             interictal_exclusion_windows.append((exclusion_start, exclusion_end))
         # else: print(f"Warning: Skipping invalid exclusion window: {exclusion_start} to {exclusion_end}") # Suppress


    # --- Create Segments and Assign Labels ---

    # print(f"Creating segments (len={segment_duration_sec}s, step={segment_duration_sec}s) from {len(synced_df)} total steps...)") # Suppress frequent print)
    segments_skipped_ictal = 0
    segments_skipped_interictal_buffer = 0 # Segments in buffer BUT NOT pre-ictal
    segments_labeled_preictal = 0
    segments_labeled_interictal = 0
    segments_total_candidates = 0 # Count segments before any skipping

    # Ensure segment_length_steps is valid before iterating
    if segment_length_steps <= 0:
        print(f"Error: Calculated segment_length_steps is {segment_length_steps}. Cannot create segments.")
        return np.array([]).reshape(0, max(1, int(segment_duration_sec * target_freq_hz))), num_features, np.array([]) # Return with default shape


    for i in tqdm(range(0, len(synced_df) - segment_length_steps + 1, step_size), desc="Segmenting Data", leave=False):
        segment_df = synced_df.iloc[i : i + segment_length_steps]
        if len(segment_df) != segment_length_steps:
            continue # Should not happen with step_size = segment_length_steps, but safety check

        segments_total_candidates += 1

        segment_start_time = segment_df.index[0]
        if target_freq_hz > 0:
             segment_end_time = segment_df.index[-1] + pd.Timedelta(seconds=0.5/target_freq_hz) # Use midpoint of last sample for end time
        else:
             segment_end_time = segment_df.index[-1] # No tolerance if freq is zero


        # Check for overlap with actual seizure (ictal) windows - SKIP
        if check_overlap(segment_start_time, segment_end_time, seizure_windows):
            segments_skipped_ictal += 1
            continue


        # Check for overlap with pre-ictal windows - LABEL 1 (PRIORITIZE)
        if check_overlap(segment_start_time, segment_end_time, pre_ictal_windows):
            segments.append(segment_df.values)
            labels.append(1)
            segments_labeled_preictal += 1
            continue


        # Check for overlap with interictal exclusion buffer - SKIP (don't label as 0 if in buffer)
        # This check ONLY happens if the segment was NOT ictal and NOT pre-ictal
        if check_overlap(segment_start_time, segment_end_time, interictal_exclusion_windows):
             segments_skipped_interictal_buffer += 1
             continue


        # If none of the above, label as Interictal (0)
        segments.append(segment_df.values)
        labels.append(0)
        segments_labeled_interictal += 1


    segments = np.array(segments)
    labels = np.array(labels)

    # print(f"Finished segmentation. Total full-length candidate segments: {segments_total_candidates}") # Suppress frequent print)
    # print(f" Segments skipped (ictal): {segments_skipped_ictal}") # Suppress frequent print)
    # print(f" Segments skipped (interictal buffer, not pre-ictal): {segments_skipped_interictal_buffer}") # Suppress frequent print)
    # print(f" Total segments included for labeling (Pre-ictal + Interictal): {len(segments)}") # Suppress frequent print)
    # print(f" Segments labeled Pre-ictal: {np.sum(labels)}, Interictal: {len(labels) - np.sum(labels)}") # Suppress frequent print)


    # Simple class balancing: Undersample majority class
    pre_ictal_indices = np.where(labels == 1)[0]
    interictal_indices = np.where(labels == 0)[0]

    min_count = min(len(pre_ictal_indices), len(interictal_indices))

    if min_count == 0:
        # print("Warning: One class has zero samples after segmentation. Cannot balance.") # Suppress frequent print)
        num_features = segments.shape[2] if segments.shape[0] > 0 else (synced_df.shape[1] if synced_df is not None else 1) # Fallback if needed
        segment_length_steps = int(segment_duration_sec * target_freq_hz)
        segment_length_steps = max(1, segment_length_steps) # Ensure at least 1
        return np.array([]).reshape(0, segment_length_steps, num_features), np.array([])


    # Only balance if both classes have samples and there's a majority to undersample
    if len(pre_ictal_indices) > 0 and len(interictal_indices) > 0 and (len(pre_ictal_indices) > min_count or len(interictal_indices) > min_count):
        # print(f"Balancing classes: Reducing majority class to {min_count} samples.") # Suppress frequent print)
        balanced_indices_pre = np.random.choice(pre_ictal_indices, min_count, replace=False)
        balanced_indices_inter = np.random.choice(interictal_indices, min_count, replace=False)
        balanced_indices = np.concatenate([balanced_indices_pre, balanced_indices_inter])
        np.random.shuffle(balanced_indices)

        segments = segments[balanced_indices]
        labels = labels[balanced_indices]
        # print(f"After balancing: Total segments: {len(segments)}, Pre-ictal: {np.sum(labels)}, Interictal: {len(labels) - np.sum(labels)}") # Suppress frequent print)

    return segments, labels

def prepare_patient_data(patient_folder, sensors_to_use, pre_ictal_window_min, pre_ictal_exclusion_buffer_min, post_ictal_buffer_min, sampling_freq_hz):
    """
    Loads raw sensor data for a patient, synchronizes, scales, and creates
    labeled segments using the specified parameters.

    Args:
        patient_folder (str): Path to the patient's data directory.
        sensors_to_use (list): List of sensor names to include.
        pre_ictal_window_min (int): Pre-ictal window duration.
        pre_ictal_exclusion_buffer_min (int): Pre-ictal exclusion buffer duration.
        post_ictal_buffer_min (int): Post-ictal buffer duration.
        sampling_freq_hz (int): Target sampling frequency.


    Returns:
        tuple: (patient_id, segments, labels, expected_seq_len, expected_num_features)
               Returns (patient_id, np.array([]), np.array([]), expected_seq_len, expected_num_features) if processing fails,
               where expected_* are calculated from inputs.
    """
    patient_id = os.path.basename(patient_folder)

    # Calculate expected shape based on inputs for consistent return shape on failure
    expected_seq_len = int(SEGMENT_DURATION_SECONDS * sampling_freq_hz)
    expected_seq_len = max(1, expected_seq_len) # Ensure at least 1

    # The number of features can only be known after synchronization and dropping NaN columns.
    # Start with the requested number, but the actual number might be less.
    expected_num_features = len(sensors_to_use) if sensors_to_use else 1 # Fallback if sensor_to_use is empty


    # Initial return with empty segments and calculated expected shape in case of early failure
    # Use a placeholder shape for the 3rd dim until we know the actual number of features
    empty_segments = np.array([]).reshape(0, expected_seq_len, expected_num_features) # Use requested num_features as fallback
    empty_labels = np.array([])


    sensor_data_dict = load_sensor_data_for_patient(patient_folder, sensors_to_use)
    # If load_sensor_data_for_patient didn't find any of the specified sensors, return empty segments but correct expected shape
    if not sensor_data_dict:
        # print(f"Skipping patient {patient_id}: Could not load any data for specified sensors {sensors_to_use}.") # Suppress frequent print)
        return patient_id, empty_segments, empty_labels, expected_seq_len, expected_num_features

    annotations_df = load_seizure_annotations(patient_folder)
    # annotations_df can be empty if no seizures, handled in create_labeled_segments

    # Synchronize and merge the loaded data, resampling to the target frequency
    synced_df = synchronize_and_merge_data(sensor_data_dict, target_freq_hz=sampling_freq_hz)
    # synced_df contains only columns for the sensors that were found and successfully merged/scaled
    # The number of columns is the actual number of features found for this patient/sensor combo
    if synced_df is None or synced_df.empty or len(synced_df.columns) == 0:
         # print(f"Skipping patient {patient_id}: Could not synchronize, merge, or lost columns for sensors {sensors_to_use}.") # Suppress frequent print)
         return patient_id, empty_segments, empty_labels, expected_seq_len, expected_num_features

    # Update expected_num_features based on actual columns found and merged for this patient/combo
    actual_num_features = len(synced_df.columns)
    # Recreate empty_segments with the correct feature count if needed later
    empty_segments = np.array([]).reshape(0, expected_seq_len, actual_num_features)


    # Create labeled segments using the synchronized data and current buffer/window parameters
    segments, labels = create_labeled_segments(
        synced_df,
        annotations_df,
        segment_duration_sec=SEGMENT_DURATION_SECONDS,
        pre_ictal_window_min=pre_ictal_window_min,
        pre_ictal_exclusion_buffer_min=pre_ictal_exclusion_buffer_min,
        post_ictal_buffer_min=post_ictal_buffer_min,
        target_freq_hz=sampling_freq_hz
    )

    # If no segments were created (e.g., due to parameters or no data after filtering)
    if len(segments) == 0:
         # print(f"Skipping patient {patient_id}: No valid segments created with current parameters.") # Suppress frequent print)
         # Ensure the returned empty segments array has the shape consistent with the processed data's features
         # Use actual_num_features derived from synced_df columns
         return patient_id, np.array([]).reshape(0, segments.shape[1] if segments.ndim > 1 else expected_seq_len, segments.shape[2] if segments.ndim > 2 else actual_num_features), np.array([]), segments.shape[1] if segments.ndim > 1 else expected_seq_len, segments.shape[2] if segments.ndim > 2 else actual_num_features


    # Segments should already have the correct shape (N, L, F) where L and F are based on inputs
    # Final check on shape consistency if needed, but relying on create_labeled_segments logic
    # print(f"Processed patient {patient_id} data shape: {segments.shape}") # Debug shape

    # Return patient_id, segments, labels, and the actual shape parameters derived from processing
    return patient_id, segments, labels, segments.shape[1], segments.shape[2]


# --- PyTorch Dataset ---

class SeizureDataset(Dataset):
    def __init__(self, segments, labels):
        """
        Args:
            segments (np.ndarray): Segments array (n_samples, seq_len, n_features).
                                   Assumes segments are already at the correct shape for the model.
            labels (np.ndarray): Labels array (n_samples,).
        """
        if segments.shape[0] == 0:
             # Determine expected shape from the input segments array, even if empty
             # Relying on prepare_patient_data returning (0, expected_seq_len, expected_num_features)
            seq_len = segments.shape[1] if segments.ndim > 1 else 1 # Fallback to 1
            num_features = segments.shape[2] if segments.ndim > 2 else 1 # Fallback to 1 feature

            self.segments = torch.empty(0, num_features, seq_len, dtype=torch.float32)
            self.labels = torch.empty(0, 1, dtype=torch.float32)
        else:
            # Ensure the segments have the correct number of dimensions (N, L, F)
            if segments.ndim == 2: # (N, L) -> add a feature dim (N, L, 1)
                segments = segments[:, :, np.newaxis]
            elif segments.ndim < 2:
                 print(f"Warning: Segments array has unexpected ndim={segments.ndim}. Cannot create dataset.")
                 # Attempt to infer shape for empty tensors based on what little is available
                 seq_len = segments.shape[1] if segments.ndim > 1 else 1
                 num_features = segments.shape[2] if segments.ndim > 2 else 1
                 self.segments = torch.empty(0, num_features, seq_len, dtype=torch.float32)
                 self.labels = torch.empty(0, 1, dtype=torch.float32)
                 return # Stop init if data is unusable


            self.segments = torch.tensor(segments, dtype=torch.float32).permute(0, 2, 1) # (N, L, F) -> (N, F, L)
            self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1) # (N,) -> (N, 1)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.segments[idx], self.labels[idx]

# --- PyTorch Model Definitions ---

class CNN_LSTM(nn.Module):
    def __init__(self, input_channels, seq_len, conv_filters, conv_kernel_size, pool_size, lstm_units, dense_units):
        super(CNN_LSTM, self).__init__()

        self.input_channels = input_channels
        self.seq_len = seq_len
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.pool_size = pool_size
        self.lstm_units = lstm_units
        self.dense_units = dense_units


        if input_channels <= 0:
            raise ValueError(f"Input channels must be positive, got {input_channels}")
        if seq_len <= 0:
             raise ValueError(f"Sequence length must be positive, got {seq_len}")
        if not conv_filters or any(f <= 0 for f in conv_filters):
             raise ValueError(f"Conv filters must be a list of positive integers, got {conv_filters}")
        if conv_kernel_size <= 0:
             raise ValueError(f"Conv kernel size must be positive, got {conv_kernel_size}")
        if pool_size <= 0:
             raise ValueError(f"Pool size must be positive, got {pool_size}")
        if lstm_units <= 0:
             raise ValueError(f"LSTM units must be positive, got {lstm_units}")
        if dense_units <= 0:
             raise ValueError(f"Dense units must be positive, got {dense_units}")


        conv_layers_list = []
        in_channels = input_channels
        current_seq_len = seq_len

        for i, out_channels in enumerate(conv_filters):
            padding = conv_kernel_size // 2 # 'same' padding approximation
            conv_layers_list.append(nn.Conv1d(in_channels, out_channels, kernel_size=conv_kernel_size, padding=padding))
            conv_layers_list.append(nn.ReLU())
            conv_layers_list.append(nn.MaxPool1d(pool_size))
            in_channels = out_channels

            # Recalculate sequence length after conv and pool
            current_seq_len = math.floor((current_seq_len + 2 * padding - (conv_kernel_size - 1) - 1) / 1 + 1) # Conv1d output length
            current_seq_len = math.floor((current_seq_len + 2 * 0 - (pool_size - 1) - 1) / pool_size + 1) # MaxPool1d output length


        self.conv_layers = nn.Sequential(*conv_layers_list)

        # Calculate the output sequence length after CNN layers dynamically
        try:
            # Create a dummy tensor on the CPU for shape calculation
            dummy_input = torch.randn(1, self.input_channels, self.seq_len, dtype=torch.float32)
            dummy_output = self.conv_layers(dummy_input)
            self.lstm_input_features = dummy_output.shape[1]
            self.lstm_input_seq_len = dummy_output.shape[2]

            if self.lstm_input_seq_len <= 0:
                 # This can happen if seq_len is too short for the filters/pooling
                 raise ValueError(f"Calculated LSTM input sequence length is zero or negative ({self.lstm_input_seq_len}). Check CNN/Pooling parameters relative to segment length.")

        except Exception as e:
             print(f"Error calculating layer output size during model init for {self.__class__.__name__} with input_channels={self.input_channels}, seq_len={self.seq_len}: {e}")
             # Re-raise the exception after printing
             raise e


        self.lstm = nn.LSTM(input_size=self.lstm_input_features,
                            hidden_size=lstm_units,
                            batch_first=True)

        self.dense_layers = nn.Sequential(
            nn.Linear(lstm_units, dense_units),
            nn.Sigmoid(),
            nn.Linear(dense_units, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        cnn_out = self.conv_layers(x)
        # Handle potential empty output after CNN if input_channels or seq_len was too small
        if cnn_out.shape[2] == 0:
             # Or return a default value, depending on desired behavior
             # Returning 0.5 (sigmoid output) might be reasonable for prediction task
             return torch.tensor([[0.5]] * x.size(0), device=x.device) # Return neutral predictions if seq_len collapses


        lstm_in = cnn_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_in)
        last_timestep_out = lstm_out[:, -1, :]
        output = self.dense_layers(last_timestep_out)
        return output


class CNN_BiLSTM(nn.Module):
    def __init__(self, input_channels, seq_len, conv_filters, conv_kernel_size, pool_size, lstm_units, dense_units):
        super(CNN_BiLSTM, self).__init__()

        self.input_channels = input_channels
        self.seq_len = seq_len
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.pool_size = pool_size
        self.lstm_units = lstm_units
        self.dense_units = dense_units

        if input_channels <= 0:
            raise ValueError(f"Input channels must be positive, got {input_channels}")
        if seq_len <= 0:
             raise ValueError(f"Sequence length must be positive, got {seq_len}")
        if not conv_filters or any(f <= 0 for f in conv_filters):
             raise ValueError(f"Conv filters must be a list of positive integers, got {conv_filters}")
        if conv_kernel_size <= 0:
             raise ValueError(f"Conv kernel size must be positive, got {conv_kernel_size}")
        if pool_size <= 0:
             raise ValueError(f"Pool size must be positive, got {pool_size}")
        if lstm_units <= 0:
             raise ValueError(f"LSTM units must be positive, got {lstm_units}")
        if dense_units <= 0:
             raise ValueError(f"Dense units must be positive, got {dense_units}")


        conv_layers_list = []
        in_channels = input_channels
        current_seq_len = seq_len

        for i, out_channels in enumerate(conv_filters):
            padding = conv_kernel_size // 2
            conv_layers_list.append(nn.Conv1d(in_channels, out_channels, kernel_size=conv_kernel_size, padding=padding))
            conv_layers_list.append(nn.ReLU())
            conv_layers_list.append(nn.MaxPool1d(pool_size))
            in_channels = out_channels

            # Recalculate sequence length after conv and pool
            current_seq_len = math.floor((current_seq_len + 2 * padding - (conv_kernel_size - 1) - 1) / 1 + 1) # Conv1d output length
            current_seq_len = math.floor((current_seq_len + 2 * 0 - (pool_size - 1) - 1) / pool_size + 1) # MaxPool1d output length


        self.conv_layers = nn.Sequential(*conv_layers_list)

        # Calculate the output sequence length after CNN layers dynamically
        try:
            # Create a dummy tensor on the CPU for shape calculation
            dummy_input = torch.randn(1, self.input_channels, self.seq_len, dtype=torch.float32)
            dummy_output = self.conv_layers(dummy_input)
            self.lstm_input_features = dummy_output.shape[1]
            self.lstm_input_seq_len = dummy_output.shape[2]

            if self.lstm_input_seq_len <= 0:
                 # This can happen if seq_len is too short for the filters/pooling
                 raise ValueError(f"Calculated LSTM input sequence length is zero or negative ({self.lstm_input_seq_len}). Check CNN/Pooling parameters relative to segment length.")

        except Exception as e:
             print(f"Error calculating layer output size during model init for {self.__class__.__name__} with input_channels={self.input_channels}, seq_len={self.seq_len}: {e}")
             # Re-raise the exception after printing
             raise e


        self.bilstm = nn.LSTM(input_size=self.lstm_input_features,
                              hidden_size=lstm_units,
                              batch_first=True,
                              bidirectional=True)

        self.dense_layers = nn.Sequential(
            nn.Linear(lstm_units * 2, dense_units),
            nn.Sigmoid(),
            nn.Linear(dense_units, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        cnn_out = self.conv_layers(x)
         # Handle potential empty output after CNN if input_channels or seq_len was too small
        if cnn_out.shape[2] == 0:
             # Or return a default value, depending on desired behavior
             return torch.tensor([[0.5]] * x.size(0), device=x.device) # Return neutral predictions if seq_len collapses


        lstm_in = cnn_out.permute(0, 2, 1)
        bilstm_out, _ = self.bilstm(lstm_in)
        last_timestep_out = bilstm_out[:, -1, :]
        output = self.dense_layers(last_timestep_out)
        return output


def get_model_class(model_type):
    """Returns the model class based on the model type string."""
    if model_type == 'CNN-LSTM':
        return CNN_LSTM
    elif model_type == 'CNN-BiLSTM':
        return CNN_BiLSTM
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# --- PyTorch Training and Evaluation ---

def train_one_epoch(model, dataloader, criterion, optimizer, device, class_weights=None):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0

    dataloader_tqdm = tqdm(dataloader, desc="Batch", leave=False)

    for inputs, labels in dataloader_tqdm:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        if class_weights is not None:
            weight_tensor = torch.zeros_like(labels)
            # Ensure class_weights are indexed correctly (0 for class 0, 1 for class 1)
            weight_tensor[labels == 0] = class_weights.get(0, 1.0) # Default to 1.0 if class not in weights
            weight_tensor[labels == 1] = class_weights.get(1, 1.0)
            loss = criterion(outputs, labels)
            loss = (loss * weight_tensor).mean()
        else:
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        dataloader_tqdm.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(dataloader.dataset)

    return epoch_loss

def evaluate_pytorch_model(model, dataloader, criterion, device):
    """Evaluates the model on a given dataloader."""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probs = []

    # Handle empty dataloader gracefully
    if len(dataloader.dataset) == 0:
         return {
            'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
            'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]
        }

    dataloader_tqdm = tqdm(dataloader, desc="Evaluating Batch", leave=False)

    with torch.no_grad():
        for inputs, labels in dataloader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            predicted = (outputs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())

            dataloader_tqdm.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(dataloader.dataset)

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)

    # Ensure metrics are calculated only if there are samples
    if len(all_labels) == 0:
         return {
            'loss': epoch_loss, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
            'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]
        }

    accuracy = accuracy_score(all_labels, all_predictions)
    # Handle cases where precision/recall/f1 might be undefined (e.g., no positive predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)

    try:
        # roc_auc_score requires at least two classes present in the evaluation set labels
        if len(np.unique(all_labels)) > 1:
            auc_roc = roc_auc_score(all_labels, all_probs)
        else:
            auc_roc = 0.0 # AUC-ROC is undefined for single class
            # print("Warning: Only one class present in evaluation set, AUC-ROC is undefined.") # Suppress frequent warning

    except ValueError: # Catch other potential ValueError
        auc_roc = 0.0
        # print("Warning: Could not compute AUC-ROC (e.g., invalid probabilities).") # Suppress frequent warning


    cm = confusion_matrix(all_labels, all_predictions).tolist()

    return {
        'loss': epoch_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'confusion_matrix': cm
    }


def train_pytorch_model(model, train_dataloader, val_dataloader, epochs, learning_rate, class_weights=None, save_best_model_path=None, desc="Training"):
    """Main training loop with validation, early stopping, and LR scheduling."""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=0.0001) # Removed verbose

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    # Only attempt training if there is data in the training dataloader
    if len(train_dataloader.dataset) == 0:
         print("Warning: Training dataloader is empty. Skipping training.")
         # Return the initial model state, no training occurred
         return model

    epoch_tqdm = tqdm(range(epochs), desc=desc, leave=True)

    for epoch in epoch_tqdm:
        start_time = time.time()

        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, DEVICE, class_weights)

        val_metrics = evaluate_pytorch_model(model, val_dataloader, criterion, DEVICE)
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']

        end_time = time.time()

        epoch_tqdm.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc:.4f}", time=f"{end_time - start_time:.2f}s")

        # Only step the scheduler if validation data is available
        if len(val_dataloader.dataset) > 0:
            scheduler.step(val_loss)

        # Only check for improvement if validation data is available
        if len(val_dataloader.dataset) > 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
            if save_best_model_path:
                 try:
                     torch.save(best_model_state, save_best_model_path)
                 except Exception as e:
                     print(f"Warning: Could not save best model state to {save_best_model_path}: {e}")

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= 5:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    # Load best weights before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        # This happens if val_loss never improved or val_dataloader was empty
        print("Warning: No best model state was saved during training (either val data empty or no improvement).")

    return model


def train_general_model_pytorch(segments, labels, model_type, conv_filters, conv_kernel_size, pool_size, lstm_units, dense_units):
    """
    Trains a general model on combined data using PyTorch.
    Accepts segments and labels directly. Uses Batch size and LR from HP.
    """
    print("\n--- Training General Model (PyTorch) ---")

    # Check for sufficient data and classes at the very start
    if len(segments) == 0 or len(labels) == 0 or len(np.unique(labels)) < 2:
        print("No data, labels, or only one class available for general training combination.")
        # Return None for model and empty metrics/dataloader
        return None, {}, None # Return None for dataloader


    print(f"Combined data shape: {segments.shape}, Labels shape: {labels.shape}")
    print(f"Combined data: Pre-ictal: {np.sum(labels)}, Interictal: {len(labels) - np.sum(labels)}")


    classes = np.unique(labels)
    if len(classes) == 2:
         class_weights_np = class_weight.compute_class_weight(
             'balanced', classes=classes, y=labels
         )
         class_weight_dict = {int(cls): float(weight) for cls, weight in zip(classes, class_weights_np)}
         print(f"Computed general class weights: {class_weight_dict}")
    else:
         class_weight_dict = None


    # Ensure enough data for splitting
    if len(segments) < 3: # Need at least 3 samples for 60/20/20 split (even with min_samples=1)
         print(f"Warning: Not enough data ({len(segments)} samples) for general training split. Skipping training.")
         return None, {}, None # Return None for dataloader


    # First split: Train vs Temp (Val+Test)
    try:
        X_train_general, X_temp, y_train_general, y_temp = train_test_split(
            segments, labels, test_size=0.4, random_state=SEED, stratify=labels
        )
    except ValueError as e:
         print(f"Warning: First data split failed: {e}. This might happen with very few samples or severe class imbalance. Skipping training.")
         return None, {}, None # Return None for dataloader


    # Ensure temp split has enough data and both classes for the next split
    if len(X_temp) == 0:
         print("Warning: Train split resulted in empty temp set. Cannot split further. Skipping training.")
         return None, {}, None # Return None for dataloader

    # Add check for number of classes in y_temp before the second split
    if len(np.unique(y_temp)) < 2:
         print(f"Warning: Temp set for validation/test split contains only one class ({np.unique(y_temp)}). Cannot perform stratified split. Skipping training.")
         return None, {}, None # Return None for dataloader


    # Second split: Val vs Test from Temp
    try:
        X_val_general, X_test_general, y_val_general, y_test_general = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
        )
    except ValueError as e:
         print(f"Warning: Second data split failed: {e}. This might happen if the temp set is too small for stratified splitting (e.g., very few samples per class). Skipping training.")
         return None, {}, None # Return None for dataloader


    # Ensure all resulting splits are not empty
    if len(X_train_general) == 0 or len(X_val_general) == 0 or len(X_test_general) == 0:
        print(f"Warning: Data split resulted in empty train ({len(X_train_general)}), val ({len(X_val_general)}), or test ({len(X_test_general)}) set. Skipping training.")
        return None, {}, None # Return None for dataloader


    print(f"General Training data shape: {X_train_general.shape}, Labels shape: {y_train_general.shape}")
    print(f"General Validation data shape: {X_val_general.shape}, Labels shape: {y_val_general.shape}")
    print(f"General Test data shape: {X_test_general.shape}, Labels shape: {y_test_general.shape}")

    train_dataset = SeizureDataset(X_train_general, y_train_general)
    val_dataset = SeizureDataset(X_val_general, y_val_general)
    test_dataset = SeizureDataset(X_test_general, y_test_general) # test_dataset is created here


    # Initialize dataloaders to None before creation in case of errors
    train_dataloader = None
    val_dataloader = None
    test_dataloader = None # <-- Initialize here

    num_workers = os.cpu_count() // 2 or 1
    persistent_workers = True if num_workers > 0 and hasattr(DataLoader, 'persistent_workers') else False

    # Get batch sizes and learning rate from current HP combination
    general_train_batch_size = current_hyperparameters['batch_size'] # General train/val batch size
    general_learning_rate = current_hyperparameters['general_lr']
    general_epochs = current_hyperparameters['general_epochs']

    # Add checks for valid batch sizes
    if general_train_batch_size <= 0:
        print(f"Error: Invalid general_train_batch_size ({general_train_batch_size}). Skipping training.")
        return None, {}, None # Return None for dataloader

    # Adjust val batch size if necessary
    general_val_batch_size = general_train_batch_size
    if len(val_dataset) > 0 and general_val_batch_size > len(val_dataset):
         print(f"Warning: General training batch_size ({general_train_batch_size}) is larger than validation dataset size ({len(val_dataset)}). Using val dataset size for val batch.")
         general_val_batch_size = len(val_dataset)

    # General test batch size (can be the same as train/val)
    general_test_batch_size = general_train_batch_size
    # Adjust test batch size if necessary
    if len(test_dataset) > 0 and general_test_batch_size > len(test_dataset):
         print(f"Warning: General test batch_size ({general_test_batch_size}) is larger than test dataset size ({len(test_dataset)}). Using test dataset size for test batch.")
         general_test_batch_size = len(test_dataset)
    # Ensure test_batch_size is at least 1 if dataset has samples
    if len(test_dataset) > 0 and general_test_batch_size <= 0:
         general_test_batch_size = 1


    try:
        train_dataloader = DataLoader(train_dataset, batch_size=general_train_batch_size, shuffle=True, num_workers=num_workers, persistent_workers=persistent_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=general_val_batch_size, shuffle=False, num_workers=num_workers, persistent_workers=persistent_workers)
        test_dataloader = DataLoader(test_dataset, batch_size=general_test_batch_size, shuffle=False, num_workers=num_workers, persistent_workers=persistent_workers) # <-- Assigned here
    except Exception as e:
        print(f"Error creating general data loaders: {e}. Skipping training.")
        # Clean up any loaders that might have been created
        del train_dataloader, val_dataloader, test_dataloader
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return None, {}, None # Return None for dataloader


    input_channels = segments.shape[2]
    seq_len = segments.shape[1]

    # Check if model parameters are valid for instantiation
    if input_channels <= 0 or seq_len <= 0 or not conv_filters or any(f <= 0 for f in conv_filters) or conv_kernel_size <= 0 or pool_size <= 0 or lstm_units <= 0 or dense_units <= 0:
        print(f"Error: Invalid model parameters for instantiation. input_channels={input_channels}, seq_len={seq_len}, conv_filters={conv_filters}, etc. Skipping training.")
        # Clean up dataloaders before returning
        del train_dataloader, val_dataloader, test_dataloader
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return None, {}, None # Return None for dataloader


    ModelClass = get_model_class(model_type)
    try:
        general_model = ModelClass(input_channels, seq_len, conv_filters, conv_kernel_size, pool_size, lstm_units, dense_units).to(DEVICE)
    except ValueError as e:
        print(f"Error instantiating model: {e}. Skipping training.")
        # Clean up dataloaders before returning
        del train_dataloader, val_dataloader, test_dataloader
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return None, {}, None # Return None for dataloader
    except Exception as e:
        print(f"An unexpected error occurred during model instantiation: {e}. Skipping training.")
        # Clean up dataloaders before returning
        del train_dataloader, val_dataloader, test_dataloader
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return None, {}, None # Return None for dataloader


    general_model_path = os.path.join(OUTPUT_DIR, 'temp_general_best_model.pth') # Use a temp name as actual saving is per combination


    # Only train if the train dataloader has data
    if train_dataloader is not None and len(train_dataloader.dataset) > 0:
        general_model = train_pytorch_model(
            general_model,
            train_dataloader,
            val_dataloader,
            epochs=general_epochs, # Use general epochs
            learning_rate=general_learning_rate, # Use general LR
            class_weights=class_weight_dict,
            save_best_model_path=general_model_path,
            desc=f"Training General ({model_type})"
        )
    else:
        print("Warning: Training dataloader is empty. Skipping general model training.")
        # If training is skipped, the initial model is returned.


    print(f"\nEvaluating general model ({model_type}) on combined test set...")
    # Only evaluate if test_dataloader has data
    if test_dataloader is not None and len(test_dataloader.dataset) > 0:
        general_metrics = evaluate_pytorch_model(general_model, test_dataloader, nn.BCELoss(), DEVICE)
        print(f"General Model Metrics: Accuracy={general_metrics['accuracy']:.4f}, "
              f"Precision={general_metrics['precision']:.4f}, Recall={general_metrics['recall']:.4f}, "
              f"F1 Score={general_metrics['f1_score']:.4f}, AUC-ROC={general_metrics['auc_roc']:.4f}")
        # print(f"Confusion Matrix:\n{general_metrics['confusion_matrix']}") # Suppress frequent print)
    else:
        print("Test dataloader is empty. Skipping general model evaluation.")
        general_metrics = {} # Return empty metrics
        # test_dataloader remains None or is already an empty DataLoader instance


    # Clean up memory
    # del train_dataloader, val_dataloader # Don't delete here, done in train_pytorch_model
    # test_dataloader is returned, do NOT delete here
    gc.collect()
    if torch.cuda.is_available():
         torch.cuda.empty_cache()

    return general_model, general_metrics, test_dataloader # <-- Return test_dataloader

def perform_personalization_pytorch(general_model_state_dict, all_patient_data, model_type, conv_filters, conv_kernel_size, pool_size, lstm_units, dense_units, personalization_epochs, personalization_learning_rate, personalization_batch_size):
    """
    Performs personalization for each patient using transfer learning (fine-tuning) in PyTorch.
    Accepts hyperparameters directly. all_patient_data contains tuples (patient_id, segments, labels, seq_len, num_features).
    """
    print("\n--- Performing Personalization (PyTorch) ---")

    if general_model_state_dict is None:
        print("General model state dict is missing. Cannot perform personalization.")
        return {}

    if not all_patient_data:
         print("No patient data available for personalization.")
         return {}


    personalization_results = {}
    ModelClass = get_model_class(model_type) # Get the correct model class


    # Wrap patient loop with tqdm
    patient_tqdm = tqdm(all_patient_data, desc="Personalizing Patients", leave=True)

    for patient_data_tuple in patient_tqdm: # Iterate through the list of patient data tuples
        patient_id, patient_segments, patient_labels, patient_seq_len, patient_num_features = patient_data_tuple # Unpack the full tuple


        patient_tqdm.set_description(f"Personalizing Patient {patient_id}")

        # Check if patient data for this combination is usable for splitting/training
        if len(patient_segments) == 0 or len(np.unique(patient_labels)) < 2:
             print(f"Skipping patient {patient_id}: No valid segments or only one class.")
             # Add entry to results with empty metrics
             personalization_results[patient_id] = {
                 "before": {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]},
                 "after": {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]}
             }
             # Clean up any potential temporary data/models if they existed
             gc.collect()
             if torch.cuda.is_available(): torch.cuda.empty_cache()
             continue


        # Ensure enough data for splitting
        if len(patient_segments) < 3: # Need at least 3 samples for 60/20/20 split
            print(f"Warning: Not enough data ({len(patient_segments)} samples) for patient {patient_id} personalization split. Skipping.")
            personalization_results[patient_id] = {
                 "before": {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]},
                 "after": {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]}
            }
            continue


        # Splitting patient data for personalization (Train, Val, Test)
        try:
            X_train_pat, X_temp_pat, y_train_pat, y_temp_pat = train_test_split(
                 patient_segments, patient_labels, test_size=0.4, random_state=SEED, stratify=patient_labels
            )
        except ValueError as e:
             print(f"Warning: Patient {patient_id} first data split failed: {e}. Skipping personalization.")
             personalization_results[patient_id] = {
                  "before": {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]},
                  "after": {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]}
             }
             continue


        if len(X_temp_pat) == 0:
             print(f"Warning: Patient {patient_id} train split resulted in empty temp set. Cannot split further. Skipping personalization.")
             personalization_results[patient_id] = {
                  "before": {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]},
                  "after": {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]}
             }
             continue

        if len(np.unique(y_temp_pat)) < 2:
            print(f"Warning: Patient {patient_id} temp set for validation/test split contains only one class ({np.unique(y_temp_pat)}). Skipping personalization.")
            personalization_results[patient_id] = {
                  "before": {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]},
                  "after": {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]}
            }
            continue


        try:
            X_val_pat, X_test_pat, y_val_pat, y_test_pat = train_test_split(
                 X_temp_pat, y_temp_pat, test_size=0.5, random_state=SEED, stratify=y_temp_pat
            )
        except ValueError as e:
            print(f"Warning: Patient {patient_id} second data split failed: {e}. Skipping personalization.")
            personalization_results[patient_id] = {
                  "before": {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]},
                  "after": {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]}
            }
            continue


        # Ensure all resulting splits are not empty
        if len(X_train_pat) == 0 or len(X_val_pat) == 0 or len(X_test_pat) == 0:
             print(f"Warning: Patient {patient_id} data split resulted in empty train ({len(X_train_pat)}), val ({len(X_val_pat)}), or test ({len(X_test_pat)}) set. Skipping personalization.")
             personalization_results[patient_id] = {
                  "before": {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]},
                  "after": {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]}
             }
             continue


        # print(f"Patient {patient_id} - Personalization Train shape: {X_train_pat.shape}, Val shape: {X_val_pat.shape}, Test shape: {X_test_pat.shape}") # Suppress frequent print)

        train_dataset_pat = SeizureDataset(X_train_pat, y_train_pat)
        val_dataset_pat = SeizureDataset(X_val_pat, y_val_pat)
        test_dataset_pat = SeizureDataset(X_test_pat, y_test_pat)

        # Initialize dataloaders to None before creation
        train_dataloader_pat = None
        val_dataloader_pat = None
        test_dataloader_pat = None


        num_workers_pat = os.cpu_count() // 4 or 1
        persistent_workers_pat = True if num_workers_pat > 0 and hasattr(DataLoader, 'persistent_workers') else False

        # Use batch_size and learning_rate from current HP combination (passed as args)
        current_personalization_batch_size = personalization_batch_size # Local variable for clarity
        current_personalization_learning_rate = personalization_learning_rate # Local variable for clarity
        current_personalization_epochs = personalization_epochs # Local variable for clarity


        # Add checks for valid batch sizes
        if current_personalization_batch_size <= 0:
             print(f"Error: Patient {patient_id} invalid personalization_batch_size ({current_personalization_batch_size}). Skipping.")
             personalization_results[patient_id] = { # Log empty results
                  "before": {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]},
                  "after": {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]}
             }
             continue


        # Adjust train/val batch size if necessary
        personalization_train_batch_size = current_personalization_batch_size
        if len(train_dataset_pat) > 0 and personalization_train_batch_size > len(train_dataset_pat):
             print(f"Warning: Patient {patient_id} personalization train batch_size ({personalization_train_batch_size}) is larger than train dataset size ({len(train_dataset_pat)}). Using dataset size.")
             personalization_train_batch_size = len(train_dataset_pat)
        # Adjust val batch size if necessary
        personalization_val_batch_size = current_personalization_batch_size
        if len(val_dataset_pat) > 0 and personalization_val_batch_size > len(val_dataset_pat):
             print(f"Warning: Patient {patient_id} personalization val batch_size ({personalization_val_batch_size}) is larger than val dataset size ({len(val_dataset_pat)}). Using dataset size.")
             personalization_val_batch_size = len(val_dataset_pat)

        # Personalization test batch size (use the personalization training batch size)
        personalized_test_batch_size = current_personalization_batch_size
        # Adjust test batch size if necessary
        if len(test_dataset_pat) > 0 and personalized_test_batch_size > len(test_dataset_pat):
            print(f"Warning: Patient {patient_id} personalized test batch_size ({personalized_test_batch_size}) is larger than test dataset size ({len(test_dataset_pat)}). Using dataset size.")
            personalized_test_batch_size = len(test_dataset_pat)
        # Ensure test_batch_size is at least 1 if dataset has samples
        if len(test_dataset_pat) > 0 and personalized_test_batch_size <= 0:
            personalized_test_batch_size = 1


        try:
            train_dataloader_pat = DataLoader(train_dataset_pat, batch_size=personalization_train_batch_size, shuffle=True, num_workers=num_workers_pat, persistent_workers=persistent_workers_pat)
            val_dataloader_pat = DataLoader(val_dataset_pat, batch_size=personalization_val_batch_size, shuffle=False, num_workers=num_workers_pat, persistent_workers=persistent_workers_pat)
            # CORRECTED LINE: Use the personalization batch size
            test_dataloader_pat = DataLoader(test_dataset_pat, batch_size=personalized_test_batch_size, shuffle=False, num_workers=num_workers_pat, persistent_workers=persistent_workers_pat) # <-- Corrected

        except Exception as e:
            print(f"Error creating patient {patient_id} data loaders: {e}. Skipping personalization.")
            # Clean up any loaders that might have been created
            del train_dataloader_pat, val_dataloader_pat, test_dataloader_pat
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            personalization_results[patient_id] = { # Log empty results
                 "before": {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]},
                 "after": {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]}
            }
            continue # Skip to next patient


        # Evaluate the general model on this patient's test data (Before Personalization)
        # print(f"Evaluating general model on patient {patient_id}'s test data (Before Personalization)...") # Suppress frequent print)
        input_channels = patient_num_features # Use num_features from prepare_patient_data tuple
        seq_len = patient_seq_len           # Use seq_len from prepare_patient_data tuple


        # Check if model parameters are valid for instantiation
        if input_channels <= 0 or seq_len <= 0 or not conv_filters or any(f <= 0 for f in conv_filters) or conv_kernel_size <= 0 or pool_size <= 0 or lstm_units <= 0 or dense_units <= 0:
            print(f"Error: Invalid model parameters for instantiation in personalization. Skipping patient {patient_id}. input_channels={input_channels}, seq_len={seq_len}, etc.")
             # Clean up dataloaders before returning
            del train_dataloader_pat, val_dataloader_pat, test_dataloader_pat
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            personalization_results[patient_id] = { # Log empty results
                 "before": {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]},
                 "after": {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]}
            }
            continue # Skip to next patient

        try:
            # Create a temporary model instance for evaluation (won't be trained)
            general_model_instance_eval = ModelClass(input_channels, seq_len, conv_filters, conv_kernel_size, pool_size, lstm_units, dense_units).to(DEVICE)
            general_model_instance_eval.load_state_dict(general_model_state_dict)
            # Only evaluate if test dataloader has data
            if test_dataloader_pat is not None and len(test_dataloader_pat.dataset) > 0:
                metrics_before = evaluate_pytorch_model(general_model_instance_eval, test_dataloader_pat, nn.BCELoss(), DEVICE)
            else:
                print(f"Warning: Patient {patient_id} test dataloader is empty. Skipping 'Before Personalization' evaluation.")
                metrics_before = {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]} # Log empty metrics

            # print(f"Before Personalization: Accuracy={metrics_before['accuracy']:.4f}, Precision={metrics_before['precision']:.4f}, Recall={metrics_before['recall']:.4f}, F1={metrics_before['f1_score']:.4f}, AUC={metrics_before['auc_roc']:.4f}") # Suppress frequent print)
            del general_model_instance_eval
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except (ValueError, RuntimeError) as e:
            print(f"Error instantiating or loading general model state or evaluating for patient {patient_id}: {e}. Skipping patient.")
             # Clean up dataloaders before returning
            del train_dataloader_pat, val_dataloader_pat, test_dataloader_pat
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            personalization_results[patient_id] = { # Log empty results
                 "before": {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]},
                 "after": {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]}
            }
            continue # Skip to next patient


        # Create a new model instance for personalization fine-tuning
        try:
            personalized_model = ModelClass(input_channels, seq_len, conv_filters, conv_kernel_size, pool_size, lstm_units, dense_units).to(DEVICE)
            personalized_model.load_state_dict(general_model_state_dict)
        except (ValueError, RuntimeError) as e:
            print(f"Error instantiating or loading general model state for personalization fine-tuning for patient {patient_id}: {e}. Skipping patient.")
             # Clean up dataloaders before returning
            del train_dataloader_pat, val_dataloader_pat, test_dataloader_pat
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            personalization_results[patient_id] = { # Log empty results, keeping before metrics
                 "before": metrics_before,
                 "after": {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]}
            }
            continue # Skip to next patient


        classes_pat = np.unique(y_train_pat)
        if len(classes_pat) == 2:
             class_weights_pat_np = class_weight.compute_class_weight(
                 'balanced', classes=classes_pat, y=y_train_pat
             )
             class_weights_pat_dict = {int(cls): float(weight) for cls, weight in zip(classes_pat, class_weights_pat_np)}
             # print(f"Computed patient {patient_id} class weights: {class_weights_pat_dict}") # Suppress frequent print)
        else:
             class_weights_pat_dict = None


        # Only attempt fine-tuning if there's training data for this patient
        if train_dataloader_pat is not None and len(train_dataloader_pat.dataset) > 0:
            # Fine-tune the model on the patient's training data
            # print(f"Fine-tuning model on patient {patient_id}'s training data...") # Suppress frequent print)
            personalized_model = train_pytorch_model(
                personalized_model,
                train_dataloader_pat,
                val_dataloader_pat,
                epochs=current_personalization_epochs, # Use personalization epochs
                learning_rate=current_personalization_learning_rate, # Use personalization LR
                class_weights=class_weights_pat_dict,
                save_best_model_path=None, # Don't save per patient during hyperparameter tuning loop
                desc=f"Fine-tuning {patient_id}"
            )
        else:
            print(f"Warning: No training data for patient {patient_id}. Skipping fine-tuning.")
            # The personalized_model instance is already initialized with general weights,
            # it just won't be trained further.

        # Evaluate the personalized model on this patient's test data (After Personalization)
        # print(f"Evaluating personalized model on patient {patient_id}'s test data (After Personalization)...") # Suppress frequent print)
        # Only evaluate if test dataloader has data
        if test_dataloader_pat is not None and len(test_dataloader_pat.dataset) > 0:
            metrics_after = evaluate_pytorch_model(personalized_model, test_dataloader_pat, nn.BCELoss(), DEVICE)
            # print(f"After Personalization: Accuracy={metrics_after['accuracy']:.4f}, Precision={metrics_after['precision']:.4f}, Recall={metrics_after['recall']:.4f}, F1={metrics_after['f1_score']:.4f}, AUC={metrics_after['auc_roc']:.4f}") # Suppress frequent print)
            # print(f"Confusion Matrix:\n{metrics_after['confusion_matrix']}") # Suppress frequent print)
        else:
            print(f"Warning: Patient {patient_id} test dataloader is empty. Skipping 'After Personalization' evaluation.")
            metrics_after = {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]} # Log empty metrics


        personalization_results[patient_id] = {
            "before": metrics_before,
            "after": metrics_after
        }

        # Clean up memory for the current patient's data/model
        del train_dataset_pat, val_dataset_pat, test_dataset_pat
        del train_dataloader_pat, val_dataloader_pat, test_dataloader_pat # <-- Delete dataloaders here
        del personalized_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    return personalization_results

def print_summary_section(title, metrics_dict, output_file):
    """Helper to print a section of the summary to a file."""
    output_file.write(f"\n--- {title} ---\n")
    if not metrics_dict:
        output_file.write("No data available.\n")
        return

    # Determine keys (metrics) from the first entry if it exists
    sample_metrics = next(iter(metrics_dict.values())) if metrics_dict else {}
    # Exclude loss and CM from table view, but they are still in the dict
    metric_keys = [key for key in sample_metrics.keys() if key not in ['loss', 'confusion_matrix']]

    if not metric_keys:
         output_file.write("No metric keys found.\n")
         return

    # Header
    header = "Key"
    separator = "---"
    for key in metric_keys:
        col_header = key.replace('_', ' ').title()
        header += f" | {col_header:<10}" # Pad header for alignment
        separator += "---" + "-" * 10
    output_file.write(header + "\n")
    output_file.write(separator + "\n")

    # Data rows
    for key, metrics in metrics_dict.items():
        row = str(key) # Use key directly (patient ID or model type)
        if metrics:
            for metric_key in metric_keys:
                 # Pad metric value strings for alignment
                 row += f" | {metrics.get(metric_key, 0.0):<10.4f}" # Use .get for safety and pad
        else:
             row += " | N/A       " * len(metric_keys) # Pad N/A for alignment
        output_file.write(row + "\n")
    output_file.write(separator + "\n")


def print_personalization_summary_section(personalization_results, output_file):
    """Prints the per-patient before/after personalization summary to a file."""
    output_file.write("\n--- Personalized Model Performance (Per Patient Summary) ---\n")
    if not personalization_results:
        output_file.write("No data available.\n")
        return

    output_file.write("Patient ID | Accuracy Before | Accuracy After | Change\n")
    output_file.write("-----------------------------------------------------\n")

    for patient_id, results in personalization_results.items():
        if 'before' in results and 'after' in results:
            acc_before = results['before'].get('accuracy', 0.0)
            acc_after = results['after'].get('accuracy', 0.0)
            change = acc_after - acc_before
            output_file.write(f"{patient_id:<10} | {acc_before:<15.4f} | {acc_after:<14.4f} | {change:<6.4f}\n") # Added padding
        else:
            output_file.write(f"{patient_id:<10} | N/A             | N/A            | N/A\n")
    output_file.write("-----------------------------------------------------\n")


# --- Main Execution ---
if __name__ == "__main__":
    # Ensure the base output directory exists at the very beginning
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Moved this up


    all_patient_folders = [f.path for f in os.scandir(DATA_ROOT_DIR) if f.is_dir() and f.name.startswith('MSEL_')]

    if not all_patient_folders:
        print(f"No patient directories starting with 'MSEL_' found in {DATA_ROOT_DIR}.")
        exit()


    # --- Define Hyperparameter Combinations to Run ---
    hp_keys = [
         'sampling_freq_hz', 'pre_ictal_window_min', 'pre_ictal_exclusion_buffer_min', 'post_ictal_buffer_min',
         'conv_filters', 'conv_kernel_size', 'pool_size', 'lstm_units', 'dense_units',
         'general_epochs', 'personalization_epochs', 'general_lr', 'personalization_lr',
         'batch_size', 'personalization_batch_size',
    ]
    hp_lists = [
        TUNABLE_SAMPLING_FREQ_HZ, TUNABLE_PRE_ICTAL_WINDOW_MINUTES, TUNABLE_PRE_ICTAL_EXCLUSION_BUFFER_MINUTES, TUNABLE_POST_ICTAL_BUFFER_MINUTES,
        TUNABLE_CONV_FILTERS, TUNABLE_CONV_KERNEL_SIZE, TUNABLE_POOL_SIZE, TUNABLE_LSTM_UNITS, TUNABLE_DENSE_UNITS,
        TUNABLE_GENERAL_MODEL_EPOCHS, TUNABLE_PERSONALIZATION_EPOCHS, TUNABLE_GENERAL_MODEL_LR, TUNABLE_PERSONALIZATION_LR,
        TUNABLE_BATCH_SIZE, TUNABLE_PERSONALIZATION_BATCH_SIZE,
    ]
    hp_ordered_dict = OrderedDict(zip(hp_keys, hp_lists))


    if ENABLE_HYPERPARAMETER_TUNING:
        hyperparameter_combinations = list(itertools.product(*hp_ordered_dict.values()))
        print(f"\n--- Hyperparameter Tuning Enabled: {len(hyperparameter_combinations)} Combinations to Run ---")
    else:
        # Use only the first value from each list as a single combination
        hyperparameter_combinations = [tuple([lst[0] for lst in hp_lists])]
        print("\n--- Hyperparameter Tuning Disabled: Running with Default Combination ---")


    # --- Loop through each Hyperparameter Combination ---
    total_hp_combos = len(hyperparameter_combinations)

    for hp_index, hp_values in enumerate(tqdm(hyperparameter_combinations, desc="Hyperparameter Combinations", leave=True)):
        # Map hp_values back to their names using the ordered keys
        current_hyperparameters = dict(zip(hp_ordered_dict.keys(), hp_values))

        # --- Validate buffer relationship for this combination BEFORE processing data ---
        if current_hyperparameters['pre_ictal_exclusion_buffer_min'] < current_hyperparameters['pre_ictal_window_min']:
            print(f"Skipping HP combination {hp_index + 1} ({current_hyperparameters}): pre_ictal_exclusion_buffer_min < pre_ictal_window_min.")
            continue # Skip this combination


        print(f"\n{'='*60}")
        print(f"RUNNING HP COMBINATION {hp_index + 1}/{total_hp_combos}")
        print(f"Details: {current_hyperparameters}") # Print full details here
        print(f"{'='*60}\n")

        # Generate a unique filename for this HP combination (shortened)
        # Format: hp_combo_X_timestamp.txt
        output_filename = os.path.join(OUTPUT_DIR, f"hp_combo_{hp_index+1}_{time.strftime('%Y%m%d_%H%M%S')}.txt")

        # Create parent directory if it doesn't exist - Redundant if OUTPUT_DIR is created at top, but harmless.
        # os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        # Diagnostic prints - REMOVE OR COMMENT OUT FOR LONG RUNS
        output_dir_for_file = os.path.dirname(output_filename)
        print(f"Attempting to create directory: {output_dir_for_file}")
        print(f"Attempting to open file: {output_filename}")


        # Use 'with' to ensure the file is closed automatically
        try:
            with open(output_filename, 'w') as output_file: # Use 'w' (write) for a new file per HP combination

                # Write initial file headers and HP details (full details)
                output_file.write(f"Seizure Prediction Results - Hyperparameter Combination {hp_index + 1}\n")
                output_file.write(f"Run Date: {time.ctime()}\n")
                output_file.write(f"Data Directory: {DATA_ROOT_DIR}\n")
                output_file.write(f"Total Patient Directories Found: {len(all_patient_folders)}\n")
                output_file.write(f"Models Run: {MODEL_TYPES_TO_RUN}\n")
                output_file.write(f"Sensor Combination Testing Enabled: {ENABLE_ADAPTIVE_SENSORS}\n")
                output_file.write(f"Hyperparameter Tuning Enabled: {ENABLE_HYPERPARAMETER_TUNING}\n")
                output_file.write("\nHyperparameters Used in this Combination:\n")
                for key, value in current_hyperparameters.items():
                    output_file.write(f"  {key}: {value}\n")
                output_file.write("=" * 80 + "\n\n")


                # --- Loop through each Sensor Combination ---
                # Use sorted keys to ensure consistent order in the output file
                for sensor_list in sorted(SENSOR_COMBINATIONS):
                    sensor_combination_name = "_".join(sensor_list)

                    print(f"\n--- Processing and Running Sensor Combination: {sensor_combination_name} ---")
                    output_file.write(f"\n{'#'*60}\n")
                    output_file.write(f"RESULTS FOR SENSOR COMBINATION: {sensor_combination_name}\n")
                    output_file.write(f"{'#'*60}\n\n")

                    # --- Process Data for this Sensor/Sampling Frequency/Buffer Combination ---
                    # This happens *inside* the HP loop and Sensor loop now
                    processed_patient_data = []
                    patients_processed_count = 0
                    for patient_folder in tqdm(all_patient_folders, desc=f"Processing Patients for {sensor_combination_name} (HP Combo {hp_index+1})", leave=False):
                        patient_data_tuple = prepare_patient_data(
                            patient_folder,
                            sensors_to_use=sensor_list,
                            pre_ictal_window_min=current_hyperparameters['pre_ictal_window_min'],
                            pre_ictal_exclusion_buffer_min=current_hyperparameters['pre_ictal_exclusion_buffer_min'],
                            post_ictal_buffer_min=current_hyperparameters['post_ictal_buffer_min'],
                            sampling_freq_hz=current_hyperparameters['sampling_freq_hz']
                        )
                        # prepare_patient_data returns (patient_id, segments, labels, expected_seq_len, expected_num_features) or None
                        # We only append if the result is not None and segments are not empty (index 1 is the segments array)
                        if patient_data_tuple is not None and len(patient_data_tuple[1]) > 0:
                            processed_patient_data.append(patient_data_tuple)
                            patients_processed_count += 1
                        # else:
                            # print(f"Skipping patient {os.path.basename(patient_folder)} for {sensor_combination_name} (no data or segments).") # Suppress frequent print)


                    if not processed_patient_data:
                        output_file.write("No valid patient data processed or created segments for this sensor/HP combination.\n\n")
                        print("No valid patient data processed or created segments for this sensor/HP combination.")
                        continue # Skip this sensor combo for this HP combination


                    output_file.write(f"Using data from {len(processed_patient_data)} patients for this sensor combination.\n\n")

                    # Get combined data and actual dimensions from the processed data of valid patients
                    # All patients processed for this combo should have segments of the same shape
                    combined_segments = np.concatenate([data[1] for data in processed_patient_data], axis=0)
                    combined_labels = np.concatenate([data[2] for data in processed_patient_data], axis=0)
                    # actual_seq_len = processed_patient_data[0][3] # Not strictly needed here, inferred by dataset
                    # actual_num_features = processed_patient_data[0][4] # Not strictly needed here, inferred by dataset


                    # --- Loop through each Model Type ---
                    for current_model_type in MODEL_TYPES_TO_RUN:
                        print(f"\n--- Running Model Type: {current_model_type} ---")
                        output_file.write(f"\n--- Model Type: {current_model_type} ---\n\n")


                        # --- Train General Model ---
                        # Pass combined data and current hyperparameters, using actual data shape
                        general_model, general_metrics, general_test_dataloader = train_general_model_pytorch(
                            combined_segments,
                            combined_labels,
                            model_type=current_model_type,
                            conv_filters=current_hyperparameters['conv_filters'],
                            conv_kernel_size=current_hyperparameters['conv_kernel_size'],
                            pool_size=current_hyperparameters['pool_size'],
                            lstm_units=current_hyperparameters['lstm_units'],
                            dense_units=current_hyperparameters['dense_units']
                            # epochs, lr, batch_size are read inside train_general_model_pytorch from current_hyperparameters
                        )

                        # Write General Model Performance to file
                        output_file.write("--- General Model Performance ---\n")
                        if general_model is not None and 'accuracy' in general_metrics:
                            output_file.write(f"Accuracy={general_metrics['accuracy']:.4f}\n")
                            output_file.write(f"Precision={general_metrics['precision']:.4f}\n")
                            output_file.write(f"Recall={general_metrics['recall']:.4f}\n")
                            output_file.write(f"F1 Score={general_metrics['f1_score']:.4f}\n")
                            output_file.write(f"AUC-ROC={general_metrics['auc_roc']:.4f}\n")
                            output_file.write(f"Confusion Matrix:\n{general_metrics['confusion_matrix']}\n")
                        else:
                            output_file.write("General model training failed or produced invalid metrics.\n")
                        output_file.write("\n")


                        if general_model is not None:
                            general_model_state = general_model.state_dict()

                            del general_model # Delete model instance
                            # Delete dataloader if it exists and is not None
                            if general_test_dataloader is not None:
                                del general_test_dataloader
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()


                            # --- Perform Personalization ---
                            # Pass the list of processed patient data for THIS sensor/HP combination
                            personalization_results = perform_personalization_pytorch(
                                general_model_state,
                                processed_patient_data, # Pass the list of (patient_id, segments, labels, ...) tuples
                                model_type=current_model_type,
                                conv_filters=current_hyperparameters['conv_filters'],
                                conv_kernel_size=current_hyperparameters['conv_kernel_size'],
                                pool_size=current_hyperparameters['pool_size'],
                                lstm_units=current_hyperparameters['lstm_units'],
                                dense_units=current_hyperparameters['dense_units'],
                                # ADD THE MISSING ARGUMENTS HERE:
                                personalization_epochs=current_hyperparameters['personalization_epochs'],
                                personalization_learning_rate=current_hyperparameters['personalization_lr'],
                                personalization_batch_size=current_hyperparameters['personalization_batch_size']
                            )

                            # --- Summarize Personalized Model Performance ---
                            # Write per-patient summary to file
                            print_personalization_summary_section(personalization_results, output_file=output_file)

                            # Calculate and Write Average Personalized Model Performance
                            metrics_after_list = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'auc_roc': []}
                            count_valid_patients = 0

                            for patient_id, results in personalization_results.items():
                                # Check if 'after' metrics exist and if the confusion matrix indicates actual evaluation
                                if 'after' in results and isinstance(results['after'], dict): # Ensure 'after' is a dict
                                    cm = results['after'].get('confusion_matrix', [[0,0],[0,0]]) # Safely get CM
                                    # Check if confusion matrix indicates samples were evaluated (>0 total samples)
                                    if isinstance(cm, list) and len(cm) == 2 and len(cm[0]) == 2 and sum(sum(row) for row in cm) > 0: # Check CM structure and sum
                                        count_valid_patients += 1
                                        metrics_after_list['accuracy'].append(results['after']['accuracy'])
                                        metrics_after_list['precision'].append(results['after']['precision'])
                                        metrics_after_list['recall'].append(results['after']['recall'])
                                        metrics_after_list['f1_score'].append(results['after']['f1_score'])
                                        metrics_after_list['auc_roc'].append(results['after']['auc_roc'])


                            output_file.write("\n--- Personalized Model Performance (Average Across Patients) ---\n")
                            if count_valid_patients > 0:
                                # Use np.mean() on the lists of metrics
                                avg_metrics = {metric: np.mean(metrics_after_list[metric]) for metric in metrics_after_list}

                                output_file.write(f"Average Accuracy={avg_metrics['accuracy']:.4f} (across {count_valid_patients} patients)\n")
                                output_file.write(f"Average Precision={avg_metrics['precision']:.4f}\n")
                                output_file.write(f"Average Recall={avg_metrics['recall']:.4f}\n")
                                output_file.write(f"Average F1 Score={avg_metrics['f1_score']:.4f}\n")
                                output_file.write(f"Average AUC-ROC={avg_metrics['auc_roc']:.4f}\n")

                            else:
                                output_file.write("No valid personalized patient results to average.\n")

                            output_file.write("\n") # Add space

                        else:
                            print(f"General model training failed for {current_model_type}. Skipping personalization.")
                            output_file.write(f"General model training failed for {current_model_type}. Skipping personalization.\n\n")

                output_file.write(f"\n--- End Model Type Section ---\n") # Separator between model types
                print(f"Finished Model Type {current_model_type}.")

                output_file.write(f"\n--- End Sensor Combination: {sensor_combination_name} ---\n") # Separator between sensor combinations
                print(f"Finished Sensor Combination {sensor_combination_name} for HP Combo {hp_index+1}.")

        except FileNotFoundError as e:
            print(f"\nFATAL ERROR: File not found or directory cannot be created: {e}")
            print(f"Check if the OUTPUT_DIR '{OUTPUT_DIR}' is valid and you have write permissions.")
            print(f"Attempted filename: {output_filename}")
            # Do not pass, as file could not be opened
            exit() # Exit if cannot write file

        except Exception as e:
            print(f"\nAN UNEXPECTED ERROR OCCURRED during HP combination {hp_index+1}: {e}")
            import traceback
            traceback.print_exc()
            # Continue to the next HP combination or exit, depending on desired robustness
            # For now, let's print error and continue
            pass # Continue to next HP combination


    print("\n--- All Runs Complete ---")
    print(f"Results saved in the '{OUTPUT_DIR}' directory.")