import gc
import glob
import itertools  # For combinations
import math
import os
import random
import time
from collections import OrderedDict # Added for better HP printing

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.utils import class_weight
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# --- Configuration ---
# Replace with the actual path to your 'data_9' folder
DATA_ROOT_DIR = 'F:\\data_9'
# Directory to save results files and models
OUTPUT_DIR = 'processed_data_pytorch_adaptive_pre_interbuffer_lovo_personalization'

# Ensure the base output directory exists early
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Data Processing Parameters ---
SEGMENT_DURATION_SECONDS = 30
# Time window before seizure onset considered pre-ictal
PRE_ICTAL_WINDOW_MINUTES = 30
# Buffer time around seizures to exclude for interictal
INTERICTAL_BUFFER_MINUTES = 180
# Define base sensors (ensure these are the possible columns after sync/scaling)
BASE_SENSORS = ['ACC', 'EDA', 'HR', 'TEMP'] # Use the actual column names after processing
# Target sampling frequency for resampling
SAMPLING_FREQ_HZ = 1


# --- Tunable Hyperparameters ---
# If ENABLE_HYPERPARAMETER_TUNING is False, only the first value from each list is used.
# Note: These are now single values based on the previous debugging run, but structured for future tuning
TUNABLE_SAMPLING_FREQ_HZ = [SAMPLING_FREQ_HZ]
TUNABLE_PRE_ICTAL_WINDOW_MINUTES = [PRE_ICTAL_WINDOW_MINUTES]
TUNABLE_INTERICTAL_BUFFER_MINUTES = [INTERICTAL_BUFFER_MINUTES]

TUNABLE_CONV_FILTERS = [[64, 128, 256]]
TUNABLE_CONV_KERNEL_SIZE = [10]
TUNABLE_POOL_SIZE = [2]
TUNABLE_LSTM_UNITS = [128]
TUNABLE_DENSE_UNITS = [64]

TUNABLE_GENERAL_MODEL_EPOCHS = [50] # Used for Overall General Model and LOPO General Model
TUNABLE_PERSONALIZATION_EPOCHS = [30] # Used for fine-tuning
TUNABLE_GENERAL_MODEL_LR = [0.001] # Used for Overall General Model and LOPO General Model
TUNABLE_PERSONALIZATION_LR = [0.0001] # Used for fine-tuning
TUNABLE_BATCH_SIZE = [32] # Used for Overall General Model and LOPO General Model Train/Val/Test
TUNABLE_PERSONALIZATION_BATCH_SIZE = [16] # Used for personalization Train/Val

# --- Model Types to Run ---
MODEL_TYPES_TO_RUN = ['CNN-LSTM', 'CNN-BiLSTM'] # Example: ['CNN-LSTM', 'CNN-BiLSTM']

# --- Sensor Combinations (Used if ENABLE_ADAPTIVE_SENSORS is True) ---
ENABLE_ADAPTIVE_SENSORS = True # Set to False to run only the full base sensor set

# Generate all combinations of 1 to 4 sensors from BASE_SENSORS if enabled
if ENABLE_ADAPTIVE_SENSORS:
    SENSOR_COMBINATIONS = []
    for i in range(1, len(BASE_SENSORS) + 1):
        for combo in itertools.combinations(BASE_SENSORS, i):
            SENSOR_COMBINATIONS.append(list(combo))
else:
    # Use the default set if adaptive sensors is disabled
    SENSOR_COMBINATIONS = [list(BASE_SENSORS)] # Ensure it's a list of lists

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
    # Mapping from BASE_SENSORS names to potential folder/file name parts
    # NOTE: This mapping might need careful checking against your actual file structure
    sensor_mapping = {
        'HR': {'folder': 'HR', 'part': 'HR'},
        'EDA': {'folder': 'EDA', 'part': 'EDA'},
        'TEMP': {'folder': 'TEMP', 'part': 'TEMP'},
        # Assuming 'Acc Mag' is derived from 'Acc' and is the column name after processing
        'Acc Mag': {'folder': 'Acc', 'part': 'Acc Mag'}
    }


    for sensor_name in sensors:
        if sensor_name not in sensor_mapping:
            # print(f"Warning: No mapping found for sensor '{sensor_name}'. Skipping.") # Suppress frequent warning
            continue

        mapping_info = sensor_mapping[sensor_name]
        attr_folder = mapping_info['folder']
        attr_name_part = mapping_info['part'] # This is the expected COLUMN name after loading/processing

        # Adjusted glob pattern to be more robust - matches any parquet file in the folder
        # A more specific pattern might be needed if other parquet files exist
        parquet_files = sorted(glob.glob(os.path.join(patient_dir, f'Empatica-{attr_folder}', '*.parquet')))


        if not parquet_files:
            # print(f"No Parquet files found for Patient in {patient_dir}, Attribute {attr_folder}") # Suppress frequent warning
            continue

        all_dfs = []
        for file_path in parquet_files:
            try:
                df = pd.read_parquet(file_path)
                # Check for 'time' and the expected data column name ('Acc Mag' or lower case sensor name)
                expected_data_col = sensor_name if sensor_name == 'Acc Mag' else sensor_name.lower()

                if 'time' in df.columns and expected_data_col in df.columns:
                    df['timestamp'] = pd.to_datetime(df['time'] / 1000, unit='s', utc=True)
                    # Rename the data column to match BASE_SENSORS exactly if needed
                    if expected_data_col != sensor_name:
                         df = df.rename(columns={expected_data_col: sensor_name})

                    df = df[['timestamp', sensor_name]] # Keep only timestamp and the correctly named sensor column
                    all_dfs.append(df)
                else:
                     # print(f"Warning: Parquet file {file_path} does not have expected 'time' or '{expected_data_col}' column. Skipping.") # Suppress
                     pass # Skip file if columns are wrong


            except Exception as e:
                print(f"Error reading Parquet file {file_path}: {e}")

        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df = combined_df.sort_values(by='timestamp').reset_index(drop=True)
            # Store with the BASE_SENSORS name as the key
            attribute_data[sensor_name] = combined_df

    return attribute_data


def load_seizure_annotations(patient_dir):
    """
    Loads and processes the SeerAnnotations CSV for a given patient.
    Converts start_time and end_time to UTC datetime objects.
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
            return pd.DataFrame(columns=['start_time', 'end_time']) # Return empty df on column error
    except FileNotFoundError:
        # print(f"Annotation file not found for {os.path.basename(patient_dir)}") # Suppress frequent warning
        return pd.DataFrame(columns=['start_time', 'end_time']) # Return empty df if not found
    except Exception as e:
        print(f"Error reading annotation file {annotation_file}: {e}")
        return pd.DataFrame(columns=['start_time', 'end_time']) # Return empty df on other errors


def synchronize_and_merge_data(sensor_data_dict, target_freq_hz):
    """
    Synchronizes sensor data, merges them, applies Robust Scaling, handles NaNs,
    and ensures columns are in BASE_SENSORS order.

    Args:
        sensor_data_dict (dict): Dictionary with sensor names (matching BASE_SENSORS) as keys and
                                 DataFrames (with 'timestamp' and data column named after sensor) as values.
        target_freq_hz (int): The target frequency in Hz for resampling.

    Returns:
        pandas.DataFrame: A single DataFrame with a datetime index containing
                          all synchronized and scaled sensor data. Columns ordered by BASE_SENSORS.
                          Returns None if input is empty or no common time found.
    """
    if not sensor_data_dict:
        return None

    resampled_data = {}
    # Use keys from sensor_data_dict, as these are the sensors actually found and loaded
    for sensor_name, df in sensor_data_dict.items():
        df = df.set_index('timestamp').sort_index()
        # Ensure target_freq_hz is not zero or negative
        if target_freq_hz <= 0:
            print(f"Error: Target frequency must be positive, got {target_freq_hz}. Cannot resample.")
            return None
        rule = f'{1/target_freq_hz}S'
        try:
            # Resample, keeping the sensor_name as the column name
            resampled_df = df[[sensor_name]].asfreq(rule) # Explicitly select the column
            resampled_data[sensor_name] = resampled_df
        except Exception as e:
             print(f"Error during resampling sensor {sensor_name} to {target_freq_hz}Hz: {e}. Skipping sensor.")
             continue


    merged_df = None
    # Iterate through resampled_data (only sensors that were successfully resampled)
    for sensor_name, df in resampled_data.items():
        if merged_df is None:
            merged_df = df
        else:
            # Use 'outer' join for maximum time coverage across all included sensors
            merged_df = merged_df.join(df, how='outer')

    if merged_df is None or merged_df.empty:
         return None

    merged_df = merged_df.sort_index()

    # Interpolate missing values
    merged_df = merged_df.interpolate(method='time')
    merged_df = merged_df.fillna(method='ffill')
    merged_df = merged_df.fillna(method='bfill')

    # Drop columns that are still all NaN after interpolation/fill (happens if a sensor had no data across the entire time range)
    merged_df = merged_df.dropna(axis=1, how='all')


    if merged_df.empty or len(merged_df.columns) == 0: # Check again if it's empty or has no columns
         # print("Warning: Merged DataFrame is empty or has no columns after handling NaNs.") # Suppress frequent print
         return None # Return None if no usable data


    # Apply Robust Scaling to all remaining data columns
    scaler = RobustScaler()
    data_cols = merged_df.columns # Use all remaining columns

    try:
        merged_df[data_cols] = scaler.fit_transform(merged_df[data_cols])
    except Exception as e:
        print(f"Error during scaling: {e}. Skipping scaling.")
        # Decide if you want to return None or unscaled data - returning None is safer if scaling is critical
        return None


    # --- Ensure consistent column order and presence of all BASE_SENSORS columns ---
    # Create a reindex list with all BASE_SENSORS
    # Reindex the merged_df to include all BASE_SENSORS columns, filling missing with NaN
    # This ensures the feature dimension is consistent across all patients, even if a sensor was missing entirely
    merged_df = merged_df.reindex(columns=BASE_SENSORS)

    # Fill NaNs introduced by reindexing (for sensors missing entirely) with 0 or another value if appropriate
    # Filling with 0 implies the signal is absent/zeroed when not recorded - review if this is appropriate
    merged_df = merged_df.fillna(0.0) # Fill NaNs introduced by reindexing with 0


    if merged_df.empty or len(merged_df.columns) != len(BASE_SENSORS): # Final check on shape
         print("Warning: Merged DataFrame is empty or has incorrect number of columns after reindexing/filling.")
         return None # Return None if final shape is wrong


    return merged_df


def create_labeled_segments(synced_df, annotations_df, segment_duration_sec, pre_ictal_window_min, interictal_buffer_min, target_freq_hz):
    """
    Creates segments from synchronized data and labels them
    as pre-ictal (1) or interictal (0) based on seizure annotations. Samples interictal
    segments to attempt class balance. Synced_df must have columns in BASE_SENSORS order.

    Args:
        synced_df (pandas.DataFrame): DataFrame with synchronized sensor data (datetime index).
                                      Must have columns in order of BASE_SENSORS (with NaNs=0).
        annotations_df (pandas.DataFrame): DataFrame with seizure start/end times.
        segment_duration_sec (int): Duration of each segment in seconds.
        pre_ictal_window_min (int): Time window before seizure onset considered pre-ictal.
        interictal_buffer_min (int): Buffer time around seizures to exclude for interictal.
        target_freq_hz (int): Frequency data was resampled to (for calculating segment steps).

    Returns:
        tuple: (segments, labels) where segments is a numpy array
               (shape: n_samples, segment_len, num_features) and labels is a numpy array (0 or 1).
               Returns (np.array([]), np.array([])) if no data or annotations.
    """
    # Ensure expected number of features based on BASE_SENSORS
    expected_num_features = len(BASE_SENSORS)
    segment_length_steps = int(segment_duration_sec * target_freq_hz)
    # Ensure segment_length_steps is at least 1 and sensible
    if segment_length_steps <= 0:
        print(f"Error: Calculated segment_length_steps is {segment_length_steps} (Duration {segment_duration_sec}s * Freq {target_freq_hz}Hz). Cannot create segments.")
        segment_length_steps = max(1, int(segment_duration_sec * target_freq_hz)) # Recalculate with max 1
        # Return empty array with correct shape if possible
        return np.array([]).reshape(0, segment_length_steps, expected_num_features), np.array([])


    if synced_df is None or annotations_df is None or synced_df.empty or len(synced_df.columns) != expected_num_features:
        print("Synced data is empty, has wrong number of columns, or annotations are missing. Cannot create segments.")
        return np.array([]).reshape(0, segment_length_steps, expected_num_features), np.array([])


    segments = []
    labels = []
    step_size = segment_length_steps # Use non-overlapping segments

    data_start_time = synced_df.index.min() if not synced_df.empty else None
    data_end_time = synced_df.index.max() if not synced_df.empty else None

    if data_start_time is None or data_end_time is None:
        print("Warning: Synced data has no time index. Cannot create segments.")
        return np.array([]).reshape(0, segment_length_steps, expected_num_features), np.array([]) # Return with default shape


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

                # Check for overlap: [seg_start, seg_end] and [win_start, win_end]
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


    # 3. Interictal Exclusion Windows (Seizure + Buffer)
    interictal_exclusion_windows = []
    buffer_timedelta = pd.Timedelta(minutes=interictal_buffer_min)
    # print(f"Defining interictal exclusion windows ({interictal_buffer_min} mins buffer around seizures)...") # Suppress frequent print)

    for _, seizure in annotations_df.iterrows():
         seizure_start = seizure['start_time']
         seizure_end = seizure['end_time']
         if seizure_start is None or seizure_end is None: continue # Skip if seizure times are missing

         exclusion_start = seizure_start - buffer_timedelta
         exclusion_end = seizure_end + buffer_timedelta
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
        return np.array([]).reshape(0, max(1, int(segment_duration_sec * target_freq_hz))), expected_num_features, np.array([]) # Return with default shape


    for i in range(0, len(synced_df) - segment_length_steps + 1, step_size): # No tqdm here, handled by outer patient loop
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

    # print(f"Finished segmentation.") # Suppress frequent print)
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
        # Return empty array with correct shape
        num_features = segments.shape[2] if segments.shape[0] > 0 else expected_num_features
        segment_length_steps = int(segment_duration_sec * target_freq_hz)
        return np.array([]).reshape(0, segments.shape[1] if segments.ndim > 1 else segment_length_steps, num_features if segments.ndim > 2 else expected_num_features), np.array([])


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


def prepare_patient_data(patient_folder):
    """
    Loads, synchronizes, scales, and creates labeled segments for a single patient.
    Segments are created with len(BASE_SENSORS) features (with NaNs=0 if missing).
    Returns (patient_id, segments, labels, found_sensors_list) or None if processing fails.
    """
    patient_id = os.path.basename(patient_folder)
    # print(f"Processing data for patient: {patient_id}") # Moved to main loop desc

    # 1. Load sensor data - Try to load all base sensors
    # load_sensor_data_for_patient now expects BASE_SENSORS and returns dict with BASE_SENSORS names as keys
    sensor_data_dict = load_sensor_data_for_patient(patient_folder, BASE_SENSORS)
    # Get the list of sensor names for which data was actually found and loaded into the dict
    found_sensors = list(sensor_data_dict.keys())


    if not sensor_data_dict:
        # print(f"Skipping patient {patient_id}: Could not load any sensor data from BASE_SENSORS.") # Moved to main loop desc
        # Return None, indicating failure to load data for any sensor
        return None

    # 2. Load annotations
    annotations_df = load_seizure_annotations(patient_folder)
    # Note: annotations_df can be empty if no seizures, handled in segment creation

    # 3. Synchronize and merge data
    # synced_df will have columns for all BASE_SENSORS, with 0s for missing ones, in sorted order
    synced_df = synchronize_and_merge_data(sensor_data_dict, target_freq_hz=SAMPLING_FREQ_HZ)
    # Ensure synced_df has the correct number of columns (all BASE_SENSORS)
    if synced_df is None or synced_df.empty or len(synced_df.columns) != len(BASE_SENSORS):
        # print(f"Skipping patient {patient_id}: Could not synchronize, merge, or lost columns.") # Moved to main loop desc
        return None

    # 4. Create labeled segments
    # Segments will have len(BASE_SENSORS) features, with scaled 0s for missing sensors
    segments, labels = create_labeled_segments(
        synced_df,
        annotations_df,
        segment_duration_sec=SEGMENT_DURATION_SECONDS,
        pre_ictal_window_min=TUNABLE_PRE_ICTAL_WINDOW_MINUTES[0], # Use first HP value
        interictal_buffer_min=TUNABLE_INTERICTAL_BUFFER_MINUTES[0], # Use first HP value
        target_freq_hz=TUNABLE_SAMPLING_FREQ_HZ[0] # Use first HP value
    )

    # Segments should have shape (N, L, len(BASE_SENSORS))
    # Calculate expected seq_len based on parameters used in create_labeled_segments
    expected_seq_len = int(SEGMENT_DURATION_SECONDS * TUNABLE_SAMPLING_FREQ_HZ[0])
    expected_seq_len = max(1, expected_seq_len)

    if len(segments) == 0 or segments.shape[2] != len(BASE_SENSORS) or segments.shape[1] != expected_seq_len:
         # print(f"Skipping patient {patient_id}: No valid segments created or incorrect shape after segmentation.") # Moved to main loop desc
         return None

    # Return patient_id, segments (with all BASE_SENSORS features), labels, AND the list of sensors that were actually found
    return patient_id, segments, labels, found_sensors


# --- PyTorch Dataset ---

class SeizureDataset(Dataset):
    def __init__(self, segments, labels, seq_len, num_features):
        """
        Args:
            segments (np.ndarray): Segments array (n_samples, seq_len, n_features).
            labels (np.ndarray): Labels array (n_samples,).
            seq_len (int): Expected sequence length.
            num_features (int): Expected number of features.
        """
        # If input segments are empty, create empty tensors with the expected shape
        if segments.shape[0] == 0:
             self.segments = torch.empty(0, num_features, seq_len, dtype=torch.float32)
             self.labels = torch.empty(0, 1, dtype=torch.float32)
        else:
            # Ensure segments have the correct shape (N, L, F)
            if segments.ndim == 2: # (N, L) -> add a feature dim (N, L, 1)
                 segments = segments[:, :, np.newaxis]
                 # Update num_features if it was expected to be 1 based on this
                 if num_features != 1:
                     print(f"Warning: Segments ndim=2 but expected num_features={num_features}. Assuming 1 feature.")
                     num_features = 1
            elif segments.ndim < 2:
                 print(f"Warning: Segments array has unexpected ndim={segments.ndim}. Cannot create dataset.")
                 self.segments = torch.empty(0, num_features, seq_len, dtype=torch.float32)
                 self.labels = torch.empty(0, 1, dtype=torch.float32)
                 return # Stop init if data is unusable


            # Ensure segments have the expected number of features
            if segments.shape[2] != num_features:
                 print(f"Warning: Segment features ({segments.shape[2]}) mismatch expected features ({num_features}). Cannot create dataset.")
                 self.segments = torch.empty(0, num_features, seq_len, dtype=torch.float32)
                 self.labels = torch.empty(0, 1, dtype=torch.float32)
                 return # Stop init

            # Ensure segments have the expected sequence length
            if segments.shape[1] != seq_len:
                 print(f"Warning: Segment length ({segments.shape[1]}) mismatch expected length ({seq_len}). Cannot create dataset.")
                 self.segments = torch.empty(0, num_features, seq_len, dtype=torch.float32)
                 self.labels = torch.empty(0, 1, dtype=torch.float32)
                 return # Stop init


            self.segments = torch.tensor(segments, dtype=torch.float32).permute(0, 2, 1) # (N, L, F) -> (N, F, L)
            self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1) # (N,) -> (N, 1)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.segments[idx], self.labels[idx]


# --- PyTorch Model Definitions ---

class CNN_LSTM(nn.Module):
    def __init__(self, input_channels, seq_len):
        super(CNN_LSTM, self).__init__()
        self.input_channels = input_channels
        self.seq_len = seq_len

        if input_channels <= 0:
            # print(f"Warning: CNN-LSTM input_channels is {input_channels}. Setting to 1.") # Suppress frequent
            input_channels = 1 # Default to 1 channel if somehow 0 or negative
        if seq_len <= 0:
            # print(f"Warning: CNN-LSTM seq_len is {seq_len}. Setting to 1.") # Suppress frequent
            seq_len = 1 # Default to 1 seq_len if somehow 0 or negative


        conv_layers_list = []
        in_channels = input_channels
        current_seq_len = seq_len

        # Dynamically build conv layers based on CONV_FILTERS list
        for i, out_channels in enumerate(TUNABLE_CONV_FILTERS[0]): # Use the first (and only) set of filters
            padding = TUNABLE_CONV_KERNEL_SIZE[0] // 2 # Use the first kernel size
            pool_size = TUNABLE_POOL_SIZE[0] # Use the first pool size

            conv_layers_list.append(nn.Conv1d(in_channels, out_channels, kernel_size=TUNABLE_CONV_KERNEL_SIZE[0], padding=padding))
            conv_layers_list.append(nn.ReLU())
            conv_layers_list.append(nn.MaxPool1d(pool_size))
            in_channels = out_channels

            # Recalculate sequence length after conv and pool
            # O_len = floor((I_len + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
            current_seq_len = math.floor((current_seq_len + 2 * padding - 1*(TUNABLE_CONV_KERNEL_SIZE[0]-1) - 1) / 1 + 1) # Conv1d output length (stride=1, dilation=1)
            current_seq_len = math.floor((current_seq_len + 2 * 0 - 1*(pool_size-1) - 1) / pool_size + 1) # MaxPool1d output length (stride=pool_size, dilation=1)


        self.conv_layers = nn.Sequential(*conv_layers_list)

        # Calculate the output sequence length after CNN layers dynamically
        try:
            # Create a dummy tensor on the CPU for shape calculation
            # Ensure dummy input has the correct seq_len after potential adjustment
            dummy_input = torch.randn(1, self.input_channels, self.seq_len, dtype=torch.float32)
            dummy_output = self.conv_layers(dummy_input)
            self.lstm_input_features = dummy_output.shape[1] # Features dimension after CNN (number of last conv filters)
            self.lstm_input_seq_len = dummy_output.shape[2] # Sequence length dimension after CNN/Pooling

            # Check if the output sequence length is valid for LSTM
            if self.lstm_input_seq_len <= 0:
                 # This can happen if seq_len is too short for the filters/pooling
                 raise ValueError(f"Calculated LSTM input sequence length is zero or negative ({self.lstm_input_seq_len}). Check CNN/Pooling parameters relative to segment length ({self.seq_len}).")

        except Exception as e:
             print(f"Error calculating layer output size during model init for {self.__class__.__name__} with input_channels={self.input_channels}, seq_len={self.seq_len}: {e}")
             # Set fallback values or re-raise - re-raising is better during development
             raise e


        self.lstm = nn.LSTM(input_size=self.lstm_input_features,
                            hidden_size=TUNABLE_LSTM_UNITS[0], # Use first LSTM units
                            batch_first=True)

        self.dense_layers = nn.Sequential(
            nn.Linear(TUNABLE_LSTM_UNITS[0], TUNABLE_DENSE_UNITS[0]), # Use first Dense units
            nn.Sigmoid(),
            nn.Linear(TUNABLE_DENSE_UNITS[0], 1), # Use first Dense units
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch_size, channels, seq_len)
        cnn_out = self.conv_layers(x) # shape: (batch_size, filters, reduced_seq_len)
        # Handle potential empty output after CNN if seq_len collapsed to 0
        if cnn_out.shape[2] == 0:
             # Or return a default value, depending on desired behavior
             # Returning 0.5 (sigmoid output) might be reasonable for prediction task
             # Ensure the tensor is on the correct device
             return torch.tensor([[0.5]] * x.size(0), device=x.device) # Return neutral predictions if seq_len collapses


        lstm_in = cnn_out.permute(0, 2, 1) # shape: (batch_size, reduced_seq_len, filters)
        lstm_out, _ = self.lstm(lstm_in) # shape: (batch_size, reduced_seq_len, LSTM_UNITS)
        last_timestep_out = lstm_out[:, -1, :] # shape: (batch_size, LSTM_UNITS)
        output = self.dense_layers(last_timestep_out) # shape: (batch_size, 1)
        return output


class CNN_BiLSTM(nn.Module):
    def __init__(self, input_channels, seq_len):
        super(CNN_BiLSTM, self).__init__()
        self.input_channels = input_channels
        self.seq_len = seq_len

        if input_channels <= 0:
            # print(f"Warning: CNN-BiLSTM input_channels is {input_channels}. Setting to 1.") # Suppress frequent
            input_channels = 1 # Default to 1 channel if somehow 0 or negative
        if seq_len <= 0:
            # print(f"Warning: CNN-BiLSTM seq_len is {seq_len}. Setting to 1.") # Suppress frequent
            seq_len = 1 # Default to 1 seq_len if somehow 0 or negative


        conv_layers_list = []
        in_channels = input_channels
        current_seq_len = seq_len

        # Dynamically build conv layers based on CONV_FILTERS list
        for i, out_channels in enumerate(TUNABLE_CONV_FILTERS[0]): # Use the first (and only) set of filters
            padding = TUNABLE_CONV_KERNEL_SIZE[0] // 2 # Use the first kernel size
            pool_size = TUNABLE_POOL_SIZE[0] # Use the first pool size

            conv_layers_list.append(nn.Conv1d(in_channels, out_channels, kernel_size=TUNABLE_CONV_KERNEL_SIZE[0], padding=padding))
            conv_layers_list.append(nn.ReLU())
            conv_layers_list.append(nn.MaxPool1d(pool_size))
            in_channels = out_channels

            # Recalculate sequence length after conv and pool
            current_seq_len = math.floor((current_seq_len + 2 * padding - 1*(TUNABLE_CONV_KERNEL_SIZE[0]-1) - 1) / 1 + 1) # Conv1d output length (stride=1, dilation=1)
            current_seq_len = math.floor((current_seq_len + 2 * 0 - 1*(pool_size-1) - 1) / pool_size + 1) # MaxPool1d output length (stride=pool_size, dilation=1)


        self.conv_layers = nn.Sequential(*conv_layers_list)


        # Calculate the output sequence length after CNN layers dynamically
        try:
            # Create a dummy tensor on the CPU for shape calculation
            # Ensure dummy input has the correct seq_len after potential adjustment
            dummy_input = torch.randn(1, self.input_channels, self.seq_len, dtype=torch.float32)
            dummy_output = self.conv_layers(dummy_input)
            self.lstm_input_features = dummy_output.shape[1] # Features dimension after CNN (number of last conv filters)
            self.lstm_input_seq_len = dummy_output.shape[2] # Sequence length dimension after CNN/Pooling

            # Check if the output sequence length is valid for LSTM
            if self.lstm_input_seq_len <= 0:
                 # This can happen if seq_len is too short for the filters/pooling
                 raise ValueError(f"Calculated LSTM input sequence length is zero or negative ({self.lstm_input_seq_len}). Check CNN/Pooling parameters relative to segment length ({self.seq_len}).")

        except Exception as e:
            print(f"Error calculating layer output size during model init for {self.__class__.__name__} with input_channels={self.input_channels}, seq_len={self.seq_len}: {e}")
            # Set fallback values or re-raise - re-raising is better during development
            raise e


        self.bilstm = nn.LSTM(input_size=self.lstm_input_features,
                              hidden_size=TUNABLE_LSTM_UNITS[0], # Use first LSTM units
                              batch_first=True,
                              bidirectional=True)

        self.dense_layers = nn.Sequential(
            nn.Linear(TUNABLE_LSTM_UNITS[0] * 2, TUNABLE_DENSE_UNITS[0]), # Use first Dense units
            nn.Sigmoid(),
            nn.Linear(TUNABLE_DENSE_UNITS[0], 1), # Use first Dense units
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch_size, channels, seq_len)
        cnn_out = self.conv_layers(x) # shape: (batch_size, filters, reduced_seq_len)
        # Handle potential empty output after CNN if seq_len collapsed to 0
        if cnn_out.shape[2] == 0:
             # Or return a default value, depending on desired behavior
             # Ensure the tensor is on the correct device
             return torch.tensor([[0.5]] * x.size(0), device=x.device) # Return neutral predictions if seq_len collapses


        lstm_in = cnn_out.permute(0, 2, 1) # shape: (batch_size, reduced_seq_len, filters)
        bilstm_out, _ = self.bilstm(lstm_in) # shape: (batch_size, reduced_seq_len, LSTM_UNITS * 2)
        last_timestep_out = bilstm_out[:, -1, :] # shape: (batch_size, LSTM_UNITS * 2)
        output = self.dense_layers(last_timestep_out) # shape: (batch_size, 1)
        return output

# Helper function to get the model class based on string name
def get_model_class(model_type):
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

    # Check if dataloader is empty
    if len(dataloader.dataset) == 0:
        # print("Warning: Training dataloader is empty.") # Suppress
        return 0.0 # Return 0 loss if no data

    dataloader_tqdm = tqdm(dataloader, desc="Batch", leave=False) # Corrected typo

    for inputs, labels in dataloader_tqdm: # Corrected typo
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        # Ensure outputs and labels have compatible shapes for BCELoss
        # BCELoss expects (N, 1) and (N, 1) or (N,) and (N,)
        # Our labels are (N, 1), outputs are (N, 1). This is fine.

        if class_weights is not None and len(class_weights) == 2:
             # Apply weights based on the label (0 or 1)
            weight_tensor = torch.zeros_like(labels, dtype=torch.float32) # Ensure dtype matches labels/loss output
            # Check if weights exist for the classes present in the batch
            if 0 in class_weights:
                weight_tensor[labels == 0] = class_weights[0]
            if 1 in class_weights:
                weight_tensor[labels == 1] = class_weights[1]

            loss = criterion(outputs, labels)
            loss = (loss * weight_tensor).mean() # Apply weights and take mean across batch
        else:
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        dataloader_tqdm.set_postfix(loss=loss.item())

    # Avoid division by zero if dataset was somehow empty despite the check
    epoch_loss = running_loss / (len(dataloader.dataset) if len(dataloader.dataset) > 0 else 1)
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
         # print("Warning: Evaluation dataloader is empty.") # Suppress
         return {
            'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
            'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]
        }

    dataloader_tqdm = tqdm(dataloader, desc="Evaluating Batch", leave=False) # Corrected typo

    with torch.no_grad():
        for inputs, labels in dataloader_tqdm: # Corrected typo
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

    all_labels = np.array(all_labels).flatten() # Ensure 1D for metrics
    all_predictions = np.array(all_predictions).flatten() # Ensure 1D for metrics
    all_probs = np.array(all_probs).flatten() # Ensure 1D for metrics


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
             # print("Warning: Only one class present in evaluation set labels, AUC-ROC is undefined.") # Suppress frequent warning

    except ValueError: # Catch other potential ValueError (e.g., invalid probabilities)
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
    # Added min_lr to scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=0.00001) # min_lr slightly lower than personalization LR


    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    # Only attempt training if there is data in the training dataloader
    if len(train_dataloader.dataset) == 0:
         print(f"Warning: Training dataloader for '{desc}' is empty. Skipping training.")
         # Return the initial model state, no training occurred
         return model

    epoch_tqdm = tqdm(range(epochs), desc=desc, leave=True)

    for epoch in epoch_tqdm:
        start_time = time.time()

        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, DEVICE, class_weights)

        # Only run validation if validation data is available
        if len(val_dataloader.dataset) > 0:
            val_metrics = evaluate_pytorch_model(model, val_dataloader, criterion, DEVICE)
            val_loss = val_metrics['loss']
            val_acc = val_metrics['accuracy']
            epoch_tqdm.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc:.4f}", time=f"{time.time() - start_time:.2f}s")

            scheduler.step(val_loss)

            # Only check for improvement if validation data is available
            if val_loss < best_val_loss:
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
                print(f"Early stopping triggered at epoch {epoch+1} for '{desc}'.")
                break

        else: # No validation data available
            epoch_tqdm.set_postfix(train_loss=f"{train_loss:.4f}", time=f"{time.time() - start_time:.2f}s")
            # Without validation data, we cannot do early stopping based on validation loss.
            # The scheduler also relies on a metric, usually validation loss.
            # We can step the scheduler with train_loss if desired, but it's less ideal.
            # For now, we skip scheduler.step and early stopping if no val data.
            # If you need robust training without val data, consider training for fixed epochs.
            pass # Skip scheduler and early stopping logic

    # Load best weights before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        # This happens if val_dataloader was empty or val_loss never improved
        print(f"Warning: No best model state was saved during training for '{desc}'. Returning final epoch state.")

    return model

# --- New Function for LOPO General Model Training ---

def train_lopo_general_model(all_processed_patient_data, excluded_patient_id, model_type, sensor_combination_indices, model_hyperparameters, general_hyperparameters):
    """
    Trains a general model on data from all patients EXCEPT the excluded one,
    using only the sensors specified by indices.

    Args:
        all_processed_patient_data (list): List of (patient_id, segments, labels, found_sensors)
                                         from initial processing (segments have len(BASE_SENSORS) features).
        excluded_patient_id (str): The ID of the patient whose data should be excluded.
        model_type (str): 'CNN-LSTM' or 'CNN-BiLSTM'.
        sensor_combination_indices (list): List of integer indices corresponding to
                                            the sensor columns to use from BASE_SENSORS.
        model_hyperparameters (dict): Dictionary containing model architecture HPs.
        general_hyperparameters (dict): Dictionary containing general model training HPs (epochs, lr, batch_size).


    Returns:
        dict: State dictionary of the trained LOPO general model.
              Returns None if training data is insufficient.
    """
    print(f"\n--- Training LOPO General Model (Excluding {excluded_patient_id}) for {model_type} ---")

    lopo_segments_raw = []
    lopo_labels_raw = []

    # Collect data from all patients EXCEPT the excluded one
    for patient_id, segments_all_sensors, labels, found_sensors in all_processed_patient_data:
        if patient_id != excluded_patient_id:
            # Slice segments to include only the features for the current combination
            # segments_all_sensors shape is (N, L, len(BASE_SENSORS))
            if len(segments_all_sensors) > 0 and len(sensor_combination_indices) > 0:
                segments_sliced = segments_all_sensors[:, :, sensor_combination_indices]
                lopo_segments_raw.append(segments_sliced)
                lopo_labels_raw.append(labels)
            # else: print(f"Skipping data from patient {patient_id} for LOPO general train (no segments or no sensors selected).") # Suppress


    if not lopo_segments_raw:
        print(f"Warning: No data available from other patients for LOPO general training (Excluding {excluded_patient_id}).")
        return None # Return None if no data for LOPO training

    lopo_segments_combined = np.concatenate(lopo_segments_raw, axis=0)
    lopo_labels_combined = np.concatenate(lopo_labels_raw, axis=0)

    print(f"LOPO Combined data shape (Excluding {excluded_patient_id}): {lopo_segments_combined.shape}")
    print(f"LOPO Combined labels shape (Excluding {excluded_patient_id}): {lopo_labels_combined.shape}")

    # Check for sufficient data and classes for LOPO training split
    if len(lopo_segments_combined) < 2 or len(np.unique(lopo_labels_combined)) < 2: # Need at least 2 samples, one of each class
         print(f"Warning: Insufficient data or only one class for LOPO general training split (Excluding {excluded_patient_id}). Skipping training.")
         return None


    # Split LOPO data into Train/Validation (e.g., 80/20)
    # Ensure enough data for splitting - min_samples=1 allows splitting even if only 1 sample per class is left after stratification
    try:
        X_train_lopo, X_val_lopo, y_train_lopo, y_val_lopo = train_test_split(
             lopo_segments_combined, lopo_labels_combined, test_size=0.2, random_state=SEED, stratify=lopo_labels_combined
        )
    except ValueError as e:
         print(f"Warning: LOPO data split failed for patient {excluded_patient_id}: {e}. This might happen with very few samples. Skipping training.")
         return None


    print(f"LOPO General Training data shape: {X_train_lopo.shape}, Labels shape: {y_train_lopo.shape}")
    print(f"LOPO General Validation data shape: {X_val_lopo.shape}, Labels shape: {y_val_lopo.shape}")

    # Ensure splits are not empty and have both classes for training/validation
    if len(X_train_lopo) == 0 or len(X_val_lopo) == 0 or len(np.unique(y_train_lopo)) < 2 or len(np.unique(y_val_lopo)) < 2:
         print(f"Warning: LOPO data split resulted in empty train/val set or single class for patient {excluded_patient_id}. Skipping training.")
         return None

    # Calculate expected seq_len and num_features for the dataset
    expected_seq_len = lopo_segments_combined.shape[1]
    expected_num_features = lopo_segments_combined.shape[2]

    lopo_train_dataset = SeizureDataset(X_train_lopo, y_train_lopo, seq_len=expected_seq_len, num_features=expected_num_features)
    lopo_val_dataset = SeizureDataset(X_val_lopo, y_val_lopo, seq_len=expected_seq_len, num_features=expected_num_features)

    num_workers = os.cpu_count() // 2 or 1
    persistent_workers = True if num_workers > 0 and hasattr(DataLoader, 'persistent_workers') else False

    # Get batch sizes and learning rate from general_hyperparameters
    general_train_batch_size = general_hyperparameters['batch_size']
    general_learning_rate = general_hyperparameters['learning_rate']
    general_epochs = general_hyperparameters['epochs']

    # Adjust batch size if larger than dataset size
    train_batch_size = general_train_batch_size
    if len(lopo_train_dataset) > 0: train_batch_size = max(1, min(train_batch_size, len(lopo_train_dataset)))
    val_batch_size = general_train_batch_size
    if len(lopo_val_dataset) > 0: val_batch_size = max(1, min(val_batch_size, len(lopo_val_dataset)))

    # Ensure batch size is at least 1 if dataset is not empty
    train_batch_size = max(1, train_batch_size) if len(lopo_train_dataset) > 0 else general_train_batch_size # Use default if empty
    val_batch_size = max(1, val_batch_size) if len(lopo_val_dataset) > 0 else general_train_batch_size


    lopo_train_dataloader = DataLoader(lopo_train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, persistent_workers=persistent_workers)
    lopo_val_dataloader = DataLoader(lopo_val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, persistent_workers=persistent_workers)

    # Calculate class weights for the LOPO training data
    classes_lopo = np.unique(y_train_lopo)
    class_weights_lopo_dict = None
    if len(classes_lopo) == 2:
         class_weights_lopo_np = class_weight.compute_class_weight(
             'balanced', classes=classes_lopo, y=y_train_lopo
         )
         class_weights_lopo_dict = {cls: weight for cls, weight in zip(classes_lopo, class_weights_lopo_np)}
         # print(f"Computed LOPO general class weights: {class_weights_lopo_dict}") # Suppress

    # Instantiate the LOPO general model with the correct input shape for this combination
    input_channels = lopo_segments_combined.shape[2]
    seq_len = lopo_segments_combined.shape[1]
    ModelClass = get_model_class(model_type)

    try:
        lopo_general_model = ModelClass(input_channels, seq_len).to(DEVICE)
    except ValueError as e:
         print(f"Error instantiating LOPO general model for {excluded_patient_id}: {e}. Skipping training.")
         # Clean up dataloaders before returning
         del lopo_train_dataloader, lopo_val_dataloader
         del lopo_train_dataset, lopo_val_dataset
         gc.collect()
         if torch.cuda.is_available(): torch.cuda.empty_cache()
         return None
    except Exception as e:
         print(f"An unexpected error occurred during LOPO model instantiation for {excluded_patient_id}: {e}. Skipping training.")
         # Clean up dataloaders before returning
         del lopo_train_dataloader, lopo_val_dataloader
         del lopo_train_dataset, lopo_val_dataset
         gc.collect()
         if torch.cuda.is_available(): torch.cuda.empty_cache()
         return None


    # Train the LOPO general model
    print(f"Starting LOPO General Model training (Excluding {excluded_patient_id})...")
    lopo_general_model = train_pytorch_model(
        lopo_general_model,
        lopo_train_dataloader,
        lopo_val_dataloader,
        epochs=general_epochs,
        learning_rate=general_learning_rate,
        class_weights=class_weights_lopo_dict,
        save_best_model_path=None, # No need to save LOPO models per patient usually
        desc=f"Training LOPO General (Excl {excluded_patient_id})"
    )

    # Clean up LOPO training dataloaders and model instance
    lopo_general_model_state = lopo_general_model.state_dict()
    del lopo_general_model, lopo_train_dataloader, lopo_val_dataloader
    del lopo_train_dataset, lopo_val_dataset
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return lopo_general_model_state


# --- Modified Personalization Function to include LOPO ---

def perform_personalization_pytorch_lopo(all_processed_patient_data, model_type, sensor_combination, general_hyperparameters, personalization_hyperparameters, model_hyperparameters):
    """
    Performs personalization for each patient IN THE PROVIDED LIST using the LOPO approach.
    This function orchestrates the LOPO general model training for each patient.

    Args:
        all_processed_patient_data (list): List of (patient_id, segments, labels, found_sensors)
                                         from initial processing (segments have len(BASE_SENSORS) features).
                                         This list contains ALL suitable patients for the current sensor combo.
        model_type (str): 'CNN-LSTM' or 'CNN-BiLSTM'.
        sensor_combination (list): List of sensor names (e.g., ['HR', 'EDA']) for the current combination.
        general_hyperparameters (dict): Dictionary containing general model training HPs (epochs, lr, batch_size).
        personalization_hyperparameters (dict): Dictionary containing personalization training HPs (epochs, lr, batch_size).
        model_hyperparameters (dict): Dictionary containing model architecture HPs.


    Returns:
        dict: Dictionary storing performance metrics before and after personalization for each patient in the list.
              Only includes patients for whom LOPO training and personalization was attempted.
    """
    combination_name = "_".join([s.lower() for s in sensor_combination]).upper()

    print(f"\n--- Performing Personalization ({model_type}) for {combination_name} using LOPO ---")

    if not all_processed_patient_data:
         print("No patient data available for personalization with LOPO.")
         return {}

    personalization_results = {}
    ModelClass = get_model_class(model_type)

    # Get indices for the sensors in the current combination (relative to BASE_SENSORS order)
    try:
        sensor_combination_indices = [BASE_SENSORS.index(s) for s in sensor_combination]
        # Sort indices to maintain consistent column order after slicing
        sensor_combination_indices.sort()
        # Ensure the indices are valid
        if any(idx < 0 or idx >= len(BASE_SENSORS) for idx in sensor_combination_indices):
             raise ValueError("Invalid sensor index generated.")
    except ValueError as e:
        print(f"Error: Sensor in combination {sensor_combination} not found or invalid index in BASE_SENSORS. {e}")
        return {} # Cannot proceed without valid sensor indices


    # Calculate expected shape of sliced segments for this combination
    expected_seq_len_sliced = int(SEGMENT_DURATION_SECONDS * TUNABLE_SAMPLING_FREQ_HZ[0]) # Use first HP value
    expected_seq_len_sliced = max(1, expected_seq_len_sliced)
    expected_num_features_sliced = len(sensor_combination_indices)


    # Wrap patient loop with tqdm
    patient_tqdm = tqdm(all_processed_patient_data, desc=f"Personalizing Patients ({model_type}, {combination_name})", leave=True)

    for patient_data_tuple in patient_tqdm: # Iterate through the list of patient data tuples
        current_patient_id, current_patient_segments_all_sensors, current_patient_labels, current_found_sensors = patient_data_tuple # Unpack the full tuple

        # Update patient progress bar description
        patient_tqdm.set_description(f"Personalizing {current_patient_id} ({model_type}, {combination_name})")

        # --- Step 1: Train LOPO General Model for the current patient ---
        # This returns the state_dict of a model trained on N-1 patients using the current sensor combination features
        lopo_general_model_state_dict = train_lopo_general_model(
             all_processed_patient_data,
             current_patient_id,
             model_type,
             sensor_combination_indices,
             model_hyperparameters, # Pass model architecture HPs
             {'epochs': general_hyperparameters['epochs'], # Pass general training HPs
              'learning_rate': general_hyperparameters['learning_rate'],
              'batch_size': general_hyperparameters['batch_size']}
        )

        if lopo_general_model_state_dict is None:
             print(f"Skipping personalization for patient {current_patient_id} ({model_type}, {combination_name}): LOPO general model training failed.")
             # Do NOT add entry to results if LOPO training failed, as we have no 'before' baseline
             continue # Move to the next patient

        # --- Step 2: Prepare and Split Target Patient Data ---
        # Slice the current patient's segments to include only the features for this combination
        if len(current_patient_segments_all_sensors) > 0 and expected_num_features_sliced > 0:
             # Ensure the number of features in segments_all_sensors is as expected before slicing
             if current_patient_segments_all_sensors.shape[2] == len(BASE_SENSORS):
                current_patient_segments_sliced = current_patient_segments_all_sensors[:, :, sensor_combination_indices]
             else:
                 print(f"Error: Patient {current_patient_id} segments_all_sensors has unexpected feature count ({current_patient_segments_all_sensors.shape[2]}). Expected {len(BASE_SENSORS)}. Skipping.")
                 continue # Move to the next patient
        else:
             print(f"Skipping patient {current_patient_id} ({model_type}, {combination_name}): No segments or no features after slicing.")
             continue # Move to the next patient


        # Check if sliced patient data is usable for splitting/training/eval
        if len(current_patient_segments_sliced) == 0 or len(current_patient_labels) != len(current_patient_segments_sliced) or len(np.unique(current_patient_labels)) < 2:
             print(f"Skipping patient {current_patient_id} ({model_type}, {combination_name}): No valid segments or only one class in patient's data after slicing/label mismatch.")
             # Do NOT add entry to results if patient data is unusable
             continue # Move to the next patient


        # Split patient's sliced data for personalization fine-tuning and testing (60/20/20)
        # Ensure enough data for splitting - need at least 3 samples for 60/20/20 stratified split
        if len(current_patient_segments_sliced) < 3:
             print(f"Warning: Not enough data ({len(current_patient_segments_sliced)} samples) for patient {current_patient_id} personalization splits. Skipping.")
             # Do NOT add entry to results if patient data is unusable for splitting
             continue


        try:
             X_train_pat, X_temp_pat, y_train_pat, y_temp_pat = train_test_split(
                  current_patient_segments_sliced, current_patient_labels, test_size=0.4, random_state=SEED, stratify=current_patient_labels
             )
             X_val_pat, X_test_pat, y_val_pat, y_test_pat = train_test_split(
                  X_temp_pat, y_temp_pat, test_size=0.5, random_state=SEED, stratify=y_temp_pat
             )
        except ValueError as e:
             print(f"Warning: Patient {current_patient_id} data split failed: {e}. Skipping personalization.")
             # Do NOT add entry to results if patient data split fails
             continue # Move to the next patient

        # Ensure resulting patient splits are not empty and have both classes
        # Check unique classes only if the split resulted in samples
        unique_y_train_pat = np.unique(y_train_pat)
        unique_y_val_pat = np.unique(y_val_pat)
        unique_y_test_pat = np.unique(y_test_pat)

        if len(X_train_pat) == 0 or len(X_val_pat) == 0 or len(X_test_pat) == 0 or len(unique_y_train_pat) < 2 or len(unique_y_val_pat) < 2 or len(unique_y_test_pat) < 2:
            print(f"Warning: Patient {current_patient_id} data split resulted in empty train ({len(X_train_pat)}), val ({len(X_val_pat)}), or test ({len(X_test_pat)}) set, or single class in one split. Skipping personalization.")
            # Do NOT add entry to results if patient data split is unusable
            continue # Move to the next patient


        # print(f"Patient {current_patient_id} Personalization Train shape: {X_train_pat.shape}, Val shape: {X_val_pat.shape}, Test shape: {X_test_pat.shape}") # Suppress

        # Create DataLoaders for the current patient's splits
        num_workers_pat = os.cpu_count() // 4 or 1
        persistent_workers_pat = True if num_workers_pat > 0 and hasattr(DataLoader, 'persistent_workers') else False

        # Get batch sizes from personalization_hyperparameters and general_hyperparameters (for test)
        personalization_train_batch_size = personalization_hyperparameters['batch_size']
        personalization_val_batch_size = personalization_hyperparameters['batch_size']
        personalized_test_batch_size = general_hyperparameters['batch_size'] # Use general batch size for final test eval

        # Adjust batch sizes if larger than dataset size, ensure min 1 if dataset not empty
        if len(X_train_pat) > 0: personalization_train_batch_size = max(1, min(personalization_train_batch_size, len(X_train_pat)))
        if len(X_val_pat) > 0: personalization_val_batch_size = max(1, min(personalization_val_batch_size, len(X_val_pat)))
        if len(X_test_pat) > 0: personalized_test_batch_size = max(1, min(personalized_test_batch_size, len(X_test_pat)))


        train_dataset_pat = SeizureDataset(X_train_pat, y_train_pat, seq_len=expected_seq_len_sliced, num_features=expected_num_features_sliced)
        val_dataset_pat = SeizureDataset(X_val_pat, y_val_pat, seq_len=expected_seq_len_sliced, num_features=expected_num_features_sliced)
        test_dataset_pat = SeizureDataset(X_test_pat, y_test_pat, seq_len=expected_seq_len_sliced, num_features=expected_num_features_sliced)

        try:
            train_dataloader_pat = DataLoader(train_dataset_pat, batch_size=personalization_train_batch_size, shuffle=True, num_workers=num_workers_pat, persistent_workers=persistent_workers_pat)
            val_dataloader_pat = DataLoader(val_dataset_pat, batch_size=personalization_val_batch_size, shuffle=False, num_workers=num_workers_pat, persistent_workers=persistent_workers_pat)
            test_dataloader_pat = DataLoader(test_dataset_pat, batch_size=personalized_test_batch_size, shuffle=False, num_workers=num_workers_pat, persistent_workers=persistent_workers_pat) # Corrected batch size name
        except Exception as e:
            print(f"Error creating patient {current_patient_id} dataloaders: {e}. Skipping personalization.")
            # Clean up
            del train_dataset_pat, val_dataset_pat, test_dataset_pat
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue # Move to next patient


        # --- Step 3: Evaluate the LOPO general model on this patient's test data (Before Personalization) ---
        print(f"Evaluating LOPO general model on patient {current_patient_id}'s test data (Before Personalization)...")
        # Model instance for evaluation needs to have the correct architecture and load the LOPO weights
        try:
            lopo_general_model_instance_eval = ModelClass(expected_num_features_sliced, expected_seq_len_sliced).to(DEVICE)
            lopo_general_model_instance_eval.load_state_dict(lopo_general_model_state_dict)
        except (ValueError, RuntimeError, Exception) as e:
             print(f"Error instantiating or loading LOPO general model state for evaluation for patient {current_patient_id}: {e}. Skipping patient.")
             # Clean up dataloaders before continuing
             del train_dataloader_pat, val_dataloader_pat, test_dataloader_pat
             del train_dataset_pat, val_dataset_pat, test_dataset_pat
             gc.collect()
             if torch.cuda.is_available(): torch.cuda.empty_cache()
             continue # Move to next patient


        metrics_before = evaluate_pytorch_model(lopo_general_model_instance_eval, test_dataloader_pat, nn.BCELoss(), DEVICE)
        print(f"Before Personalization: Accuracy={metrics_before['accuracy']:.4f}, Precision={metrics_before['precision']:.4f}, Recall={metrics_before['recall']:.4f}, F1={metrics_before['f1_score']:.4f}, AUC={metrics_before['auc_roc']:.4f}")

        del lopo_general_model_instance_eval
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


        # --- Step 4: Create and Fine-tune a new model instance for personalization ---
        try:
            personalized_model = ModelClass(expected_num_features_sliced, expected_seq_len_sliced).to(DEVICE)
            # Load the LOPO general model state as the starting point
            personalized_model.load_state_dict(lopo_general_model_state_dict)
        except (ValueError, RuntimeError, Exception) as e:
             print(f"Error instantiating or loading LOPO general model state for fine-tuning for patient {current_patient_id}: {e}. Skipping patient.")
             # Clean up dataloaders before continuing
             del train_dataloader_pat, val_dataloader_pat, test_dataloader_pat
             del train_dataset_pat, val_dataset_pat, test_dataset_pat
             gc.collect()
             if torch.cuda.is_available(): torch.cuda.empty_cache()
             # Store before metrics, but indicate personalization failed
             personalization_results[current_patient_id] = {
                 "before": metrics_before,
                 "after": {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]}
             }
             continue # Move to next patient


        # Calculate class weights for the patient's fine-tuning data
        classes_pat = np.unique(y_train_pat)
        class_weights_pat_dict = None
        if len(classes_pat) == 2:
             class_weights_pat_np = class_weight.compute_class_weight(
                 'balanced', classes=classes_pat, y=y_train_pat
             )
             class_weights_pat_dict = {cls: weight for cls, weight in zip(classes_pat, class_weights_pat_np)}
             # print(f"Computed patient {current_patient_id} fine-tuning class weights: {class_weights_pat_dict}") # Suppress
        else:
             # print(f"Warning: Only one class for patient {current_patient_id} fine-tuning data. No class weights applied.") # Suppress
             pass # No class weights if only one class


        # Only attempt fine-tuning if there's training data for this patient
        if len(train_dataset_pat) > 0:
            print(f"Starting fine-tuning for patient {current_patient_id}...")
            personalized_model = train_pytorch_model(
                personalized_model,
                train_dataloader_pat,
                val_dataloader_pat,
                epochs=personalization_hyperparameters['epochs'],
                learning_rate=personalization_hyperparameters['learning_rate'],
                class_weights=class_weights_pat_dict,
                save_best_model_path=None, # Don't save personalized models here, just get the final state
                desc=f"Fine-tuning {current_patient_id}"
            )
        else:
            print(f"Warning: No fine-tuning data for patient {current_patient_id}. Skipping fine-tuning.")
            # The personalized_model instance is already initialized with LOPO general weights,
            # it just won't be trained further.

        # --- Step 5: Evaluate the personalized model on this patient's test data (After Personalization) ---
        print(f"Evaluating personalized model on patient {current_patient_id}'s test data (After Personalization)...")
        metrics_after = evaluate_pytorch_model(personalized_model, test_dataloader_pat, nn.BCELoss(), DEVICE)
        print(f"After Personalization: Accuracy={metrics_after['accuracy']:.4f}, Precision={metrics_after['precision']:.4f}, Recall={metrics_after['recall']:.4f}, F1={metrics_after['f1_score']:.4f}, AUC={metrics_after['auc_roc']:.4f}")


        # Store results for this patient
        personalization_results[current_patient_id] = {
            "before": metrics_before,
            "after": metrics_after
        }

        # Clean up memory for the current patient's data/model/dataloaders
        del train_dataset_pat, val_dataset_pat, test_dataset_pat
        del train_dataloader_pat, val_dataloader_pat, test_dataloader_pat
        del personalized_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    return personalization_results

# --- Helper to get sensor indices and filter patients for a combination ---
# This function is slightly repurposed. It now finds which patients have ALL required sensors
# and gets the indices for slicing from the full BASE_SENSORS segment array.
def get_patients_and_indices_for_combination(all_processed_patient_data, sensor_combination):
    """
    Filters patients from the full list to those having all sensors in the combination,
    and gets the correct column indices for slicing their segments.

    Args:
        all_processed_patient_data (list): List of (patient_id, segments, labels, found_sensors)
                                        from initial processing (segments have len(BASE_SENSORS) features).
        sensor_combination (list): List of sensor names (e.g., ['HR', 'EDA']) for the current combination.

    Returns:
        tuple: (patients_suitable_for_combination, sensor_combination_indices)
               patients_suitable_for_combination: list of (patient_id, segments_all_sensors, labels, found_sensors)
                                                 subset of the input list.
               sensor_combination_indices: list of integer indices corresponding to
                                            the sensor columns to use from BASE_SENSORS.
               Returns ([], []) if no patients are suitable or invalid combination.
    """
    combination_name = "_".join(sensor_combination).upper()

    print(f"\nChecking patients for sensor combination: {combination_name}")

    patients_suitable_for_combination = []

    # Get indices for the sensors in the current combination (relative to BASE_SENSORS order)
    try:
        # Ensure sensors in the combination are in BASE_SENSORS and get their indices
        sensor_combination_indices = [BASE_SENSORS.index(s) for s in sensor_combination]
        # Sort indices to maintain consistent column order after slicing
        sensor_combination_indices.sort()
    except ValueError as e:
        print(f"Error: Sensor '{e}' in combination {sensor_combination} not found in BASE_SENSORS {BASE_SENSORS}. Cannot process this combination.")
        return [], [] # Cannot proceed with invalid combination
    except Exception as e:
        print(f"An unexpected error occurred getting sensor indices for combination {sensor_combination}: {e}")
        return [], [] # Cannot proceed with invalid combination


    for patient_data_tuple in all_processed_patient_data:
         patient_id, segments_all_sensors, labels, found_sensors = patient_data_tuple
         # Check if the patient has *all* sensors required for this combination
         # `found_sensors` is the list of sensor names actually found for this patient
         if all(s in found_sensors for s in sensor_combination):
              # Check if segments have the correct number of features (should be len(BASE_SENSORS))
              # And if there are actual segments and both classes present
              if segments_all_sensors.shape[2] == len(BASE_SENSORS) and len(segments_all_sensors) > 0 and len(np.unique(labels)) > 1:
                 patients_suitable_for_combination.append(patient_data_tuple) # Append the full patient data tuple
              # else: print(f"Skipping patient {patient_id} for combination {combination_name}: Segments shape mismatch ({segments_all_sensors.shape[2]} vs {len(BASE_SENSORS)}) or no segments/single class.") # Suppress
         # else: print(f"Skipping patient {patient_id} for combination {combination_name}: Missing required sensors {set(sensor_combination) - set(found_sensors)}.") # Suppress


    if not patients_suitable_for_combination:
        print(f"No patients found with all sensors for combination: {combination_name}. Skipping this combination.")
        return [], [] # Return empty if no suitable patients

    print(f"Found {len(patients_suitable_for_combination)} patients suitable for combination: {combination_name}.")
    return patients_suitable_for_combination, sensor_combination_indices


def print_personalization_summary(personalization_results, output_file=None):
    """ Prints a summary table of personalization results to console or file. """
    # Determine where to print (console or file)
    def print_func(*args, **kwargs):
        if output_file:
            print(*args, **kwargs, file=output_file)
        else:
            print(*args, **kwargs)

    print_func("\n--- Personalized Model Performance (Per Patient Summary) ---")
    if not personalization_results:
         print_func("No personalization results available.")
         return

    print_func("Patient ID | Accuracy Before | Accuracy After | Change")
    print_func("-----------------------------------------------------")

    total_change = 0
    count_valid_patients = 0

    for patient_id, results in personalization_results.items():
        # Check if both 'before' and 'after' metrics exist and are valid dictionaries
        if isinstance(results.get('before'), dict) and isinstance(results.get('after'), dict):
            acc_before = results['before'].get('accuracy', 0.0)
            acc_after = results['after'].get('accuracy', 0.0)
            change = acc_after - acc_before

            # Check if the 'after' evaluation confusion matrix indicates data was processed
            cm_after = results['after'].get('confusion_matrix', [[0,0],[0,0]])
            if isinstance(cm_after, list) and len(cm_after) == 2 and len(cm_after[0]) == 2 and sum(sum(row) for row in cm_after) > 0:
                 print_func(f"{patient_id:<10} | {acc_before:.4f}        | {acc_after:.4f}       | {change:.4f}")
                 total_change += change
                 count_valid_patients += 1
            else:
                 # Patient was in results dict, but after evaluation had no data (e.g., empty test set)
                 print_func(f"{patient_id:<10} | {acc_before:.4f}        | N/A            | N/A") # Show before, but N/A for after if evaluation failed
                 # Do NOT include in average change calculation
                 # print_func(f"--- Debug: Patient {patient_id} skipped average calculation due to empty after test set.") # Suppress

        else:
             # Patient was in results dict but metrics structure is unexpected (e.g., LOPO failed earlier)
             print_func(f"{patient_id:<10} | N/A             | N/A            | N/A") # Indicate missing data
             # Do NOT include in average change calculation
             # print_func(f"--- Debug: Patient {patient_id} skipped average calculation due to missing metrics.") # Suppress


    print_func("-----------------------------------------------------")
    if count_valid_patients > 0:
        average_change = total_change / count_valid_patients
        print_func(f"Average Accuracy Improvement (across {count_valid_patients} patients with valid evaluation data): {average_change:.4f}")
    else:
        print_func("No valid personalized patient results to summarize average improvement.")


# --- Main Execution ---
if __name__ == "__main__":
    # Ensure the base output directory exists at the very beginning
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_patient_folders = [f.path for f in os.scandir(DATA_ROOT_DIR) if f.is_dir() and f.name.startswith('MSEL_')]

    if not all_patient_folders:
        print(f"No patient directories starting with 'MSEL_' found in {DATA_ROOT_DIR}.")
        exit()

    print(f"Found {len(all_patient_folders)} patient directories.")

    # --- Step 0: Process data for all patients (load, sync, segment using ALL BASE_SENSORS) ---
    # processed_patient_data will store (patient_id, segments_all_sensors, labels, found_sensors_list)
    # segments_all_sensors will have shape (N, L, len(BASE_SENSORS)) with 0s for missing original sensors
    processed_patient_data = []
    print("\n--- Starting Initial Patient Data Processing ---")
    # Use the first HP values for data processing parameters
    current_sampling_freq_hz = TUNABLE_SAMPLING_FREQ_HZ[0]
    current_pre_ictal_window_min = TUNABLE_PRE_ICTAL_WINDOW_MINUTES[0]
    current_interictal_buffer_min = TUNABLE_INTERICTAL_BUFFER_MINUTES[0]
    expected_seq_len_initial_processing = int(SEGMENT_DURATION_SECONDS * current_sampling_freq_hz)
    expected_seq_len_initial_processing = max(1, expected_seq_len_initial_processing)
    expected_num_features_initial_processing = len(BASE_SENSORS)


    for patient_folder in tqdm(all_patient_folders, desc="Initial Patient Data Processing", leave=True):
        # Note: prepare_patient_data now uses the first values from TUNABLE_s
        patient_data = prepare_patient_data(patient_folder)
        if patient_data:
             # Check the shape of the segments returned by prepare_patient_data
             patient_id, segments, labels, found_sensors = patient_data
             if segments.shape[1] == expected_seq_len_initial_processing and segments.shape[2] == expected_num_features_initial_processing:
                processed_patient_data.append(patient_data)
             # else: print(f"Skipped patient {patient_id}: Segments shape mismatch after processing ({segments.shape} vs expected {(len(segments), expected_seq_len_initial_processing, expected_num_features_initial_processing)})") # Suppress
        # else: print(f"Skipped initial processing for patient {os.path.basename(patient_folder)}.") # Suppress

    if not processed_patient_data:
        print("No valid patient data was processed with current data parameters. Exiting.")
        exit()
    else:
        print(f"\nSuccessfully processed initial data for {len(processed_patient_data)} patients.")

    # Store the actual sequence length and feature count determined during initial processing
    # All segments in processed_patient_data should have the same shape
    if processed_patient_data:
        actual_seq_len = processed_patient_data[0][1].shape[1]
        actual_num_features_all_sensors = processed_patient_data[0][1].shape[2] # Should be len(BASE_SENSORS)
        print(f"Actual segment shape after initial processing: (N, {actual_seq_len}, {actual_num_features_all_sensors})")
    else:
        actual_seq_len = int(SEGMENT_DURATION_SECONDS * TUNABLE_SAMPLING_FREQ_HZ[0]) # Fallback calculation
        actual_seq_len = max(1, actual_seq_len)
        actual_num_features_all_sensors = len(BASE_SENSORS) # Fallback
        print(f"Warning: No processed data, using fallback segment shape: (N, {actual_seq_len}, {actual_num_features_all_sensors})")


    # --- Phase 1: Train and Evaluate Overall General Model (on all data) ---
    print(f"\n{'='*60}")
    print("PHASE 1: TRAINING & EVALUATING OVERALL GENERAL MODEL")
    print(f"{'='*60}\n")

    # Combine data from all processed patients for the overall general model
    overall_general_segments_raw = []
    overall_general_labels_raw = []
    # Ensure we use segments with all BASE_SENSORS features
    for patient_id, segments_all_sensors, labels, found_sensors in processed_patient_data:
         overall_general_segments_raw.append(segments_all_sensors)
         overall_general_labels_raw.append(labels)

    overall_general_segments_combined = np.concatenate(overall_general_segments_raw, axis=0)
    overall_general_labels_combined = np.concatenate(overall_general_labels_raw, axis=0)

    # Check for sufficient data for overall general training
    if len(overall_general_segments_combined) == 0 or len(np.unique(overall_general_labels_combined)) < 2:
        print("No data or only one class available for Overall General Model training. Skipping Phase 1.")
        # Skip phase 1 and its reporting if no data
        overall_general_model_metrics = {}
        # Create a dummy model state or None to indicate failure, will affect Phase 2 starting point if needed
        overall_general_model_state = None # Indicate failure/skip
    else:

        print(f"Overall General Combined data shape: {overall_general_segments_combined.shape}")
        print(f"Overall General Combined labels shape: {overall_general_labels_combined.shape}")

        # Perform 60/20/20 split for Overall General Model
        # Ensure enough data for splitting - min_samples=1 for stratify
        if len(overall_general_segments_combined) < 3: # Need at least 3 samples for 60/20/20
             print(f"Warning: Not enough data ({len(overall_general_segments_combined)} samples) for Overall General Model split. Skipping training.")
             overall_general_model_metrics = {}
             overall_general_model_state = None # Indicate failure/skip
        else:
            try:
                X_train_og, X_temp_og, y_train_og, y_temp_og = train_test_split(
                    overall_general_segments_combined, overall_general_labels_combined, test_size=0.4, random_state=SEED, stratify=overall_general_labels_combined
                )
                X_val_og, X_test_og, y_val_og, y_test_og = train_test_split(
                    X_temp_og, y_temp_og, test_size=0.5, random_state=SEED, stratify=y_temp_og
                )
            except ValueError as e:
                 print(f"Warning: Overall General Model data split failed: {e}. Skipping training.")
                 overall_general_model_metrics = {}
                 overall_general_model_state = None # Indicate failure/skip
            except Exception as e:
                 print(f"An unexpected error occurred during Overall General Model data split: {e}. Skipping training.")
                 overall_general_model_metrics = {}
                 overall_general_model_state = None # Indicate failure/skip


        # Only proceed if splits are valid and contain both classes
        if 'X_train_og' in locals() and len(X_train_og) > 0 and len(X_val_og) > 0 and len(X_test_og) > 0 and len(np.unique(y_train_og)) > 1 and len(np.unique(y_val_og)) > 1 and len(np.unique(y_test_og)) > 1:

            print(f"Overall General Train shape: {X_train_og.shape}, Val shape: {X_val_og.shape}, Test shape: {X_test_og.shape}")

            # Use the actual shape determined during initial processing
            overall_general_train_dataset = SeizureDataset(X_train_og, y_train_og, seq_len=actual_seq_len, num_features=actual_num_features_all_sensors)
            overall_general_val_dataset = SeizureDataset(X_val_og, y_val_og, seq_len=actual_seq_len, num_features=actual_num_features_all_sensors)
            overall_general_test_dataset = SeizureDataset(X_test_og, y_test_og, seq_len=actual_seq_len, num_features=actual_num_features_all_sensors)

            num_workers_og = os.cpu_count() // 2 or 1
            persistent_workers_og = True if num_workers_og > 0 and hasattr(DataLoader, 'persistent_workers') else False

            # Adjust batch sizes if larger than dataset size, ensure min 1 if dataset not empty
            og_train_batch_size = TUNABLE_BATCH_SIZE[0]
            if len(overall_general_train_dataset) > 0: og_train_batch_size = max(1, min(og_train_batch_size, len(overall_general_train_dataset)))
            og_val_batch_size = TUNABLE_BATCH_SIZE[0]
            if len(overall_general_val_dataset) > 0: og_val_batch_size = max(1, min(og_val_batch_size, len(overall_general_val_dataset)))
            og_test_batch_size = TUNABLE_BATCH_SIZE[0]
            if len(overall_general_test_dataset) > 0: og_test_batch_size = max(1, min(og_test_batch_size, len(overall_general_test_dataset)))


            overall_general_train_dataloader = DataLoader(overall_general_train_dataset, batch_size=og_train_batch_size, shuffle=True, num_workers=num_workers_og, persistent_workers=persistent_workers_og)
            overall_general_val_dataloader = DataLoader(overall_general_val_dataset, batch_size=og_val_batch_size, shuffle=False, num_workers=num_workers_og, persistent_workers=persistent_workers_og)
            overall_general_test_dataloader = DataLoader(overall_general_test_dataset, batch_size=og_test_batch_size, shuffle=False, num_workers=num_workers_og, persistent_workers=persistent_workers_og)

            # Calculate class weights for the overall general training data
            classes_og = np.unique(y_train_og)
            class_weights_og_dict = None
            if len(classes_og) == 2:
                 class_weights_og_np = class_weight.compute_class_weight(
                     'balanced', classes=classes_og, y=y_train_og
                 )
                 class_weights_og_dict = {cls: weight for cls, weight in zip(classes_og, class_weights_og_np)}
                 print(f"Computed Overall General class weights: {class_weights_og_dict}")
            else:
                 print("Warning: Only one class for Overall General training data. No class weights applied.")
                 pass # No class weights

            # Instantiate the Overall General Model (using BASE_SENSORS features)
            input_channels_og = actual_num_features_all_sensors
            seq_len_og = actual_seq_len
            ModelClass_og = get_model_class(MODEL_TYPES_TO_RUN[0]) # Using the first model type for overall general model

            try:
                overall_general_model = ModelClass_og(input_channels_og, seq_len_og).to(DEVICE)
            except (ValueError, Exception) as e:
                print(f"Error instantiating Overall General Model: {e}. Skipping training.")
                # Clean up dataloaders
                del overall_general_train_dataloader, overall_general_val_dataloader, overall_general_test_dataloader
                del overall_general_train_dataset, overall_general_val_dataset, overall_general_test_dataset
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                overall_general_model_metrics = {}
                overall_general_model_state = None # Indicate failure/skip

            # Only attempt training if dataloader is not empty
            if 'overall_general_train_dataloader' in locals() and len(overall_general_train_dataloader.dataset) > 0:
                 print("\nStarting Overall General Model training...")
                 overall_general_model = train_pytorch_model(
                     overall_general_model,
                     overall_general_train_dataloader,
                     overall_general_val_dataloader,
                     epochs=TUNABLE_GENERAL_MODEL_EPOCHS[0],
                     learning_rate=TUNABLE_GENERAL_MODEL_LR[0],
                     class_weights=class_weights_og_dict,
                     save_best_model_path=os.path.join(OUTPUT_DIR, f'overall_general_model_{MODEL_TYPES_TO_RUN[0]}.pth'),
                     desc=f"Training Overall General ({MODEL_TYPES_TO_RUN[0]})"
                 )
                 overall_general_model_state = overall_general_model.state_dict() # Get state dict after training/loading best

                 print("\nEvaluating Overall General Model on its test set...")
                 overall_general_model_metrics = evaluate_pytorch_model(overall_general_model, overall_general_test_dataloader, nn.BCELoss(), DEVICE)
                 print(f"Overall General Model Metrics: Accuracy={overall_general_model_metrics['accuracy']:.4f}, "
                       f"Precision={overall_general_model_metrics['precision']:.4f}, Recall={overall_general_model_metrics['recall']:.4f}, "
                       f"F1 Score={overall_general_model_metrics['f1_score']:.4f}, AUC-ROC={overall_general_model_metrics['auc_roc']:.4f}")

                 # Clean up Overall General Model resources
                 del overall_general_model, overall_general_train_dataloader, overall_general_val_dataloader, overall_general_test_dataloader
                 del overall_general_train_dataset, overall_general_val_dataset, overall_general_test_dataset
                 gc.collect()
                 if torch.cuda.is_available(): torch.cuda.empty_cache()

            else:
                 print("Overall General training dataloader is empty. Skipping training and evaluation.")
                 overall_general_model_metrics = {}
                 overall_general_model_state = None # Indicate failure/skip


    # --- Define the main output file ---
    # Use a timestamp for a unique filename for the entire run
    output_filename = os.path.join(OUTPUT_DIR, f'seizure_prediction_results_{time.strftime("%Y%m%d_%H%M%S")}.txt')

    with open(output_filename, 'w') as output_file:
        output_file.write(f"Seizure Prediction Results\n")
        output_file.write(f"Run Date: {time.ctime()}\n")
        output_file.write(f"Data Directory: {DATA_ROOT_DIR}\n")
        output_file.write(f"Processed {len(processed_patient_data)} patients initially.\n")
        output_file.write(f"Models Run for Personalization: {MODEL_TYPES_TO_RUN}\n")
        output_file.write(f"Sensor Combination Testing Enabled: {ENABLE_ADAPTIVE_SENSORS}\n")
        # Print the actual HP values used (first value from lists)
        output_file.write(f"\nHyperparameters Used:\n")
        output_file.write(f"  SEGMENT_DURATION_SECONDS: {SEGMENT_DURATION_SECONDS}\n")
        output_file.write(f"  PRE_ICTAL_WINDOW_MINUTES: {TUNABLE_PRE_ICTAL_WINDOW_MINUTES[0]}\n")
        output_file.write(f"  INTERICTAL_BUFFER_MINUTES: {TUNABLE_INTERICTAL_BUFFER_MINUTES[0]}\n")
        output_file.write(f"  BASE_SENSORS: {BASE_SENSORS}\n")
        output_file.write(f"  SAMPLING_FREQ_HZ: {TUNABLE_SAMPLING_FREQ_HZ[0]}\n")
        output_file.write(f"  CONV_FILTERS: {TUNABLE_CONV_FILTERS[0]}\n")
        output_file.write(f"  CONV_KERNEL_SIZE: {TUNABLE_CONV_KERNEL_SIZE[0]}\n")
        output_file.write(f"  POOL_SIZE: {TUNABLE_POOL_SIZE[0]}\n")
        output_file.write(f"  LSTM_UNITS: {TUNABLE_LSTM_UNITS[0]}\n")
        output_file.write(f"  DENSE_UNITS: {TUNABLE_DENSE_UNITS[0]}\n")
        output_file.write(f"  GENERAL_MODEL_EPOCHS (LOPO & Overall): {TUNABLE_GENERAL_MODEL_EPOCHS[0]}\n")
        output_file.write(f"  PERSONALIZATION_EPOCHS (Fine-tune): {TUNABLE_PERSONALIZATION_EPOCHS[0]}\n")
        output_file.write(f"  GENERAL_MODEL_LR (LOPO & Overall): {TUNABLE_GENERAL_MODEL_LR[0]}\n")
        output_file.write(f"  PERSONALIZATION_LR (Fine-tune): {TUNABLE_PERSONALIZATION_LR[0]}\n")
        output_file.write(f"  BATCH_SIZE (General/LOPO/Patient Test): {TUNABLE_BATCH_SIZE[0]}\n")
        output_file.write(f"  PERSONALIZATION_BATCH_SIZE (Patient Train/Val): {TUNABLE_PERSONALIZATION_BATCH_SIZE[0]}\n")


        if not ENABLE_ADAPTIVE_SENSORS:
            output_file.write(f"Running only for Base Sensor Set: {BASE_SENSORS}\n")
        output_file.write("=" * 80 + "\n\n")

        # --- Write Overall General Model Results to file ---
        output_file.write(f"\n{'='*60}\n")
        output_file.write("PHASE 1 RESULTS: OVERALL GENERAL MODEL\n")
        output_file.write(f"{'='*60}\n\n")

        output_file.write(f"Model Type: {MODEL_TYPES_TO_RUN[0]} (using all BASE_SENSORS)\n") # Assuming first model type for overall
        output_file.write("--- General Model Performance (on Overall General Test Set) ---\n")
        if overall_general_model_metrics:
            output_file.write(f"Accuracy={overall_general_model_metrics['accuracy']:.4f}\n")
            output_file.write(f"Precision={overall_general_model_metrics['precision']:.4f}\n")
            output_file.write(f"Recall={overall_general_model_metrics['recall']:.4f}\n")
            output_file.write(f"F1 Score={overall_general_model_metrics['f1_score']:.4f}\n")
            output_file.write(f"AUC-ROC={overall_general_model_metrics['auc_roc']:.4f}\n")
            output_file.write(f"Confusion Matrix:\n{overall_general_model_metrics['confusion_matrix']}\n")
        else:
            output_file.write("Overall General Model training failed or was skipped due to insufficient data.\n")
        output_file.write("\n")


        # --- Phase 2: Per-Patient Personalization (with LOPO) ---
        print(f"\n{'='*60}")
        print("PHASE 2: PER-PATIENT PERSONALIZATION (using LOPO)")
        print(f"{'='*60}\n")
        output_file.write(f"\n{'='*60}\n")
        output_file.write("PHASE 2 RESULTS: PER-PATIENT PERSONALIZATION (using LOPO)\n")
        output_file.write(f"{'='*60}\n\n")


        all_personalization_results_by_combo_model = {} # Stores results nested by model_type and sensor_combination_name

        # --- Loop through each model type for personalization ---
        for current_model_type in MODEL_TYPES_TO_RUN:
            all_personalization_results_by_combo_model[current_model_type] = {} # Nested dict

            # --- Loop through each sensor combination for personalization ---
            for current_combination in SENSOR_COMBINATIONS:
                combination_name = "_".join(current_combination).upper() # Consistent naming

                print(f"\n{'='*40}")
                print(f"RUNNING PERSONALIZATION FOR: Model {current_model_type} + Sensors {combination_name}")
                print(f"{'='*40}\n")

                output_file.write(f"\n\n{'#'*40}\n")
                output_file.write(f"PERSONALIZATION RESULTS FOR MODEL: {current_model_type}, SENSORS: {combination_name}\n")
                output_file.write(f"{'#'*40}\n\n")

                # Get list of patients suitable for this combination and the sensor indices
                patients_suitable_for_combination, sensor_combination_indices = get_patients_and_indices_for_combination(
                     processed_patient_data,
                     current_combination # Pass the list of sensor names
                )

                if not patients_suitable_for_combination:
                     print(f"Skipping personalization for {current_model_type} + {combination_name}: No suitable patients found.")
                     output_file.write(f"Skipping personalization for {current_model_type} + {combination_name}: No suitable patients found.\n\n")
                     all_personalization_results_by_combo_model[current_model_type][combination_name] = {
                          'personalization_results': {},
                          'avg_personalized_metrics': None,
                          'num_suitable_patients': len(patients_suitable_for_combination)
                     }
                     continue # Move to the next combination


                print(f"Proceeding with personalization for {len(patients_suitable_for_combination)} suitable patients.")
                output_file.write(f"Proceeding with personalization for {len(patients_suitable_for_combination)} suitable patients.\n\n")

                # Get hyperparameters for personalization and general model (for LOPO)
                general_hps_for_lopo = {
                    'epochs': TUNABLE_GENERAL_MODEL_EPOCHS[0],
                    'learning_rate': TUNABLE_GENERAL_MODEL_LR[0],
                    'batch_size': TUNABLE_BATCH_SIZE[0] # Use general batch size for LOPO training
                }
                personalization_hps = {
                    'epochs': TUNABLE_PERSONALIZATION_EPOCHS[0],
                    'learning_rate': TUNABLE_PERSONALIZATION_LR[0],
                    'batch_size': TUNABLE_PERSONALIZATION_BATCH_SIZE[0] # Use personalization batch size for fine-tuning
                }
                model_hps = {
                    'conv_filters': TUNABLE_CONV_FILTERS[0],
                    'conv_kernel_size': TUNABLE_CONV_KERNEL_SIZE[0],
                    'pool_size': TUNABLE_POOL_SIZE[0],
                    'lstm_units': TUNABLE_LSTM_UNITS[0],
                    'dense_units': TUNABLE_DENSE_UNITS[0],
                }


                # --- Perform Personalization for this combination (with LOPO handled inside) ---
                # Pass the *full* processed_patient_data list and the sensor indices
                personalization_results = perform_personalization_pytorch_lopo(
                    patients_suitable_for_combination, # Pass only the suitable patients
                    current_model_type,
                    current_combination, # Pass the list of sensor names
                    general_hps_for_lopo, # Pass general HPs for LOPO training inside
                    personalization_hps, # Pass personalization HPs for fine-tuning inside
                    model_hps # Pass model architecture HPs
                )

                all_personalization_results_by_combo_model[current_model_type][combination_name] = {
                     'personalization_results': personalization_results, # Store per-patient results
                     'num_suitable_patients': len(patients_suitable_for_combination) # Store patient count
                }


                # --- Summarize Personalized Model Performance for this combination ---
                # Write per-patient summary to file
                print_personalization_summary(personalization_results, output_file=output_file)
                # Print per-patient summary to console too for monitoring
                print_personalization_summary(personalization_results, output_file=None)

                # Calculate and Write Average Personalized Model Performance for this combination
                metrics_after_list = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'auc_roc': []}
                count_valid_patients_pers = 0

                for patient_id, results in personalization_results.items():
                    # Check if the 'after' personalization metrics are valid for this patient
                    if isinstance(results.get('after'), dict) and 'accuracy' in results.get('after', {}):
                        # Check if the test set for this patient had data (via CM)
                         cm_after = results['after'].get('confusion_matrix', [[0,0],[0,0]])
                         if isinstance(cm_after, list) and len(cm_after) == 2 and len(cm_after[0]) == 2 and sum(sum(row) for row in cm_after) > 0:
                             count_valid_patients_pers += 1
                             metrics_after_list['accuracy'].append(results['after']['accuracy'])
                             metrics_after_list['precision'].append(results['after']['precision'])
                             metrics_after_list['recall'].append(results['after']['recall'])
                             metrics_after_list['f1_score'].append(results['after']['f1_score'])
                             metrics_after_list['auc_roc'].append(results['after']['auc_roc'])
                         # else: print(f"--- Debug: Patient {patient_id} CM check failed for averaging.") # Suppress
                    # else: print(f"--- Debug: Patient {patient_id} 'after' metrics missing or invalid for averaging.") # Suppress


                output_file.write("\n--- Personalized Model Performance (Average Across Patients) ---\n")
                if count_valid_patients_pers > 0:
                    avg_metrics = {metric: np.mean(metrics_after_list[metric]) for metric in metrics_after_list}
                    all_personalization_results_by_combo_model[current_model_type][combination_name]['avg_personalized_metrics'] = avg_metrics # Store for final comparison

                    output_file.write(f"Average Accuracy={avg_metrics['accuracy']:.4f} (across {count_valid_patients_pers} patients with valid evaluation data)\n")
                    output_file.write(f"Average Precision={avg_metrics['precision']:.4f}\n")
                    output_file.write(f"Average Recall={avg_metrics['recall']:.4f}\n")
                    output_file.write(f"Average F1 Score={avg_metrics['f1_score']:.4f}\n")
                    output_file.write(f"Average AUC-ROC={avg_metrics['auc_roc']:.4f}\n")

                else:
                    output_file.write("No valid personalized patient results to average.\n")
                    all_personalization_results_by_combo_model[current_model_type][combination_name]['avg_personalized_metrics'] = None # Store None

                output_file.write("\n") # Add space


        # --- Final Personalization Comparison Summary Table ---
        output_file.write(f"\n\n{'='*60}\n")
        output_file.write(f"SUMMARY OF PERSONALIZATION RESULTS FOR MODEL: {current_model_type}\n")
        output_file.write(f"{'='*60}\n\n")

        output_file.write("Personalized Model Performance Comparison (Average Across Patients):\n")
        output_file.write("Sensors    | Patients | Avg Acc  | Avg Prec  | Avg Rec  | Avg F1   | Avg AUC  | Avg Acc Change\n")
        output_file.write("--------------------------------------------------------------------------------------------\n")

        # Sort combinations by name for consistent table output
        for combo_name in sorted(all_personalization_results_by_combo_model[current_model_type].keys()):
             combo_results = all_personalization_results_by_combo_model[current_model_type].get(combo_name, {}) # Use .get
             avg_metrics = combo_results.get('avg_personalized_metrics', None)
             num_suitable_patients = combo_results.get('num_suitable_patients', 0)

             # Calculate average change for this combination if possible
             total_change_combo = 0
             count_valid_patients_combo_change = 0
             for patient_id, pers_results in combo_results.get('personalization_results', {}).items():
                  if isinstance(pers_results.get('before'), dict) and isinstance(pers_results.get('after'), dict):
                       acc_before = pers_results['before'].get('accuracy', 0.0)
                       acc_after = pers_results['after'].get('accuracy', 0.0)
                       cm_after = pers_results['after'].get('confusion_matrix', [[0,0],[0,0]])
                       if isinstance(cm_after, list) and len(cm_after) == 2 and len(cm_after[0]) == 2 and sum(sum(row) for row in cm_after) > 0:
                            total_change_combo += (acc_after - acc_before)
                            count_valid_patients_combo_change += 1

             avg_change_combo = total_change_combo / count_valid_patients_combo_change if count_valid_patients_combo_change > 0 else 0.0

             if avg_metrics is not None:
                 output_file.write(f"{combo_name:<10} | {num_suitable_patients:<8} | {avg_metrics['accuracy']:.4f} | {avg_metrics['precision']:.4f} | {avg_metrics['recall']:.4f} | {avg_metrics['f1_score']:.4f} | {avg_metrics['auc_roc']:.4f} | {avg_change_combo:.4f}\n")
             else:
                 output_file.write(f"{combo_name:<10} | {num_suitable_patients:<8} | N/A      | N/A       | N/A      | N/A      | N/A      | N/A\n")

        output_file.write("--------------------------------------------------------------------------------------------\n\n")


    print("\n--- All Runs Complete ---")
    print(f"Results saved in the '{OUTPUT_DIR}' directory.")
    print(f"Output file: {output_filename}")