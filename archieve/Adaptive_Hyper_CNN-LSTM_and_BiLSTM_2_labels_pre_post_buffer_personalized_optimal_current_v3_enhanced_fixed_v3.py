import concurrent.futures  # Import for parallel processing
import gc
import glob
import itertools  # For combinations
import logging  # Import logging module
import math
import os
import pickle
import random
import sys  # Import sys for logging to console
import time
from collections import OrderedDict  # Added for better HP printing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.utils import class_weight
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# --- Configuration ---
# Replace with the actual path to your 'data_9' folder
DATA_ROOT_DIR = "F:\\data_9"
# Directory to save results files and models
OUTPUT_DIR = "processed_data_pytorch_adaptive_pre_post_buffer_lovo_personalization_v2_enhanced"

# Ensure the base output directory exists early
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Feature Flags ---
# Set to True to run all model types in MODEL_TYPES_TO_RUN; False to run only the first one
RUN_ALL_MODEL_TYPES = True
# Set to True to run all sensor combinations; False to run only the full BASE_SENSORS set
ENABLE_ADAPTIVE_SENSORS = False
# Set to True to iterate through all combinations of TUNABLE_ hyperparameters; False to use only the first value from each list
ENABLE_TUNABLE_HYPERPARAMETERS = True
# Set to True to run Phase 2 (Personalization/LOPO); False to only run Phase 1 (Overall General Model)
ENABLE_PERSONALIZATION = False

REUSE_PROCESSED_DATA = True 

# --- Data Processing Parameters ---
SEGMENT_DURATION_SECONDS = 30

# Tunable Buffer and Data Parameters
TUNABLE_PRE_ICTAL_WINDOW_MINUTES = [
    55
]  # Time window before seizure onset considered pre-ictal

TUNABLE_PRE_ICTAL_EXCLUSION_BUFFER_MINUTES = [
    180
]  # Must be >= the corresponding TUNABLE_PRE_ICTAL_WINDOW_MINUTES.
TUNABLE_POST_ICTAL_BUFFER_MINUTES = [
    180
]  # Buffer time *after* seizure end to exclude for clean interictal.

# Target sampling frequency for resampling
TUNABLE_SAMPLING_FREQ_HZ = [1]

# Define base sensors (ensure these are the possible columns after sync/scaling)
BASE_SENSORS = ['HR', 'EDA', 'TEMP', 'ACC']


# --- Tunable Hyperparameters ---
# These lists define the values to iterate through if ENABLE_TUNABLE_HYPERPARAMETERS is True.
# If ENABLE_TUNABLE_HYPERPARAMETERS is False, only the first value from each list is used.
    # Add FIXED FROM THERE
TUNABLE_CONV_FILTERS = [    
    # [16,32,64],
    # [16,32,64,128],
    [32, 64, 128],
    [64, 128, 256],
    # [32, 64, 128, 256]
]  # Example: Added another filter combination

    # Add FIXED FROM THERE
TUNABLE_CONV_KERNEL_SIZE = [
    10   # Smaller common odd kernel size
]  # Example: Added another kernel size

TUNABLE_POOL_SIZE = [2]  # Example: Added another pool size

    # Add FIXED FROM THERE
TUNABLE_LSTM_UNITS = [
    # 64,   # Original and used in previous runs
    128   # Explore more LSTM units to increase recurrent capacity
]  # Example: Added another LSTM unit size

    # Add FIXED FROM THERE
TUNABLE_DENSE_UNITS = [
    # 32,   # Original and used in previous runs
    # 64    # Explore more Dense units for higher-level feature processing
    128
]  # Example: Added another Dense unit size

TUNABLE_GENERAL_MODEL_EPOCHS = [
    100
]  # Used for Overall General Model and LOPO General Model
TUNABLE_PERSONALIZATION_EPOCHS = [30]  # Used for fine-tuning

    # the sweetspot macam between 0.00001 to 0.0001
TUNABLE_GENERAL_MODEL_LR = [0.00001,0.00005,0.0001,0.0005,0.001,0.005]  # Used for Overall General Model and LOPO General Model

    # Add FIXED FROM THERE
TUNABLE_PERSONALIZATION_LR = [
    0.0001,   # Original and used in previous runs
    # 0.00005,  # Smaller learning rate for personalized fine-tuning
    # 0.0002    # Larger learning rate for personalized fine-tuning
]  # Used for fine-tuning
TUNABLE_BATCH_SIZE = [
    32
]  # Used for Overall General Model and LOPO General Model Train/Val/Test
TUNABLE_PERSONALIZATION_BATCH_SIZE = [16]  # Used for personalization Train/Val

    # TO CHANGE
    # 0.1,0.2 no overfit & decent underfit, can try 0.01,0.05
TUNABLE_DROPOUT_RATE = [0.1,0.2,0.3,0.4,0.5] # Recommended values to start tuning

    # TO CHANGE
TUNABLE_WEIGHT_DECAY_GENERAL = [
    0,
    # 0.00002,
    # 0.00003,
    # 0.00004,
    # 0.00005
    # 0.0005,
    # 0.001
]  # For the Overall General Model and LOPO General Model

    # TO CHANGE
TUNABLE_WEIGHT_DECAY_PERSONALIZATION = [
    0
]  # For personalization (fine-tuning)

# --- Model Types to Run ---
MODEL_TYPES_TO_RUN = ["CNN-LSTM"]  # Example: ['CNN-LSTM', 'CNN-BiLSTM']

def get_data_config_identifier(pre_ictal_win, pre_ictal_excl, post_ictal_buf, sampling_freq):
    """Generates a unique string identifier for a given set of data processing parameters."""
    return f"piw_{pre_ictal_win}_pieb_{pre_ictal_excl}_pib_{post_ictal_buf}_sf_{sampling_freq}"


# --- Sensor Combinations ---
# Generate all combinations of 1 to len(BASE_SENSORS) sensors from BASE_SENSORS
ALL_SENSOR_COMBINATIONS = []
for i in range(1, len(BASE_SENSORS) + 1):
    for combo in itertools.combinations(BASE_SENSORS, i):
        ALL_SENSOR_COMBINATIONS.append(list(combo))

# --- Data Loading Limit ---
# Set to None to include all patients, or an integer to limit the number of patients
MAX_PATIENTS_TO_INCLUDE = (
    10  # Example: 5 (to process only the first 5 patients found)
)

# List of patient IDs to explicitly exclude from the run
PATIENTS_TO_EXCLUDE = []
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
# Removed: print(f"Using device: {DEVICE}")

# Function to generate a unique identifier for the data processing config
def get_data_config_identifier(pre_ictal_win, pre_ictal_excl, post_ictal_buf, sampling_freq):
    return f"piw_{pre_ictal_win}_pieb_{pre_ictal_excl}_pib_{post_ictal_buf}_sf_{sampling_freq}"

FIXED_DATA_PROCESSING_CONFIG = {
    "pre_ictal_window_min": TUNABLE_PRE_ICTAL_WINDOW_MINUTES[0],
    "pre_ictal_exclusion_buffer_min": TUNABLE_PRE_ICTAL_EXCLUSION_BUFFER_MINUTES[0],
    "post_ictal_buffer_min": TUNABLE_POST_ICTAL_BUFFER_MINUTES[0],
    "sampling_freq_hz": TUNABLE_SAMPLING_FREQ_HZ[0],
}

# --- Data Loading and Preprocessing ---

def load_sensor_data_for_patient(patient_dir, sensors):
    """
    Loads Parquet data for specified sensors for a given patient,
    concatenates it, sorts by timestamp, and converts to UTC.
    Does NOT apply scaling yet. Filters to only include specified sensors.
    Uses the file naming convention and column structure from the provided file.

    Args:
        patient_dir (str): The directory for the specific patient.
        sensors (list): List of sensor names (e.g., ['HR', 'EDA']) from BASE_SENSORS.

    Returns:
        dict: A dictionary where keys are attribute names (e.g., 'HR')
              and values are Pandas DataFrames containing the raw data for the
              specified sensors. Returns an empty dict if no data is found
              for any of the specified sensors.
    """
    attribute_data = {}
    # Use the sensor_mapping provided by the user
    sensor_mapping = {"HR": "HR", "EDA": "EDA", "TEMP": "TEMP", "ACC": "Acc Mag"}

    patient_id = os.path.basename(patient_dir)
    logging.debug(f"[{patient_id}] Starting load_sensor_data_for_patient for sensors: {sensors}") # Added debug log

    for (
        sensor_name
    ) in sensors:  # sensor_name will be from BASE_SENSORS (e.g., 'ACC', 'EDA')
        if sensor_name not in sensor_mapping:
            logging.debug(f"[{patient_id}] Sensor {sensor_name} not in sensor_mapping. Skipping.") # Added debug log
            continue

        # Use the BASE_SENSORS name for the folder name (e.g., 'Empatica-ACC')
        # This assumes the folder structure is Empatica-<BASE_SENSORS_Name>
        attr_folder = sensor_name
        # Use the mapped name for the file pattern and column check
        attr_name_part = sensor_mapping[
            sensor_name
        ]  # This is the expected part in the FILENAME and the actual COLUMN name ('data')

        # Use the file naming convention from the provided file
        # Pattern: <patient_id>_Empatica-<attr_folder>_<attr_name_part>_segment_*.parquet
        parquet_files = sorted(
            glob.glob(
                os.path.join(
                    patient_dir,
                    f"Empatica-{attr_folder}",
                    f"{patient_id}_Empatica-{attr_folder}_{attr_name_part}_segment_*.parquet",
                )
            )
        )

        if not parquet_files:
            logging.info(
                f"[{patient_id}] No Parquet files found for Attribute {attr_folder} with pattern {patient_id}_Empatica-{attr_folder}_{attr_name_part}_segment_*.parquet"
            )  # Uncommented warning
            continue

        all_dfs = []
        logging.debug(f"[{patient_id}] Found {len(parquet_files)} parquet files for {sensor_name}. Reading...") # Added debug log
        for file_path in parquet_files:
            try:
                df = pd.read_parquet(file_path)
                # Check for 'time' and 'data' columns as seen in the provided file's logic
                if "time" in df.columns and "data" in df.columns:
                    df["timestamp"] = pd.to_datetime(
                        df["time"] / 1000, unit="s", utc=True
                    )
                    # Rename the 'data' column to the BASE_SENSORS name (sensor_name - uppercase)
                    # This keeps consistency with BASE_SENSORS list and downstream logic
                    df = df.rename(columns={"data": sensor_name})

                    df = df[
                        ["timestamp", sensor_name]
                    ]  # Keep only timestamp and the correctly named sensor column
                    all_dfs.append(df)
                    logging.debug(f"[{patient_id}] Successfully read file: {os.path.basename(file_path)}") # Added debug log
                else:
                    logging.info(
                        f"[{patient_id}] Warning: Parquet file {file_path} does not have expected 'time' and 'data' columns. Skipping."
                    )  # Uncommented warning
                    pass  # Skip file if columns are wrong

            except Exception as e:
                logging.error(f"[{patient_id}] Error reading Parquet file {file_path}: {e}")

        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df = combined_df.sort_values(by="timestamp").reset_index(drop=True)
            # Store with the BASE_SENSORS name (uppercase) as the key
            attribute_data[sensor_name] = combined_df
            logging.debug(f"[{patient_id}] Combined and sorted data for {sensor_name}. Shape: {combined_df.shape}") # Added debug log
        else:
             logging.debug(f"[{patient_id}] No valid dataframes found for {sensor_name} after reading files.") # Added debug log


    logging.info(
        f"[{patient_id}] Finished load_sensor_data_for_patient. Found sensors: {list(attribute_data.keys())}"
    )  # Uncommented print
    return attribute_data


def load_seizure_annotations(patient_dir):
    """
    Loads and processes the SeerAnnotations CSV for a given patient.
    Converts start_time and end_time to UTC datetime objects.
    """
    patient_id = os.path.basename(patient_dir)
    annotation_file = os.path.join(
        patient_dir, f"{os.path.basename(patient_dir)}_SeerAnnotations.csv"
    )
    logging.debug(f"[{patient_id}] Starting load_seizure_annotations from: {annotation_file}") # Added debug log
    try:
        annotations_df = pd.read_csv(annotation_file)
        if (
            "start_time" in annotations_df.columns
            and "end_time" in annotations_df.columns
        ):
            annotations_df["start_time"] = pd.to_datetime(
                annotations_df["start_time"] / 1000, unit="s", utc=True
            )
            annotations_df["end_time"] = pd.to_datetime(
                annotations_df["end_time"] / 1000, unit="s", utc=True
            )
            # Ensure end_time is at least 1 second after start_time to avoid zero-duration seizures
            annotations_df["end_time"] = annotations_df.apply(
                lambda row: row["end_time"]
                if row["end_time"] > row["start_time"]
                else row["start_time"] + pd.Timedelta(seconds=1),
                axis=1,
            )
            logging.debug(f"[{patient_id}] Successfully loaded annotations. Found {len(annotations_df)} seizures.") # Added debug log
            return annotations_df[["start_time", "end_time"]]
        else:
            logging.error(
                f"[{patient_id}] Error: Annotation file {annotation_file} does not have expected 'start_time' and 'end_time' columns."
            )
            return pd.DataFrame(
                columns=["start_time", "end_time"]
            )  # Return empty df on column error
    except FileNotFoundError:
        logging.info(f"[{patient_id}] No annotation file found at: {annotation_file}. Assuming no seizures.") # Added info log for no file
        return pd.DataFrame(
            columns=["start_time", "end_time"]
        )  # Return empty df if not found
    except Exception as e:
        logging.error(f"[{patient_id}] Error reading annotation file {annotation_file}: {e}")
        return pd.DataFrame(
            columns=["start_time", "end_time"]
        )  # Return empty df on other errors


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
    # logging.debug("Starting synchronize_and_merge_data") # Added debug log - patient_id not available here

    if not sensor_data_dict:
        logging.debug("sensor_data_dict is empty. Skipping synchronization.") # Added debug log
        return None

    resampled_data = {}
    # Use keys from sensor_data_dict, as these are the sensors actually found and loaded
    for sensor_name, df in sensor_data_dict.items():
        df = df.set_index("timestamp").sort_index()
        # Ensure target_freq_hz is not zero or negative
        if target_freq_hz <= 0:
            logging.error(
                f"Error: Target frequency must be positive, got {target_freq_hz}. Cannot resample."
            )
            return None
        # --- FIX: Changed 'S' to 's' for FutureWarning ---
        rule = f"{1/target_freq_hz}s"
        # --- END FIX ---
        try:
            # Resample, keeping the sensor_name as the column name
            resampled_df = df[[sensor_name]].asfreq(
                rule
            )  # Explicitly select the column
            resampled_data[sensor_name] = resampled_df
            logging.debug(f"Resampled sensor {sensor_name} to {target_freq_hz}Hz. Shape: {resampled_df.shape}") # Added debug log
        except Exception as e:
            logging.error(
                f"Error during resampling sensor {sensor_name} to {target_freq_hz}Hz: {e}. Skipping sensor."
            )
            continue

    merged_df = None
    # Iterate through resampled_data (only sensors that were successfully resampled)
    logging.debug(f"Merging {len(resampled_data)} resampled dataframes.") # Added debug log
    for sensor_name, df in resampled_data.items():
        if merged_df is None:
            merged_df = df
        else:
            # Use 'outer' join for maximum time coverage across all included sensors
            merged_df = merged_df.join(df, how="outer")

    if merged_df is None or merged_df.empty:
        logging.warning("Merged DataFrame is empty after joining.") # Added warning log
        return None

    merged_df = merged_df.sort_index()
    logging.debug(f"Merged DataFrame shape before interpolation: {merged_df.shape}") # Added debug log

    # Interpolate missing values
    merged_df = merged_df.interpolate(method="time")
    merged_df = merged_df.fillna(method="ffill")
    merged_df = merged_df.fillna(method="bfill")
    logging.debug("Interpolated and filled NaNs.") # Added debug log


    # Drop columns that are still all NaN after interpolation/fill (happens if a sensor had no data across the entire time range)
    cols_before_dropna = merged_df.columns.tolist()
    merged_df = merged_df.dropna(axis=1, how="all")
    cols_after_dropna = merged_df.columns.tolist()
    if cols_before_dropna != cols_after_dropna:
        logging.debug(f"Dropped columns with all NaNs: {set(cols_before_dropna) - set(cols_after_dropna)}") # Added debug log


    if (
        merged_df.empty or len(merged_df.columns) == 0
    ):  # Check again if it's empty or has no columns
        logging.warning("Merged DataFrame is empty or has no columns after dropping all-NaN columns.") # Added warning log
        return None  # Return None if no usable data

    # Apply Robust Scaling to all remaining data columns
    scaler = RobustScaler()
    data_cols = merged_df.columns  # Use all remaining columns
    logging.debug(f"Applying Robust Scaling to columns: {data_cols.tolist()}") # Added debug log

    try:
        # Ensure data_cols is not empty before scaling
        if not data_cols.empty:
            merged_df[data_cols] = scaler.fit_transform(merged_df[data_cols])
            logging.debug("Successfully applied Robust Scaling.") # Added debug log
        else:
            logging.warning("No columns remaining to apply scaling.") # Added warning log

    except Exception as e:
        logging.error(f"Error during scaling: {e}. Skipping scaling.")
        # Decide if you want to return None or unscaled data - returning None is safer if scaling is critical
        return None

    # --- Ensure consistent column order and presence of all BASE_SENSORS columns ---
    # Create a reindex list with all BASE_SENSORS
    # Reindex the merged_df to include all BASE_SENSORS columns, filling missing with NaN
    # This ensures the feature dimension is consistent across all patients, even if a sensor was missing entirely
    logging.debug(f"Reindexing columns to match BASE_SENSORS order: {BASE_SENSORS}") # Added debug log
    merged_df = merged_df.reindex(columns=BASE_SENSORS)
    logging.debug(f"Shape after reindexing: {merged_df.shape}") # Added debug log


    # Fill NaNs introduced by reindexing (for sensors missing entirely) with 0 or another value if appropriate
    # Filling with 0 implies the signal is absent/zeroed when not recorded - review if this is appropriate
    merged_df = merged_df.fillna(0.0)  # Fill NaNs introduced by reindexing with 0
    logging.debug("Filled NaNs introduced by reindexing with 0.0.") # Added debug log


    if merged_df.empty or len(merged_df.columns) != len(
        BASE_SENSORS
    ):  # Final check on shape
        logging.warning(
            "Warning: Merged DataFrame is empty or has incorrect number of columns after reindexing/filling."
        )  # Uncommented warning
        return None  # Return None if final shape is wrong
    
    logging.debug("Finished synchronize_and_merge_data successfully.") # Added debug log
    return merged_df


def create_labeled_segments(
    synced_df,
    annotations_df,
    segment_duration_sec,
    pre_ictal_window_min,
    pre_ictal_exclusion_buffer_min,
    post_ictal_buffer_min,
    target_freq_hz,
):
    """
    Creates segments from synchronized data and labels them
    as pre-ictal (1) or interictal (0) based on seizure annotations. Samples interictal
    segments to attempt class balance. Uses asymmetrical buffers. Synced_df must have columns in BASE_SENSORS order.

    Args:
        synced_df (pandas.DataFrame): DataFrame with synchronized sensor data (datetime index).
                                        Must have columns in order of BASE_SENSORS (with NaNs if missing).
        annotations_df (pandas.DataFrame): DataFrame with seizure start/end times.
        segment_duration_sec (int): Duration of each segment in seconds.
        pre_ictal_window_min (int): Time window before seizure onset considered pre-ictal.
        pre_ictal_exclusion_buffer_min (int): Buffer time *before* seizure onset to exclude for clean interictal. Must be >= pre_ictal_window_min.
        post_ictal_buffer_min (int): Buffer time *after* seizure end to exclude for clean interictal.
        target_freq_hz (int): Frequency data was resampled to (for calculating segment steps).


    Returns:
        tuple: (segments, labels) where segments is a numpy array
                (shape: n_samples, segment_len, num_features) and labels is a numpy array (0 or 1).
                Returns (np.array([]), np.array([])) with appropriate shape if no data or annotations.
    """
    # Need default shape even if inputs are None/empty or invalid
    num_features = (
        synced_df.shape[1]
        if synced_df is not None and not synced_df.empty
        else len(BASE_SENSORS)
    )  # Fallback to len(BASE_SENSORS)
    segment_length_steps = int(segment_duration_sec * target_freq_hz)
    # Ensure segment_length_steps is at least 1 and sensible
    if segment_length_steps <= 0:
        logging.error(
            f"Error: Calculated segment_length_steps is {segment_length_steps} (Duration {segment_duration_sec}s * Freq {target_freq_hz}Hz). Cannot create segments."
        )
        segment_length_steps = 1  # Default to 1 step if calculation is bad

    # Validate buffer relationship
    if pre_ictal_exclusion_buffer_min < pre_ictal_window_min:
        logging.error(
            f"Error: pre_ictal_exclusion_buffer_min ({pre_ictal_exclusion_buffer_min}) must be >= pre_ictal_window_min ({pre_ictal_window_min}). Skipping segmentation."
        )
        # Return with default shape on error
        return np.array([]).reshape(0, segment_length_steps, num_features), np.array([])

    if (
        synced_df is None
        or synced_df.empty
        or len(synced_df.columns) != len(BASE_SENSORS)
    ):
        logging.warning(
            "Synced data is empty, has wrong number of columns, or annotations are missing. Cannot create segments."
        )  # Uncommented warning
        # Return with default shape on error
        num_features = len(BASE_SENSORS)  # Default to len(BASE_SENSORS) features
        return np.array([]).reshape(0, segment_length_steps, num_features), np.array(
            []
        )  # Return with default shape

    # Ensure annotations are not None, even if empty
    if annotations_df is None:
        annotations_df = pd.DataFrame(columns=["start_time", "end_time"])

    segments = []
    labels = []
    step_size = segment_length_steps  # Use non-overlapping segments

    data_start_time = synced_df.index.min() if not synced_df.empty else None
    data_end_time = synced_df.index.max() if not synced_df.empty else None

    if data_start_time is None or data_end_time is None:
        logging.warning(
            "Warning: Synced data has no time index. Cannot create segments."
        )  # Uncommented warning
        num_features = len(BASE_SENSORS)  # Default to len(BASE_SENSORS) features
        return np.array([]).reshape(0, segment_length_steps, num_features), np.array(
            []
        )  # Return with default shape

    # Helper to check overlap (inclusive of boundaries)
    def check_overlap(seg_start, seg_end, windows):
        for win_start, win_end in windows:
            # Ensure window is valid before checking overlap
            if win_start is not None and win_end is not None and win_start < win_end:
                # Use half sample period tolerance for boundary check
                if target_freq_hz > 0:  # Avoid division by zero
                    overlap_tolerance = pd.Timedelta(seconds=0.5 / target_freq_hz)
                else:
                    overlap_tolerance = pd.Timedelta(
                        seconds=0
                    )  # No tolerance if freq is zero

                if (
                    max(seg_start, win_start)
                    < min(seg_end, win_end) + overlap_tolerance
                ):
                    return True
        return False

    # --- Define Time Windows ---

    # 1. Actual Seizure (Ictal) Windows
    seizure_windows = []
    logging.debug("Defining ictal windows...")  # Uncommented print
    for i, seizure in annotations_df.iterrows():
        seizure_start = seizure["start_time"]
        seizure_end = seizure["end_time"]
        if (
            seizure_start is not None
            and seizure_end is not None
            and seizure_start < seizure_end
        ):
            seizure_windows.append((seizure_start, seizure_end))
        else:
            logging.warning(
                f"Warning: Skipping zero or negative duration seizure annotation: {seizure_start} to {seizure_end}"
            )  # Uncommented warning

    # 2. Pre-ictal Windows (Positive Class)
    pre_ictal_windows = []
    logging.debug(
        f"Defining pre-ictal windows ({pre_ictal_window_min} mins before seizure onset)..."
    )  # Uncommented print
    for i, seizure in annotations_df.iterrows():
        seizure_start = seizure["start_time"]
        if seizure_start is None:
            continue  # Skip if seizure start is missing

        pre_ictal_start_uncapped = seizure_start - pd.Timedelta(
            minutes=pre_ictal_window_min
        )
        # End just before seizure starts (half sample tolerance)
        if target_freq_hz > 0:
            pre_ictal_end = seizure_start - pd.Timedelta(seconds=0.5 / target_freq_hz)
        else:
            pre_ictal_end = seizure_start  # No tolerance if freq is zero

        # Cap the pre-ictal start at the beginning of the available data
        pre_ictal_start = max(data_start_time, pre_ictal_start_uncapped)

        # Ensure the capped window is valid
        if (
            pre_ictal_start is not None
            and pre_ictal_end is not None
            and pre_ictal_start < pre_ictal_end
        ):
            # Ensure pre-ictal window does *not* overlap with the seizure itself
            if not check_overlap(pre_ictal_start, pre_ictal_end, seizure_windows):
                pre_ictal_windows.append((pre_ictal_start, pre_ictal_end))
                logging.debug(
                    f" Seizure {i+1}: Pre-ictal window added: {pre_ictal_start} to {pre_ictal_end}"
                )  # Uncommented print
            else:
                logging.debug(
                    f" Seizure {i+1}: Calculated pre-ictal window overlaps with seizure window. Skipped."
                )  # Uncommented print
        else:
            logging.debug(
                f" Seizure {i+1}: Calculated pre-ictal window or capped window is invalid. Skipped."
            )  # Uncommented print

    # 3. Interictal Exclusion Windows (Asymmetrical Buffer)
    # These define areas NOT suitable for clean interictal samples
    interictal_exclusion_windows = []
    buffer_before_timedelta = pd.Timedelta(minutes=pre_ictal_exclusion_buffer_min)
    buffer_after_timedelta = pd.Timedelta(minutes=post_ictal_buffer_min)
    logging.debug(
        f"Defining interictal exclusion windows ({pre_ictal_exclusion_buffer_min} mins before, {post_ictal_buffer_min} mins after)..."
    )  # Uncommented print

    for _, seizure in annotations_df.iterrows():
        seizure_start = seizure["start_time"]
        seizure_end = seizure["end_time"]
        if seizure_start is None or seizure_end is None:
            continue  # Skip if seizure times are missing

        exclusion_start = seizure_start - buffer_before_timedelta
        exclusion_end = seizure_end + buffer_after_timedelta
        # Ensure exclusion window is valid
        if (
            exclusion_start is not None
            and exclusion_end is not None
            and exclusion_start < exclusion_end
        ):
            interictal_exclusion_windows.append((exclusion_start, exclusion_end))
        else:
            logging.warning(
                f"Warning: Skipping invalid exclusion window: {exclusion_start} to {exclusion_end}"
            )  # Uncommented warning

    # --- Create Segments and Assign Labels ---

    logging.info(
        f"Creating segments (len={segment_duration_sec}s, step={segment_duration_sec}s) from {len(synced_df)} total steps...)"
    )  # Uncommented print
    segments_skipped_ictal = 0
    segments_skipped_interictal_buffer = 0  # Segments in buffer BUT NOT pre-ictal
    segments_labeled_preictal = 0
    segments_labeled_interictal = 0
    segments_total_candidates = 0  # Count segments before any skipping

    # Ensure segment_length_steps is valid before iterating
    if segment_length_steps <= 0:
        logging.error(
            f"Error: Calculated segment_length_steps is {segment_length_steps}. Cannot create segments."
        )
        num_features = (
            synced_df.shape[2]
            if synced_df is not None and not synced_df.empty
            else len(BASE_SENSORS)
        )  # Fallback
        segment_length_steps = int(segment_duration_sec * target_freq_hz)
        segment_length_steps = max(1, segment_length_steps)  # Ensure at least 1
        return np.array([]).reshape(
            0, max(1, int(segment_duration_sec * target_freq_hz)), num_features
        ), np.array(
            []
        )  # Return with default shape

    for i in range(
        0, len(synced_df) - segment_length_steps + 1, step_size
    ):  # No tqdm here, handled by outer patient loop
        segment_df = synced_df.iloc[i : i + segment_length_steps]
        if len(segment_df) != segment_length_steps:
            continue  # Should not happen with step_size = segment_length_steps, but safety check

        segments_total_candidates += 1

        segment_start_time = segment_df.index[0]
        if target_freq_hz > 0:
            segment_end_time = segment_df.index[-1] + pd.Timedelta(
                seconds=0.5 / target_freq_hz
            )  # Use midpoint of last sample for end time
        else:
            segment_end_time = segment_df.index[-1]  # No tolerance if freq is zero

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
        if check_overlap(
            segment_start_time, segment_end_time, interictal_exclusion_windows
        ):
            segments_skipped_interictal_buffer += 1
            continue

        # If none of the above, label as Interictal (0)
        segments.append(segment_df.values)
        labels.append(0)
        segments_labeled_interictal += 1

    segments = np.array(segments)
    labels = np.array(labels)

    logging.info(
        f"Finished segmentation. Total full-length candidate segments: {segments_total_candidates}"
    )  # Uncommented print
    logging.info(
        f" Segments skipped (ictal): {segments_skipped_ictal}"
    )  # Uncommented print
    logging.info(
        f" Segments skipped (interictal buffer, not pre-ictal): {segments_skipped_interictal_buffer}"
    )  # Uncommented print
    logging.info(
        f" Total segments included for labeling (Pre-ictal + Interictal): {len(segments)}"
    )  # Uncommented print)
    logging.info(
        f" Segments labeled Pre-ictal: {np.sum(labels)}, Interictal: {len(labels) - np.sum(labels)}"
    )  # Uncommented print)

    # Simple class balancing: Undersample majority class
    pre_ictal_indices = np.where(labels == 1)[0]
    interictal_indices = np.where(labels == 0)[0]

    min_count = min(len(pre_ictal_indices), len(interictal_indices))

    if min_count == 0:
        logging.warning(
            "Warning: One class has zero samples after segmentation. Cannot balance."
        )  # Uncommented warning
        num_features = (
            segments.shape[2]
            if segments.shape[0] > 0
            else (synced_df.shape[1] if synced_df is not None else len(BASE_SENSORS))
        )  # Fallback
        segment_length_steps = int(segment_duration_sec * target_freq_hz)
        segment_length_steps = max(1, segment_length_steps)  # Ensure at least 1
        return np.array([]).reshape(0, segment_length_steps, num_features), np.array([])

    # Only balance if both classes have samples and there's a majority to undersample
    if (
        len(pre_ictal_indices) > 0
        and len(interictal_indices) > 0
        and (len(pre_ictal_indices) > min_count or len(interictal_indices) > min_count)
    ):
        logging.info(
            f"Balancing classes: Reducing majority class to {min_count} samples."
        )  # Uncommented print
        balanced_indices_pre = np.random.choice(
            pre_ictal_indices, min_count, replace=False
        )
        balanced_indices_inter = np.random.choice(
            interictal_indices, min_count, replace=False
        )
        balanced_indices = np.concatenate(
            [balanced_indices_pre, balanced_indices_inter]
        )
        np.random.shuffle(balanced_indices)

        segments = segments[balanced_indices]
        labels = labels[balanced_indices]
        logging.info(
            f"After balancing: Total segments: {len(segments)}, Pre-ictal: {np.sum(labels)}, Interictal: {len(labels) - np.sum(labels)}"
        )  # Uncommented print)

    return segments, labels


def prepare_patient_data(
    patient_folder,
    current_sampling_freq_hz,
    current_pre_ictal_window_min,
    current_pre_ictal_exclusion_buffer_min,
    current_post_ictal_buffer_min,
):
    """
    Loads, synchronizes, scales, and creates labeled segments for a single patient.
    Segments are created with len(BASE_SENSORS) features (with NaNs=0 if missing).
    Returns (patient_id, segments, labels, found_sensors_list) or None if processing fails.
    Accepts data processing parameters as arguments.
    """
    patient_id = os.path.basename(patient_folder)
    logging.info(f"--- Starting data processing for patient: {patient_id} ---") # Added info log

    # 1. Load sensor data - Try to load all base sensors
    logging.debug(f"[{patient_id}] Calling load_sensor_data_for_patient...") # Added debug log
    sensor_data_dict = load_sensor_data_for_patient(patient_folder, BASE_SENSORS)
    # Get the list of sensor names for which data was actually found and loaded into the dict
    found_sensors = list(sensor_data_dict.keys())
    logging.debug(f"[{patient_id}] load_sensor_data_for_patient returned. Found sensors: {found_sensors}") # Added debug log


    if not sensor_data_dict:
        logging.warning(f"[{patient_id}] Skipping patient: Could not load any sensor data from BASE_SENSORS.") # Changed print to logging.warning
        return None

    # 2. Load annotations
    logging.debug(f"[{patient_id}] Calling load_seizure_annotations...") # Added debug log
    annotations_df = load_seizure_annotations(patient_folder)
    logging.debug(f"[{patient_id}] load_seizure_annotations returned. {len(annotations_df)} annotations found.") # Added debug log

    # 3. Synchronize and merge data
    logging.debug(f"[{patient_id}] Calling synchronize_and_merge_data...") # Added debug log
    # synced_df will have columns for all BASE_SENSORS, with 0s for missing ones, in sorted order
    synced_df = synchronize_and_merge_data(
        sensor_data_dict, target_freq_hz=current_sampling_freq_hz
    )
    logging.debug(f"[{patient_id}] synchronize_and_merge_data returned. synced_df is None: {synced_df is None}, shape: {synced_df.shape if synced_df is not None else 'N/A'}") # Added debug log

    # Explicitly delete intermediate objects to free memory
    del sensor_data_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    # Ensure synced_df has the correct number of columns (all BASE_SENSORS)
    if (
        synced_df is None
        or synced_df.empty
        or len(synced_df.columns) != len(BASE_SENSORS)
    ):
        logging.warning(f"[{patient_id}] Skipping patient: Could not synchronize, merge, or lost columns.") # Changed print to logging.warning
        return None

    # 4. Create labeled segments
    logging.debug(f"[{patient_id}] Calling create_labeled_segments...") # Added debug log
    segments, labels = create_labeled_segments(
        synced_df,
        annotations_df,
        segment_duration_sec=SEGMENT_DURATION_SECONDS,
        pre_ictal_window_min=current_pre_ictal_window_min,  # Use the current HP value
        pre_ictal_exclusion_buffer_min=current_pre_ictal_exclusion_buffer_min,  # Use the current HP value
        post_ictal_buffer_min=current_post_ictal_buffer_min,  # Use the current HP value
        target_freq_hz=current_sampling_freq_hz,  # Use the current HP value
    )
    logging.debug(f"[{patient_id}] create_labeled_segments returned. Segments shape: {segments.shape}, Labels shape: {labels.shape}") # Added debug log

    # Explicitly delete intermediate objects to free memory
    del synced_df, annotations_df
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    # Segments should have shape (N, L, len(BASE_SENSORS))
    # Calculate expected seq_len based on parameters used in create_labeled_segments
    expected_seq_len = int(SEGMENT_DURATION_SECONDS * current_sampling_freq_hz)
    expected_seq_len = max(1, expected_seq_len)
    expected_num_features = len(BASE_SENSORS)

    if (
        len(segments) == 0
        or segments.shape[2] != expected_num_features
        or segments.shape[1] != expected_seq_len
    ):
        logging.warning(f"[{patient_id}] Skipping patient: No valid segments created or incorrect shape after segmentation. Expected shape (N, {expected_seq_len}, {expected_num_features}), got {segments.shape}.") # Changed print to logging.warning
        return None

    logging.info(f"--- Finished data processing for patient: {patient_id} successfully. Segment shape: {segments.shape} ---") # Added info log

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
            if segments.ndim == 2:  # (N, L) -> add a feature dim (N, L, 1)
                segments = segments[:, :, np.newaxis]
                # Update num_features if it was expected to be 1 based on this
                if num_features != 1:
                    logging.warning(
                        f"Warning: Segments ndim=2 but expected num_features={num_features}. Assuming 1 feature."
                    )  # Uncommented warning
                    num_features = 1
            elif segments.ndim < 2:
                logging.warning(
                    f"Warning: Segments array has unexpected ndim={segments.ndim}. Cannot create dataset."
                )  # Uncommented warning
                self.segments = torch.empty(
                    0, num_features, seq_len, dtype=torch.float32
                )
                self.labels = torch.empty(0, 1, dtype=torch.float32)
                return  # Stop init if data is unusable

            # Ensure segments have the expected number of features
            if segments.shape[2] != num_features:
                logging.warning(
                    f"Warning: Segment features ({segments.shape[2]}) mismatch expected features ({num_features}). Cannot create dataset."
                )  # Uncommented warning
                self.segments = torch.empty(
                    0, num_features, seq_len, dtype=torch.float32
                )
                self.labels = torch.empty(0, 1, dtype=torch.float32)
                return  # Stop init

            # Ensure segments have the expected sequence length
            if segments.shape[1] != seq_len:
                logging.warning(
                    f"Warning: Segment length ({segments.shape[1]}) mismatch expected length ({seq_len}). Cannot create dataset."
                )  # Uncommented warning
                self.segments = torch.empty(
                    0, num_features, seq_len, dtype=torch.float32
                )
                self.labels = torch.empty(0, 1, dtype=torch.float32)
                return  # Stop init

            self.segments = torch.tensor(segments, dtype=torch.float32).permute(
                0, 2, 1
            )  # (N, L, F) -> (N, F, L)
            self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(
                1
            )  # (N,) -> (N, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.segments[idx], self.labels[idx]


# --- PyTorch Model Definitions ---


class CNN_LSTM(nn.Module):
    def __init__(
        self,
        input_channels,
        seq_len,
        conv_filters,
        conv_kernel_size,
        pool_size,
        lstm_units,
        dense_units,
        dropout_rate=0.5,
    ):
        super(CNN_LSTM, self).__init__()
        self.input_channels = input_channels
        self.seq_len = seq_len
        self.conv_filters = conv_filters  # Now taken as argument
        self.conv_kernel_size = conv_kernel_size  # Now taken as argument
        self.pool_size = pool_size  # Now taken as argument
        self.lstm_units = lstm_units  # Now taken as argument
        self.dense_units = dense_units  # Now taken as argument
        self.dropout_rate = dropout_rate # <--- Store dropout rate

        if input_channels <= 0:
            input_channels = 1  # Default to 1 channel if somehow 0 or negative
        if seq_len <= 0:
            seq_len = 1  # Default to 1 seq_len if somehow 0 or negative
        if not conv_filters:  # Ensure filter list is not empty
            conv_filters = [32]  # Default filter if list is empty

        conv_layers_list = []
        in_channels = input_channels
        current_seq_len = seq_len

        # Dynamically build conv layers based on conv_filters list
        for i, out_channels in enumerate(conv_filters):
            # Ensure kernel size and pool size are valid
            kernel_size = max(1, conv_kernel_size)
            pool_size = max(1, pool_size)

            padding = kernel_size // 2  # Calculate padding

            conv_layers_list.append(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=kernel_size, padding=padding
                )
            )
            conv_layers_list.append(nn.BatchNorm1d(out_channels)) # <--- ADD BATCH NORM
            conv_layers_list.append(nn.ReLU())
            conv_layers_list.append(nn.MaxPool1d(pool_size))
            # Add Dropout after each pooling layer in CNN blocks
            conv_layers_list.append(nn.Dropout(self.dropout_rate)) # <--- ADD DROPOUT

            in_channels = out_channels

            # Recalculate sequence length after conv and pool
            current_seq_len = math.floor(
                (current_seq_len + 2 * padding - 1 * (kernel_size - 1) - 1) / 1 + 1
            )  # After Conv1d (stride=1)
            current_seq_len = math.floor(
                (current_seq_len + 2 * 0 - 1 * (pool_size - 1) - 1) / pool_size + 1
            )  # After MaxPool1d (stride=pool_size)

        self.conv_layers = nn.Sequential(*conv_layers_list)

        # Calculate the output sequence length after CNN layers dynamically
        try:
            # Create a dummy tensor on the CPU for shape calculation
            dummy_input = torch.randn(
                1, self.input_channels, self.seq_len, dtype=torch.float32
            )
            dummy_output = self.conv_layers(dummy_input)
            self.lstm_input_features = dummy_output.shape[
                1
            ]  # Features dimension after CNN (number of last conv filters)
            self.lstm_input_seq_len = dummy_output.shape[
                2
            ]  # Sequence length dimension after CNN/Pooling

            # Check if the output sequence length is valid for LSTM
            if self.lstm_input_seq_len <= 0:
                # This can happen if seq_len is too short for the filters/pooling
                raise ValueError(
                    f"Calculated LSTM input sequence length is zero or negative ({self.lstm_input_seq_len}). Check CNN/Pooling parameters relative to segment length ({self.seq_len})."
                )

        except Exception as e:
            logging.error(
                f"Error calculating layer output size during model init for {self.__class__.__name__} with input_channels={self.input_channels}, seq_len={self.seq_len}: {e}"
            )
            # Set fallback values or re-raise - re-raising is better during development
            raise e

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_features,
            hidden_size=lstm_units,  # Use lstm_units argument
            batch_first=True,
        )
        # Add Dropout after LSTM/BiLSTM layer
        self.lstm_dropout = nn.Dropout(self.dropout_rate) # <--- ADD DROPOUT

        self.dense_layers = nn.Sequential(
            nn.Linear(
                lstm_units, dense_units
            ),  # Use lstm_units and dense_units arguments
            nn.ReLU(),
            nn.Linear(dense_units, 1),  # Use dense_units argument
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (batch_size, channels, seq_len)
        cnn_out = self.conv_layers(x)  # shape: (batch_size, filters, reduced_seq_len)
        # Handle potential empty output after CNN if seq_len collapsed to 0
        if cnn_out.shape[2] == 0:
            # Or return a default value, depending on desired behavior
            # Returning 0.5 (sigmoid output) might be reasonable for prediction task
            # Ensure the tensor is on the correct device
            return torch.tensor(
                [[0.5]] * x.size(0), device=x.device
            )  # Return neutral predictions if seq_len collapses

        lstm_in = cnn_out.permute(
            0, 2, 1
        )  # shape: (batch_size, reduced_seq_len, filters)
        lstm_out, _ = self.lstm(
            lstm_in
        )  # shape: (batch_size, reduced_seq_len, LSTM_UNITS)
        lstm_out = self.lstm_dropout(lstm_out) # <--- APPLY DROPOUT HERE
        last_timestep_out = lstm_out[:, -1, :]  # shape: (batch_size, LSTM_UNITS)
        output = self.dense_layers(last_timestep_out)  # shape: (batch_size, 1)
        return output


class CNN_BiLSTM(nn.Module):
    def __init__(
        self,
        input_channels,
        seq_len,
        conv_filters,
        conv_kernel_size,
        pool_size,
        lstm_units,
        dense_units,
        dropout_rate=0.5,
    ):
        super(CNN_BiLSTM, self).__init__()
        self.input_channels = input_channels
        self.seq_len = seq_len
        self.conv_filters = conv_filters  # Now taken as argument
        self.conv_kernel_size = conv_kernel_size  # Now taken as argument
        self.pool_size = pool_size  # Now taken as argument
        self.lstm_units = lstm_units  # Now taken as argument
        self.dense_units = dense_units  # Now taken as argument
        self.dropout_rate = dropout_rate
        
        if input_channels <= 0:
            input_channels = 1  # Default to 1 channel if somehow 0 or negative
        if seq_len <= 0:
            seq_len = 1  # Default to 1 seq_len if somehow 0 or negative
        if not conv_filters:  # Ensure filter list is not empty
            conv_filters = [32]  # Default filter if list is empty

        conv_layers_list = []
        in_channels = input_channels
        current_seq_len = seq_len

        # Dynamically build conv layers based on conv_filters list
        for i, out_channels in enumerate(conv_filters):
            # Ensure kernel size and pool size are valid
            kernel_size = max(1, conv_kernel_size)
            pool_size = max(1, pool_size)

            padding = kernel_size // 2  # Calculate padding

            conv_layers_list.append(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=kernel_size, padding=padding
                )
            )
            conv_layers_list.append(nn.BatchNorm1d(out_channels)) # <--- ADDED: Batch Normalization
            conv_layers_list.append(nn.ReLU())
            conv_layers_list.append(nn.MaxPool1d(pool_size))
            conv_layers_list.append(nn.Dropout(self.dropout_rate)) # <--- ADDED: Dropout after pooling

            in_channels = out_channels

            # Recalculate sequence length after conv and pool
            current_seq_len = math.floor(
                (current_seq_len + 2 * padding - 1 * (kernel_size - 1) - 1) / 1 + 1
            )  # After Conv1d (stride=1)
            current_seq_len = math.floor(
                (current_seq_len + 2 * 0 - 1 * (pool_size - 1) - 1) / pool_size + 1
            )  # After MaxPool1d (stride=pool_size)

        self.conv_layers = nn.Sequential(*conv_layers_list)

        # Calculate the output sequence length after CNN layers dynamically
        try:
            # Create a dummy tensor on the CPU for shape calculation
            dummy_input = torch.randn(
                1, self.input_channels, self.seq_len, dtype=torch.float32
            )
            dummy_output = self.conv_layers(dummy_input)
            self.lstm_input_features = dummy_output.shape[
                1
            ]  # Features dimension after CNN (number of last conv filters)
            self.lstm_input_seq_len = dummy_output.shape[
                2
            ]  # Sequence length dimension after CNN/Pooling

            # Check if the output sequence length is valid for LSTM
            if self.lstm_input_seq_len <= 0:
                # This can happen if seq_len is too short for the filters/pooling
                raise ValueError(
                    f"Calculated LSTM input sequence length is zero or negative ({self.lstm_input_seq_len}). Check CNN/Pooling parameters relative to segment length ({self.seq_len})."
                )

        except Exception as e:
            logging.error(
                f"Error calculating layer output size during model init for {self.__class__.__name__} with input_channels={self.input_channels}, seq_len={self.seq_len}: {e}"
            )  # Changed print to logging.error
            # Set fallback values or re-raise - re-raising is better during development
            raise e

        self.bilstm = nn.LSTM(
            input_size=self.lstm_input_features,
            hidden_size=lstm_units,  # Use lstm_units argument
            batch_first=True,
            bidirectional=True,
        )
        self.bilstm_dropout = nn.Dropout(self.dropout_rate) # <--- ADDED: Dropout layer for BiLSTM output

        self.dense_layers = nn.Sequential(
            nn.Linear(
                lstm_units * 2, dense_units
            ),  # Use lstm_units and dense_units arguments, input size is doubled
            nn.ReLU(),
            nn.Linear(dense_units, 1),  # Use dense_units argument
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (batch_size, channels, seq_len)
        cnn_out = self.conv_layers(x)  # shape: (batch_size, filters, reduced_seq_len)
        # Handle potential empty output after CNN if seq_len collapsed to 0
        if cnn_out.shape[2] == 0:
            # Or return a default value, depending on desired behavior
            # Returning 0.5 (sigmoid output) might be reasonable for prediction task
            # Ensure the tensor is on the correct device
            return torch.tensor(
                [[0.5]] * x.size(0), device=x.device
            )  # Return neutral predictions if seq_len collapses

        lstm_in = cnn_out.permute(
            0, 2, 1
        )  # shape: (batch_size, reduced_seq_len, filters)
        bilstm_out, _ = self.bilstm(
            lstm_in
        )  # shape: (batch_size, reduced_seq_len, LSTM_UNITS * 2)
        bilstm_out = self.bilstm_dropout(bilstm_out) # <--- APPLIED DROPOUT
        last_timestep_out = bilstm_out[:, -1, :]  # shape: (batch_size, LSTM_UNITS * 2)
        output = self.dense_layers(last_timestep_out)  # shape: (batch_size, 1)
        return output


# Helper function to get the model class based on string name
def get_model_class(model_type):
    if model_type == "CNN-LSTM":
        return CNN_LSTM
    elif model_type == "CNN-BiLSTM":
        return CNN_BiLSTM
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# --- PyTorch Training and Evaluation ---


def calculate_metrics(all_labels, all_predictions, all_probs):
    """Calculates and returns a dictionary of evaluation metrics."""
    # Ensure inputs are numpy arrays and flattened
    all_labels = np.array(all_labels).flatten()
    all_predictions = np.array(all_predictions).flatten()
    all_probs = np.array(all_probs).flatten()

    # Ensure metrics are calculated only if there are samples
    if len(all_labels) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "auc_roc": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
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
            auc_roc = 0.0  # AUC-ROC is undefined for single class
            logging.warning(
                "Warning: Only one class present in evaluation set labels, AUC-ROC is undefined."
            )  # Uncommented warning

    except ValueError:  # Catch other potential ValueError (e.g., invalid probabilities)
        auc_roc = 0.0
        logging.warning(
            "Warning: Could not compute AUC-ROC (e.g., invalid probabilities)."
        )  # Uncommented warning

    cm = confusion_matrix(all_labels, all_predictions).tolist()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_roc": auc_roc,
        "confusion_matrix": cm,
    }


def train_one_epoch(
    model, train_dataloader, criterion, optimizer, device, class_weights=None
):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    # Define criterion - Use BCELoss as models now output probabilities
    criterion = nn.BCELoss() # <-- CHANGE THIS LINE BACK

    # Check if dataloader is empty
    if len(train_dataloader.dataset) == 0:
        return 0.0 # Return 0 loss if no data

    dataloader_tqdm = tqdm(train_dataloader, desc="Batch", leave=False)


    for inputs, labels in dataloader_tqdm:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs) # Model now outputs probabilities

        # Ensure outputs and labels have compatible shapes for BCELoss
        # BCELoss expects (N, 1) and (N, 1) or (N,) and (N,)
        # Our labels are (N, 1), outputs are (N, 1). This is fine.

        # --- Implement manual class weighting using BCELoss ---
        if class_weights is not None and len(class_weights) == 2:
            # Apply weights based on the label (0 or 1)
            # Ensure weight_tensor has the same shape as labels
            weight_tensor = torch.zeros_like(labels, dtype=torch.float32)
            # Check if weights exist for the classes present in the batch
            if 0 in class_weights:
                weight_tensor[labels == 0] = class_weights[0]
            if 1 in class_weights:
                weight_tensor[labels == 1] = class_weights[1]

            loss = criterion(outputs, labels)
            # Manually apply weights and take mean across batch
            loss = (loss * weight_tensor).mean() # <-- Apply weights manually
        else:
            # Use standard BCELoss if no class weights
            loss = criterion(outputs, labels)
        # --- End manual class weighting ---


        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0) # Keep track of total samples for average loss

        dataloader_tqdm.set_postfix(loss=loss.item())

    # Avoid division by zero if dataset was somehow empty despite the check
    epoch_loss = running_loss / (total_samples if total_samples > 0 else 1) # Use total_samples
    return epoch_loss

def evaluate_pytorch_model(model, dataloader, criterion, device):
    """Evaluates the model on a given dataloader and returns loss and detailed metrics."""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probs = []

    # Handle empty dataloader gracefully
    if len(dataloader.dataset) == 0:
        metrics = calculate_metrics(all_labels, all_predictions, all_probs)
        return {'loss': 0.0, **metrics, 'all_probs': [], 'all_labels': []}


    dataloader_tqdm = tqdm(dataloader, desc="Evaluating Batch", leave=False)

    # Define criterion for evaluation - Use BCELoss
    eval_criterion = nn.BCELoss()

    with torch.no_grad():
        for inputs, labels in dataloader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs) # Model now outputs probabilities (0-1)

            # Calculate loss using BCELoss
            loss = eval_criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            # Predictions based on probability threshold (0.5)
            predicted = (outputs > 0.5).float()

            # --- AMENDMENT START ---
            # Flatten the numpy array from the batch and convert to a list of standard Python integers
            all_labels.extend(labels.cpu().numpy().flatten().tolist())
            # --- AMENDMENT END ---

            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy()) # <-- Store outputs (probabilities) for AUC

            dataloader_tqdm.set_postfix(loss=loss.item())


    epoch_loss = running_loss / len(dataloader.dataset)

    # Calculate detailed metrics using the helper function
    metrics = calculate_metrics(all_labels, all_predictions, all_probs)

    # Return metrics including probabilities and labels
    return {'loss': epoch_loss, **metrics, 'all_probs': all_probs, 'all_labels': all_labels}

def train_pytorch_model(
    model,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    epochs,
    learning_rate,
    class_weights=None,
    save_best_model_path=None,
    desc="Training",
    device=torch.device("cpu"),
    weight_decay=0.0,
):
    """Main training loop with validation, early stopping, LR scheduling, and returns metrics for all sets."""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=0.00001
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None

    # <--- ADD THIS SECTION FOR HISTORY COLLECTION ---
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'train_f1_score': [],
        'train_auc_roc': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1_score': [],
        'val_auc_roc': []
    }
    # --- END ADDITION ---

    if len(train_dataloader.dataset) == 0:
        logging.warning(
            f"Warning: Training dataloader for '{desc}' is empty. Skipping training."
        )
        model.to(device)
        # Evaluate with dummy metrics or on current state if no training occurred
        train_metrics = evaluate_pytorch_model(
            model, train_dataloader, criterion, device
        )
        val_metrics = evaluate_pytorch_model(model, val_dataloader, criterion, device)
        test_metrics = evaluate_pytorch_model(model, test_dataloader, criterion, device)
        return model, {"train": train_metrics, "val": val_metrics, "test": test_metrics, "history": history}
    
    epoch_tqdm = tqdm(range(epochs), desc=desc, leave=True)

    for epoch in epoch_tqdm:
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_dataloader, criterion, optimizer, device, class_weights
        )
        history['train_loss'].append(train_loss)

        # Evaluate on train set to get per-epoch metrics (adds overhead)
        # You might want to do this less frequently if your training set is very large
        train_metrics_epoch = evaluate_pytorch_model(model, train_dataloader, criterion, device)
        history['train_accuracy'].append(train_metrics_epoch['accuracy'])
        history['train_f1_score'].append(train_metrics_epoch['f1_score'])
        history['train_auc_roc'].append(train_metrics_epoch['auc_roc'])

        if len(val_dataloader.dataset) > 0:
            val_metrics = evaluate_pytorch_model(
                model, val_dataloader, criterion, device
            )
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]
            
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_metrics['accuracy'])
            history['val_f1_score'].append(val_metrics['f1_score'])
            history['val_auc_roc'].append(val_metrics['auc_roc'])

            epoch_tqdm.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                val_acc=f"{val_acc:.4f}",
                time=f"{time.time() - start_time:.2f}s",
            )

            scheduler.step(val_loss)

            # <--- CORRECTED LINE ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss # Corrected from "best_val_loss = best_val_loss"
                epochs_without_improvement = 0
                best_model_state = model.state_dict()
                if save_best_model_path:
                    try:
                        os.makedirs(
                            os.path.dirname(save_best_model_path), exist_ok=True
                        )
                        torch.save(best_model_state, save_best_model_path)
                    except Exception as e:
                        logging.warning(
                            f"Warning: Could not save best model state to {save_best_model_path}: {e}"
                        )

            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= 10:
                logging.info(
                    f"Early stopping triggered at epoch {epoch+1} for '{desc}'."
                )
                break

        else:
            epoch_tqdm.set_postfix(
                train_loss=f"{train_loss:.4f}", time=f"{time.time() - start_time:.2f}s"
            )
            # Append None for val metrics if no val data
            history['val_loss'].append(None)
            history['val_accuracy'].append(None)
            history['val_f1_score'].append(None)
            history['val_auc_roc'].append(None)
            pass

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logging.info(f"Loaded best model state based on validation loss for '{desc}'.")
    else:
        logging.warning(
            f"Warning: No best model state was saved during training for '{desc}'. Returning final epoch state."
        )

    eval_criterion = nn.BCELoss()
    train_metrics = evaluate_pytorch_model(
        model, train_dataloader, eval_criterion, device
    )
    val_metrics = evaluate_pytorch_model(model, val_dataloader, eval_criterion, device)
    test_metrics = evaluate_pytorch_model(
        model, test_dataloader, eval_criterion, device
    )

    return model, {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "history": history # <--- RETURN HISTORY HERE
    }


# --- New Function for LOPO General Model Training ---


def train_lopo_general_model(
    all_processed_patient_data,
    excluded_patient_id,
    model_type,
    sensor_combination_indices,
    model_hyperparameters,
    general_hyperparameters,
    current_hp_combo_str,
    device=torch.device("cpu"),
):
    """
    Trains a general model on data from all patients EXCEPT the excluded one,
    using only the sensors specified by indices. Saves the trained model state.

    Args:
        all_processed_patient_data (list): List of (patient_id, segments, labels, found_sensors)
                                         from initial processing (segments have len(BASE_SENSORS) features).
                                         This list is passed to the child process.
        excluded_patient_id (str): The ID of the patient whose data should be excluded.
        model_type (str): 'CNN-LSTM' or 'CNN-BiLSTM'.
        sensor_combination_indices (list): List of integer indices corresponding to
                                            the sensor columns to use from BASE_SENSORS.
        model_hyperparameters (dict): Dictionary containing model architecture HPs.
        general_hyperparameters (dict): Dictionary containing general model training HPs (epochs, lr, batch_size).
        current_hp_combo_str (str): String representation of the current HP combination for saving.
        device (torch.device): The device to train on (cuda or cpu).


    Returns:
        tuple: (State dictionary of the trained LOPO general model, metrics dictionary)
              Returns (None, None) if training data is insufficient or training fails.
    """
    logging.info(f"--- Training LOPO General Model (Excluding {excluded_patient_id}) for {model_type} ---") # Removed direct print

    lopo_segments_raw = []
    lopo_labels_raw = []

    # Collect data from all patients EXCEPT the excluded one
    for patient_data_tuple in all_processed_patient_data:
        patient_id, segments_all_sensors, labels, found_sensors = patient_data_tuple
        if patient_id != excluded_patient_id:
            # Slice segments to include only the features for the current combination
            # segments_all_sensors shape is (N, L, len(BASE_SENSORS))
            if (
                len(segments_all_sensors) > 0
                and len(sensor_combination_indices) > 0
                and segments_all_sensors.shape[2] == len(BASE_SENSORS)
            ):
                segments_sliced = segments_all_sensors[:, :, sensor_combination_indices]
                lopo_segments_raw.append(segments_sliced)
                lopo_labels_raw.append(labels)
            # else: logging.warning(f"Skipping patient {patient_id} data for LOPO general training (Excluding {excluded_patient_id}): Shape mismatch or no data after slicing.") # Uncommented print and changed to logging.warning

    if not lopo_segments_raw:
        logging.warning(
            f"Warning: No data available from other patients for LOPO general training (Excluding {excluded_patient_id})."
        )  # Changed print to logging.warning
        return None, None  # Return None if no data for LOPO training

    lopo_segments_combined = np.concatenate(lopo_segments_raw, axis=0)
    lopo_labels_combined = np.concatenate(lopo_labels_raw, axis=0)

    # logging.info(f"LOPO Combined data shape (Excluding {excluded_patient_id}): {lopo_segments_combined.shape}") # Removed direct print
    # logging.info(f"LOPO Combined labels shape (Excluding {excluded_patient_id}): {lopo_labels_combined.shape}") # Removed direct print

    # Check for sufficient data and classes for LOPO training split
    # Need at least 3 samples total for 60/20/20 split, and at least one of each class in the total combined data
    if len(lopo_segments_combined) < 3 or len(np.unique(lopo_labels_combined)) < 2:
        logging.warning(
            f"Warning: Insufficient data ({len(lopo_segments_combined)} samples) or only one class ({np.unique(lopo_labels_combined)}) for LOPO general training split (Excluding {excluded_patient_id}). Skipping training."
        )  # Changed print to logging.warning
        return None, None

    try:
        # Split LOPO data into Train/Validation/Test (e.g., 60/20/20)
        # Ensure enough data for splitting - train_test_split needs at least 2 samples for test_size > 0
        # And requires at least 2 samples per class for stratify
        # We need 3 splits (train, val, test), so at least 3 samples are needed overall.
        # For stratification, we need at least one sample of each class in each split.
        # Let's assume the total combined data has at least one of each class (checked above).
        # The split might still result in a split with only one class if the total sample count is very low.

        # Attempt split 1: train vs temp
        X_train_lopo, X_temp_lopo, y_train_lopo, y_temp_lopo = train_test_split(
            lopo_segments_combined,
            lopo_labels_combined,
            test_size=0.4,
            random_state=SEED,
            stratify=lopo_labels_combined,
        )
        # Attempt split 2: val vs test from temp
        X_val_lopo, X_test_lopo, y_val_lopo, y_test_lopo = train_test_split(
            X_temp_lopo,
            y_temp_lopo,
            test_size=0.5,
            random_state=SEED,
            stratify=y_temp_lopo,  # Use stratify on y_temp_lopo
        )
    except ValueError as e:
        logging.warning(
            f"Warning: LOPO data split failed for patient {excluded_patient_id}: {e}. This might happen with very few samples or imbalanced small datasets leading to single-class splits. Skipping training."
        )  # Changed print to logging.warning
        return None, None
    except Exception as e:
        logging.error(
            f"An unexpected error occurred during LOPO general data split for patient {excluded_patient_id}: {e}. Skipping training."
        )
        return None, None

    # logging.info(f"LOPO General Training data shape: {X_train_lopo.shape}, Labels shape: {y_train_lopo.shape}") # Removed direct print
    # logging.info(f"LOPO General Validation data shape: {X_val_lopo.shape}, Labels shape: {y_val_lopo.shape}") # Removed direct print
    # logging.info(f"LOPO General Test data shape: {X_test_lopo.shape}, Labels shape: {y_test_lopo.shape}") # Added test set info

    # Ensure splits are not empty and have both classes for training/validation/test
    # Check unique classes in each split explicitly
    unique_y_train_lopo = np.unique(y_train_lopo)
    unique_y_val_lopo = np.unique(y_val_lopo)
    unique_y_test_lopo = np.unique(y_test_lopo)

    if (
        len(X_train_lopo) == 0
        or len(X_val_lopo) == 0
        or len(X_test_lopo) == 0
        or len(unique_y_train_lopo) < 2
        or len(unique_y_val_lopo) < 2
        or len(unique_y_test_lopo) < 2
    ):
        logging.warning(
            f"Warning: LOPO data split resulted in empty train ({len(X_train_lopo)}), val ({len(X_val_lopo)}), or test ({len(X_test_lopo)}) set, or single class in one split ({excluded_patient_id}). Skipping training."
        )  # Changed print to logging.warning
        return None, None

    # Calculate expected seq_len and num_features for the dataset
    expected_seq_len = lopo_segments_combined.shape[1]
    expected_num_features = lopo_segments_combined.shape[2]

    lopo_train_dataset = SeizureDataset(
        X_train_lopo,
        y_train_lopo,
        seq_len=expected_seq_len,
        num_features=expected_num_features,
    )
    lopo_val_dataset = SeizureDataset(
        X_val_lopo,
        y_val_lopo,
        seq_len=expected_seq_len,
        num_features=expected_num_features,
    )
    lopo_test_dataset = SeizureDataset(
        X_test_lopo,
        y_test_lopo,
        seq_len=expected_seq_len,
        num_features=expected_num_features,
    )  # Added test dataset

    # Use a smaller number of workers for dataloaders within each parallel process
    # Reduced num_workers to 0 to avoid potential issues with shared memory/resources in multiprocessing
    num_workers = 0
    persistent_workers = False  # persistent_workers requires num_workers > 0

    # Get batch sizes and learning rate from general_hyperparameters
    general_train_batch_size = general_hyperparameters["batch_size"]
    general_learning_rate = general_hyperparameters["learning_rate"]
    general_epochs = general_hyperparameters["epochs"]

    # Adjust batch size if larger than dataset size
    train_batch_size = general_train_batch_size
    if len(lopo_train_dataset) > 0:
        train_batch_size = max(1, min(train_batch_size, len(lopo_train_dataset)))
    val_batch_size = general_train_batch_size
    if len(lopo_val_dataset) > 0:
        val_batch_size = max(1, min(val_batch_size, len(lopo_val_dataset)))
    test_batch_size = general_train_batch_size  # Use general batch size for test eval
    if len(lopo_test_dataset) > 0:
        test_batch_size = max(1, min(test_batch_size, len(lopo_test_dataset)))

    lopo_train_dataloader = DataLoader(
        lopo_train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    lopo_val_dataloader = DataLoader(
        lopo_val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    lopo_test_dataloader = DataLoader(
        lopo_test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )  # Added test dataloader

    # Calculate class weights for the LOPO training data
    # Ensure weights are calculated only if there are samples in the training set
    class_weights_lopo_dict = None
    if len(y_train_lopo) > 0:
        classes_lopo = np.unique(y_train_lopo)
        if len(classes_lopo) == 2:
            class_weights_lopo_np = class_weight.compute_class_weight(
                "balanced", classes=classes_lopo, y=y_train_lopo
            )
            class_weights_lopo_dict = {
                cls: weight for cls, weight in zip(classes_lopo, class_weights_lopo_np)
            }
            logging.info(
                f"Computed LOPO general class weights (Excluding {excluded_patient_id}): {class_weights_lopo_dict}"
            )  # Changed print to logging.info
        # else: logging.warning(f"Warning: Only one class for LOPO general training data (Excluding {excluded_patient_id}). No class weights applied.") # Uncommented print and changed to logging.warning

    # Instantiate the LOPO general model with the correct input shape and hyperparameters for this combination
    input_channels = lopo_segments_combined.shape[2]
    seq_len = lopo_segments_combined.shape[1]
    ModelClass = get_model_class(model_type)

    try:
        lopo_general_model = ModelClass(
            input_channels,
            seq_len,
            conv_filters=model_hyperparameters["conv_filters"],
            conv_kernel_size=model_hyperparameters["conv_kernel_size"],
            pool_size=model_hyperparameters["pool_size"],
            lstm_units=model_hyperparameters["lstm_units"],
            dense_units=model_hyperparameters["dense_units"],
            dropout_rate=model_hyperparameters["dropout_rate"],
        ).to(device)
    except (ValueError, Exception) as e:
        logging.error(
            f"Error instantiating LOPO general model for {excluded_patient_id} ({model_type}, {current_hp_combo_str}): {e}. Skipping training."
        )  # Changed print to logging.error
        # Clean up dataloaders before returning
        del (
            lopo_train_dataloader,
            lopo_val_dataloader,
            lopo_test_dataloader,
        )  # Added test dataloader
        del (
            lopo_train_dataset,
            lopo_val_dataset,
            lopo_test_dataset,
        )  # Added test dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None, None

    # Define save path for the LOPO general model for this patient/combo/HP
    # lopo_model_save_dir = os.path.join(
    #     OUTPUT_DIR, current_hp_combo_str, model_type, "lopo_general"
    # )
    lopo_model_save_dir = os.path.join(
        OUTPUT_DIR, model_type, "lopo_general"
    )
    lopo_model_save_path = os.path.join(
        lopo_model_save_dir, f"excl_{excluded_patient_id}.pth"
    )  # Saved by excluded patient ID

    # Train the LOPO general model
    logging.info(
        f"Starting LOPO General Model training (Excluding {excluded_patient_id}) for {model_type} ({current_hp_combo_str})..."
    )  # Changed print to logging.info
    lopo_general_model, lopo_general_metrics = train_pytorch_model(
        lopo_general_model,
        lopo_train_dataloader,
        lopo_val_dataloader,
        lopo_test_dataloader,  # Pass test dataloader
        epochs=general_epochs,
        learning_rate=general_learning_rate,
        class_weights=class_weights_lopo_dict,
        save_best_model_path=lopo_model_save_path,  # Save the LOPO general model
        desc=f"Training LOPO General (Excl {excluded_patient_id})",
        device=device,
        weight_decay=general_hyperparameters["weight_decay"], # <--- PASS weight_decay HERE
    )
    
    lopo_model_plot_dir = os.path.join(OUTPUT_DIR, model_type, "lopo_general", excluded_patient_id, "plots")
    os.makedirs(lopo_model_plot_dir, exist_ok=True) # Ensure plot directory exists

    if 'history' in lopo_general_metrics:
        plot_training_history(
            lopo_general_metrics['history'],
            f'LOPO General Model (Excl {excluded_patient_id}, {model_type}, HP Combo {current_hp_combo_str})',
            lopo_model_plot_dir,
            f'excl_{excluded_patient_id}_lopo_general'
        )

    # Clean up LOPO training dataloaders and model instance
    lopo_general_model_state = lopo_general_model.state_dict()
    del (
        lopo_general_model,
        lopo_train_dataloader,
        lopo_val_dataloader,
        lopo_test_dataloader,
    )  # Added test dataloader
    del lopo_train_dataset, lopo_val_dataset, lopo_test_dataset  # Added test dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return lopo_general_model_state, lopo_general_metrics


# Define a new function to process a single patient's personalization (intended for parallel execution)
def process_single_patient_personalization(
    patient_data_tuple,  # Tuple for the current patient (id, segments_all_sensors, labels, found_sensors)
    all_processed_patient_data,  # Full list of processed data for ALL patients
    model_type,
    sensor_combination,
    sensor_combination_indices,
    general_hyperparameters,
    personalization_hyperparameters,
    model_hyperparameters,
    expected_seq_len_sliced,
    expected_num_features_sliced,
    current_hp_combo_str,  # Pass HP combo string for saving
    device_name,  # Pass device name as string
):
    """
    Processes personalization for a single patient within the LOPO framework.
    This function is intended to be run in parallel for each patient.
    It trains the LOPO general model, splits patient data, evaluates before/after personalization.
    Saves the personalized model state.
    Returns (patient_id, results_dict or None)
    """
    # Set the device within the child process
    device = torch.device(device_name)

    (
        current_patient_id,
        current_patient_segments_all_sensors,
        current_patient_labels,
        current_found_sensors,
    ) = patient_data_tuple

    combo_name = "_".join([s.lower() for s in sensor_combination]).upper()
    logging.info(f"Starting personalization for patient {current_patient_id} ({model_type}, {combo_name}, {current_hp_combo_str})") # Removed direct print

    # --- Step 1: Train LOPO General Model for the current patient ---
    # This returns the state_dict of a model trained on N-1 patients using the current sensor combination features
    # Pass the full list of all patients' data to the LOPO training function
    lopo_general_model_state_dict, lopo_general_metrics = train_lopo_general_model(
        all_processed_patient_data,  # Pass the full list, train_lopo_general_model will filter
        current_patient_id,
        model_type,
        sensor_combination_indices,
        model_hyperparameters,
        general_hyperparameters,
        current_hp_combo_str,  # Pass HP combo string
        device,  # Pass device to LOPO training
    )

    if lopo_general_model_state_dict is None:
        logging.warning(
            f"Skipping personalization for patient {current_patient_id} ({model_type}, {combo_name}, {current_hp_combo_str}): LOPO general model training failed."
        )  # Changed print to logging.warning
        # Return None for results if LOPO general training failed
        return (current_patient_id, None)

    # --- Step 2: Prepare and Split Target Patient Data ---
    # Slice the current patient's segments to include only the features for this combination
    if (
        len(current_patient_segments_all_sensors) > 0
        and expected_num_features_sliced > 0
    ):
        if current_patient_segments_all_sensors.shape[2] == len(BASE_SENSORS):
            current_patient_segments_sliced = current_patient_segments_all_sensors[
                :, :, sensor_combination_indices
            ]
        else:
            logging.error(
                f"Error: Patient {current_patient_id} segments_all_sensors has unexpected feature count ({current_patient_segments_all_sensors.shape[2]}). Expected {len(BASE_SENSORS)}. Skipping."
            )  # Changed print to logging.error
            return (current_patient_id, None)
    else:
        logging.warning(
            f"Skipping patient {current_patient_id} ({model_type}, {combo_name}, {current_hp_combo_str}): No segments or no features after slicing."
        )  # Changed print to logging.warning
        return (current_patient_id, None)

    # Check if sliced patient data is usable for splitting/training/eval
    # Need at least 3 samples total for 60/20/20 split, and at least one of each class in the patient's data
    if (
        len(current_patient_segments_sliced) < 3
        or len(np.unique(current_patient_labels)) < 2
    ):
        logging.warning(
            f"Warning: Insufficient data ({len(current_patient_segments_sliced)} samples) or only one class ({np.unique(current_patient_labels)}) for patient {current_patient_id} personalization splits ({model_type}, {combo_name}, {current_hp_combo_str}). Skipping."
        )  # Changed print to logging.warning
        return (current_patient_id, None)

    try:
        # Split patient's sliced data for personalization fine-tuning and testing (60/20/20)
        # Ensure enough data for splitting and stratification (at least 2 samples per class for test_size > 0)
        X_train_pat, X_temp_pat, y_train_pat, y_temp_pat = train_test_split(
            current_patient_segments_sliced,
            current_patient_labels,
            test_size=0.4,
            random_state=SEED,
            stratify=current_patient_labels,
        )
        X_val_pat, X_test_pat, y_val_pat, y_test_pat = train_test_split(
            X_temp_pat,
            y_temp_pat,
            test_size=0.5,
            random_state=SEED,
            stratify=y_temp_pat,  # Stratify on y_temp_pat
        )
    except ValueError as e:
        logging.warning(
            f"Warning: Patient {current_patient_id} data split failed ({model_type}, {combo_name}, {current_hp_combo_str}): {e}. This might happen with very few samples or imbalanced small datasets leading to single-class splits. Skipping personalization."
        )  # Changed print to logging.warning
        return (current_patient_id, None)
    except Exception as e:
        logging.error(
            f"An unexpected error occurred during patient {current_patient_id} data split ({model_type}, {combo_name}, {current_hp_combo_str}): {e}. Skipping personalization."
        )  # Changed print to logging.error
        return (current_patient_id, None)

    # Ensure resulting patient splits are not empty and have both classes
    unique_y_train_pat = np.unique(y_train_pat)
    unique_y_val_pat = np.unique(y_val_pat)
    unique_y_test_pat = np.unique(y_test_pat)

    if (
        len(X_train_pat) == 0
        or len(X_val_pat) == 0
        or len(X_test_pat) == 0
        or len(unique_y_train_pat) < 2
        or len(unique_y_val_pat) < 2
        or len(unique_y_test_pat) < 2
    ):
        logging.warning(
            f"Warning: Patient {current_patient_id} data split resulted in empty train ({len(X_train_pat)}), val ({len(X_val_pat)}), or test ({len(X_test_pat)}) set, or single class in one split ({model_type}, {combo_name}, {current_hp_combo_str}). Skipping personalization."
        )  # Changed print to logging.warning
        return (current_patient_id, None)

    # Create DataLoaders for the current patient's splits
    # Reduced num_workers to 0 to avoid potential issues with shared memory/resources in multiprocessing
    num_workers_pat = 0
    persistent_workers_pat = False  # persistent_workers requires num_workers > 0

    personalization_train_batch_size = personalization_hyperparameters["batch_size"]
    personalization_val_batch_size = personalization_hyperparameters["batch_size"]
    personalized_test_batch_size = general_hyperparameters[
        "batch_size"
    ]  # Use general batch size for final test eval

    if len(X_train_pat) > 0:
        personalization_train_batch_size = max(
            1, min(personalization_train_batch_size, len(X_train_pat))
        )
    if len(X_val_pat) > 0:
        personalization_val_batch_size = max(
            1, min(personalization_val_batch_size, len(X_val_pat))
        )
    if len(X_test_pat) > 0:
        personalized_test_batch_size = max(
            1, min(personalized_test_batch_size, len(X_test_pat))
        )

    train_dataset_pat = SeizureDataset(
        X_train_pat,
        y_train_pat,
        seq_len=expected_seq_len_sliced,
        num_features=expected_num_features_sliced,
    )
    val_dataset_pat = SeizureDataset(
        X_val_pat,
        y_val_pat,
        seq_len=expected_seq_len_sliced,
        num_features=expected_num_features_sliced,
    )
    test_dataset_pat = SeizureDataset(
        X_test_pat,
        y_test_pat,
        seq_len=expected_seq_len_sliced,
        num_features=expected_num_features_sliced,
    )

    try:
        train_dataloader_pat = DataLoader(
            train_dataset_pat,
            batch_size=personalization_train_batch_size,
            shuffle=True,
            num_workers=num_workers_pat,
            persistent_workers=persistent_workers_pat,
        )
        val_dataloader_pat = DataLoader(
            val_dataset_pat,
            batch_size=personalization_val_batch_size,
            shuffle=False,
            num_workers=num_workers_pat,
            persistent_workers=persistent_workers_pat,
        )
        test_dataloader_pat = DataLoader(
            test_dataset_pat,
            batch_size=personalized_test_batch_size,
            shuffle=False,
            num_workers=num_workers_pat,
            persistent_workers=persistent_workers_pat,
        )  # Corrected batch size name
    except Exception as e:
        logging.error(
            f"Error creating patient {current_patient_id} dataloaders ({model_type}, {combo_name}, {current_hp_combo_str}): {e}. Skipping personalization."
        )  # Changed print to logging.error
        del train_dataset_pat, val_dataset_pat, test_dataset_pat
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (current_patient_id, None)

    # --- Step 3: Evaluate the LOPO general model on this patient's test data (Before Personalization) ---
    logging.info(f"Evaluating LOPO general model on patient {current_patient_id}'s test data (Before Personalization)...") # Removed direct print
    ModelClass = get_model_class(model_type)  # Get ModelClass within the process

    try:
        lopo_general_model_instance_eval = ModelClass(
            expected_num_features_sliced,
            expected_seq_len_sliced,
            conv_filters=model_hyperparameters["conv_filters"],
            conv_kernel_size=model_hyperparameters["conv_kernel_size"],
            pool_size=model_hyperparameters["pool_size"],
            lstm_units=model_hyperparameters["lstm_units"],
            dense_units=model_hyperparameters["dense_units"],
        ).to(device)
        lopo_general_model_instance_eval.load_state_dict(lopo_general_model_state_dict)
    except (ValueError, RuntimeError, Exception) as e:
        logging.error(
            f"Error instantiating or loading LOPO general model state for evaluation for patient {current_patient_id} ({model_type}, {combo_name}, {current_hp_combo_str}): {e}. Skipping patient."
        )  # Changed print to logging.error
        del train_dataloader_pat, val_dataloader_pat, test_dataloader_pat
        del train_dataset_pat, val_dataset_pat, test_dataset_pat
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Return before metrics with placeholder after metrics to indicate failure
        return (
            current_patient_id,
            {
                "before": {
                    "loss": 0.0,
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "auc_roc": 0.0,
                    "confusion_matrix": [[0, 0], [0, 0]],
                },
                "after": {
                    "loss": 0.0,
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "auc_roc": 0.0,
                    "confusion_matrix": [[0, 0], [0, 0]],
                },
                "lopo_general_metrics": lopo_general_metrics,
            },
        )  # Include LOPO general metrics even if personalization fails

    metrics_before = evaluate_pytorch_model(
        lopo_general_model_instance_eval, test_dataloader_pat, nn.BCELoss(), device
    )
    logging.info(
        f"Patient {current_patient_id} ({model_type}, {combo_name}, {current_hp_combo_str}) - Before Personalization Metrics: Acc={metrics_before['accuracy']:.4f}, Prec={metrics_before['precision']:.4f}, Rec={metrics_before['recall']:.4f}, F1={metrics_before['f1_score']:.4f}, AUC={metrics_before['auc_roc']:.4f}"
    )  # Changed print to logging.info
    
    # --- Add Plotting for Before Personalization ---
    # plot_dir_pers = os.path.join(OUTPUT_DIR, current_hp_combo_str, model_type, combo_name, 'personalized', current_patient_id, 'plots')
    plot_dir_pers = os.path.join(OUTPUT_DIR, model_type, combo_name, 'personalized', current_patient_id, 'plots')
    os.makedirs(plot_dir_pers, exist_ok=True) # Ensure plot directory exists

    # AUC-ROC Plot (Before Personalization)
    # --- IMPORTANT: Need to modify evaluate_pytorch_model to return all_probs and all_labels ---
    if 'all_probs' in metrics_before and 'all_labels' in metrics_before:
        plot_auc_roc(
            metrics_before['all_probs'],
            metrics_before['all_labels'],
            f'Before Personalization AUC-ROC (Patient {current_patient_id}, {combo_name})',
            os.path.join(plot_dir_pers, 'before_personalization_auc_roc.png')
        )
    else:
        logging.warning(f"Skipping Before Personalization AUC-ROC plot for Patient {current_patient_id}: 'all_probs' or 'all_labels' not found in metrics.")


    # Confusion Matrix Plot (Before Personalization)
    plot_confusion_matrix(
        metrics_before.get('confusion_matrix', [[0,0],[0,0]]),
        ['Interictal (0)', 'Pre-ictal (1)'], # Class names
        f'Before Personalization Confusion Matrix (Patient {current_patient_id}, {combo_name})',
        os.path.join(plot_dir_pers, 'before_personalization_confusion_matrix.png')
    )
    # --- End Plotting for Before Personalization ---
    
    del lopo_general_model_instance_eval
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Step 4: Create and Fine-tune a new model instance for personalization ---
    try:
        personalized_model = ModelClass(
            expected_num_features_sliced,
            expected_seq_len_sliced,
            conv_filters=model_hyperparameters["conv_filters"],
            conv_kernel_size=model_hyperparameters["conv_kernel_size"],
            pool_size=model_hyperparameters["pool_size"],
            lstm_units=model_hyperparameters["lstm_units"],
            dense_units=model_hyperparameters["dense_units"],
        ).to(device)
        # Load the LOPO general model state as the starting point
        personalized_model.load_state_dict(lopo_general_model_state_dict)
    except (ValueError, RuntimeError, Exception) as e:
        logging.error(
            f"Error instantiating or loading LOPO general model state for fine-tuning for patient {current_patient_id} ({model_type}, {combo_name}, {current_hp_combo_str}): {e}. Skipping patient."
        )  # Changed print to logging.error
        del train_dataloader_pat, val_dataloader_pat, test_dataloader_pat
        del train_dataset_pat, val_dataset_pat, test_dataset_pat
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Store before metrics, but indicate personalization failed with placeholder after metrics
        return (
            current_patient_id,
            {
                "before": metrics_before,
                "after": {
                    "loss": 0.0,
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "auc_roc": 0.0,
                    "confusion_matrix": [[0, 0], [0, 0]],
                },
                "lopo_general_metrics": lopo_general_metrics,
            },
        )  # Include LOPO general metrics

    # Calculate class weights for the patient's fine-tuning data
    # Ensure weights are calculated only if there are samples in the training set
    class_weights_pat_dict = None
    if len(y_train_pat) > 0:
        classes_pat = np.unique(y_train_pat)
        if len(classes_pat) == 2:
            class_weights_pat_np = class_weight.compute_class_weight(
                "balanced", classes=classes_pat, y=y_train_pat
            )
            class_weights_pat_dict = {
                cls: weight for cls, weight in zip(classes_pat, class_weights_pat_np)
            }
        # --- Add Plotting for Before Personalization ---
    plot_dir_pers = os.path.join(OUTPUT_DIR, model_type, combo_name, 'personalized', current_patient_id, 'plots')
    os.makedirs(plot_dir_pers, exist_ok=True) # Ensure plot directory exists

    # AUC-ROC Plot (Before Personalization)
    if 'all_probs' in metrics_before and 'all_labels' in metrics_before:
        plot_auc_roc(
            metrics_before['all_probs'],
            metrics_before['all_labels'],
            f'Before Personalization AUC-ROC (Patient {current_patient_id}, {combo_name})',
            os.path.join(plot_dir_pers, 'before_personalization_auc_roc.png')
        )
        # <--- ADD CALL TO PLOT PROBABILITY DISTRIBUTION FOR BEFORE PERSONALIZATION ---
        plot_probability_distribution(
            metrics_before['all_probs'],
            metrics_before['all_labels'],
            f'Before Personalization Probability Distribution (Patient {current_patient_id}, {combo_name})',
            os.path.join(plot_dir_pers, 'before_personalization_prob_dist.png')
        )
        # --- END ADDITION ---
    else:
        logging.warning(f"Skipping Before Personalization AUC-ROC/ProbDist plot for Patient {current_patient_id}: 'all_probs' or 'all_labels' not found in metrics.")

    # Confusion Matrix Plot (Before Personalization)
    plot_confusion_matrix(
        metrics_before.get('confusion_matrix', [[0,0],[0,0]]),
        ['Interictal (0)', 'Pre-ictal (1)'], # Class names
        f'Before Personalization Confusion Matrix (Patient {current_patient_id}, {combo_name})',
        os.path.join(plot_dir_pers, 'before_personalization_confusion_matrix.png')
    )
    # Only attempt fine-tuning if there's training data for this patient
    if len(train_dataset_pat) > 0:
        logging.info(f"Starting fine-tuning for patient {current_patient_id}...") # Removed direct print

        # Define save path for the personalized model for this patient/combo/HP
        # personalized_model_save_dir = os.path.join(
        #     OUTPUT_DIR,timestamp_str, current_hp_combo_str, model_type, combo_name, "personalized"
        # )
        personalized_model_save_dir = os.path.join(
            OUTPUT_DIR,timestamp_str, model_type, combo_name, "personalized"
        )
        personalized_model_save_path = os.path.join(
            personalized_model_save_dir, f"patient_{current_patient_id}.pth"
        )

        personalized_model, personalized_metrics = train_pytorch_model(
            personalized_model,
            train_dataloader_pat,
            val_dataloader_pat,
            test_dataloader_pat,  # Pass test dataloader
            epochs=personalization_hyperparameters["epochs"],
            learning_rate=personalization_hyperparameters["learning_rate"],
            class_weights=class_weights_pat_dict,
            save_best_model_path=personalized_model_save_path,  # Save the personalized model
            desc=f"Fine-tuning {current_patient_id}",
            device=device,
            weight_decay=personalization_hyperparameters["weight_decay"],
        )
        if 'history' in personalized_metrics:
            plot_training_history(
            personalized_metrics['history'],
            f'Personalized Model (Patient {current_patient_id}, {combo_name})',
            plot_dir_pers,
            f'patient_{current_patient_id}_personalized'
        )
    else:
        logging.warning(
            f"Warning: No fine-tuning data for patient {current_patient_id} ({model_type}, {combo_name}, {current_hp_combo_str}). Skipping fine-tuning."
        )  # Changed print to logging.warning
        # The personalized_model instance is already initialized with LOPO general weights,
        # it just won't be trained further.
        personalized_metrics = {
            "train": evaluate_pytorch_model(
                personalized_model, train_dataloader_pat, nn.BCELoss(), device
            ),
            "val": evaluate_pytorch_model(
                personalized_model, val_dataloader_pat, nn.BCELoss(), device
            ),
            "test": evaluate_pytorch_model(
                personalized_model, test_dataloader_pat, nn.BCELoss(), device
            ),
        }  # Evaluate with the non-fine-tuned model

    # --- Step 5: Evaluate the personalized model on this patient's test data (After Personalization) ---
    # The evaluation is already done inside train_pytorch_model when it returns metrics for all sets
    metrics_after = personalized_metrics["test"]

    logging.info(
        f"Patient {current_patient_id} ({model_type}, {combo_name}, {current_hp_combo_str}) - After Personalization Metrics: Acc={metrics_after['accuracy']:.4f}, Prec={metrics_after['precision']:.4f}, Rec={metrics_after['recall']:.4f}, F1={metrics_after['f1_score']:.4f}, AUC={metrics_after['auc_roc']:.4f}"
    )  # Changed print to logging.info

    # --- Add Plotting for After Personalization ---
    # plot_dir_pers is already defined above
    # AUC-ROC Plot (After Personalization)
    # --- IMPORTANT: Need to modify evaluate_pytorch_model to return all_probs and all_labels ---
    if 'all_probs' in metrics_after and 'all_labels' in metrics_after:
        plot_auc_roc(
            metrics_after['all_probs'],
            metrics_after['all_labels'],
            f'After Personalization AUC-ROC (Patient {current_patient_id}, {combo_name})',
            os.path.join(plot_dir_pers, 'after_personalization_auc_roc.png')
        )
        plot_probability_distribution(
            metrics_after['all_probs'],
            metrics_after['all_labels'],
            f'After Personalization Probability Distribution (Patient {current_patient_id}, {combo_name})',
            os.path.join(plot_dir_pers, 'after_personalization_prob_dist.png')
        )
    else:
        logging.warning(f"Skipping After Personalization AUC-ROC plot for Patient {current_patient_id}: 'all_probs' or 'all_labels' not found in metrics.")

    # Confusion Matrix Plot (After Personalization)
    plot_confusion_matrix(
        metrics_after.get('confusion_matrix', [[0,0],[0,0]]),
        ['Interictal (0)', 'Pre-ictal (1)'], # Class names
        f'After Personalization Confusion Matrix (Patient {current_patient_id}, {combo_name})',
        os.path.join(plot_dir_pers, 'after_personalization_confusion_matrix.png')
    )
    # --- End Plotting for After Personalization ---
    # Clean up memory for the current patient's data/model/dataloaders
    del train_dataset_pat, val_dataset_pat, test_dataset_pat
    del train_dataloader_pat, val_dataloader_pat, test_dataloader_pat
    del personalized_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Return the results for this patient, including all metrics for before/after and LOPO general
    return (
        current_patient_id,
        {
            "before": metrics_before,
            "after": metrics_after,
            "lopo_general_metrics": lopo_general_metrics,
            "personalized_metrics_all_sets": personalized_metrics,
        },
    )


# Modify the perform_personalization_pytorch_lopo function to use ProcessPoolExecutor
def perform_personalization_pytorch_lopo(
    all_processed_patient_data,
    model_type,
    sensor_combination,
    general_hyperparameters,
    personalization_hyperparameters,
    model_hyperparameters,
    current_hp_combo_str,
    device_name,
):
    """
    Orchestrates parallel personalization for each patient using the LOPO approach.

    Args:
        all_processed_patient_data (list): List of (patient_id, segments, labels, found_sensors)
                                         from initial processing (segments have len(BASE_SENSORS) features).
                                         This list contains ALL suitable patients for the current sensor combo.
                                         This will be passed to each child process.
        model_type (str): 'CNN-LSTM' or 'CNN-BiLSTM'.
        sensor_combination (list): List of sensor names (e.g., ['HR', 'EDA']) for the current combination.
        general_hyperparameters (dict): Dictionary containing general model training HPs (epochs, lr, batch_size).
        personalization_hyperparameters (dict): Dictionary containing personalization training HPs (epochs, lr, batch_size).
        model_hyperparameters (dict): Dictionary containing model architecture HPs.
        current_hp_combo_str (str): String representation of the current HP combination for saving.
        device_name (str): The name of the device ('cuda' or 'cpu').

    Returns:
        dict: Dictionary storing performance metrics before and after personalization for each patient in the list.
              Only includes patients for whom LOPO training and personalization was attempted.
    """
    combination_name = "_".join([s.lower() for s in sensor_combination]).upper()

    logging.info(
        f"--- Performing Personalization ({model_type}) for {combination_name} with HP: {current_hp_combo_str} using LOPO (Parallel) ---"
    )  # Changed print to logging.info

    if not all_processed_patient_data:
        logging.warning(
            "No patient data available for personalization with LOPO."
        )  # Changed print to logging.warning
        return {}

    personalization_results = {}
    ModelClass = get_model_class(model_type)

    # Get indices for the sensors in the current combination
    try:
        sensor_combination_indices = [BASE_SENSORS.index(s) for s in sensor_combination]
        sensor_combination_indices.sort()
        if any(
            idx < 0 or idx >= len(BASE_SENSORS) for idx in sensor_combination_indices
        ):
            raise ValueError("Invalid sensor index generated.")
    except ValueError as e:
        logging.error(
            f"Error: Sensor in combination {sensor_combination} not found or invalid index in BASE_SENSORS. {e}"
        )  # Changed print to logging.error
        return {}
    except Exception as e:
        logging.error(
            f"An unexpected error occurred getting sensor indices for combination {sensor_combination}: {e}"
        )  # Changed print to logging.error
        return {}

    # Calculate expected shape of sliced segments for this combination
    # Need to use the sampling frequency from the current HP combination
    current_sampling_freq_hz = model_hyperparameters.get(
        "sampling_freq_hz", TUNABLE_SAMPLING_FREQ_HZ[0]
    )  # Get from HP dict or use default
    expected_seq_len_sliced = int(SEGMENT_DURATION_SECONDS * current_sampling_freq_hz)
    expected_seq_len_sliced = max(1, expected_seq_len_sliced)
    expected_num_features_sliced = len(sensor_combination_indices)

    # Filter patients suitable for this combination BEFORE submitting to the pool
    # get_patients_and_indices_for_combination needs the full list to filter
    (
        patients_suitable_for_combination,
        sensor_combination_indices,
    ) = get_patients_and_indices_for_combination(
        all_processed_patient_data,  # Pass the full list to filter
        sensor_combination,  # Pass the list of sensor names
    )

    if not patients_suitable_for_combination:
        logging.warning(
            f"Skipping personalization for {model_type} + {combination_name} with HP: {current_hp_combo_str}: No suitable patients found."
        )  # Changed print to logging.warning
        return {}

    logging.info(
        f"Initiating parallel personalization for {len(patients_suitable_for_combination)} suitable patients for combination: {combination_name} with HP: {current_hp_combo_str}."
    )  # Changed print to logging.info

# --- Explicitly create personalization directories for this combo/HP ---
    # personalized_model_save_dir_base = os.path.join(
    #     OUTPUT_DIR, current_hp_combo_str, model_type, combination_name, "personalized"
    # )
    # plot_dir_pers_base = os.path.join(
    #     OUTPUT_DIR, current_hp_combo_str, model_type, combination_name, "personalized", "plots"
    # ) # Base plots dir for all personalized patients
    personalized_model_save_dir_base = os.path.join(
        OUTPUT_DIR, model_type, combination_name, "personalized"
    )
    plot_dir_pers_base = os.path.join(
        OUTPUT_DIR, model_type, combination_name, "personalized", "plots"
    ) # Base plots dir for all personalized patients
    # The actual patient-specific plot dirs will be created inside process_single_patient_personalization

    try:
        os.makedirs(personalized_model_save_dir_base, exist_ok=True)
        # We won't create the base plots dir here, as each patient will have their own plots subdir
        os.makedirs(plot_dir_pers_base, exist_ok=True)
        logging.info(f"Created base personalization directories for HP: {current_hp_combo_str}, Model: {model_type}, Sensors: {combination_name}")
    except Exception as e:
        logging.error(f"Error creating base personalization directories for HP: {current_hp_combo_str}, Model: {model_type}, Sensors: {combination_name}: {e}. Skipping personalization for this combo.")
        return {} # Skip personalization for this combination if base dirs fail

    # Use ProcessPoolExecutor for CPU-bound tasks (model training)
    # Determine max_workers - Significantly reduced to mitigate memory issues
    # Start with a small number like 1 or 2 and increase if system allows.
    max_workers = 2  # Reduced from os.cpu_count() - 2
    # Ensure we don't create more workers than suitable patients
    max_workers = min(max_workers, len(patients_suitable_for_combination))
    # Ensure at least 1 worker if there are suitable patients
    max_workers = (
        max(1, max_workers) if len(patients_suitable_for_combination) > 0 else 0
    )

    # Futures will hold the result when ready
    futures = []
    # Only create the executor if there are workers to run tasks
    if max_workers > 0:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            # Use tqdm around the submission loop to show progress of submitting tasks
            for patient_data_tuple in tqdm(
                patients_suitable_for_combination,
                desc=f"Submitting patient tasks ({model_type}, {combination_name}, {current_hp_combo_str})",
                leave=False,
            ):
                # Submit the processing of a single patient to the pool
                future = executor.submit(
                    process_single_patient_personalization,
                    patient_data_tuple,  # Data for this specific patient
                    all_processed_patient_data,  # Full list of all patients (needed by train_lopo_general_model in child process)
                    model_type,
                    sensor_combination,  # Pass sensor names
                    sensor_combination_indices,  # Pass sensor indices
                    general_hyperparameters,
                    personalization_hyperparameters,
                    model_hyperparameters,
                    expected_seq_len_sliced,
                    expected_num_features_sliced,
                    current_hp_combo_str,  # Pass HP combo string
                    device_name,  # Pass the device name string
                )
                futures.append(future)

            # Collect results as they complete
            # Use tqdm around as_completed for a progress bar of completed tasks
            personalization_results_list = []  # Collect results in a list first
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"Collecting patient results ({model_type}, {combination_name}, {current_hp_combo_str})",
            ):
                try:
                    # The result is a tuple: (patient_id, results_dict or None)
                    patient_id, patient_results = future.result()
                    if patient_results is not None:
                        personalization_results_list.append(
                            (patient_id, patient_results)
                        )
                    else:
                        logging.warning(
                            f"Personalization failed or skipped for patient {patient_id} in a parallel process ({model_type}, {combination_name}, {current_hp_combo_str})."
                        )  # Changed print to logging.warning

                except Exception as exc:
                    logging.error(
                        f"A patient processing generated an exception: {exc} ({model_type}, {combination_name}, {current_hp_combo_str})"
                    )  # Changed print to logging.error
                    # You might want to identify which patient failed if possible,
                    # but ProcessPoolExecutor hides details well.

        # Aggregate results into a dictionary after collecting all
        personalization_results = {
            patient_id: results for patient_id, results in personalization_results_list
        }

        logging.info(
            f"Finished parallel personalization for combination: {combination_name} with HP: {current_hp_combo_str}. Processed {len(personalization_results)} patients successfully."
        )  # Changed print to logging.info
    else:
        logging.warning(
            f"No workers available for parallel processing for combination: {combination_name} with HP: {current_hp_combo_str}. Skipping."
        )  # Changed print to logging.warning
        personalization_results = {}  # Return empty results if no workers

    return personalization_results


# --- Helper to get sensor indices and filter patients for a combination ---
# This function is slightly repurposed. It now finds which patients have ALL required sensors
# and gets the correct column indices for slicing from the full BASE_SENSORS segment array.
# This function is now called BEFORE the parallelization loop.
def get_patients_and_indices_for_combination(
    all_processed_patient_data, sensor_combination
):
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

    logging.info(f"Checking patients for sensor combination: {combination_name}") # Changed print to logging.info

    patients_suitable_for_combination = []

    # Get indices for the sensors in the current combination (relative to BASE_SENSORS order)
    try:
        # Ensure sensors in the combination are in BASE_SENSORS and get their indices
        sensor_combination_indices = [BASE_SENSORS.index(s) for s in sensor_combination]
        # Sort indices to maintain consistent column order after slicing
        sensor_combination_indices.sort()
    except ValueError as e:
        logging.error(
            f"Error: Sensor '{e}' in combination {sensor_combination} not found in BASE_SENSORS {BASE_SENSORS}. Cannot process this combination."
        )  # Changed print to logging.error
        return [], []  # Cannot proceed with invalid combination
    except Exception as e:
        logging.error(
            f"An unexpected error occurred getting sensor indices for combination {sensor_combination}: {e}"
        )  # Changed print to logging.error
        return [], []  # Cannot proceed with invalid combination

    for patient_data_tuple in all_processed_patient_data:
        patient_id, segments_all_sensors, labels, found_sensors = patient_data_tuple
        # Check if the patient has *all* sensors required for this combination
        # `found_sensors` is the list of sensor names actually found for this patient (uppercase)
        if all(s in found_sensors for s in sensor_combination):
            # Check if segments have the correct number of features (should be len(BASE_SENSORS))
            # And if there are actual segments and both classes present
            if (
                segments_all_sensors.shape[2] == len(BASE_SENSORS)
                and len(segments_all_sensors) > 0
                and len(np.unique(labels)) > 1
            ):
                patients_suitable_for_combination.append(
                    patient_data_tuple
                )  # Append the full patient data tuple
            # else: logging.info(f"Skipping patient {patient_id} for combination {combination_name}: Segments shape mismatch ({segments_all_sensors.shape[2]} vs {len(BASE_SENSORS)}) or no segments/single class.") # Uncommented print and changed to logging.info
        # else: logging.info(f"Skipping patient {patient_id} for combination {combination_name}: Missing required sensors {set(sensor_combination) - set(found_sensors)}.") # Uncommented print and changed to logging.info

    if not patients_suitable_for_combination:
        logging.warning(
            f"No patients found with all sensors for combination: {combination_name}. Skipping this combination."
        )  # Changed print to logging.warning
        return [], []  # Return empty if no suitable patients

    # logging.info(f"Found {len(patients_suitable_for_combination)} patients suitable for combination: {combination_name}.") # Changed print to logging.info
    return patients_suitable_for_combination, sensor_combination_indices


def format_metrics_for_summary(metrics_dict, prefix=""):
    """Formats a dictionary of metrics for printing in the summary file."""
    if not metrics_dict:
        return f"{prefix}Loss: N/A, Acc: N/A, Prec: N/A, Rec: N/A, F1: N/A, AUC: N/A, CM: N/A"

    loss = metrics_dict.get("loss", "N/A")
    acc = metrics_dict.get("accuracy", "N/A")
    prec = metrics_dict.get("precision", "N/A")
    rec = metrics_dict.get("recall", "N/A")
    f1 = metrics_dict.get("f1_score", "N/A")
    auc = metrics_dict.get("auc_roc", "N/A")
    cm = metrics_dict.get("confusion_matrix", [[0, 0], [0, 0]])

    # Format CM nicely
    cm_str = f"[[{cm[0][0]}, {cm[0][1]}], [{cm[1][0]}, {cm[1][1]}]]"

    # Format metrics with 4 decimal places if they are numbers
    loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss)
    acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else str(acc)
    prec_str = f"{prec:.4f}" if isinstance(prec, (int, float)) else str(prec)
    rec_str = f"{rec:.4f}" if isinstance(rec, (int, float)) else str(rec)
    f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else str(f1)
    auc_str = f"{auc:.4f}" if isinstance(auc, (int, float)) else str(auc)

    return f"{prefix}Loss: {loss_str}, Acc: {acc_str}, Prec: {prec_str}, Rec: {rec_str}, F1: {f1_str}, AUC: {auc_str}, CM: {cm_str}"


def print_personalization_summary(personalization_results, output_file=None):
    """Prints a summary table of personalization results to console or file. Includes detailed metrics."""
    # Determine where to print (console or file)
    def print_func(*args, **kwargs):
        if output_file:
            print(*args, **kwargs, file=output_file)
        else:
            # Use logging for console output
            logging.info(*args, **kwargs)

    print_func("--- Personalized Model Performance (Per Patient Summary) ---")
    if not personalization_results:
        print_func("No personalization results available.")
        return

    # Sort results by patient ID for consistent output
    sorted_patient_ids = sorted(personalization_results.keys())

    # Print header with more detailed metrics
    print_func(
        "Patient ID | Before (Test) Metrics                                                                 | After (Test) Metrics                                                                  | Acc Change"
    )
    print_func(
        "-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    )

    total_change = 0
    count_valid_patients = 0

    for patient_id in sorted_patient_ids:
        results = personalization_results[patient_id]
        # Check if both 'before' and 'after' metrics exist and are valid dictionaries
        if isinstance(results.get("before"), dict) and isinstance(
            results.get("after"), dict
        ):
            metrics_before_test = results["before"]
            metrics_after_test = results["after"]

            # Check if the 'after' evaluation confusion matrix indicates data was processed
            cm_after = metrics_after_test.get("confusion_matrix", [[0, 0], [0, 0]])
            if (
                isinstance(cm_after, list)
                and len(cm_after) == 2
                and len(cm_after[0]) == 2
                and sum(sum(row) for row in cm_after) > 0
            ):
                acc_before = metrics_before_test.get("accuracy", 0.0)
                acc_after = metrics_after_test.get("accuracy", 0.0)
                change = acc_after - acc_before

                before_str = format_metrics_for_summary(
                    metrics_before_test, prefix="Test: "
                )
                after_str = format_metrics_for_summary(
                    metrics_after_test, prefix="Test: "
                )

                print_func(
                    f"{patient_id:<10} | {before_str:<85} | {after_str:<85} | {change:.4f}"
                )
                total_change += change
                count_valid_patients += 1
            else:
                # Patient was in results dict, but after evaluation had no data (e.g., empty test set)
                before_str = format_metrics_for_summary(
                    metrics_before_test, prefix="Test: "
                )
                print_func(
                    f"{patient_id:<10} | {before_str:<85} | N/A                                                                   | N/A"
                )  # Show before, but N/A for after if evaluation failed
                # Do NOT include in average change calculation
                logging.info(
                    f"--- Debug: Patient {patient_id} skipped average calculation due to empty after test set."
                )  # Uncommented print and changed to logging.info

        else:
            # Patient was in results dict but metrics structure is unexpected (e.g., LOPO failed earlier in the parallel process)
            print_func(
                f"{patient_id:<10} | N/A                                                                   | N/A                                                                   | N/A"
            )  # Indicate missing data
            # Do NOT include in average change calculation
            logging.info(
                f"--- Debug: Patient {patient_id} skipped average calculation due to missing metrics."
            )  # Uncommented print and changed to logging.info

    print_func(
        "-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    )
    if count_valid_patients > 0:
        average_change = total_change / count_valid_patients
        print_func(
            f"Average Accuracy Improvement (across {count_valid_patients} patients with valid evaluation data): {average_change:.4f}"
        )
    else:
        print_func(
            "No valid personalized patient results to summarize average improvement."
        )


def plot_auc_roc(all_probs, all_labels, title, save_path):
    """Generates and saves an AUC-ROC curve plot."""
    try:
        # Ensure there are samples and both classes are present
        if len(all_labels) > 0 and len(np.unique(all_labels)) > 1:
            fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend(loc="lower right")
            plt.tight_layout()

            # Ensure directory exists before saving
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close() # Close the plot to free memory
            logging.info(f"Saved AUC-ROC plot to {save_path}")
        else:
            logging.warning(f"Skipping AUC-ROC plot for '{title}': Insufficient data or only one class.")
    except Exception as e:
        logging.error(f"Error generating or saving AUC-ROC plot '{title}': {e}")


def plot_confusion_matrix(cm, classes, title, save_path):
    """Generates and saves a Confusion Matrix plot."""
    try:
        # Ensure the confusion matrix is valid and has data
        if isinstance(cm, list) and len(cm) == 2 and len(cm[0]) == 2 and sum(sum(row) for row in cm) > 0:
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.title(title)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()

            # Ensure directory exists before saving
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close() # Close the plot to free memory
            logging.info(f"Saved Confusion Matrix plot to {save_path}")
        else:
            logging.warning(f"Skipping Confusion Matrix plot for '{title}': Invalid or empty confusion matrix.")
    except Exception as e:
        logging.error(f"Error generating or saving Confusion Matrix plot '{title}': {e}")
        
def plot_training_history(history, title_prefix, save_dir, filename_suffix):
    """Generates and saves plots of training history (loss, accuracy, F1-score, AUC-ROC) over epochs."""
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    if history['val_loss'] and any(v is not None for v in history['val_loss']): # Only plot if validation data exists
        plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title(f'{title_prefix} - Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'loss_history_{filename_suffix}.png'))
    plt.close()
    logging.info(f"Saved loss history plot to {os.path.join(save_dir, f'loss_history_{filename_suffix}.png')}")

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_accuracy'], label='Train Accuracy')
    if history['val_accuracy'] and any(v is not None for v in history['val_accuracy']): # Only plot if validation data exists
        plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{title_prefix} - Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'accuracy_history_{filename_suffix}.png'))
    plt.close()
    logging.info(f"Saved accuracy history plot to {os.path.join(save_dir, f'accuracy_history_{filename_suffix}.png')}")

    # Plot F1-score
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_f1_score'], label='Train F1-score')
    if history['val_f1_score'] and any(v is not None for v in history['val_f1_score']): # Only plot if validation data exists
        plt.plot(epochs, history['val_f1_score'], label='Validation F1-score')
    plt.title(f'{title_prefix} - F1-score Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'f1_history_{filename_suffix}.png'))
    plt.close()
    logging.info(f"Saved F1-score history plot to {os.path.join(save_dir, f'f1_history_{filename_suffix}.png')}")

    # Plot AUC-ROC
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_auc_roc'], label='Train AUC-ROC')
    if history['val_auc_roc'] and any(v is not None for v in history['val_auc_roc']): # Only plot if validation data exists
        plt.plot(epochs, history['val_auc_roc'], label='Validation AUC-ROC')
    plt.title(f'{title_prefix} - AUC-ROC Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('AUC-ROC')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'auc_history_{filename_suffix}.png'))
    plt.close()
    logging.info(f"Saved AUC-ROC history plot to {os.path.join(save_dir, f'auc_history_{filename_suffix}.png')}")


def plot_probability_distribution(all_probs, all_labels, title, save_path):
    """Generates and saves a histogram/KDE plot of model predicted probabilities on the test set."""
    try:
        # Ensure all_probs and all_labels are not empty and have consistent lengths
        if not all_probs or not all_labels or len(all_probs) != len(all_labels):
            logging.warning(f"Skipping probability distribution plot for '{title}': Invalid or empty data available.")
            return

        # --- AMENDMENT START ---
        # Ensure labels are standard Python integers or strings for hashing
        # Convert numpy array labels to a list of standard Python integers first
        python_labels = [int(label) for label in np.array(all_labels).flatten()]

        df_probs = pd.DataFrame({
            'Probability': all_probs,
            # Use the list of standard Python integers to create the 'True Label' column
            'True Label': ['Pre-ictal (1)' if label == 1 else 'Interictal (0)' for label in python_labels]
        })
        # --- AMENDMENT END ---

        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=df_probs,
            x='Probability',
            hue='True Label',
            kde=True, # Adds a Kernel Density Estimate plot
            bins=50, # More bins for smoother distribution
            palette={'Interictal (0)': 'skyblue', 'Pre-ictal (1)': 'salmon'},
            stat='density', # Normalize histogram to show density
            common_norm=False # Ensure each hue is normalized separately
        )
        plt.title(title)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.xlim([0, 1])
        plt.legend(title='True Label')
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Saved probability distribution plot to {save_path}")
    except Exception as e:
        logging.error(f"Error generating or saving probability distribution plot '{title}': {e}")

# ... (Keep all imports, configurations, feature flags, tunable parameters,
#      model definitions, training/evaluation functions, LOPO functions,
#      helper functions, and plotting functions as they are) ...


# --- Main Execution ---
if __name__ == "__main__":
    # Ensure the base output directory exists at the very beginning
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Configure Logging ---
    # Use a timestamp for a unique filename for the entire run
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(
        OUTPUT_DIR, f"seizure_prediction_results_{timestamp_str}_v3enhanced.log"
    )
    summary_output_filename = os.path.join(
        OUTPUT_DIR, f"seizure_prediction_summary_{timestamp_str}_v3evnhanced.txt"
    )

    # Configure the root logger
    # Add a check to prevent adding handlers multiple times if the script is somehow re-imported
    # This can happen in some environments even with __main__ guard
    if not logging.getLogger().handlers: # Check if handlers already exist
        logging.basicConfig(
            level=logging.INFO,  # Set the minimum logging level
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_filename),  # Log to file
                logging.StreamHandler(sys.stdout),  # Log to console (stdout)
            ],
        )


    # Log the start of the run
    logging.info("--- Seizure Prediction Run Started ---")
    logging.info(f"Run Date: {time.ctime()}")
    logging.info(f"Data Directory: {DATA_ROOT_DIR}")
    logging.info(f"Output Directory: {OUTPUT_DIR}")
    logging.info(f"Using device: {DEVICE}")  # Log device info at the start
    logging.info(f"Run All Model Types: {RUN_ALL_MODEL_TYPES}")
    logging.info(f"Adaptive Sensor Testing Enabled: {ENABLE_ADAPTIVE_SENSORS}")
    logging.info(f"Tunable Hyperparameters Enabled: {ENABLE_TUNABLE_HYPERPARAMETERS}")
    logging.info(f"Personalization Enabled: {ENABLE_PERSONALIZATION}")
    logging.info(f"REUSE_PROCESSED_DATA: {REUSE_PROCESSED_DATA}") # Log the new flag
    logging.info(
        f"Max Patients to Include: {'All' if MAX_PATIENTS_TO_INCLUDE is None else MAX_PATIENTS_TO_INCLUDE}"
    )
    logging.info(f"Base Sensors: {BASE_SENSORS}")
    logging.info(f"Segment Duration (seconds): {SEGMENT_DURATION_SECONDS}")

    all_patient_folders = [
        f.path
        for f in os.scandir(DATA_ROOT_DIR)
        if f.is_dir() and f.name.startswith("MSEL_")
    ]
    all_patient_folders.sort()  # Sort to ensure consistent patient order

    if not all_patient_folders:
        logging.error(
            f"No patient directories starting with 'MSEL_' found in {DATA_ROOT_DIR}. Exiting."
        )
        sys.exit()

    # --- Filter out excluded patients ---
    original_patient_count = len(all_patient_folders)
    all_patient_folders = [
        folder
        for folder in all_patient_folders
        if os.path.basename(folder) not in PATIENTS_TO_EXCLUDE
    ]
    if original_patient_count > len(all_patient_folders):
        logging.info(
            f"Excluded {original_patient_count - len(all_patient_folders)} patient(s) based on PATIENTS_TO_EXCLUDE list."
        )
    # --- End filtering ---

    # --- Apply the patient limit if set (MOVED BEFORE CACHING LOGIC) ---
    if (
        MAX_PATIENTS_TO_INCLUDE is not None
        and isinstance(MAX_PATIENTS_TO_INCLUDE, int)
        and MAX_PATIENTS_TO_INCLUDE > 0
    ):
        # Select the first MAX_PATIENTS_TO_INCLUDE from the already filtered list
        all_patient_folders_current_run = all_patient_folders[:MAX_PATIENTS_TO_INCLUDE]
        logging.info(
            f"Limiting run to the first {MAX_PATIENTS_TO_INCLUDE} patient directories."
        )
    else:
        # If MAX_PATIENTS_TO_INCLUDE is None or invalid, include all filtered patients
        all_patient_folders_current_run = all_patient_folders
        logging.info("Including all filtered patient directories in this run.")

    num_patients_in_run = len(all_patient_folders_current_run)
    logging.info(
        f"Total {num_patients_in_run} patient directories included for this run."
    )
    # --- END MOVED PATIENT LIMITING ---


    # <--- MODIFIED CACHING LOGIC START ---

    # Define fixed data processing parameters used if REUSE_PROCESSED_DATA is True
    # These are always the first value from their respective TUNABLE lists for caching purposes.
    # This ensures a consistent cache file path for a fixed data processing configuration.
    FIXED_DATA_PROCESSING_CONFIG = {
        "pre_ictal_window_min": TUNABLE_PRE_ICTAL_WINDOW_MINUTES[0],
        "pre_ictal_exclusion_buffer_min": TUNABLE_PRE_ICTAL_EXCLUSION_BUFFER_MINUTES[0],
        "post_ictal_buffer_min": TUNABLE_POST_ICTAL_BUFFER_MINUTES[0],
        "sampling_freq_hz": TUNABLE_SAMPLING_FREQ_HZ[0],
    }

    # Construct the path for cached processed data based on FIXED_DATA_PROCESSING_CONFIG
    # AND the number of patients included in this run.
    data_config_id = get_data_config_identifier(
        FIXED_DATA_PROCESSING_CONFIG["pre_ictal_window_min"],
        FIXED_DATA_PROCESSING_CONFIG["pre_ictal_exclusion_buffer_min"],
        FIXED_DATA_PROCESSING_CONFIG["post_ictal_buffer_min"],
        FIXED_DATA_PROCESSING_CONFIG["sampling_freq_hz"]
    )
    # Include the number of patients in the cache filename
    cache_filename = f"processed_patient_data_n{num_patients_in_run}_{data_config_id}.pkl"
    # Use a dedicated cache directory within OUTPUT_DIR
    PROCESSED_DATA_CACHE_DIR = os.path.join(OUTPUT_DIR, "cached_processed_data")
    PROCESSED_DATA_CACHE_PATH = os.path.join(PROCESSED_DATA_CACHE_DIR, cache_filename)

    all_processed_patient_data = [] # Initialize list to store processed data for all patients

    # --- Data Processing Phase (potentially cached) ---
    print(PROCESSED_DATA_CACHE_PATH, "oiii")
    processed_data_loaded_from_cache = False
    # Check for cache only if REUSE_PROCESSED_DATA is True
    if REUSE_PROCESSED_DATA and os.path.exists(PROCESSED_DATA_CACHE_PATH):
        logging.info(f"Attempting to load cached processed data from: {PROCESSED_DATA_CACHE_PATH}")
        try:
            with open(PROCESSED_DATA_CACHE_PATH, 'rb') as f:
                cached_data = pickle.load(f)

            # --- VERIFY CACHED DATA MATCHES CURRENT RUN PATIENTS ---
            # Check if the loaded data has the correct number of patients
            if isinstance(cached_data, list) and len(cached_data) == num_patients_in_run:
                 # Optional: Also verify patient IDs match? More robust but adds overhead.
                 # For now, just checking the count is a good start.
                 all_processed_patient_data = cached_data
                 logging.info(f"Successfully loaded cached data for {len(all_processed_patient_data)} patients.")
                 processed_data_loaded_from_cache = True
            else:
                 logging.warning(f"Cached data found at {PROCESSED_DATA_CACHE_PATH} has {len(cached_data) if isinstance(cached_data, list) else 'an invalid format'} patients, but {num_patients_in_run} patients are included in the current run. Reprocessing data.")
                 all_processed_patient_data = [] # Clear invalid cached data
            # --- END VERIFICATION ---

        except Exception as e:
            logging.error(f"Error loading cached data from {PROCESSED_DATA_CACHE_PATH}: {e}. Falling back to reprocessing data.")
            all_processed_patient_data = [] # Ensure list is empty if load failed

    # If data wasn't loaded from cache, or REUSE_PROCESSED_DATA is False, process the data
    if not processed_data_loaded_from_cache:
        if REUSE_PROCESSED_DATA: # If flag was true but path didn't exist or load failed or verification failed
            logging.info(f"Cached processed data not found, load failed, or patient count mismatch. Processing data now using config: {data_config_id} for {num_patients_in_run} patients.")
        else: # If flag is false
            logging.info("REUSE_PROCESSED_DATA is False. Processing data without caching.")

        # The list of patient folders to process is already filtered by MAX_PATIENTS_TO_INCLUDE and PATIENTS_TO_EXCLUDE
        patient_dirs_to_process = all_patient_folders_current_run

        # Process patients using the FIXED_DATA_PROCESSING_CONFIG (regardless of ENABLE_TUNABLE_HYPERPARAMETERS)
        # This is because the cache is based on a fixed config. If tuning data HPs, caching is off.
        logging.info(f"Processing data for {len(patient_dirs_to_process)} patients using fixed data processing config: {data_config_id}")
        # Removed ThreadPoolExecutor for this initial sequential processing phase
        for patient_dir in tqdm(patient_dirs_to_process, desc="Processing Patients"):
            result = prepare_patient_data(
                patient_dir,
                current_sampling_freq_hz=FIXED_DATA_PROCESSING_CONFIG["sampling_freq_hz"],
                current_pre_ictal_window_min=FIXED_DATA_PROCESSING_CONFIG["pre_ictal_window_min"],
                current_pre_ictal_exclusion_buffer_min=FIXED_DATA_PROCESSING_CONFIG["pre_ictal_exclusion_buffer_min"],
                current_post_ictal_buffer_min=FIXED_DATA_PROCESSING_CONFIG["post_ictal_buffer_min"]
            )
            if result is not None:
                all_processed_patient_data.append(result)


        # Save processed data to cache if REUSE_PROCESSED_DATA was True and data was just processed
        # Also ensure that the number of successfully processed patients matches the number intended for the run
        if REUSE_PROCESSED_DATA and not processed_data_loaded_from_cache and len(all_processed_patient_data) == num_patients_in_run:
            os.makedirs(PROCESSED_DATA_CACHE_DIR, exist_ok=True)
            try:
                with open(PROCESSED_DATA_CACHE_PATH, 'wb') as f:
                    pickle.dump(all_processed_patient_data, f)
                logging.info(f"Processed data for {len(all_processed_patient_data)} patients cached to: {PROCESSED_DATA_CACHE_PATH}")
            except Exception as e:
                logging.error(f"Error saving processed data cache to {PROCESSED_DATA_CACHE_PATH}: {e}")
        elif REUSE_PROCESSED_DATA and not processed_data_loaded_from_cache and len(all_processed_patient_data) != num_patients_in_run:
             logging.warning(f"Processed data for {len(all_processed_patient_data)} patients, but {num_patients_in_run} were intended for this run. Skipping cache save to {PROCESSED_DATA_CACHE_PATH} to avoid mismatch.")


    # Check if any patients were successfully processed (either loaded or newly processed)
    # This check now also verifies that the number of processed patients matches the intended number for the run
    if not all_processed_patient_data or len(all_processed_patient_data) != num_patients_in_run:
        logging.error(f"No patient data processed successfully for the intended {num_patients_in_run} patients (either from cache or new processing). Processed {len(all_processed_patient_data)}. Exiting.")
        sys.exit(1)

    # <--- MODIFIED CACHING LOGIC END ---


    # --- Prepare Hyperparameter Combinations (Modified to handle caching) ---
    # This part remains largely the same, as it generates all HP combinations
    # based on the TUNABLE lists. The data processing HPs within the combo
    # will be fixed to the cached values if REUSE_PROCESSED_DATA is True.
    # The actual data used for training/evaluation will be from all_processed_patient_data,
    # which is already filtered by MAX_PATIENTS_TO_INCLUDE.
    hp_combinations = []
    # Define the base parameter lists including ALL tunable parameters
    base_hp_param_lists = {
        "conv_filters": TUNABLE_CONV_FILTERS,
        "conv_kernel_size": TUNABLE_CONV_KERNEL_SIZE,
        "pool_size": TUNABLE_POOL_SIZE,
        "lstm_units": TUNABLE_LSTM_UNITS,
        "dense_units": TUNABLE_DENSE_UNITS,
        "general_model_epochs": TUNABLE_GENERAL_MODEL_EPOCHS,
        "personalization_epochs": TUNABLE_PERSONALIZATION_EPOCHS,
        "general_model_lr": TUNABLE_GENERAL_MODEL_LR,
        "personalization_lr": TUNABLE_PERSONALIZATION_LR,
        "batch_size": TUNABLE_BATCH_SIZE,
        "personalization_batch_size": TUNABLE_PERSONALIZATION_BATCH_SIZE,
        "dropout_rate": TUNABLE_DROPOUT_RATE,
        "general_model_weight_decay": TUNABLE_WEIGHT_DECAY_GENERAL,
        "personalization_weight_decay": TUNABLE_WEIGHT_DECAY_PERSONALIZATION,
        # Data processing HPs are ALWAYS included in the hp_combo for consistency in extraction later.
        # Their values will be adjusted based on REUSE_PROCESSED_DATA below.
        "sampling_freq_hz": TUNABLE_SAMPLING_FREQ_HZ,
        "pre_ictal_window_min": TUNABLE_PRE_ICTAL_WINDOW_MINUTES,
        "pre_ictal_exclusion_buffer_min": TUNABLE_PRE_ICTAL_EXCLUSION_BUFFER_MINUTES,
        "post_ictal_buffer_min": TUNABLE_POST_ICTAL_BUFFER_MINUTES,
    }

    if ENABLE_TUNABLE_HYPERPARAMETERS:
        hp_param_lists_for_product = base_hp_param_lists.copy()
        if REUSE_PROCESSED_DATA:
            # If reusing processed data, fix the data processing HPs to their single cached values
            # This prevents iterating over data processing HPs in the grid search when caching is enabled.
            hp_param_lists_for_product["sampling_freq_hz"] = [FIXED_DATA_PROCESSING_CONFIG["sampling_freq_hz"]]
            hp_param_lists_for_product["pre_ictal_window_min"] = [FIXED_DATA_PROCESSING_CONFIG["pre_ictal_window_min"]]
            hp_param_lists_for_product["pre_ictal_exclusion_buffer_min"] = [FIXED_DATA_PROCESSING_CONFIG["pre_ictal_exclusion_buffer_min"]]
            hp_param_lists_for_product["post_ictal_buffer_min"] = [FIXED_DATA_PROCESSING_CONFIG["post_ictal_buffer_min"]]
        # Else (if not REUSE_PROCESSED_DATA), the data processing HPs remain as their full TUNABLE_ lists
        # and will be included in the grid search combinations.

        keys, values = zip(*hp_param_lists_for_product.items())
        for bundle in itertools.product(*values):
            hp_combinations.append(dict(zip(keys, bundle)))
    else:
        # If ENABLE_TUNABLE_HYPERPARAMETERS is False, we only run one combination.
        single_combo = {}
        for key, value_list in base_hp_param_lists.items():
            single_combo[key] = value_list[0] # Use the first value from each TUNABLE_ list as the default

        # If REUSE_PROCESSED_DATA is True, override the data processing HPs with the fixed values
        # for this single combination.
        if REUSE_PROCESSED_DATA:
            single_combo.update(FIXED_DATA_PROCESSING_CONFIG)

        hp_combinations.append(single_combo)

    logging.info(
        f"Prepared {len(hp_combinations)} hyperparameter combination(s) to test."
    )
    # <--- END MODIFIED HYPERPARAMETER PREPARATION ---


    # --- Outer loop for Hyperparameter Combinations ---
    all_results = {}  # Dictionary to store all results for all HP combos
    start_time_overall = time.time()

    # Iterate through each hyperparameter combination
    # The hp_combinations list now correctly reflects whether data processing HPs are fixed or tuned
    for hp_idx, current_hp_combo in enumerate(tqdm(hp_combinations, desc="Overall HP Combinations")):
        # Create a string representation of the current HP combination for logging and saving
        # Include data processing HPs in the description for clarity, even if fixed
        hp_combo_desc_parts = []
        for k in ["sampling_freq_hz", "pre_ictal_window_min", "conv_filters", "lstm_units", "batch_size"]:
            if k in current_hp_combo:
                 # Handle list values for conv_filters
                 value_str = str(current_hp_combo[k]).replace('[', '').replace(']', '').replace(', ', '-')
                 hp_combo_desc_parts.append(f"{k}-{value_str}")
            # else: logging.warning(f"Warning: Key '{k}' not found in current_hp_combo.") # Should not happen with base_hp_param_lists

        current_hp_combo_str = f"HP_Combo_{hp_idx}_" + "_".join(hp_combo_desc_parts)


        logging.info(f"{'='*80}")
        logging.info(
            f"STARTING RUN FOR HYPERPARAMETER COMBINATION {hp_idx+1}/{len(hp_combinations)}"
        )
        logging.info(f"Parameters: {OrderedDict(sorted(current_hp_combo.items()))}") # Print sorted HPs
        logging.info(f"{'='*80}")

        all_results[current_hp_combo_str] = {}  # Store results for this HP combo

        # Extract current HP values for clarity and passing
        # These values are now extracted from the current_hp_combo, which accounts for caching
        current_sampling_freq_hz = current_hp_combo["sampling_freq_hz"]
        current_pre_ictal_window_min = current_hp_combo["pre_ictal_window_min"]
        current_pre_ictal_exclusion_buffer_min = current_hp_combo[
            "pre_ictal_exclusion_buffer_min"
        ]
        current_post_ictal_buffer_min = current_hp_combo["post_ictal_buffer_min"]
        current_conv_filters = current_hp_combo["conv_filters"]
        current_conv_kernel_size = current_hp_combo["conv_kernel_size"]
        current_pool_size = current_hp_combo["pool_size"]
        current_lstm_units = current_hp_combo["lstm_units"]
        current_dense_units = current_hp_combo["dense_units"]
        current_general_model_epochs = current_hp_combo["general_model_epochs"]
        current_personalization_epochs = current_hp_combo["personalization_epochs"]
        current_general_model_lr = current_hp_combo["general_model_lr"]
        current_personalization_lr = current_hp_combo["personalization_lr"]
        current_batch_size = current_hp_combo["batch_size"] # Corrected from current_batch_combo
        current_personalization_batch_size = current_hp_combo[
            "personalization_batch_size"
        ]
        current_dropout_rate= current_hp_combo["dropout_rate"]
        current_general_model_weight_decay = current_hp_combo["general_model_weight_decay"]
        current_personalization_weight_decay = current_hp_combo["personalization_weight_decay"]

        # Store model and general HPs in dictionaries to pass to functions
        model_hyperparameters = {
            "conv_filters": current_conv_filters,
            "conv_kernel_size": current_conv_kernel_size,
            "pool_size": current_pool_size,
            "lstm_units": current_lstm_units,
            "dense_units": current_dense_units,
            "sampling_freq_hz": current_sampling_freq_hz,  # Include sampling freq here for model init
            "dropout_rate": current_dropout_rate,
        }
        general_hyperparameters = {
            "epochs": current_general_model_epochs,
            "learning_rate": current_general_model_lr,
            "batch_size": current_batch_size,
            "weight_decay": current_general_model_weight_decay,
        }
        personalization_hyperparameters = {
            "epochs": current_personalization_epochs,
            "learning_rate": current_personalization_lr,
            "batch_size": current_personalization_batch_size,
            "weight_decay": current_personalization_weight_decay,
        }

        # --- Data Processing is now done BEFORE this loop if REUSE_PROCESSED_DATA is True ---
        # The `all_processed_patient_data` list is available here, containing data
        # processed with the FIXED_DATA_PROCESSING_CONFIG if caching was enabled and successful,
        # or the data processed with the CURRENT HP combination if caching was off or failed.
        # This list is already filtered by MAX_PATIENTS_TO_INCLUDE.

        # Determine which model types to run (same as before, but inside HP loop)
        models_to_run = (
            MODEL_TYPES_TO_RUN if RUN_ALL_MODEL_TYPES else [MODEL_TYPES_TO_RUN[0]]
        )

        # Determine which sensor combinations to run (same as before, but inside HP loop)
        sensor_combinations_to_run = (
            ALL_SENSOR_COMBINATIONS if ENABLE_ADAPTIVE_SENSORS else [list(BASE_SENSORS)]
        )

        # --- Loops for Model Types and Sensor Combinations (Inner Loops) ---
        for current_model_type in models_to_run:
            all_results[current_hp_combo_str][
                current_model_type
            ] = {}  # Store results for this model type

            for current_combination in sensor_combinations_to_run:
                combination_name = "_".join(
                    current_combination
                ).upper()  # Consistent naming
                all_results[current_hp_combo_str][current_model_type][
                    combination_name
                ] = {}  # Store results for this combo

                logging.info(f"{'='*40}")
                logging.info(
                    f"RUNNING: Model {current_model_type} + Sensors {combination_name} with HP: {current_hp_combo_str}"
                )
                logging.info(f"{'='*40}")

                # Get list of patients suitable for this combination and the sensor indices
                # This function filters from the already processed `all_processed_patient_data` list
                # which is already limited by MAX_PATIENTS_TO_INCLUDE.
                (
                    patients_suitable_for_combination,
                    sensor_combination_indices,
                ) = get_patients_and_indices_for_combination(
                    all_processed_patient_data,  # Use the list populated before the HP loop (already limited)
                    current_combination,  # Pass the list of sensor names
                )

                if not patients_suitable_for_combination:
                    logging.warning(
                        f"Skipping run for {current_model_type} + {combination_name} with HP: {current_hp_combo_str}: No suitable patients found with required sensors in processed data."
                    )
                    all_results[current_hp_combo_str][current_model_type][
                        combination_name
                    ]["num_suitable_patients"] = 0
                    all_results[current_hp_combo_str][current_model_type][
                        combination_name
                    ]["overall_general"] = {"metrics": {}, "num_suitable_patients": 0}
                    if ENABLE_PERSONALIZATION:
                        all_results[current_hp_combo_str][current_model_type][
                            combination_name
                        ]["personalization"] = {
                            "personalization_results": {},
                            "avg_personalized_metrics": None,
                            "num_suitable_patients": 0,
                        }
                    continue  # Move to the next combination

                all_results[current_hp_combo_str][current_model_type][combination_name][
                    "num_suitable_patients"
                ] = len(patients_suitable_for_combination)
                logging.info(
                    f"Proceeding with {len(patients_suitable_for_combination)} suitable patients for this run."
                )

                # --- Phase 1: Train and Evaluate Overall General Model ---
                logging.info(f"{'--'*30}")
                logging.info("PHASE 1: TRAINING & EVALUATING OVERALL GENERAL MODEL")
                logging.info(f"{'--'*30}")

                # Combine data from suitable patients for the overall general model for this combination
                overall_general_segments_raw = []
                overall_general_labels_raw = []
                # We also need to keep track of patient IDs for the LOPO split later
                overall_general_patient_ids = [] # This list is not strictly needed for Overall General training, but good for tracking

                for patient_data_tuple in patients_suitable_for_combination:
                    # patient_data_tuple is (patient_id, segments_all_sensors, labels, found_sensors)
                    patient_id, segments_all_sensors, labels, found_sensors = patient_data_tuple

                    # Slice segments to include only the features for the current combination
                    # segments_all_sensors shape is (N, L, len(BASE_SENSORS)) - based on initial processing
                    # We use sensor_combination_indices obtained from get_patients_and_indices_for_combination
                    if (
                        len(segments_all_sensors) > 0
                        and len(sensor_combination_indices) > 0
                        and segments_all_sensors.shape[2] == len(BASE_SENSORS)
                    ):
                        segments_sliced = segments_all_sensors[
                            :, :, sensor_combination_indices
                        ]
                        overall_general_segments_raw.append(segments_sliced)
                        overall_general_labels_raw.append(labels)
                        # Add patient ID for each segment (will be repeated) - useful for debugging/tracking
                        # overall_general_patient_ids.extend([patient_id] * len(labels)) # Uncomment if needed for debugging

                    # else: logging.warning(f"Skipping patient {patient_id} data for Overall General training: Shape mismatch or no data after slicing.")

                # Check if any data was collected for the overall general model for this combination
                if not overall_general_segments_raw:
                    logging.warning(
                        f"No segments available for Overall General Model training ({current_model_type}, {combination_name}, {current_hp_combo_str}). Skipping."
                    )
                    overall_general_results_by_combo_model_run = {
                        "metrics": {},
                        "num_suitable_patients": len(patients_suitable_for_combination),
                    }
                    overall_general_model_state = None  # Indicate failure/skip
                else:
                    overall_general_segments_combined = np.concatenate(
                        overall_general_segments_raw, axis=0
                    )
                    overall_general_labels_combined = np.concatenate(
                        overall_general_labels_raw, axis=0
                    )
                    # Convert patient IDs list to numpy array (if used)
                    # overall_general_patient_ids = np.array(overall_general_patient_ids)


                    # Check for sufficient data for overall general training for this combination
                    # Need at least 3 samples total for 60/20/20 split, and at least one of each class in the total combined data
                    if (
                        len(overall_general_segments_combined) < 3
                        or len(np.unique(overall_general_labels_combined)) < 2
                    ):
                        logging.warning(
                            f"Not enough data ({len(overall_general_segments_combined)} samples) or only one class ({np.unique(overall_general_labels_combined)}) available for Overall General Model training ({current_model_type}, {combination_name}, {current_hp_combo_str}). Skipping training."
                        )
                        overall_general_results_by_combo_model_run = {
                            "metrics": {},
                            "num_suitable_patients": len(patients_suitable_for_combination),
                        }
                        overall_general_model_state = None  # Indicate failure/skip
                    else:
                        logging.info(
                            f"Overall General Combined data shape ({current_model_type}, {combination_name}, {current_hp_combo_str}): {overall_general_segments_combined.shape}"
                        )
                        logging.info(
                            f"Overall General Combined labels shape ({current_model_type}, {combination_name}, {current_hp_combo_str}): {overall_general_labels_combined.shape}"
                        )

                        # Perform 60/20/20 split for Overall General Model
                        try:
                            # Ensure enough data for splitting and stratification (at least 2 samples per class for test_size > 0)
                            X_train_og, X_temp_og, y_train_og, y_temp_og = train_test_split(
                                overall_general_segments_combined,
                                overall_general_labels_combined,
                                test_size=0.4,
                                random_state=SEED,
                                stratify=overall_general_labels_combined,
                            )
                            X_val_og, X_test_og, y_val_og, y_test_og = train_test_split(
                                X_temp_og,
                                y_temp_og,
                                test_size=0.5,
                                random_state=SEED,
                                stratify=y_temp_og,  # Stratify on y_temp_og
                            )
                        except ValueError as e:
                            logging.warning(
                                f"Warning: Overall General Model data split failed ({current_model_type}, {combination_name}, {current_hp_combo_str}): {e}. This might happen with very few samples or imbalanced small datasets leading to single-class splits. Skipping training."
                            )
                            overall_general_model_metrics = {}
                            overall_general_model_state = None  # Indicate failure/skip
                        except Exception as e:
                            logging.error(
                                f"An unexpected error occurred during Overall General Model data split ({current_model_type}, {combination_name}, {current_hp_combo_str}): {e}. Skipping training."
                            )
                            overall_general_model_metrics = {}
                            overall_general_model_state = None  # Indicate failure/skip

                        # Only proceed if splits are valid and contain both classes
                        unique_y_train_og = np.unique(y_train_og)
                        unique_y_val_og = np.unique(y_val_og)
                        unique_y_test_og = np.unique(y_test_og)

                        if (
                            "X_train_og" in locals()
                            and len(X_train_og) > 0
                            and len(X_val_og) > 0
                            and len(X_test_og) > 0
                            and len(unique_y_train_og) > 1
                            and len(unique_y_val_og) > 1
                            and len(unique_y_test_og) > 1
                        ):

                            logging.info(
                                f"Overall General Train shape ({current_model_type}, {combination_name}, {current_hp_combo_str}): {X_train_og.shape}, Val shape: {X_val_og.shape}, Test shape: {X_test_og.shape}"
                            )

                            # Use the actual shape determined from the sliced data for this combination
                            input_channels_og = overall_general_segments_combined.shape[2]
                            seq_len_og = overall_general_segments_combined.shape[1]

                            overall_general_train_dataset = SeizureDataset(
                                X_train_og,
                                y_train_og,
                                seq_len=seq_len_og, # Pass shape info to Dataset
                                num_features=input_channels_og, # Pass shape info to Dataset
                            )
                            overall_general_val_dataset = SeizureDataset(
                                X_val_og,
                                y_val_og,
                                seq_len=seq_len_og, # Pass shape info to Dataset
                                num_features=input_channels_og, # Pass shape info to Dataset
                            )
                            overall_general_test_dataset = SeizureDataset(
                                X_test_og,
                                y_test_og,
                                seq_len=seq_len_og, # Pass shape info to Dataset
                                num_features=input_channels_og, # Pass shape info to Dataset
                            )

                            # Reduced num_workers to 0 to mitigate memory issues
                            num_workers_og = 0
                            persistent_workers_og = (
                                False  # persistent_workers requires num_workers > 0
                            )

                            # Adjust batch sizes if larger than dataset size, ensure min 1 if dataset not empty
                            og_train_batch_size = current_batch_size
                            if len(overall_general_train_dataset) > 0:
                                og_train_batch_size = max(
                                    1,
                                    min(
                                        og_train_batch_size,
                                        len(overall_general_train_dataset),
                                    ),
                                )
                            og_val_batch_size = current_batch_size
                            if len(overall_general_val_dataset) > 0:
                                og_val_batch_size = max(
                                    1,
                                    min(
                                        og_val_batch_size, len(overall_general_val_dataset)
                                    ),
                                )
                            og_test_batch_size = current_batch_size
                            if len(overall_general_test_dataset) > 0:
                                og_test_batch_size = max(
                                    1,
                                    min(
                                        og_test_batch_size,
                                        len(overall_general_test_dataset),
                                    ),
                                )

                            overall_general_train_dataloader = DataLoader(
                                overall_general_train_dataset,
                                batch_size=og_train_batch_size,
                                shuffle=True,
                                num_workers=num_workers_og,
                                persistent_workers=persistent_workers_og,
                            )
                            overall_general_val_dataloader = DataLoader(
                                overall_general_val_dataset,
                                batch_size=og_val_batch_size,
                                shuffle=False,
                                num_workers=num_workers_og,
                                persistent_workers=persistent_workers_og,
                            )
                            overall_general_test_dataloader = DataLoader(
                                overall_general_test_dataset,
                                batch_size=og_test_batch_size,
                                shuffle=False,
                                num_workers=num_workers_og,
                                persistent_workers=persistent_workers_og,
                            )

                            # Calculate class weights for the overall general training data
                            # Ensure weights are calculated only if there are samples in the training set
                            class_weights_og_tensor = None # Initialize to None
                            if len(y_train_og) > 0:
                                classes_og = np.unique(y_train_og)
                                if len(classes_og) == 2:
                                    class_weights_og_np = class_weight.compute_class_weight(
                                        "balanced", classes=classes_og, y=y_train_og
                                    )
                                    class_weights_og_tensor = torch.tensor(class_weights_og_np, dtype=torch.float32).to(
                                        DEVICE
                                    )
                                    logging.info(
                                        f"Computed Overall General class weights ({current_model_type}, {combination_name}, {current_hp_combo_str}): {{0: {class_weights_og_np[0]:.4f}, 1: {class_weights_og_np[1]:.4f}}}"
                                    )
                                else:
                                    # Handle case where training set has only one class (unlikely with stratification but safe)
                                    logging.warning(f"Training set for Overall General Model has only one class ({classes_og}). Using uniform weights.")
                                    class_weights_og_tensor = torch.tensor([1.0, 1.0], dtype=torch.float32).to(DEVICE)
                            else:
                                logging.warning(f"Overall General training set is empty. Cannot compute class weights. Using uniform weights.")
                                class_weights_og_tensor = torch.tensor([1.0, 1.0], dtype=torch.float32).to(DEVICE)


                            # Instantiate the Overall General Model with current HPs
                            ModelClass_og = get_model_class(current_model_type)

                            try:
                                overall_general_model = ModelClass_og(
                                    input_channels=input_channels_og, # Use num_features for the current sensor combo
                                    seq_len=seq_len_og, # Use actual sequence length
                                    conv_filters=current_conv_filters,
                                    conv_kernel_size=current_conv_kernel_size,
                                    pool_size=current_pool_size,
                                    lstm_units=current_lstm_units,
                                    dense_units=current_dense_units,
                                    dropout_rate=current_dropout_rate, # Pass dropout rate
                                ).to(DEVICE)
                            except (ValueError, Exception) as e:
                                logging.error(
                                    f"Error instantiating Overall General Model ({current_model_type}, {combination_name}, {current_hp_combo_str}): {e}. Skipping training."
                                )
                                # Clean up dataloaders
                                del (
                                    overall_general_train_dataloader,
                                    overall_general_val_dataloader,
                                    overall_general_test_dataloader,
                                )
                                del (
                                    overall_general_train_dataset,
                                    overall_general_val_dataset,
                                    overall_general_test_dataset,
                                )
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                overall_general_model_metrics = {}
                                overall_general_model_state = None  # Indicate failure/skip

                            # Only attempt training if model instance was created successfully
                            if "overall_general_model" in locals() and overall_general_model is not None:

                                logging.info(
                                    f"Starting Overall General Model training ({current_model_type}, {combination_name}, {current_hp_combo_str})..."
                                )

                                # Define save path for the Overall General Model (ensure directories exist)
                                overall_general_model_save_dir = os.path.join(
                                    OUTPUT_DIR,
                                    timestamp_str,
                                    current_model_type,
                                    combination_name,
                                )
                                # Save model within a subfolder named by HP combo index
                                overall_general_model_save_path = os.path.join(
                                    overall_general_model_save_dir,
                                    f"hp_combo_{hp_idx+1}", # Include HP combo index in the folder name
                                    f"overall_general_model.pth", # Consistent filename within HP folder
                                )
                                # Plot directory within the HP combo folder
                                plot_dir_og = os.path.join(overall_general_model_save_dir, f"hp_combo_{hp_idx+1}", 'plots')

                                try:
                                    os.makedirs(os.path.dirname(overall_general_model_save_path), exist_ok=True) # Ensure model save directory exists
                                    os.makedirs(plot_dir_og, exist_ok=True) # Ensure plot directory exists
                                    logging.info(f"Created output directories for HP: {current_hp_combo_str}, Model: {current_model_type}, Sensors: {combination_name}")
                                except Exception as e:
                                    logging.error(f"Error creating output directories for HP: {current_hp_combo_str}, Model: {current_model_type}, Sensors: {combination_name}: {e}. Skipping this run.")
                                    # Store skip reason and continue to the next combination
                                    all_results[current_hp_combo_str][current_model_type][combination_name]['status'] = 'Directory Creation Failed'
                                    # Clean up dataloaders and model before continuing
                                    del (
                                        overall_general_train_dataloader,
                                        overall_general_val_dataloader,
                                        overall_general_test_dataloader,
                                    )
                                    del (
                                        overall_general_train_dataset,
                                        overall_general_val_dataset,
                                        overall_general_test_dataset,
                                    )
                                    del overall_general_model # Delete model instance
                                    gc.collect()
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    continue # Skip this combination run

                                # Define the criterion (loss function) with class weights
                                # Use BCELoss as the final layer is Sigmoid
                                criterion_og = nn.BCELoss()
                                # If you were using BCEWithLogitsLoss (output is logits), you'd pass pos_weight
                                # criterion_og = nn.BCEWithLogitsLoss(pos_weight=class_weights_og_tensor[1]) # Assuming class 1 is the positive class

                                (
                                    overall_general_model,
                                    overall_general_metrics,
                                ) = train_pytorch_model(
                                    overall_general_model,
                                    overall_general_train_dataloader,
                                    overall_general_val_dataloader,
                                    overall_general_test_dataloader,  # Pass test dataloader
                                    epochs=current_general_model_epochs,
                                    learning_rate=current_general_model_lr,
                                    class_weights=class_weights_og_tensor, # Pass class weights tensor
                                    save_best_model_path=overall_general_model_save_path,  # Save the overall general model
                                    desc=f"Training Overall General ({current_model_type}, {combination_name}, HP {hp_idx+1})",
                                    device=DEVICE,
                                    weight_decay=current_general_model_weight_decay, # Pass weight_decay
                                )

                                # Plot Training History
                                if 'history' in overall_general_metrics:
                                    plot_training_history(
                                        overall_general_metrics['history'],
                                        f'Overall General Model ({current_model_type}, {combination_name}, HP {hp_idx+1})',
                                        plot_dir_og,
                                        f'overall_general_hp_{hp_idx+1}'
                                    )


                                final_train_loss_from_history = overall_general_metrics['history']['train_loss'][-1] if overall_general_metrics['history']['train_loss'] else 0.0

                                # Log other training metrics (Accuracy, F1, AUC) from the final evaluation on the train set
                                # These are still useful to see the performance on the training data after training
                                logging.info(
                                    f"Overall General Model Training Metrics ({current_model_type}, {combination_name}, HP {hp_idx+1}) - Train (Final Eval): Acc={overall_general_metrics['train']['accuracy']:.4f}, Prec={overall_general_metrics['train']['precision']:.4f}, Rec={overall_general_metrics['train']['recall']:.4f}, F1={overall_general_metrics['train']['f1_score']:.4f}, AUC={overall_general_metrics['train']['auc_roc']:.4f}"
                                )                                
                                # Log the final epoch's average training loss from history
                                logging.info(
                                    f"Overall General Model Training Metrics ({current_model_type}, {combination_name}, HP {hp_idx+1}) - Train Loss (Final Epoch): {final_train_loss_from_history:.4f}"
                                )
                                logging.info(
                                    f"Overall General Model Validation Metrics ({current_model_type}, {combination_name}, HP {hp_idx+1}) - Val: {format_metrics_for_summary(overall_general_metrics['val'])}"
                                )
                                logging.info(
                                    f"Overall General Model Testing Metrics ({current_model_type}, {combination_name}, HP {hp_idx+1}) - Test: {format_metrics_for_summary(overall_general_metrics['test'])}"
                                )

                                # --- Add Plotting for Overall General Model (Test Set) ---
                                overall_general_test_metrics_data = overall_general_metrics['test']
                                overall_general_test_probs = overall_general_test_metrics_data.get('all_probs', []) # Assuming evaluate_pytorch_model returns 'all_probs'
                                overall_general_test_labels = overall_general_test_metrics_data.get('all_labels', []) # Assuming evaluate_pytorch_model returns 'all_labels'
                                overall_general_test_cm = overall_general_test_metrics_data.get('confusion_matrix', [[0,0],[0,0]])

                                # AUC-ROC Plot
                                if 'all_probs' in overall_general_test_metrics_data and 'all_labels' in overall_general_test_metrics_data:
                                    plot_auc_roc(
                                        overall_general_test_metrics_data['all_probs'],
                                        overall_general_test_metrics_data['all_labels'],
                                        f'Overall General Model AUC-ROC ({current_model_type}, {timestamp_str}, {combination_name}, HP {hp_idx+1})',
                                        os.path.join(plot_dir_og, f'overall_general_hp_{hp_idx+1}_auc_roc.png')
                                    )
                                    plot_probability_distribution(
                                        overall_general_test_metrics_data['all_probs'],
                                        overall_general_test_metrics_data['all_labels'],
                                        f'Overall General Model Probability Distribution (Test Set) ({current_model_type}, {combination_name}, HP {hp_idx+1})',
                                        os.path.join(plot_dir_og, f'overall_general_hp_{hp_idx+1}_prob_dist.png')
                                    )
                                else:
                                    logging.warning("Skipping Overall General AUC-ROC & Probability Distribution plot: 'all_probs' or 'all_labels' not found in test metrics.")


                                # Confusion Matrix Plot
                                plot_confusion_matrix(
                                    overall_general_test_cm,
                                    ['Interictal (0)', 'Pre-ictal (1)'], # Class names
                                    f'Overall General Model Confusion Matrix ({current_model_type},  {timestamp_str}, {combination_name}, HP {hp_idx+1})',
                                    os.path.join(plot_dir_og, f'overall_general_hp_{hp_idx+1}_confusion_matrix.png')
                                )
                                # --- End Plotting for Overall General Model ---

                                overall_general_results_by_combo_model_run = {
                                    'metrics': overall_general_metrics,
                                    'num_suitable_patients': len(patients_suitable_for_combination)
                                }

                                # Clean up Overall General Model resources
                                del (
                                    overall_general_model,
                                    overall_general_train_dataloader,
                                    overall_general_val_dataloader,
                                    overall_general_test_dataloader,
                                )
                                del (
                                    overall_general_train_dataset,
                                    overall_general_val_dataset,
                                    overall_general_test_dataset,
                                )
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()

                            else:
                                logging.warning(
                                    f"Overall General training dataloader ({current_model_type}, {combination_name}, {current_hp_combo_str}) is empty. Skipping training and evaluation."
                                )
                                overall_general_model_metrics = {}
                                # overall_general_model_state = None  # Indicate failure/skip - not needed if model instance is deleted
                                overall_general_results_by_combo_model_run = {
                                    "metrics": {},
                                    "num_suitable_patients": len(
                                        patients_suitable_for_combination
                                    ),
                                }

                # Store Overall General results for this run
                all_results[current_hp_combo_str][current_model_type][combination_name][
                    "overall_general"
                ] = overall_general_results_by_combo_model_run


                # --- Phase 2: Per-Patient Personalization (with LOPO) ---
                if ENABLE_PERSONALIZATION:
                    logging.info(f"{'--'*30}")
                    logging.info("PHASE 2: PER-PATIENT PERSONALIZATION (using LOPO)")
                    logging.info(f"{'--'*30}")

                    # Perform Personalization for this combination (with LOPO handled inside and now parallel)
                    # Pass the *full* all_processed_patient_data list (processed with FIXED HPs)
                    personalization_results = perform_personalization_pytorch_lopo(
                        all_processed_patient_data,  # Pass the list of all patients processed with FIXED HPs
                        current_model_type,
                        current_combination,  # Pass the list of sensor names
                        general_hyperparameters,  # Pass general HPs for LOPO training inside
                        personalization_hyperparameters,  # Pass personalization HPs for fine-tuning inside
                        model_hyperparameters,  # Pass model architecture HPs
                        current_hp_combo_str,  # Pass HP combo string
                        DEVICE.type,  # Pass the device name as a string
                    )

                    all_results[current_hp_combo_str][current_model_type][
                        combination_name
                    ]["personalization"] = {
                        "personalization_results": personalization_results,  # Store per-patient results
                        "num_suitable_patients": len(
                            patients_suitable_for_combination
                        ),  # Store patient count (those that were attempted)
                    }

                    # --- Summarize Personalized Model Performance for this combination ---
                    # Write per-patient summary to the summary text file
                    with open(
                        summary_output_filename, "a"
                    ) as summary_file:  # Use 'a' for append mode
                        summary_file.write(f"\n\n{'#'*60}\n")
                        summary_file.write(
                            f"PERSONALIZATION RESULTS FOR HP: {current_hp_combo_str}, MODEL: {current_model_type}, SENSORS: {combination_name}\n"
                        )
                        summary_file.write(
                            f"Hyperparameters: {OrderedDict(sorted(current_hp_combo.items()))}\n" # Print sorted HPs
                        )
                        summary_file.write(f"{'#'*60}\n\n")
                        print_personalization_summary(
                            personalization_results, output_file=summary_file
                        )

                    # Print per-patient summary to console too for monitoring (using logging)
                    print_personalization_summary(
                        personalization_results, output_file=None
                    )

                    # Calculate and Write Average Personalized Model Performance for this combination
                    metrics_after_list = {
                        "accuracy": [],
                        "precision": [],
                        "recall": [],
                        "f1_score": [],
                        "auc_roc": [],
                        "sensitivity": [], # Added sensitivity
                        "specificity": [], # Added specificity
                    }
                    count_valid_patients_pers = 0

                    for patient_id, results in personalization_results.items():
                        # Check if the 'after' personalization metrics are valid for this patient
                        if isinstance(
                            results.get("after"), dict
                        ) and "accuracy" in results.get("after", {}):
                            # Check if the test set for this patient had data (via CM)
                            cm_after = results["after"].get(
                                "confusion_matrix", [[0, 0], [0, 0]]
                            )
                            if (
                                isinstance(cm_after, list)
                                and len(cm_after) == 2
                                and len(cm_after[0]) == 2
                                and sum(sum(row) for row in cm_after) > 0
                            ):
                                count_valid_patients_pers += 1
                                metrics_after_list["accuracy"].append(
                                    results["after"]["accuracy"]
                                )
                                metrics_after_list["precision"].append(
                                    results["after"]["precision"]
                                )
                                metrics_after_list["recall"].append(
                                    results["after"]["recall"]
                                )
                                metrics_after_list["f1_score"].append(
                                    results["after"]["f1_score"]
                                )
                                metrics_after_list["auc_roc"].append(
                                    results["after"]["auc_roc"]
                                )
                                # Calculate and append Sensitivity and Specificity if CM is valid
                                if len(cm_after) == 2 and len(cm_after[0]) == 2:
                                    tn, fp, fn, tp = cm_after[0][0], cm_after[0][1], cm_after[1][0], cm_after[1][1]
                                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                                    metrics_after_list["sensitivity"].append(sensitivity)
                                    metrics_after_list["specificity"].append(specificity)
                                else:
                                    metrics_after_list["sensitivity"].append(0.0)
                                    metrics_after_list["specificity"].append(0.0)

                            # else: logging.info(f"--- Debug: Patient {patient_id} CM check failed for averaging.")
                        # else: logging.info(f"--- Debug: Patient {patient_id} 'after' metrics missing or invalid for averaging.")

                    with open(
                        summary_output_filename, "a"
                    ) as summary_file:  # Use 'a' for append mode
                        summary_file.write(
                            "\n--- Personalized Model Performance (Average Across Patients) ---\n"
                        )
                        if count_valid_patients_pers > 0:
                            avg_metrics = {
                                metric: np.mean(metrics_after_list[metric])
                                for metric in metrics_after_list
                            }
                            all_results[current_hp_combo_str][current_model_type][
                                combination_name
                            ]["personalization"][
                                "avg_personalized_metrics"
                            ] = avg_metrics  # Store for final comparison

                            summary_file.write(
                                f"Average Accuracy={avg_metrics['accuracy']:.4f} (across {count_valid_patients_pers} patients with valid evaluation data)\n"
                            )
                            summary_file.write(
                                f"Average Precision={avg_metrics['precision']:.4f}\n"
                            )
                            summary_file.write(
                                f"Average Recall={avg_metrics['recall']:.4f}\n"
                            )
                            summary_file.write(
                                f"Average F1 Score={avg_metrics['f1_score']:.4f}\n"
                            )
                            summary_file.write(
                                f"Average AUC-ROC={avg_metrics['auc_roc']:.4f}\n"
                            )
                            summary_file.write(
                                f"Average Sensitivity={avg_metrics['sensitivity']:.4f}\n"
                            )
                            summary_file.write(
                                f"Average Specificity={avg_metrics['specificity']:.4f}\n"
                            )

                        else:
                            summary_file.write(
                                "No valid personalized patient results to average.\n"
                            )
                            all_results[current_hp_combo_str][current_model_type][
                                combination_name
                            ]["personalization"][
                                "avg_personalized_metrics"
                            ] = None  # Store None

                        summary_file.write("\n")  # Add space

                else:
                    logging.info(
                        f"Personalization (Phase 2) is disabled. Skipping for {current_model_type} + {combination_name} with HP: {current_hp_combo_str}."
                    )
                    all_results[current_hp_combo_str][current_model_type][
                        combination_name
                    ][
                        "personalization"
                    ] = None  # Indicate personalization was skipped


    # --- Final Summary Table for all runs ---
    logging.info(f"\n\n{'='*80}\n")
    logging.info("GENERATING FINAL SUMMARY TABLE...")
    logging.info(f"{'='*80}\n")

    try:
        with open(summary_output_filename, "w") as summary_file: # Use 'w' to overwrite for the final table
            summary_file.write(f"Experiment Summary - {timestamp_str}\n")
            summary_file.write(f"Total execution time: {time.time() - start_time_overall:.2f} seconds\n\n")

            # Write Feature Flags
            summary_file.write("--- Feature Flags ---\n")
            summary_file.write(f"RUN_ALL_MODEL_TYPES: {RUN_ALL_MODEL_TYPES}\n")
            summary_file.write(f"ENABLE_ADAPTIVE_SENSORS: {ENABLE_ADAPTIVE_SENSORS}\n")
            summary_file.write(f"ENABLE_TUNABLE_HYPERPARAMETERS: {ENABLE_TUNABLE_HYPERPARAMETERS}\n")
            summary_file.write(f"ENABLE_PERSONALIZATION: {ENABLE_PERSONALIZATION}\n")
            summary_file.write(f"REUSE_PROCESSED_DATA: {REUSE_PROCESSED_DATA}\n")
            summary_file.write("\n")

            # Write fixed data processing config if REUSE_PROCESSED_DATA is True
            if REUSE_PROCESSED_DATA:
                summary_file.write("--- Cached Data Processing Configuration ---\n")
                summary_file.write(f"  Pre-ictal Window: {FIXED_DATA_PROCESSING_CONFIG['pre_ictal_window_min']} min\n")
                summary_file.write(f"  Pre-ictal Exclusion Buffer: {FIXED_DATA_PROCESSING_CONFIG['pre_ictal_exclusion_buffer_min']} min\n")
                summary_file.write(f"  Post-ictal Buffer: {FIXED_DATA_PROCESSING_CONFIG['post_ictal_buffer_min']} min\n")
                summary_file.write(f"  Sampling Frequency: {FIXED_DATA_PROCESSING_CONFIG['sampling_freq_hz']} Hz\n")
                # Include the number of patients in the cached data path log
                summary_file.write(f"  Cached Data Path (for {num_patients_in_run} patients): {PROCESSED_DATA_CACHE_PATH}\n")
                summary_file.write("\n")

            # Write TUNABLE Hyperparameters
            summary_file.write("--- Tunable Hyperparameters Settings ---\n")
            # Exclude data processing HPs from this list if REUSE_PROCESSED_DATA is True,
            # as they are listed in the Cached Data Processing Configuration section.
            tunable_hp_for_summary = {
                "TUNABLE_CONV_FILTERS": TUNABLE_CONV_FILTERS,
                "TUNABLE_CONV_KERNEL_SIZE": TUNABLE_CONV_KERNEL_SIZE,
                "TUNABLE_POOL_SIZE": TUNABLE_POOL_SIZE,
                "TUNABLE_LSTM_UNITS": TUNABLE_LSTM_UNITS,
                "TUNABLE_DENSE_UNITS": TUNABLE_DENSE_UNITS,
                "TUNABLE_GENERAL_MODEL_EPOCHS": TUNABLE_GENERAL_MODEL_EPOCHS,
                "TUNABLE_PERSONALIZATION_EPOCHS": TUNABLE_PERSONALIZATION_EPOCHS,
                "TUNABLE_GENERAL_MODEL_LR": TUNABLE_GENERAL_MODEL_LR,
                "TUNABLE_PERSONALIZATION_LR": TUNABLE_PERSONALIZATION_LR,
                "TUNABLE_BATCH_SIZE": TUNABLE_BATCH_SIZE,
                "TUNABLE_PERSONALIZATION_BATCH_SIZE": TUNABLE_PERSONALIZATION_BATCH_SIZE,
                "TUNABLE_DROPOUT_RATE": TUNABLE_DROPOUT_RATE,
                "TUNABLE_WEIGHT_DECAY_GENERAL": TUNABLE_WEIGHT_DECAY_GENERAL,
                "TUNABLE_WEIGHT_DECAY_PERSONALIZATION": TUNABLE_WEIGHT_DECAY_PERSONALIZATION,
            }
            if not REUSE_PROCESSED_DATA:
                 # Include data processing HPs in this list if they were actually tuned
                 tunable_hp_for_summary.update({
                    "TUNABLE_PRE_ICTAL_WINDOW_MINUTES": TUNABLE_PRE_ICTAL_WINDOW_MINUTES,
                    "TUNABLE_PRE_ICTAL_EXCLUSION_BUFFER_MINUTES": TUNABLE_PRE_ICTAL_EXCLUSION_BUFFER_MINUTES,
                    "TUNABLE_POST_ICTAL_BUFFER_MINUTES": TUNABLE_POST_ICTAL_BUFFER_MINUTES,
                    "TUNABLE_SAMPLING_FREQ_HZ": TUNABLE_SAMPLING_FREQ_HZ,
                 })

            for param_name, values in tunable_hp_for_summary.items():
                summary_file.write(f"  {param_name}: {values}\n")
            summary_file.write("\n")

            # Write Model Types and Sensor Combinations
            summary_file.write(f"MODEL_TYPES_TO_RUN: {MODEL_TYPES_TO_RUN}\n")
            if ENABLE_ADAPTIVE_SENSORS:
                summary_file.write(f"ALL_SENSOR_COMBINATIONS ({len(ALL_SENSOR_COMBINATIONS)} total):\n")
                for combo in ALL_SENSOR_COMBINATIONS:
                    summary_file.write(f"  - {'+'.join(combo)}\n")
            else:
                summary_file.write(f"BASE_SENSORS: {BASE_SENSORS}\n")
            summary_file.write("\n")


            # --- Results Table Header ---
            summary_file.write("--- Results per Hyperparameter Combination, Model Type, and Sensor Combination ---\n")
            summary_file.write(
                "  HP Combo | Patients | Overall General Model (Validation)                      | Overall General Model (Test)                      | Average Personalized Model (Test)                 | Avg Personalization Change\n"
            )
            summary_file.write(
                "  Idx      | Suitable | Acc  | Prec | Rec  | F1   | AUC  | Sens | Spec | Acc  | Prec | Rec  | F1   | AUC  | Sens | Spec | Acc  | Prec | Rec  | F1   | AUC  | Sens | Spec | Acc  | Prec | Rec  | F1   | AUC  | Sens | Spec\n"
            )
            summary_file.write(
                "  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n"
            )

            # Iterate through HP combinations (sorted by index for consistent output)
            for hp_combo_str in sorted(all_results.keys(), key=lambda x: int(x.split('_')[2])):
                hp_results = all_results[hp_combo_str]
                hp_combo_idx = int(hp_combo_str.split('_')[2]) + 1 # Extract 1-based index

                # Iterate through model types
                for model_type in sorted(hp_results.keys()):
                    model_results = hp_results[model_type]

                    # Iterate through sensor combinations (sorted alphabetically)
                    for combo_name in sorted(model_results.keys()):
                        combo_results = model_results[combo_name]
                        num_suitable_patients = combo_results.get(
                            "num_suitable_patients", 0
                        )

                        # Overall General Metrics for this combo
                        overall_general_val_metrics = (
                            combo_results.get("overall_general", {})
                            .get("metrics", {})
                            .get("val", {})
                        )
                        overall_general_test_metrics = (
                            combo_results.get("overall_general", {})
                            .get("metrics", {})
                            .get("test", {})
                        )

                        overall_general_val_metrics_str = (
                            f"{overall_general_val_metrics.get('accuracy', 0.0):.2f} | {overall_general_val_metrics.get('precision', 0.0):.2f} | "
                            f"{overall_general_val_metrics.get('recall', 0.0):.2f} | {overall_general_val_metrics.get('f1_score', 0.0):.2f} | "
                            f"{overall_general_val_metrics.get('auc_roc', 0.0):.2f} | {overall_general_val_metrics.get('sensitivity', 0.0):.2f} | "
                            f"{overall_general_val_metrics.get('specificity', 0.0):.2f}"
                        )
                        overall_general_test_metrics_str = (
                            f"{overall_general_test_metrics.get('accuracy', 0.0):.2f} | {overall_general_test_metrics.get('precision', 0.0):.2f} | "
                            f"{overall_general_test_metrics.get('recall', 0.0):.2f} | {overall_general_test_metrics.get('f1_score', 0.0):.2f} | "
                            f"{overall_general_test_metrics.get('auc_roc', 0.0):.2f} | {overall_general_test_metrics.get('sensitivity', 0.0):.2f} | "
                            f"{overall_general_test_metrics.get('specificity', 0.0):.2f}"
                        )


                        # Personalized Metrics for this combo (average)
                        personalization_data = combo_results.get(
                            "personalization", None
                        )
                        if personalization_data is not None:
                            avg_personalized_metrics = personalization_data.get(
                                "avg_personalized_metrics", None
                            )
                            if avg_personalized_metrics:
                                avg_personalized_metrics_str = (
                                    f"{avg_personalized_metrics.get('accuracy', 0.0):.2f} | {avg_personalized_metrics.get('precision', 0.0):.2f} | "
                                    f"{avg_personalized_metrics.get('recall', 0.0):.2f} | {avg_personalized_metrics.get('f1_score', 0.0):.2f} | "
                                    f"{avg_personalized_metrics.get('auc_roc', 0.0):.2f} | {avg_personalized_metrics.get('sensitivity', 0.0):.2f} | "
                                    f"{avg_personalized_metrics.get('specificity', 0.0):.2f}"
                                )

                                # Calculate overall average change for this combo across patients
                                total_change_combo = 0
                                count_valid_patients_combo_change = 0
                                for patient_id, pers_results in personalization_data.get(
                                    "personalization_results", {}
                                ).items():
                                    if isinstance(
                                        pers_results.get("before"), dict
                                    ) and isinstance(pers_results.get("after"), dict):
                                        cm_after = pers_results["after"].get(
                                            "confusion_matrix", [[0, 0], [0, 0]]
                                        )
                                        if (
                                            isinstance(cm_after, list)
                                            and len(cm_after) == 2
                                            and len(cm_after[0]) == 2
                                            and sum(sum(row) for row in cm_after) > 0
                                        ):
                                            acc_before = pers_results["before"].get(
                                                "accuracy", 0.0
                                            )
                                            acc_after = pers_results["after"].get(
                                                "accuracy", 0.0
                                            )
                                            total_change_combo += acc_after - acc_before
                                            count_valid_patients_combo_change += 1
                                avg_change_combo = (
                                    total_change_combo / count_valid_patients_combo_change
                                    if count_valid_patients_combo_change > 0
                                    else 0.0
                                )
                                avg_change_combo_str = f"{avg_change_combo:.4f}"
                            else:
                                avg_personalized_metrics_str = "Personalization Failed (N/A)                              " # To match column width
                                avg_change_combo_str = "N/A"
                        else:
                            avg_personalized_metrics_str = "Personalization Disabled (N/A)                                    "
                            avg_change_combo_str = "N/A"

                        # --- MODIFIED FILE WRITE LINE ---
                        summary_file.write(
                            f"  {combo_name:<10} | {num_suitable_patients:<8} | {overall_general_val_metrics_str:<85} | {overall_general_test_metrics_str:<85} | {avg_personalized_metrics_str:<85} | {avg_change_combo_str}\n"
                        )

                    summary_file.write(
                        "  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n\n"
                    ) # Increased length for separator
    except Exception as e:
        logging.error(f"An error occurred while writing the final summary file: {e}")

    logging.info("--- All Runs Complete ---")
    logging.info(f"Results saved in the '{OUTPUT_DIR}' directory.")
    logging.info(f"Log file: {log_filename}")
    logging.info(f"Summary file: {summary_output_filename}")
