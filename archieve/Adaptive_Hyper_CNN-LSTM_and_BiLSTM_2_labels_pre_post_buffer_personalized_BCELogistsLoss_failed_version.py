import concurrent.futures  # Import for parallel processing
import gc
import glob
import itertools  # For combinations
import logging  # Import logging module
import math
import os
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
OUTPUT_DIR = "processed_data_pytorch_adaptive_pre_post_buffer_lovo_personalization_v2"

# Ensure the base output directory exists early
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Feature Flags ---
# Set to True to run all model types in MODEL_TYPES_TO_RUN; False to run only the first one
RUN_ALL_MODEL_TYPES = True
# Set to True to run all sensor combinations; False to run only the full BASE_SENSORS set
ENABLE_ADAPTIVE_SENSORS = False
# Set to True to iterate through all combinations of TUNABLE_ hyperparameters; False to use only the first value from each list
ENABLE_TUNABLE_HYPERPARAMETERS = False
# Set to True to run Phase 2 (Personalization/LOPO); False to only run Phase 1 (Overall General Model)
ENABLE_PERSONALIZATION = False


# --- Data Processing Parameters ---
SEGMENT_DURATION_SECONDS = 30

# Tunable Buffer and Data Parameters
TUNABLE_PRE_ICTAL_WINDOW_MINUTES = [
    30
]  # Time window before seizure onset considered pre-ictal
TUNABLE_PRE_ICTAL_EXCLUSION_BUFFER_MINUTES = [
    120
]  # (Originally 180) Buffer time *before* seizure onset to exclude for clean interictal. Must be >= the corresponding TUNABLE_PRE_ICTAL_WINDOW_MINUTES.
TUNABLE_POST_ICTAL_BUFFER_MINUTES = [
    180
]  # Buffer time *after* seizure end to exclude for clean interictal.

# Define base sensors (ensure these are the possible columns after sync/scaling)
BASE_SENSORS = ['HR', 'EDA', 'TEMP', 'ACC'] # Exclude BVP as requested

# Target sampling frequency for resampling
TUNABLE_SAMPLING_FREQ_HZ = [1]  # Made tunable


# --- Tunable Hyperparameters ---
# These lists define the values to iterate through if ENABLE_TUNABLE_HYPERPARAMETERS is True.
# If ENABLE_TUNABLE_HYPERPARAMETERS is False, only the first value from each list is used.
TUNABLE_CONV_FILTERS = [
    [64, 128, 256],
    [32, 64, 128],
]  # Example: Added another filter combination
TUNABLE_CONV_KERNEL_SIZE = [10, 5]  # Example: Added another kernel size
TUNABLE_POOL_SIZE = [2, 3]  # Example: Added another pool size
TUNABLE_LSTM_UNITS = [128, 64]  # Example: Added another LSTM unit size
TUNABLE_DENSE_UNITS = [64, 32]  # Example: Added another Dense unit size

TUNABLE_GENERAL_MODEL_EPOCHS = [
    50
]  # Used for Overall General Model and LOPO General Model
TUNABLE_PERSONALIZATION_EPOCHS = [30]  # Used for fine-tuning
TUNABLE_GENERAL_MODEL_LR = [
    0.001
]  # Used for Overall General Model and LOPO General Model
TUNABLE_PERSONALIZATION_LR = [0.0001]  # Used for fine-tuning
TUNABLE_BATCH_SIZE = [
    32
]  # Used for Overall General Model and LOPO General Model Train/Val/Test
TUNABLE_PERSONALIZATION_BATCH_SIZE = [16]  # Used for personalization Train/Val

# --- Model Types to Run ---
MODEL_TYPES_TO_RUN = ["CNN-LSTM", "CNN-BiLSTM"]  # Example: ['CNN-LSTM', 'CNN-BiLSTM']

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

    for (
        sensor_name
    ) in sensors:  # sensor_name will be from BASE_SENSORS (e.g., 'ACC', 'EDA')
        if sensor_name not in sensor_mapping:
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
                f"No Parquet files found for Patient {patient_id}, Attribute {attr_folder} with pattern {patient_id}_Empatica-{attr_folder}_{attr_name_part}_segment_*.parquet"
            )  # Uncommented warning
            continue

        all_dfs = []
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
                else:
                    logging.info(
                        f"Warning: Parquet file {file_path} does not have expected 'time' and 'data' columns. Skipping."
                    )  # Uncommented warning
                    pass  # Skip file if columns are wrong

            except Exception as e:
                logging.error(f"Error reading Parquet file {file_path}: {e}")

        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df = combined_df.sort_values(by="timestamp").reset_index(drop=True)
            # Store with the BASE_SENSORS name (uppercase) as the key
            attribute_data[sensor_name] = combined_df

    logging.info(
        f"Loaded data for patient {patient_id}. Found sensors: {list(attribute_data.keys())}"
    )  # Uncommented print
    return attribute_data


def load_seizure_annotations(patient_dir):
    """
    Loads and processes the SeerAnnotations CSV for a given patient.
    Converts start_time and end_time to UTC datetime objects.
    """
    annotation_file = os.path.join(
        patient_dir, f"{os.path.basename(patient_dir)}_SeerAnnotations.csv"
    )
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
            return annotations_df[["start_time", "end_time"]]
        else:
            logging.error(
                f"Error: Annotation file {annotation_file} does not have expected 'start_time' and 'end_time' columns."
            )
            return pd.DataFrame(
                columns=["start_time", "end_time"]
            )  # Return empty df on column error
    except FileNotFoundError:
        return pd.DataFrame(
            columns=["start_time", "end_time"]
        )  # Return empty df if not found
    except Exception as e:
        logging.error(f"Error reading annotation file {annotation_file}: {e}")
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
    if not sensor_data_dict:
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
        rule = f"{1/target_freq_hz}S"
        try:
            # Resample, keeping the sensor_name as the column name
            resampled_df = df[[sensor_name]].asfreq(
                rule
            )  # Explicitly select the column
            resampled_data[sensor_name] = resampled_df
        except Exception as e:
            logging.error(
                f"Error during resampling sensor {sensor_name} to {target_freq_hz}Hz: {e}. Skipping sensor."
            )
            continue

    merged_df = None
    # Iterate through resampled_data (only sensors that were successfully resampled)
    for sensor_name, df in resampled_data.items():
        if merged_df is None:
            merged_df = df
        else:
            # Use 'outer' join for maximum time coverage across all included sensors
            merged_df = merged_df.join(df, how="outer")

    if merged_df is None or merged_df.empty:
        return None

    merged_df = merged_df.sort_index()

    # Interpolate missing values
    merged_df = merged_df.interpolate(method="time")
    merged_df = merged_df.fillna(method="ffill")
    merged_df = merged_df.fillna(method="bfill")

    # Drop columns that are still all NaN after interpolation/fill (happens if a sensor had no data across the entire time range)
    merged_df = merged_df.dropna(axis=1, how="all")

    if (
        merged_df.empty or len(merged_df.columns) == 0
    ):  # Check again if it's empty or has no columns
        return None  # Return None if no usable data

    # Apply Robust Scaling to all remaining data columns
    scaler = RobustScaler()
    data_cols = merged_df.columns  # Use all remaining columns

    try:
        merged_df[data_cols] = scaler.fit_transform(merged_df[data_cols])
    except Exception as e:
        logging.error(f"Error during scaling: {e}. Skipping scaling.")
        # Decide if you want to return None or unscaled data - returning None is safer if scaling is critical
        return None

    # --- Ensure consistent column order and presence of all BASE_SENSORS columns ---
    # Create a reindex list with all BASE_SENSORS
    # Reindex the merged_df to include all BASE_SENSORS columns, filling missing with NaN
    # This ensures the feature dimension is consistent across all patients, even if a sensor was missing entirely
    merged_df = merged_df.reindex(columns=BASE_SENSORS)

    # Fill NaNs introduced by reindexing (for sensors missing entirely) with 0 or another value if appropriate
    # Filling with 0 implies the signal is absent/zeroed when not recorded - review if this is appropriate
    merged_df = merged_df.fillna(0.0)  # Fill NaNs introduced by reindexing with 0

    if merged_df.empty or len(merged_df.columns) != len(
        BASE_SENSORS
    ):  # Final check on shape
        logging.warning(
            "Warning: Merged DataFrame is empty or has incorrect number of columns after reindexing/filling."
        )  # Uncommented warning
        return None  # Return None if final shape is wrong

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
        return np.array([]).reshape(0, segment_length_steps, num_features), np.array([])

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
    logging.info("Defining ictal windows...")  # Uncommented print
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
    logging.info(
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
                logging.info(
                    f" Seizure {i+1}: Pre-ictal window added: {pre_ictal_start} to {pre_ictal_end}"
                )  # Uncommented print
            else:
                logging.info(
                    f" Seizure {i+1}: Calculated pre-ictal window overlaps with seizure window. Skipped."
                )  # Uncommented print
        else:
            logging.info(
                f" Seizure {i+1}: Calculated pre-ictal window or capped window is invalid. Skipped."
            )  # Uncommented print

    # 3. Interictal Exclusion Windows (Asymmetrical Buffer)
    # These define areas NOT suitable for clean interictal samples
    interictal_exclusion_windows = []
    buffer_before_timedelta = pd.Timedelta(minutes=pre_ictal_exclusion_buffer_min)
    buffer_after_timedelta = pd.Timedelta(minutes=post_ictal_buffer_min)
    logging.info(
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
        return np.array([]).reshape(
            0, max(1, int(segment_duration_sec * target_freq_hz))
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
    )  # Uncommented print
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
    # logging.info(f"Processing data for patient: {patient_id}") # Moved to main loop desc

    # 1. Load sensor data - Try to load all base sensors
    # load_sensor_data_for_patient now expects BASE_SENSORS and returns dict with BASE_SENSORS names as keys
    sensor_data_dict = load_sensor_data_for_patient(patient_folder, BASE_SENSORS)
    # Get the list of sensor names for which data was actually found and loaded into the dict
    found_sensors = list(sensor_data_dict.keys())

    if not sensor_data_dict:
        # logging.warning(f"Skipping patient {patient_id}: Could not load any sensor data from BASE_SENSORS.") # Moved to main loop desc
        # Return None, indicating failure to load data for any sensor
        return None

    # 2. Load annotations
    annotations_df = load_seizure_annotations(patient_folder)
    # Note: annotations_df can be empty if no seizures, handled in segment creation

    # 3. Synchronize and merge data
    # synced_df will have columns for all BASE_SENSORS, with 0s for missing ones, in sorted order
    synced_df = synchronize_and_merge_data(
        sensor_data_dict, target_freq_hz=current_sampling_freq_hz
    )
    # Ensure synced_df has the correct number of columns (all BASE_SENSORS)
    if (
        synced_df is None
        or synced_df.empty
        or len(synced_df.columns) != len(BASE_SENSORS)
    ):
        # logging.warning(f"Skipping patient {patient_id}: Could not synchronize, merge, or lost columns.") # Moved to main loop desc
        return None

    # 4. Create labeled segments

    segments, labels = create_labeled_segments(
        synced_df,
        annotations_df,
        segment_duration_sec=SEGMENT_DURATION_SECONDS,
        pre_ictal_window_min=current_pre_ictal_window_min,  # Use the current HP value
        pre_ictal_exclusion_buffer_min=current_pre_ictal_exclusion_buffer_min,  # Use the current HP value
        post_ictal_buffer_min=current_post_ictal_buffer_min,  # Use the current HP value
        target_freq_hz=current_sampling_freq_hz,  # Use the current HP value
    )

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
        # logging.warning(f"Skipping patient {patient_id}: No valid segments created or incorrect shape after segmentation.") # Moved to main loop desc
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
    ):
        super(CNN_LSTM, self).__init__()
        self.input_channels = input_channels
        self.seq_len = seq_len
        self.conv_filters = conv_filters  # Now taken as argument
        self.conv_kernel_size = conv_kernel_size  # Now taken as argument
        self.pool_size = pool_size  # Now taken as argument
        self.lstm_units = lstm_units  # Now taken as argument
        self.dense_units = dense_units  # Now taken as argument

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
            conv_layers_list.append(nn.ReLU())
            conv_layers_list.append(nn.MaxPool1d(pool_size))
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

        self.dense_layers = nn.Sequential(
            nn.Linear(
                lstm_units, dense_units
            ),  # Use lstm_units and dense_units arguments
            nn.Sigmoid(),
            nn.Linear(dense_units, 1),  # Use dense_units argument
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
    ):
        super(CNN_BiLSTM, self).__init__()
        self.input_channels = input_channels
        self.seq_len = seq_len
        self.conv_filters = conv_filters  # Now taken as argument
        self.conv_kernel_size = conv_kernel_size  # Now taken as argument
        self.pool_size = pool_size  # Now taken as argument
        self.lstm_units = lstm_units  # Now taken as argument
        self.dense_units = dense_units  # Now taken as argument

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
            conv_layers_list.append(nn.ReLU())
            conv_layers_list.append(nn.MaxPool1d(pool_size))
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

        self.dense_layers = nn.Sequential(
            nn.Linear(
                lstm_units * 2, dense_units
            ),  # Use lstm_units and dense_units arguments, input size is doubled
            nn.Sigmoid(),
            nn.Linear(dense_units, 1),  # Use dense_units argument
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


def train_one_epoch(model, train_dataloader, criterion, optimizer, device, class_weights=None):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    # Define criterion - Use BCEWithLogitsLoss as models now output logits
    criterion = nn.BCEWithLogitsLoss() # <-- CHANGE THIS LINE

    # Move class weights to device and configure criterion if they exist
    if class_weights is not None:
        # Assuming class_weights_dict is {0: weight_0, 1: weight_1}
        if 0 in class_weights and 1 in class_weights and class_weights[0] > 0:
            pos_weight = torch.tensor(class_weights[1] / class_weights[0], dtype=torch.float32).to(device)
            # Use BCEWithLogitsLoss with pos_weight
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # <-- This is already BCEWithLogitsLoss


    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs) # Model now outputs logits

        # The criterion is already set to BCEWithLogitsLoss (potentially with pos_weight)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    return running_loss / total_samples

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

    # Define criterion for evaluation - Use BCEWithLogitsLoss
    eval_criterion = nn.BCEWithLogitsLoss() # <-- CHANGE THIS LINE

    with torch.no_grad():
        for inputs, labels in dataloader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs) # Model now outputs logits

            # Calculate loss using BCEWithLogitsLoss
            loss = eval_criterion(outputs, labels) # <-- Use the new eval_criterion
            running_loss += loss.item() * inputs.size(0)

            # --- Apply Sigmoid to logits to get probabilities for metrics and prediction ---
            probabilities = torch.sigmoid(outputs) # <-- ADD THIS LINE

            # Predictions based on probability threshold (0.5)
            predicted = (probabilities > 0.5).float() # <-- Use probabilities for thresholding

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy()) # <-- Store probabilities (from sigmoid) for AUC

            dataloader_tqdm.set_postfix(loss=loss.item())


    epoch_loss = running_loss / len(dataloader.dataset)

    # Calculate detailed metrics using the helper function
    metrics = calculate_metrics(all_labels, all_predictions, all_probs)

    # Return metrics including probabilities and labels
    return {'loss': epoch_loss, **metrics, 'all_probs': all_probs, 'all_labels': all_labels}

def train_pytorch_model(model, train_dataloader, val_dataloader, test_dataloader, epochs, learning_rate, class_weights=None, save_best_model_path=None, desc="Training", device=torch.device("cpu")):
    """Main training loop with validation, early stopping, LR scheduling, and returns metrics for all sets."""
    # Define initial criterion (will be overridden in train_one_epoch if class_weights are used)
    # This initial definition isn't strictly necessary anymore as train_one_epoch defines it,
    # but keep it consistent if you like. The important part is the one used in evaluation calls.
    criterion = nn.BCEWithLogitsLoss() # <-- CHANGE THIS LINE (can also remove this line if you prefer)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=0.00001)

    # ... (rest of the training loop) ...

    # Evaluate on train, val, and test sets after training
    # Use BCEWithLogitsLoss for evaluation
    eval_criterion = nn.BCEWithLogitsLoss() # <-- CHANGE THIS LINE

    train_metrics = evaluate_pytorch_model(model, train_dataloader, eval_criterion, device)
    val_metrics = evaluate_pytorch_model(model, val_dataloader, eval_criterion, device)
    test_metrics = evaluate_pytorch_model(model, test_dataloader, eval_criterion, device)

    return model, {'train': train_metrics, 'val': val_metrics, 'test': test_metrics}
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

    # Only attempt fine-tuning if there's training data for this patient
    if len(train_dataset_pat) > 0:
        # logging.info(f"Starting fine-tuning for patient {current_patient_id}...") # Removed direct print

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
                desc=f"Submitting patient tasks ({model_type}, {combo_name}, {current_hp_combo_str})",
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
                desc=f"Collecting patient results ({model_type}, {combo_name}, {current_hp_combo_str})",
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
                            f"Personalization failed or skipped for patient {patient_id} in a parallel process ({model_type}, {combo_name}, {current_hp_combo_str})."
                        )  # Changed print to logging.warning

                except Exception as exc:
                    logging.error(
                        f"A patient processing generated an exception: {exc} ({model_type}, {combo_name}, {current_hp_combo_str})"
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
        
        
# --- Main Execution ---
if __name__ == "__main__":
    # Ensure the base output directory exists at the very beginning
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Configure Logging ---
    # Use a timestamp for a unique filename for the entire run
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(
        OUTPUT_DIR, f"seizure_prediction_results_{timestamp_str}.log"
    )
    summary_output_filename = os.path.join(
        OUTPUT_DIR, f"seizure_prediction_summary_{timestamp_str}.txt"
    )

    # Configure the root logger
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
        )  # Changed print to logging.error
        sys.exit()  # Use sys.exit to exit the script cleanly

    # Apply the patient limit if set
    if (
        MAX_PATIENTS_TO_INCLUDE is not None
        and isinstance(MAX_PATIENTS_TO_INCLUDE, int)
        and MAX_PATIENTS_TO_INCLUDE > 0
    ):
        all_patient_folders = all_patient_folders[:MAX_PATIENTS_TO_INCLUDE]
        logging.info(
            f"Limiting run to the first {MAX_PATIENTS_TO_INCLUDE} patient directories."
        )  # Changed print to logging.info

    logging.info(
        f"Including {len(all_patient_folders)} patient directories in this run."
    )  # Changed print to logging.info

    # --- Prepare Hyperparameter Combinations ---
    # Create a list of dictionaries, each representing one combination of tunable HPs
    hp_combinations = []
    if ENABLE_TUNABLE_HYPERPARAMETERS:
        # Get lists of values for each tunable HP
        hp_param_lists = {
            "sampling_freq_hz": TUNABLE_SAMPLING_FREQ_HZ,
            "pre_ictal_window_min": TUNABLE_PRE_ICTAL_WINDOW_MINUTES,
            "pre_ictal_exclusion_buffer_min": TUNABLE_PRE_ICTAL_EXCLUSION_BUFFER_MINUTES,
            "post_ictal_buffer_min": TUNABLE_POST_ICTAL_BUFFER_MINUTES,
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
        }
        # Generate combinations
        keys, values = zip(*hp_param_lists.items())
        for bundle in itertools.product(*values):
            hp_combinations.append(dict(zip(keys, bundle)))
    else:
        # If tuning is disabled, create one combination using the first value of each list
        hp_combinations.append(
            {
                "sampling_freq_hz": TUNABLE_SAMPLING_FREQ_HZ[0],
                "pre_ictal_window_min": TUNABLE_PRE_ICTAL_WINDOW_MINUTES[0],
                "pre_ictal_exclusion_buffer_min": TUNABLE_PRE_ICTAL_EXCLUSION_BUFFER_MINUTES[
                    0
                ],
                "post_ictal_buffer_min": TUNABLE_POST_ICTAL_BUFFER_MINUTES[0],
                "conv_filters": TUNABLE_CONV_FILTERS[0],
                "conv_kernel_size": TUNABLE_CONV_KERNEL_SIZE[0],
                "pool_size": TUNABLE_POOL_SIZE[0],
                "lstm_units": TUNABLE_LSTM_UNITS[0],
                "dense_units": TUNABLE_DENSE_UNITS[0],
                "general_model_epochs": TUNABLE_GENERAL_MODEL_EPOCHS[0],
                "personalization_epochs": TUNABLE_PERSONALIZATION_EPOCHS[0],
                "general_model_lr": TUNABLE_GENERAL_MODEL_LR[0],
                "personalization_lr": TUNABLE_PERSONALIZATION_LR[0],
                "batch_size": TUNABLE_BATCH_SIZE[0],
                "personalization_batch_size": TUNABLE_PERSONALIZATION_BATCH_SIZE[0],
            }
        )

    logging.info(
        f"Prepared {len(hp_combinations)} hyperparameter combination(s) to test."
    )

    # --- Outer loop for Hyperparameter Combinations ---
    all_results = {}  # Dictionary to store all results for all HP combos

    for hp_idx, current_hp_combo in enumerate(hp_combinations):
        # Create a string representation of the current HP combination for logging and saving
        hp_combo_desc = (
            "_".join(
                [
                    f"{k}-{v}"
                    for k, v in current_hp_combo.items()
                    if k
                    in [
                        "sampling_freq_hz",
                        "pre_ictal_window_min",
                        "conv_filters",
                        "lstm_units",
                        "batch_size",
                    ]
                ]
            )
            .replace("[", "")
            .replace("]", "")
            .replace(", ", "-")
        )
        current_hp_combo_str = f"HP_Combo_{hp_idx}_{hp_combo_desc}"

        logging.info(f"{'='*80}")
        logging.info(
            f"STARTING RUN FOR HYPERPARAMETER COMBINATION {hp_idx+1}/{len(hp_combinations)}"
        )
        logging.info(f"Parameters: {current_hp_combo}")
        logging.info(f"{'='*80}")

        all_results[current_hp_combo_str] = {}  # Store results for this HP combo

        # Extract current HP values for clarity and passing
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
        current_batch_size = current_hp_combo["batch_size"]
        current_personalization_batch_size = current_hp_combo[
            "personalization_batch_size"
        ]

        # Store model and general HPs in dictionaries to pass to functions
        model_hyperparameters = {
            "conv_filters": current_conv_filters,
            "conv_kernel_size": current_conv_kernel_size,
            "pool_size": current_pool_size,
            "lstm_units": current_lstm_units,
            "dense_units": current_dense_units,
            "sampling_freq_hz": current_sampling_freq_hz,  # Include sampling freq here for model init
        }
        general_hyperparameters = {
            "epochs": current_general_model_epochs,
            "learning_rate": current_general_model_lr,
            "batch_size": current_batch_size,
        }
        personalization_hyperparameters = {
            "epochs": current_personalization_epochs,
            "learning_rate": current_personalization_lr,  # Fixed variable name
            "batch_size": current_personalization_batch_size,
        }

        # --- Step 0: Process data for included patients using the CURRENT HP combination ---
        # This needs to be done INSIDE the HP loop because data processing parameters are now tunable.
        processed_patient_data = []
        logging.info(
            f"--- Starting Patient Data Processing for HP Combination: {current_hp_combo_str} ---"
        )

        # Calculate expected shape based on current HP data parameters
        current_expected_seq_len_initial_processing = int(
            SEGMENT_DURATION_SECONDS * current_sampling_freq_hz
        )
        current_expected_seq_len_initial_processing = max(
            1, current_expected_seq_len_initial_processing
        )
        current_expected_num_features_initial_processing = len(BASE_SENSORS)

        for patient_folder in tqdm(
            all_patient_folders,
            desc=f"Data Processing ({current_hp_combo_str})",
            leave=True,
        ):
            patient_data = prepare_patient_data(
                patient_folder,
                current_sampling_freq_hz,
                current_pre_ictal_window_min,
                current_pre_ictal_exclusion_buffer_min,
                current_post_ictal_buffer_min,
            )
            if patient_data:
                patient_id, segments, labels, found_sensors = patient_data
                if (
                    segments.shape[1] == current_expected_seq_len_initial_processing
                    and segments.shape[2]
                    == current_expected_num_features_initial_processing
                ):
                    processed_patient_data.append(patient_data)
                # else: logging.info(f"Skipped patient {patient_id}: Segments shape mismatch after processing with current HPs ({segments.shape} vs expected {(len(segments), current_expected_seq_len_initial_processing, current_expected_num_features_initial_processing)})") # Uncommented print and changed to logging.info
            # else: logging.info(f"Skipped initial processing for patient {os.path.basename(patient_folder)} with current HPs.") # Uncommented print and changed to logging.info

        if not processed_patient_data:
            logging.error(
                f"No valid patient data was processed with HP Combination: {current_hp_combo_str}. Skipping this combination."
            )
            continue  # Move to the next HP combination if no data was processed

        logging.info(
            f"Successfully processed initial data for {len(processed_patient_data)} patients with HP Combination: {current_hp_combo_str}."
        )

        # Store the actual sequence length and feature count determined during initial processing
        if processed_patient_data:
            actual_seq_len_initial = processed_patient_data[0][1].shape[1]
            actual_num_features_all_sensors = processed_patient_data[0][1].shape[2]
            logging.info(
                f"Actual segment shape after processing with current HPs: (N, {actual_seq_len_initial}, {actual_num_features_all_sensors})"
            )
        else:
            actual_seq_len_initial = current_expected_seq_len_initial_processing  # Use calculated expected as fallback
            actual_num_features_all_sensors = current_expected_num_features_initial_processing  # Use calculated expected as fallback
            logging.warning(
                f"Warning: No processed data for shape check, using calculated expected shape as fallback: (N, {actual_seq_len_initial}, {actual_num_features_all_sensors})"
            )

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
                # This is done BEFORE training/personalization
                (
                    patients_suitable_for_combination,
                    sensor_combination_indices,
                ) = get_patients_and_indices_for_combination(
                    processed_patient_data,  # Pass the list of patients processed with CURRENT HPs
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
                for patient_data_tuple in patients_suitable_for_combination:
                    (
                        patient_id,
                        segments_all_sensors,
                        labels,
                        found_sensors,
                    ) = patient_data_tuple
                    # Slice segments to include only the features for the current combination
                    # segments_all_sensors shape is (N, L, len(BASE_SENSORS)) - based on initial processing
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
                    # else: logging.warning(f"Skipping patient {patient_id} data for Overall General training: Shape mismatch or no data after slicing.") # Uncommented print and changed to logging.warning

                overall_general_segments_combined = np.concatenate(
                    overall_general_segments_raw, axis=0
                )
                overall_general_labels_combined = np.concatenate(
                    overall_general_labels_raw, axis=0
                )

                # Check for sufficient data for overall general training for this combination
                # Need at least 3 samples total for 60/20/20 split, and at least one of each class in the total combined data
                if (
                    len(overall_general_segments_combined) < 3
                    or len(np.unique(overall_general_labels_combined)) < 2
                ):
                    logging.warning(
                        f"No data ({len(overall_general_segments_combined)} samples) or only one class ({np.unique(overall_general_labels_combined)}) available for Overall General Model training ({current_model_type}, {combination_name}, {current_hp_combo_str}). Skipping."
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
                            seq_len=seq_len_og,
                            num_features=input_channels_og,
                        )
                        overall_general_val_dataset = SeizureDataset(
                            X_val_og,
                            y_val_og,
                            seq_len=seq_len_og,
                            num_features=input_channels_og,
                        )
                        overall_general_test_dataset = SeizureDataset(
                            X_test_og,
                            y_test_og,
                            seq_len=seq_len_og,
                            num_features=input_channels_og,
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
                        class_weights_og_dict = None
                        if len(y_train_og) > 0:
                            classes_og = np.unique(y_train_og)
                            if len(classes_og) == 2:
                                class_weights_og_np = class_weight.compute_class_weight(
                                    "balanced", classes=classes_og, y=y_train_og
                                )
                                class_weights_og_dict = {
                                    cls: weight
                                    for cls, weight in zip(
                                        classes_og, class_weights_og_np
                                    )
                                }
                                logging.info(
                                    f"Computed Overall General class weights ({current_model_type}, {combination_name}, {current_hp_combo_str}): {class_weights_og_dict}"
                                )
                            # else: logging.warning(f"Warning: Only one class for Overall General training data ({current_model_type}, {combination_name}, {current_hp_combo_str}). No class weights applied.")

                        # Instantiate the Overall General Model with current HPs
                        ModelClass_og = get_model_class(current_model_type)

                        try:
                            overall_general_model = ModelClass_og(
                                input_channels_og,
                                seq_len_og,
                                conv_filters=current_conv_filters,
                                conv_kernel_size=current_conv_kernel_size,
                                pool_size=current_pool_size,
                                lstm_units=current_lstm_units,
                                dense_units=current_dense_units,
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

                        # Only attempt training if dataloader is not empty
                        if (
                            "overall_general_train_dataloader" in locals()
                            and len(overall_general_train_dataloader.dataset) > 0
                        ):
                            logging.info(
                                f"Starting Overall General Model training ({current_model_type}, {combination_name}, {current_hp_combo_str})..."
                            )

                            # Define save path for the Overall General Model
                            overall_general_model_save_dir = os.path.join(
                                OUTPUT_DIR,
                                timestamp_str,
                                # current_hp_combo_str,
                                current_model_type,
                                combination_name,
                            )
                            overall_general_model_save_path = os.path.join(
                                overall_general_model_save_dir,
                                "overall_general_model.pth",
                            )
                            plot_dir_og = os.path.join(OUTPUT_DIR, current_model_type, combination_name, 'plots')
                            # plot_dir_og = os.path.join(OUTPUT_DIR, current_hp_combo_str, current_model_type, combination_name, 'plots')

                            # try:
                            #     os.makedirs(overall_general_model_save_dir, exist_ok=True)
                            #     os.makedirs(plot_dir_og, exist_ok=True)
                            #     logging.info(f"Created output directories for HP: {current_hp_combo_str}, Model: {current_model_type}, Sensors: {combination_name}")
                            # except Exception as e:
                            #     logging.error(f"Error creating output directories for HP: {current_hp_combo_str}, Model: {current_model_type}, Sensors: {combination_name}: {e}. Skipping this run.")
                            #     # Store skip reason and continue to the next combination
                            #     all_results[current_hp_combo_str][current_model_type][combination_name]['status'] = 'Directory Creation Failed'
                            #     continue # Skip this combination run
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
                                class_weights=class_weights_og_dict,
                                save_best_model_path=overall_general_model_save_path,  # Save the overall general model
                                desc=f"Training Overall General ({current_model_type}, {combination_name}, {current_hp_combo_str})",
                                device=DEVICE,
                            )
                            
                            logging.info(
                                f"Overall General Model Training Metrics ({current_model_type}, {combination_name}, {current_hp_combo_str}) - Train: {format_metrics_for_summary(overall_general_metrics['train'])}"
                            )
                            logging.info(
                                f"Overall General Model Validation Metrics ({current_model_type}, {combination_name}, {current_hp_combo_str}) - Val: {format_metrics_for_summary(overall_general_metrics['val'])}"
                            )
                            logging.info(
                                f"Overall General Model Testing Metrics ({current_model_type}, {combination_name}, {current_hp_combo_str}) - Test: {format_metrics_for_summary(overall_general_metrics['test'])}"
                            )

                            # --- Add Plotting for Overall General Model (Test Set) ---
                            overall_general_test_metrics = overall_general_metrics['test']
                            overall_general_test_probs = overall_general_metrics['test'].get('all_probs', []) # Assuming evaluate_pytorch_model returns 'all_probs'
                            overall_general_test_labels = overall_general_metrics['test'].get('all_labels', []) # Assuming evaluate_pytorch_model returns 'all_labels'
                            overall_general_test_cm = overall_general_metrics['test'].get('confusion_matrix', [[0,0],[0,0]])

                            os.makedirs(plot_dir_og, exist_ok=True) # Ensure plot directory exists

                            # AUC-ROC Plot
                            # Need actual probabilities and labels from the test set evaluation
                            # You'll need to modify evaluate_pytorch_model to return all_probs and all_labels
                            # For now, using placeholders assuming they are in the metrics dict
                            # --- IMPORTANT: You need to modify evaluate_pytorch_model to return all_probs and all_labels ---
                            # See step 5 below.
                            if 'all_probs' in overall_general_test_metrics and 'all_labels' in overall_general_test_metrics:
                                plot_auc_roc(
                                    overall_general_test_metrics['all_probs'],
                                    overall_general_test_metrics['all_labels'],
                                    f'Overall General Model AUC-ROC ({current_model_type}, {combination_name})',
                                    os.path.join(plot_dir_og, f'{timestamp_str}_overall_general_auc_roc.png')
                                )
                            else:
                                logging.warning("Skipping Overall General AUC-ROC plot: 'all_probs' or 'all_labels' not found in test metrics.")


                            # Confusion Matrix Plot
                            plot_confusion_matrix(
                                overall_general_test_cm,
                                ['Interictal (0)', 'Pre-ictal (1)'], # Class names
                                f'Overall General Model Confusion Matrix ({current_model_type}, {combination_name})',
                                os.path.join(plot_dir_og, f'{timestamp_str}_overall_general_confusion_matrix.png')
                            )
                            # --- End Plotting for Overall General Model ---

                            overall_general_results_by_combo_model_run = {
                                'metrics': overall_general_metrics,
                                'num_suitable_patients': len(patients_suitable_for_combination)
                            }
                            
                            overall_general_model_state = (
                                overall_general_model.state_dict()
                            )  # Get state dict after training/loading best

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
                            overall_general_model_state = None  # Indicate failure/skip
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
                    # Pass the *full* processed_patient_data list and the sensor indices
                    personalization_results = perform_personalization_pytorch_lopo(
                        processed_patient_data,  # Pass the list of patients processed with CURRENT HPs
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
                            f"Hyperparameters: {current_hp_combo}\n"
                        )  # Add HP combo to file header
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
                            # else: logging.info(f"--- Debug: Patient {patient_id} CM check failed for averaging.") # Uncommented print and changed to logging.info
                        # else: logging.info(f"--- Debug: Patient {patient_id} 'after' metrics missing or invalid for averaging.") # Uncommented print and changed to logging.info

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
        with open(summary_output_filename, "a") as summary_file:
            summary_file.write(f"\n\n{'='*80}\n")
            summary_file.write("FINAL SUMMARY OF ALL RUNS\n")
            summary_file.write(f"{'='*80}\n\n")

            # Iterate through HP combinations
            for hp_combo_str, hp_results in all_results.items():
                summary_file.write(
                    f"\n--- Results for Hyperparameter Combination: {hp_combo_str} ---\n"
                )
                # Iterate through model types
                for model_type, model_results in hp_results.items():
                    summary_file.write(f"\n  Model Type: {model_type}\n")
                    # --- MODIFIED HEADER LINE ---
                    summary_file.write(
                        "  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n"
                    ) # Increased length to accommodate more metrics
                    summary_file.write(
                        "  Sensors    | Patients | Overall General (Val) Metrics                                                                 | Overall General (Test) Metrics                                                                | Avg Personalized (Test) Metrics                                                              | Avg Acc Change\n"
                    ) # Added header for Validation metrics
                    summary_file.write(
                        "  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n"
                    ) # Increased length for separator


                    # Iterate through sensor combinations
                    for combo_name in sorted(model_results.keys()):
                        combo_results = model_results[combo_name]
                        num_suitable_patients = combo_results.get(
                            "num_suitable_patients", 0
                        )

                        # Overall General Metrics for this combo
                        # --- MODIFIED RETRIEVAL TO GET BOTH VAL AND TEST METRICS ---
                        overall_general_val_metrics = ( # <--- Get validation metrics
                            combo_results.get("overall_general", {})
                            .get("metrics", {})
                            .get("val", {})
                        )
                        overall_general_test_metrics = ( # <--- Get test metrics
                            combo_results.get("overall_general", {})
                            .get("metrics", {})
                            .get("test", {})
                        )

                        # --- MODIFIED STRING FORMATTING TO INCLUDE BOTH ---
                        overall_general_val_metrics_str = ( # <--- Format validation metrics
                            format_metrics_for_summary(
                                overall_general_val_metrics, prefix="Val: "
                            )
                            if overall_general_val_metrics
                            else "N/A"
                        )
                        overall_general_test_metrics_str = ( # <--- Format test metrics
                            format_metrics_for_summary(
                                overall_general_test_metrics, prefix="Test: "
                            )
                            if overall_general_test_metrics
                            else "N/A"
                        )


                        # Personalized Metrics for this combo (average)
                        personalization_data = combo_results.get(
                            "personalization", None
                        )
                        if personalization_data is not None:
                            avg_personalized_metrics = personalization_data.get(
                                "avg_personalized_metrics", None
                            )
                            avg_personalized_metrics_str = (
                                format_metrics_for_summary(
                                    avg_personalized_metrics, prefix="Avg Test: "
                                )
                                if avg_personalized_metrics
                                else "N/A"
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
                            avg_personalized_metrics_str = "Personalization Disabled"
                            avg_change_combo_str = "N/A"

                        # --- MODIFIED FILE WRITE LINE ---
                        summary_file.write(
                            f"  {combo_name:<10} | {num_suitable_patients:<8} | {overall_general_val_metrics_str:<85} | {overall_general_test_metrics_str:<85} | {avg_personalized_metrics_str:<85} | {avg_change_combo_str}\n" # <--- Added overall_general_val_metrics_str
                        )

                    summary_file.write(
                        "  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n\n"
                    ) # Increased length for separator

    logging.info("--- All Runs Complete ---")
    logging.info(f"Results saved in the '{OUTPUT_DIR}' directory.")
    logging.info(f"Log file: {log_filename}")
    logging.info(f"Summary file: {summary_output_filename}")