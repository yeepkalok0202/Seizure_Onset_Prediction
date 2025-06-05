import gc
import glob
import itertools  # For combinations
import math
import os
import random
import time

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
DATA_ROOT_DIR = 'F:\\data_9'
OUTPUT_DIR = 'processed_data_pytorch_adaptive_interbuffer_old'
SEGMENT_DURATION_SECONDS = 30
PRE_ICTAL_WINDOW_MINUTES = 30 # Adjusted based on previous debugging
INTERICTAL_BUFFER_MINUTES = 180
BASE_SENSORS = ['ACC', 'EDA', 'HR', 'TEMP'] # Sorted for consistent column order
SAMPLING_FREQ_HZ = 1

# Model Hyperparameters (Placeholders - TUNE THESE)
CONV_FILTERS = [64, 128, 256]
CONV_KERNEL_SIZE = 10
POOL_SIZE = 2
LSTM_UNITS = 128
DENSE_UNITS = 64
GENERAL_MODEL_EPOCHS = 50
PERSONALIZATION_EPOCHS = 30
GENERAL_MODEL_LR = 0.001
PERSONALIZATION_LR = 0.0001
BATCH_SIZE = 32
PERSONALIZATION_BATCH_SIZE = 16

# Feature Flag for Sensor Combinations
ENABLE_SENSOR_COMBINATIONS = True # Set to False to run only the full base sensor set


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Data Loading and Preprocessing ---

def load_sensor_data_for_patient(patient_dir, sensors):
    """
    Loads Parquet data for specified sensors for a given patient,
    concatenates it, sorts by timestamp, and converts to UTC.
    Does NOT apply scaling yet.

    Args:
        patient_dir (str): The directory for the specific patient.
        sensors (list): List of sensor names (e.g., ['HR', 'EDA']).

    Returns:
        dict: A dictionary where keys are attribute names (e.g., 'HR')
              and values are Pandas DataFrames containing the raw data.
              Returns an empty dict if no data is found for any sensor.
    """
    attribute_data = {}
    sensor_mapping = {
        'HR': 'HR', 'EDA': 'EDA', 'TEMP': 'TEMP', 'ACC': 'Acc Mag'
    }

    for sensor_name in sensors:
        if sensor_name not in sensor_mapping:
            print(f"Warning: No mapping found for sensor '{sensor_name}'. Skipping.") # Suppress frequent warning
            continue

        attr_folder = sensor_name
        attr_name_part = sensor_mapping[sensor_name]

        parquet_files = sorted(glob.glob(os.path.join(patient_dir, f'Empatica-{attr_folder}', f'{os.path.basename(patient_dir)}_Empatica-{attr_folder}_{attr_name_part}_segment_*.parquet')))

        if not parquet_files:
            print(f"No Parquet files found for Patient in {patient_dir}, Attribute {attr_folder}") # Suppress frequent warning
            continue

        all_dfs = []
        for file_path in parquet_files:
            try:
                df = pd.read_parquet(file_path)
                if 'time' in df.columns and 'data' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['time'] / 1000, unit='s', utc=True)
                    df = df.rename(columns={'data': sensor_name.lower()})
                    df = df[['timestamp', sensor_name.lower()]]
                    all_dfs.append(df)
                else:
                    print(f"Warning: Parquet file {file_path} does not have expected 'time' and 'data' columns. Skipping.")

            except Exception as e:
                print(f"Error reading Parquet file {file_path}: {e}")

        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df = combined_df.sort_values(by='timestamp').reset_index(drop=True)
            attribute_data[sensor_name.lower()] = combined_df

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
            annotations_df['end_time'] = annotations_df.apply(lambda row: row['end_time'] if row['end_time'] > row['start_time'] else row['start_time'] + pd.Timedelta(seconds=1), axis=1)
            return annotations_df[['start_time', 'end_time']]
        else:
            print(f"Error: Annotation file {annotation_file} does not have expected 'start_time' and 'end_time' columns.")
            return None
    except FileNotFoundError:
        return pd.DataFrame(columns=['start_time', 'end_time'])
    except Exception as e:
        print(f"Error reading annotation file {annotation_file}: {e}")
        return None


def synchronize_and_merge_data(sensor_data_dict, target_freq_hz=1):
    """
    Synchronizes sensor data, merges them, applies Robust Scaling, handles NaNs,
    and ensures columns are in a consistent order (sorted BASE_SENSORS).

    Args:
        sensor_data_dict (dict): Dictionary with sensor names as keys and
                                 DataFrames (with 'timestamp' and data column) as values.
        target_freq_hz (int): The target frequency in Hz for resampling.

    Returns:
        pandas.DataFrame: A single DataFrame with a datetime index containing
                          all synchronized and scaled sensor data. Columns ordered by BASE_SENSORS.
                          Returns None if input is empty or no common time found.
    """
    if not sensor_data_dict:
        return None

    resampled_data = {}
    for sensor_name, df in sensor_data_dict.items():
        df = df.set_index('timestamp').sort_index()
        rule = f'{1/target_freq_hz}S'
        resampled_df = df.asfreq(rule)
        resampled_data[sensor_name] = resampled_df

    merged_df = None
    for sensor_name, df in resampled_data.items():
        if merged_df is None:
            merged_df = df
        else:
            merged_df = merged_df.join(df, how='outer')

    if merged_df is None:
         return None

    merged_df = merged_df.sort_index()

    merged_df = merged_df.interpolate(method='time')
    merged_df = merged_df.fillna(method='ffill')
    merged_df = merged_df.fillna(method='bfill')

    # --- Ensure consistent column order and fill missing sensors with NaN if needed ---
    # Create a reindex list with all BASE_SENSORS, using the lower case names
    base_sensor_cols = [s.lower() for s in BASE_SENSORS]
    # Reindex the merged_df to include all base sensor columns, filling missing with NaN
    merged_df = merged_df.reindex(columns=base_sensor_cols)


    if merged_df.empty or merged_df.isnull().values.all(): # Check if it's empty or all NaNs after reindexing
         print("Warning: Merged DataFrame is empty or all NaNs after reindexing. Skipping scaling.")
         return None # Return None if no usable data

    # Apply Robust Scaling only to columns that are not all NaN
    scaler = RobustScaler()
    valid_cols = merged_df.columns[~merged_df.isnull().all()] # Get columns that are NOT all NaN
    if not valid_cols.empty:
        merged_df[valid_cols] = scaler.fit_transform(merged_df[valid_cols])
    else:
        print("Warning: No valid columns left to scale after reindexing and NaN check.")
        return None # Return None if no columns can be scaled


    return merged_df


def create_labeled_segments(synced_df, annotations_df, segment_duration_sec, pre_ictal_window_min, interictal_buffer_min, target_freq_hz):
    """
    Creates segments from synchronized data and labels them
    as pre-ictal (1) or interictal (0) based on seizure annotations. Samples interictal
    segments to attempt class balance. Synced_df must have columns in BASE_SENSORS order.

    Args:
        synced_df (pandas.DataFrame): DataFrame with synchronized sensor data (datetime index).
                                      Must have columns in order of BASE_SENSORS (with NaNs if missing).
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
    if synced_df is None or annotations_df is None or synced_df.empty or len(synced_df.columns) != len(BASE_SENSORS):
        print("Synced data is empty, has wrong number of columns, or annotations are missing. Cannot create segments.")
        num_features = len(BASE_SENSORS) # Expected number of features is always len(BASE_SENSORS) now
        segment_length_steps = int(segment_duration_sec * target_freq_hz)
        return np.array([]).reshape(0, segment_length_steps, num_features), np.array([])

    segments = []
    labels = []
    segment_length_steps = int(segment_duration_sec * target_freq_hz)
    step_size = segment_length_steps # Use non-overlapping segments

    data_start_time = synced_df.index.min()
    data_end_time = synced_df.index.max()
    # print(f"Data range: {data_start_time} to {data_end_time}") # Suppress repeated print


    # Helper to check overlap (inclusive of boundaries)
    def check_overlap(seg_start, seg_end, windows):
        for win_start, win_end in windows:
            if max(seg_start, win_start) <= min(seg_end, win_end):
                return True
        return False


    # --- Define Time Windows ---

    # 1. Actual Seizure (Ictal) Windows
    seizure_windows = []
    for i, seizure in annotations_df.iterrows():
         seizure_start = seizure['start_time']
         seizure_end = seizure['end_time']
         if seizure_start < seizure_end:
             seizure_windows.append((seizure_start, seizure_end))

    # 2. Pre-ictal Windows
    pre_ictal_windows = []
    # print(f"Defining pre-ictal windows ({pre_ictal_window_min} mins before seizure onset)...") # Suppress repeated print
    for i, seizure in annotations_df.iterrows():
        seizure_start = seizure['start_time']
        pre_ictal_start_uncapped = seizure_start - pd.Timedelta(minutes=pre_ictal_window_min)
        pre_ictal_end = seizure_start - pd.Timedelta(seconds=1)

        # Cap the pre-ictal start at the beginning of the available data
        pre_ictal_start = max(data_start_time, pre_ictal_start_uncapped)

        # Ensure the capped window is valid AND overlaps with the data range
        if pre_ictal_start < pre_ictal_end and max(pre_ictal_start, data_start_time) <= min(pre_ictal_end, data_end_time):
             # Ensure pre-ictal window does *not* overlap with the seizure itself
             if not check_overlap(pre_ictal_start, pre_ictal_end, seizure_windows):
                 pre_ictal_windows.append((pre_ictal_start, pre_ictal_end))
                 # print(f" Seizure {i+1}: Pre-ictal window added: {pre_ictal_start} to {pre_ictal_end}") # Suppress repeated print
             # else: print(f" Seizure {i+1}: Calculated pre-ictal window overlaps with seizure window. Skipped.") # Suppress
        # else: print(f" Seizure {i+1}: Calculated pre-ictal window is invalid or outside data range. Skipped.") # Suppress


    # 3. Interictal Exclusion Windows (Seizure + Buffer)
    interictal_exclusion_windows = []
    buffer_timedelta = pd.Timedelta(minutes=interictal_buffer_min)
    # print(f"Defining interictal exclusion windows ({interictal_buffer_min} mins buffer around seizures)...") # Suppress
    for _, seizure in annotations_df.iterrows():
         exclusion_start = seizure['start_time'] - buffer_timedelta
         exclusion_end = seizure['end_time'] + buffer_timedelta
         interictal_exclusion_windows.append((exclusion_start, exclusion_end))


    # --- Create Segments and Assign Labels ---

    # print(f"Creating segments (len={segment_duration_sec}s, step={segment_duration_sec}s) from {len(synced_df)} total steps...") # Suppress
    segments_skipped_ictal = 0
    segments_skipped_interictal_buffer = 0
    segments_labeled_preictal = 0
    segments_labeled_interictal = 0
    segments_total_candidates = 0

    for i in range(0, len(synced_df) - segment_length_steps + 1, step_size): # No tqdm here, handled by outer patient loop
        segment_df = synced_df.iloc[i : i + segment_length_steps]
        if len(segment_df) != segment_length_steps:
            continue

        segments_total_candidates += 1

        segment_start_time = segment_df.index[0]
        segment_end_time = segment_df.index[-1]

        if check_overlap(segment_start_time, segment_end_time, seizure_windows):
            segments_skipped_ictal += 1
            continue

        if check_overlap(segment_start_time, segment_end_time, pre_ictal_windows):
            segments.append(segment_df.values)
            labels.append(1)
            segments_labeled_preictal += 1
            continue

        if check_overlap(segment_start_time, segment_end_time, interictal_exclusion_windows):
             segments_skipped_interictal_buffer += 1
             continue

        segments.append(segment_df.values)
        labels.append(0)
        segments_labeled_interictal += 1


    segments = np.array(segments)
    labels = np.array(labels)

    # print(f"Finished segmentation.") # Suppress
    # print(f" Segments skipped (ictal): {segments_skipped_ictal}") # Suppress
    # print(f" Segments skipped (interictal buffer, not pre-ictal): {segments_skipped_interictal_buffer}") # Suppress
    # print(f" Total segments included for labeling (Pre-ictal + Interictal): {len(segments)}") # Suppress
    # print(f" Segments labeled Pre-ictal: {np.sum(labels)}, Interictal: {len(labels) - np.sum(labels)}") # Suppress


    # Simple class balancing: Undersample majority class
    pre_ictal_indices = np.where(labels == 1)[0]
    interictal_indices = np.where(labels == 0)[0]

    min_count = min(len(pre_ictal_indices), len(interictal_indices))

    if min_count == 0:
        # print("Warning: One class has zero samples after segmentation. Cannot balance.") # Suppress
        num_features = segments.shape[2] if segments.shape[0] > 0 else len(BASE_SENSORS)
        segment_length_steps = int(segment_duration_sec * target_freq_hz)
        return np.array([]).reshape(0, segment_length_steps, num_features), np.array([])


    if len(pre_ictal_indices) > min_count or len(interictal_indices) > min_count:
        # print(f"Balancing classes: Reducing majority class to {min_count} samples.") # Suppress
        balanced_indices_pre = np.random.choice(pre_ictal_indices, min_count, replace=False)
        balanced_indices_inter = np.random.choice(interictal_indices, min_count, replace=False)
        balanced_indices = np.concatenate([balanced_indices_pre, balanced_indices_inter])
        np.random.shuffle(balanced_indices)

        segments = segments[balanced_indices]
        labels = labels[balanced_indices]
        # print(f"After balancing: Total segments: {len(segments)}, Pre-ictal: {np.sum(labels)}, Interictal: {len(labels) - np.sum(labels)}") # Suppress

    return segments, labels

def prepare_patient_data(patient_folder):
    """
    Loads, synchronizes, scales, and creates labeled segments for a single patient.
    Returns the list of sensors found for this patient.
    Segments are created with len(BASE_SENSORS) features (with NaNs if missing).
    """
    patient_id = os.path.basename(patient_folder)
    # print(f"Processing data for patient: {patient_id}") # Moved to main loop desc

    # 1. Load sensor data - Try to load all base sensors
    sensor_data_dict = load_sensor_data_for_patient(patient_folder, BASE_SENSORS)
    # Get the list of sensor names for which data was actually found
    found_sensors = list(sensor_data_dict.keys())

    if not sensor_data_dict:
        # print(f"Skipping patient {patient_id}: Could not load any sensor data.") # Moved to main loop desc
        # Return None, indicating failure to load data for any sensor
        return None

    # 2. Load annotations
    annotations_df = load_seizure_annotations(patient_folder)
    # Note: annotations_df can be empty if no seizures, handled in segment creation

    # 3. Synchronize and merge data
    # synced_df will have columns for all BASE_SENSORS, with NaNs for missing ones, in sorted order
    synced_df = synchronize_and_merge_data(sensor_data_dict, target_freq_hz=SAMPLING_FREQ_HZ)
    if synced_df is None or synced_df.empty:
        # print(f"Skipping patient {patient_id}: Could not synchronize or merge sensor data.") # Moved to main loop desc
        return None

    # 4. Create labeled segments
    # Segments will have len(BASE_SENSORS) features, with scaled NaNs for missing sensors
    segments, labels = create_labeled_segments(
        synced_df,
        annotations_df,
        segment_duration_sec=SEGMENT_DURATION_SECONDS,
        pre_ictal_window_min=PRE_ICTAL_WINDOW_MINUTES,
        interictal_buffer_min=INTERICTAL_BUFFER_MINUTES,
        target_freq_hz=SAMPLING_FREQ_HZ
    )

    if len(segments) == 0:
         # print(f"Skipping patient {patient_id}: No valid segments created.") # Moved to main loop desc
         return None

    # Return segments (with all BASE_SENSORS features), labels, AND the list of sensors that were actually found
    return patient_id, segments, labels, found_sensors


# --- PyTorch Dataset ---

class SeizureDataset(Dataset):
    def __init__(self, segments, labels):
        """
        Args:
            segments (np.ndarray): Segments array (n_samples, seq_len, n_features).
                                   Expected features are already selected/ordered.
            labels (np.ndarray): Labels array (n_samples,).
        """
        if segments.shape[0] == 0:
            # Determine the correct number of features and seq_len from the shape if possible, default otherwise
            n_features = segments.shape[2] if segments.shape[0] > 0 else len(BASE_SENSORS) # Default to BASE_SENSORS if unknown
            seq_len = segments.shape[1] if segments.shape[0] > 0 else int(SEGMENT_DURATION_SECONDS * SAMPLING_FREQ_HZ) # Default
            self.segments = torch.empty(0, n_features, seq_len, dtype=torch.float32)
            self.labels = torch.empty(0, 1, dtype=torch.float32)
        else:
            # Assume segments already has the correct features selected and ordered
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

        # Need at least one input channel
        if input_channels == 0:
            raise ValueError("input_channels must be greater than 0")

        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, CONV_FILTERS[0], kernel_size=CONV_KERNEL_SIZE, padding=CONV_KERNEL_SIZE // 2),
            nn.ReLU(),
            nn.MaxPool1d(POOL_SIZE),
            nn.Conv1d(CONV_FILTERS[0], CONV_FILTERS[1], kernel_size=CONV_KERNEL_SIZE, padding=CONV_KERNEL_SIZE // 2),
            nn.ReLU(),
            nn.MaxPool1d(POOL_SIZE),
            nn.Conv1d(CONV_FILTERS[1], CONV_FILTERS[2], kernel_size=CONV_KERNEL_SIZE, padding=CONV_KERNEL_SIZE // 2),
            nn.ReLU(),
            nn.MaxPool1d(POOL_SIZE)
        )

        # Calculate the output sequence length after CNN layers dynamically
        try:
            # Create a dummy tensor with the correct device and dtype for shape calculation
            dummy_input = torch.randn(1, self.input_channels, self.seq_len, dtype=torch.float32)
            dummy_output = self.conv_layers(dummy_input)
            self.lstm_input_features = dummy_output.shape[1]
            self.lstm_input_seq_len = dummy_output.shape[2]
        except Exception as e:
             print(f"Error calculating LSTM input size for CNN_LSTM with input_channels={input_channels}, seq_len={seq_len}: {e}")
             # Fallback or re-raise - for robustness, let's re-raise if this fails
             raise e


        self.lstm = nn.LSTM(input_size=self.lstm_input_features,
                            hidden_size=LSTM_UNITS,
                            batch_first=True)

        self.dense_layers = nn.Sequential(
            nn.Linear(LSTM_UNITS, DENSE_UNITS),
            nn.Sigmoid(),
            nn.Linear(DENSE_UNITS, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch_size, channels, seq_len)
        cnn_out = self.conv_layers(x) # shape: (batch_size, filters, reduced_seq_len)
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

        # Need at least one input channel
        if input_channels == 0:
            raise ValueError("input_channels must be greater than 0")


        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, CONV_FILTERS[0], kernel_size=CONV_KERNEL_SIZE, padding=CONV_KERNEL_SIZE // 2),
            nn.ReLU(),
            nn.MaxPool1d(POOL_SIZE),
            nn.Conv1d(CONV_FILTERS[0], CONV_FILTERS[1], kernel_size=CONV_KERNEL_SIZE, padding=CONV_KERNEL_SIZE // 2),
            nn.ReLU(),
            nn.MaxPool1d(POOL_SIZE),
            nn.Conv1d(CONV_FILTERS[1], CONV_FILTERS[2], kernel_size=CONV_KERNEL_SIZE, padding=CONV_KERNEL_SIZE // 2),
            nn.ReLU(),
            nn.MaxPool1d(POOL_SIZE)
        )

        # Calculate the output sequence length after CNN layers dynamically
        try:
            # Create a dummy tensor with the correct device and dtype for shape calculation
            dummy_input = torch.randn(1, self.input_channels, self.seq_len, dtype=torch.float32)
            dummy_output = self.conv_layers(dummy_input)
            self.lstm_input_features = dummy_output.shape[1]
            self.lstm_input_seq_len = dummy_output.shape[2]
        except Exception as e:
            print(f"Error calculating LSTM input size for CNN_BiLSTM with input_channels={input_channels}, seq_len={seq_len}: {e}")
            # Fallback or re-raise
            raise e


        self.bilstm = nn.LSTM(input_size=self.lstm_input_features,
                              hidden_size=LSTM_UNITS,
                              batch_first=True,
                              bidirectional=True)

        self.dense_layers = nn.Sequential(
            nn.Linear(LSTM_UNITS * 2, DENSE_UNITS),
            nn.Sigmoid(),
            nn.Linear(DENSE_UNITS, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch_size, channels, seq_len)
        cnn_out = self.conv_layers(x) # shape: (batch_size, filters, reduced_seq_len)
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

# train_one_epoch and evaluate_pytorch_model remain the same

def train_one_epoch(model, dataloader, criterion, optimizer, device, class_weights=None):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    # all_labels = [] # Not needed for loss calc or batch tqdm update
    # all_predictions = [] # Not needed for loss calc or batch tqdm update

    dataloader_tqdm = tqdm(dataloader, desc="Batch", leave=False)

    for inputs, labels in dataloader_tqdm:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        if class_weights is not None:
            weight_tensor = torch.zeros_like(labels)
            weight_tensor[labels == 0] = class_weights[0]
            weight_tensor[labels == 1] = class_weights[1]
            loss = criterion(outputs, labels)
            loss = (loss * weight_tensor).mean()
        else:
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # predicted = (outputs > 0.5).float() # Not needed for loss calc or batch tqdm update
        # all_labels.extend(labels.cpu().numpy()) # Not needed for loss calc or batch tqdm update
        # all_predictions.extend(predicted.cpu().numpy()) # Not needed for loss calc or batch tqdm update

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
        auc_roc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc_roc = 0.0
        # print("Warning: Only one class present in evaluation set, AUC-ROC is undefined.") # Suppress


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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=0.0001) # Removed verbose=1

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    epoch_tqdm = tqdm(range(epochs), desc=desc, leave=True)

    for epoch in epoch_tqdm:
        start_time = time.time()

        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, DEVICE, class_weights)

        val_metrics = evaluate_pytorch_model(model, val_dataloader, criterion, DEVICE)
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']

        end_time = time.time()

        epoch_tqdm.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc:.4f}", time=f"{end_time - start_time:.2f}s")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
            if save_best_model_path:
                 torch.save(best_model_state, save_best_model_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= 5:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    # Load best weights before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        # This might happen if training finishes after 0 epochs (e.g., instant early stop, though unlikely with patience>0)
        # or if val_loss never improved.
        print("Warning: No best model state was saved during training.")


    return model


def train_general_model_pytorch(combined_segments, combined_labels, model_type, sensor_combination_name):
    """
    Trains a general model on combined data using PyTorch.
    Takes pre-combined and sliced data for a specific sensor combination.

    Args:
        combined_segments (np.ndarray): Segments array for this combination (N, L, F_comb).
        combined_labels (np.ndarray): Labels array for this combination (N,).
        model_type (str): 'CNN-LSTM' or 'CNN-BiLSTM'.
        sensor_combination_name (str): Name of the sensor combination for logging.

    Returns:
        torch.nn.Module: The trained general model.
        dict: Metrics on the general test set.
        DataLoader: General test DataLoader (for potential later use)
    """
    print(f"\n--- Training General Model ({model_type}) for {sensor_combination_name} ---")

    if len(combined_segments) == 0 or len(np.unique(combined_labels)) < 2:
        print("No data or only one class available for general training combination.")
        # Return dummy model and empty metrics/dataloader
        num_features = combined_segments.shape[2] if len(combined_segments) > 0 else len(BASE_SENSORS)
        seq_len = combined_segments.shape[1] if len(combined_segments) > 0 else int(SEGMENT_DURATION_SECONDS * SAMPLING_FREQ_HZ)
        ModelClass = get_model_class(model_type)
        dummy_model = ModelClass(num_features if num_features > 0 else 1, seq_len).to(DEVICE) # Create a dummy model to avoid errors later
        empty_dataset = SeizureDataset(np.array([]).reshape(0, seq_len, num_features if num_features > 0 else 1), np.array([]))
        empty_dataloader = DataLoader(empty_dataset, batch_size=BATCH_SIZE)

        return dummy_model, {}, empty_dataloader


    print(f"Combined data shape: {combined_segments.shape}, Labels shape: {combined_labels.shape}")
    print(f"Combined data: Pre-ictal: {np.sum(combined_labels)}, Interictal: {len(combined_labels) - np.sum(combined_labels)}")


    classes = np.unique(combined_labels)
    if len(classes) == 2:
         class_weights_np = class_weight.compute_class_weight(
             'balanced', classes=classes, y=combined_labels
         )
         class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights_np)}
         print(f"Computed general class weights: {class_weight_dict}")
    else:
         class_weights_dict = None
         print("Warning: Cannot compute class weights for general model (less than 2 classes).")


    X_train_general, X_temp, y_train_general, y_temp = train_test_split(
        combined_segments, combined_labels, test_size=0.4, random_state=SEED, stratify=combined_labels
    )
    X_val_general, X_test_general, y_val_general, y_test_general = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )

    print(f"General Training data shape: {X_train_general.shape}, Labels shape: {y_train_general.shape}")
    print(f"General Validation data shape: {X_val_general.shape}, Labels shape: {y_val_general.shape}")
    print(f"General Test data shape: {X_test_general.shape}, Labels shape: {y_test_general.shape}")

    train_dataset = SeizureDataset(X_train_general, y_train_general)
    val_dataset = SeizureDataset(X_val_general, y_val_general)
    test_dataset = SeizureDataset(X_test_general, y_test_general)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() // 2 or 1, persistent_workers=True if os.cpu_count() > 1 else False)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 2 or 1, persistent_workers=True if os.cpu_count() > 1 else False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 2 or 1, persistent_workers=True if os.cpu_count() > 1 else False)

    input_channels = combined_segments.shape[2] # Number of features = size of combination
    seq_len = combined_segments.shape[1]

    ModelClass = get_model_class(model_type)
    general_model = ModelClass(input_channels, seq_len).to(DEVICE)

    general_model_path = os.path.join(OUTPUT_DIR, f'general_best_model_{model_type}_{sensor_combination_name}.pth')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Only attempt training if there's actual training data
    if len(train_dataset) > 0:
        general_model = train_pytorch_model(
            general_model,
            train_dataloader,
            val_dataloader,
            epochs=GENERAL_MODEL_EPOCHS,
            learning_rate=GENERAL_MODEL_LR,
            class_weights=class_weight_dict,
            save_best_model_path=general_model_path,
            desc=f"Training General ({model_type}, {sensor_combination_name})"
        )
    else:
        print(f"Warning: No training data for {model_type} + {sensor_combination_name}. Skipping training.")
        # Still create a model, just don't train it
        pass # The model is already initialized above


    print(f"\nEvaluating general model ({model_type}) for {sensor_combination_name} on combined test set...")
    general_metrics = evaluate_pytorch_model(general_model, test_dataloader, nn.BCELoss(), DEVICE) # Pass device
    print(f"General Model Metrics: Accuracy={general_metrics['accuracy']:.4f}, "
          f"Precision={general_metrics['precision']:.4f}, Recall={general_metrics['recall']:.4f}, "
          f"F1 Score={general_metrics['f1_score']:.4f}, AUC-ROC={general_metrics['auc_roc']:.4f}")
    # print(f"Confusion Matrix:\n{general_metrics['confusion_matrix']}") # Suppress detailed CM in console

    del train_dataloader, val_dataloader
    gc.collect()
    if torch.cuda.is_available():
         torch.cuda.empty_cache()

    return general_model, general_metrics, test_dataloader


def perform_personalization_pytorch(general_model_state_dict, patients_data_for_combination_sliced, model_type, sensor_combination_name):
    """
    Performs personalization for each patient IN THE PROVIDED LIST using transfer learning.
    Uses data already sliced to the specific sensor combination.

    Args:
        general_model_state_dict (dict): The state_dict of the pre-trained general model.
        patients_data_for_combination_sliced (list): List of (patient_id, segments_sliced, labels, found_sensors) tuples
                                                    *already filtered* to contain patients suitable for this combination,
                                                    and segments are already sliced to the correct columns.
        model_type (str): 'CNN-LSTM' or 'CNN-BiLSTM'.
        sensor_combination_name (str): Name of the sensor combination for logging.

    Returns:
        dict: Dictionary storing performance metrics before and after personalization for each patient in the list.
    """
    print(f"\n--- Performing Personalization ({model_type}) for {sensor_combination_name} ---")

    if general_model_state_dict is None:
        print("General model state dict is missing. Cannot perform personalization.")
        return {}

    if not patients_data_for_combination_sliced:
         print("No patients available for this combination. Skipping personalization.")
         return {}

    personalization_results = {}
    ModelClass = get_model_class(model_type)

    # Get input channels and seq_len from the first patient's sliced data
    # Assumes all patients in this list have data sliced to the same shape
    num_features = patients_data_for_combination_sliced[0][1].shape[2]
    seq_len = patients_data_for_combination_sliced[0][1].shape[1]


    # Wrap patient loop with tqdm
    patient_tqdm = tqdm(patients_data_for_combination_sliced, desc=f"Personalizing Patients ({model_type}, {sensor_combination_name})", leave=True)

    for patient_id, patient_segments, patient_labels, patient_found_sensors in patient_tqdm:
        # patient_segments here are already sliced to the current combination

        # Update patient progress bar description
        patient_tqdm.set_description(f"Personalizing {patient_id} ({model_type}, {sensor_combination_name})")

        # Check if patient data for this combination is usable for splitting/training
        if len(patient_segments) == 0 or len(np.unique(patient_labels)) < 2:
             print(f"Skipping patient {patient_id} ({model_type}, {sensor_combination_name}): No valid segments or only one class after slicing/processing.")
             # Add entry to results with empty metrics
             personalization_results[patient_id] = {
                 "before": {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]},
                 "after": {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]}
             }
             # Clean up any potential temporary data/models if they existed
             gc.collect()
             if torch.cuda.is_available(): torch.cuda.empty_cache()
             continue


        # Split patient's sliced data for personalization fine-tuning and testing
        # Use the pre-sliced patient_segments and patient_labels
        X_train_pat, X_temp_pat, y_train_pat, y_temp_pat = train_test_split(
             patient_segments, patient_labels, test_size=0.4, random_state=SEED, stratify=patient_labels
        )
        X_val_pat, X_test_pat, y_val_pat, y_test_pat = train_test_split(
             X_temp_pat, y_temp_pat, test_size=0.5, random_state=SEED, stratify=y_temp_pat
        )

        print(f"Patient {patient_id} ({model_type}, {sensor_combination_name}) - Personalization Train shape: {X_train_pat.shape}, Val shape: {X_val_pat.shape}, Test shape: {X_test_pat.shape}")


        train_dataset_pat = SeizureDataset(X_train_pat, y_train_pat)
        val_dataset_pat = SeizureDataset(X_val_pat, y_val_pat)
        test_dataset_pat = SeizureDataset(X_test_pat, y_test_pat)

        train_dataloader_pat = DataLoader(train_dataset_pat, batch_size=PERSONALIZATION_BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() // 4 or 1, persistent_workers=True if os.cpu_count() > 1 else False)
        val_dataloader_pat = DataLoader(val_dataset_pat, batch_size=PERSONALIZATION_BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 4 or 1, persistent_workers=True if os.cpu_count() > 1 else False)
        test_dataloader_pat = DataLoader(test_dataset_pat, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 4 or 1, persistent_workers=True if os.cpu_count() > 1 else False)


        # Evaluate the general model on this patient's test data (Before Personalization)
        print(f"Evaluating general model ({model_type}, {sensor_combination_name}) on patient {patient_id}'s test data (Before Personalization)...")
        # Model instance for evaluation needs to have the same architecture as the general model
        general_model_instance_eval = ModelClass(num_features, seq_len).to(DEVICE)
        general_model_instance_eval.load_state_dict(general_model_state_dict)

        metrics_before = evaluate_pytorch_model(general_model_instance_eval, test_dataloader_pat, nn.BCELoss(), DEVICE) # Pass device
        print(f"Before Personalization ({model_type}, {sensor_combination_name}): Accuracy={metrics_before['accuracy']:.4f}, Precision={metrics_before['precision']:.4f}, Recall={metrics_before['recall']:.4f}, F1={metrics_before['f1_score']:.4f}, AUC={metrics_before['auc_roc']:.4f}")
        del general_model_instance_eval
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


        # Create a new model instance for personalization fine-tuning
        personalized_model = ModelClass(num_features, seq_len).to(DEVICE)
        personalized_model.load_state_dict(general_model_state_dict)

        classes_pat = np.unique(y_train_pat)
        if len(classes_pat) == 2:
             class_weights_pat_np = class_weight.compute_class_weight(
                 'balanced', classes=classes_pat, y=y_train_pat
             )
             class_weights_pat_dict = {cls: weight for cls, weight in zip(classes_pat, class_weights_pat_np)}
             # print(f"Computed patient {patient_id} ({model_type}, {sensor_combination_name}) class weights: {class_weights_pat_dict}") # Suppress detailed print
        else:
             class_weights_pat_dict = None
             # print(f"Warning: Only one class for patient {patient_id} training data ({model_type}, {sensor_combination_name}). No class weights applied.") # Suppress

        # Only attempt fine-tuning if there's training data for this patient
        if len(train_dataset_pat) > 0:
            # Fine-tune the model on the patient's training data
            print(f"Fine-tuning model ({model_type}, {sensor_combination_name}) on patient {patient_id}'s training data...")
            personalized_model_path = os.path.join(OUTPUT_DIR, f'{patient_id}_{model_type}_{sensor_combination_name}_personalized_best_model.pth')
            personalized_model = train_pytorch_model(
                personalized_model,
                train_dataloader_pat,
                val_dataloader_pat,
                epochs=PERSONALIZATION_EPOCHS,
                learning_rate=PERSONALIZATION_LR,
                class_weights=class_weights_pat_dict,
                save_best_model_path=personalized_model_path,
                desc=f"Fine-tuning {patient_id} ({model_type}, {sensor_combination_name})"
            )
        else:
            print(f"Warning: No training data for patient {patient_id} ({model_type}, {sensor_combination_name}). Skipping fine-tuning.")
            # The personalized_model instance is already initialized with general weights,
            # it just won't be trained.

        # Evaluate the personalized model on this patient's test data (After Personalization)
        print(f"Evaluating personalized model ({model_type}, {sensor_combination_name}) on patient {patient_id}'s test data (After Personalization)...")
        metrics_after = evaluate_pytorch_model(personalized_model, test_dataloader_pat, nn.BCELoss(), DEVICE) # Pass device
        print(f"After Personalization ({model_type}, {sensor_combination_name}): Accuracy={metrics_after['accuracy']:.4f}, Precision={metrics_after['precision']:.4f}, Recall={metrics_after['recall']:.4f}, F1={metrics_after['f1_score']:.4f}, AUC={metrics_after['auc_roc']:.4f}")
        # print(f"Confusion Matrix:\n{metrics_after['confusion_matrix']}") # Suppress detailed CM in console

        personalization_results[patient_id] = {
            "before": metrics_before,
            "after": metrics_after
        }

        del train_dataset_pat, val_dataset_pat, test_dataset_pat
        del train_dataloader_pat, val_dataloader_pat, test_dataloader_pat
        del personalized_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    return personalization_results


def get_combined_data_for_combination(processed_patients_data, sensor_combination):
    """
    Filters patients to those having all sensors in the combination,
    selects only the specified sensor columns from their segments, and combines.

    Args:
        processed_patients_data (list): List of (patient_id, segments, labels, found_sensors)
                                        from initial processing (segments have len(BASE_SENSORS) features).
        sensor_combination (list): List of sensor names (e.g., ['HR', 'EDA']) for the current combination.

    Returns:
        tuple: (combined_segments, combined_labels, patients_in_combination_data_sliced)
               combined_segments: np.ndarray (N, L, F_comb)
               combined_labels: np.ndarray (N,)
               patients_in_combination_data_sliced: list of (patient_id, segments_sliced, labels, found_sensors)
                                                    (segments_sliced is the view for this combination)
    """
    combination_lower = [s.lower() for s in sensor_combination]
    combination_name = "_".join(combination_lower).upper() # e.g., HR_EDA

    print(f"\nFiltering patients and preparing data for combination: {combination_name}")

    patients_in_combination_data_sliced = []
    all_segments_for_combination = []
    all_labels_for_combination = []

    # Get indices for the sensors in the current combination (relative to BASE_SENSORS order)
    try:
        sensor_indices = [BASE_SENSORS.index(s.upper()) for s in sensor_combination]
    except ValueError as e:
        print(f"Error: Sensor in combination {sensor_combination} not found in BASE_SENSORS. {e}")
        return np.array([]), np.array([]), []


    for patient_id, segments_all_sensors, labels, found_sensors in processed_patients_data:
        # Check if the patient has *all* sensors required for this combination
        if all(s.lower() in found_sensors for s in sensor_combination):
            # Slice segments to include only the features for the current combination
            segments_for_combination = segments_all_sensors[:, :, sensor_indices]

            # Only include if the sliced data still has valid segments and both classes are present
            if len(segments_for_combination) > 0 and len(np.unique(labels)) > 1:
                 all_segments_for_combination.append(segments_for_combination)
                 all_labels_for_combination.append(labels)
                 # Store the sliced data for this patient for the personalization step
                 # Note: patients_data_for_combination_sliced contains segments already filtered by columns
                 patients_in_combination_data_sliced.append((patient_id, segments_for_combination, labels, found_sensors))
            # else: print(f"Skipping patient {patient_id} for combination {combination_name}: No segments or single class after slicing.")
        # else: print(f"Skipping patient {patient_id} for combination {combination_name}: Missing required sensors.")


    if not all_segments_for_combination:
        print(f"No patients found with all sensors for combination: {combination_name}. Skipping this combination.")
        return np.array([]), np.array([]), []

    combined_segments = np.concatenate(all_segments_for_combination, axis=0)
    combined_labels = np.concatenate(all_labels_for_combination, axis=0)

    print(f"Combined data prepared for {combination_name}: {combined_segments.shape[0]} samples from {len(patients_in_combination_data_sliced)} patients.")

    return combined_segments, combined_labels, patients_in_combination_data_sliced

def print_personalization_summary(personalization_results, output_file=None):
    """ Prints a summary table of personalization results to console or file. """
    # Determine where to print (console or file)
    def print_func(*args, **kwargs):
        if output_file:
            print(*args, **kwargs, file=output_file)
        else:
            print(*args, **kwargs)

    print_func("\n--- Personalized Model Performance (Per Patient Summary) ---")
    print_func("Patient ID | Accuracy Before | Accuracy After | Change")
    print_func("-----------------------------------------------------")
    total_change = 0
    count = 0
    for patient_id, results in personalization_results.items():
        # Ensure metrics are valid before printing/counting
        if 'accuracy' in results['before'] and 'accuracy' in results['after']:
            acc_before = results['before']['accuracy']
            acc_after = results['after']['accuracy']
            change = acc_after - acc_before
            print_func(f"{patient_id:<10} | {acc_before:.4f}        | {acc_after:.4f}       | {change:.4f}")

            # Only include patients for whom data was processed (metrics > 0) in the average change calculation
            # This heuristic might need review - maybe average across all *attempted* personalizations, but ignore errors?
            # Let's average only patients where both before/after metrics are available and test set was non-empty.
            # A test set with 0 samples will typically result in 0 accuracy, but a non-empty test set failing the model might too.
            # Using a simple check that accuracy is not None/missing might be better.
            if results['after']['accuracy'] is not None: # Assuming if accuracy is there, other metrics are too
                # Check if the test set for this patient was likely non-empty
                # This is tricky without knowing the dataset size, but a non-zero accuracy implies data was processed.
                # Or, perhaps check if the confusion matrix size indicates data > 0
                cm = results['after'].get('confusion_matrix', [[0,0],[0,0]])
                if len(cm) == 2 and len(cm[0]) == 2 and sum(sum(row) for row in cm) > 0:
                    # Only count if Confusion Matrix indicates samples were evaluated
                    total_change += change # Keep average change calculation too if needed elsewhere
                    count += 1
                else: print_func(f"--- Debug: Patient {patient_id} skipped average calculation due to empty test set/CM")


        else:
            print_func(f"{patient_id:<10} | N/A             | N/A            | N/A") # Indicate missing data


    print_func("-----------------------------------------------------")
    if count > 0:
        average_change = total_change / count
        print_func(f"Average Accuracy Improvement (across {count} patients with valid data): {average_change:.4f}")
    else:
        print_func("No valid personalized patient results to summarize average improvement.")
        
# --- Main Execution ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_patient_folders = [f.path for f in os.scandir(DATA_ROOT_DIR) if f.is_dir() and f.name.startswith('MSEL_')]

    if not all_patient_folders:
        print(f"No patient directories starting with 'MSEL_' found in {DATA_ROOT_DIR}.")
    else:
        print(f"Found {len(all_patient_folders)} patient directories.")

    # --- Step 0: Process data for all patients (load, sync, segment using ALL found sensors) ---
    # processed_patient_data will store (patient_id, segments_all_found, labels, found_sensors_list)
    # segments_all_found will have shape (N, L, len(BASE_SENSORS)) with NaNs for missing original sensors
    processed_patient_data = []
    for patient_folder in tqdm(all_patient_folders, desc="Initial Patient Data Processing"):
        patient_data = prepare_patient_data(patient_folder)
        if patient_data:
            processed_patient_data.append(patient_data)

    if not processed_patient_data:
        print("No valid patient data was processed. Exiting.")
    else:
        print(f"\nSuccessfully processed initial data for {len(processed_patient_data)} patients.")

        model_types_to_run = ['CNN-LSTM', 'CNN-BiLSTM']

        # --- Generate Sensor Combinations if enabled ---
        if ENABLE_SENSOR_COMBINATIONS:
            sensor_combinations = []
            # Generate combinations of lengths 1 to len(BASE_SENSORS)
            for i in range(1, len(BASE_SENSORS) + 1):
                for combo in itertools.combinations(BASE_SENSORS, i):
                    sensor_combinations.append(list(combo))
            print(f"\nEnabled sensor combinations feature. Will run {len(sensor_combinations)} combinations.")
            print(f"Combinations: {[', '.join(c) for c in sensor_combinations]}")
        else:
            # If not enabled, only run the full set (assuming patients have all of BASE_SENSORS)
            sensor_combinations = [list(BASE_SENSORS)]
            print("\nSensor combinations feature is OFF. Running only for the full base sensor set.")


        all_results = {} # Stores results nested by model_type and sensor_combination_name

        # --- Define the output file ---
        output_filename = os.path.join(OUTPUT_DIR, f'seizure_prediction_results_{time.strftime("%Y%m%d_%H%M%S")}.txt')

        with open(output_filename, 'w') as output_file:
            output_file.write(f"Seizure Prediction Results\n")
            output_file.write(f"Run Date: {time.ctime()}\n")
            output_file.write(f"Data Directory: {DATA_ROOT_DIR}\n")
            output_file.write(f"Processed {len(processed_patient_data)} patients initially.\n")
            output_file.write(f"Models Run: {model_types_to_run}\n")
            output_file.write(f"Sensor Combination Testing Enabled: {ENABLE_SENSOR_COMBINATIONS}\n")
            output_file.write(f"Hyperparameter Tuning:\n"
                            f"  SEGMENT_DURATION_SECONDS: {SEGMENT_DURATION_SECONDS}\n"
                            f"  PRE_ICTAL_WINDOW_MINUTES: {PRE_ICTAL_WINDOW_MINUTES}\n"
                            f"  INTERICTAL_BUFFER_MINUTES: {INTERICTAL_BUFFER_MINUTES}\n"
                            f"  BASE_SENSORS: {BASE_SENSORS}\n"
                            f"  SAMPLING_FREQ_HZ: {SAMPLING_FREQ_HZ}\n"
                            f"  CONV_FILTERS: {CONV_FILTERS}\n"
                            f"  CONV_KERNEL_SIZE: {CONV_KERNEL_SIZE}\n"
                            f"  POOL_SIZE: {POOL_SIZE}\n"
                            f"  LSTM_UNITS: {LSTM_UNITS}\n"
                            f"  DENSE_UNITS: {DENSE_UNITS}\n"
                            f"  GENERAL_MODEL_EPOCHS: {GENERAL_MODEL_EPOCHS}\n"
                            f"  PERSONALIZATION_EPOCHS: {PERSONALIZATION_EPOCHS}\n"
                            f"  GENERAL_MODEL_LR: {GENERAL_MODEL_LR}\n"
                            f"  PERSONALIZATION_LR: {PERSONALIZATION_LR}\n"
                            f"  BATCH_SIZE: {BATCH_SIZE}\n"
                            f"  PERSONALIZATION_BATCH_SIZE: {PERSONALIZATION_BATCH_SIZE}\n")
            if not ENABLE_SENSOR_COMBINATIONS:
                output_file.write(f"Running only for Base Sensor Set: {BASE_SENSORS}\n")
            output_file.write("=" * 50 + "\n\n")

            # --- Loop through each model type ---
            for current_model_type in model_types_to_run:
                all_results[current_model_type] = {} # Nested dictionary for this model type

                # --- Loop through each sensor combination ---
                for current_combination in sensor_combinations:
                    combination_name = "_".join([s.lower() for s in current_combination]).upper() # Consistent naming

                    print(f"\n{'='*40}")
                    print(f"RUNNING: Model {current_model_type} + Sensors {combination_name}")
                    print(f"{'='*40}\n")

                    output_file.write(f"\n\n{'#'*40}\n")
                    output_file.write(f"RESULTS FOR MODEL: {current_model_type}, SENSORS: {combination_name}\n")
                    output_file.write(f"{'#'*40}\n\n")


                    # --- Get data combined ONLY for this combination and suitable patients ---
                    # This returns (combined_segments, combined_labels, patients_in_combination_data_sliced)
                    # patients_in_combination_data_sliced contains segments already sliced to the current combination size
                    combined_segments, combined_labels, patients_in_combination_data_sliced = get_combined_data_for_combination(
                         processed_patient_data,
                         current_combination
                    )

                    num_patients_in_combination = len(patients_in_combination_data_sliced)
                    all_results[current_model_type][combination_name] = {
                        'general_metrics': {}, # Will be populated later
                        'personalization_results': {}, # Will be populated later
                        'avg_personalized_metrics': None, # Will be calculated later
                        'num_patients_in_combination': num_patients_in_combination # Store patient count
                    }


                    if len(combined_segments) == 0 or num_patients_in_combination == 0:
                        print(f"Skipping training/personalization for {current_model_type} + {combination_name}: No data or no suitable patients.")
                        output_file.write(f"Skipping training/personalization for {current_model_type} + {combination_name}: No data or no suitable patients.\n\n")
                        continue # Move to the next combination


                    output_file.write(f"Using data from {num_patients_in_combination} patients for this combination.\n")


                    # --- Step 1: Train the General Model for this combination ---
                    # Pass the combined data directly
                    general_model, general_metrics, general_test_dataloader = train_general_model_pytorch(
                        combined_segments,
                        combined_labels,
                        model_type=current_model_type,
                        sensor_combination_name=combination_name
                    )

                    all_results[current_model_type][combination_name]['general_metrics'] = general_metrics

                    # Write General Model Performance to file
                    output_file.write("--- General Model Performance ---\n")
                    if 'accuracy' in general_metrics:
                        output_file.write(f"Accuracy={general_metrics['accuracy']:.4f}\n")
                        output_file.write(f"Precision={general_metrics['precision']:.4f}\n")
                        output_file.write(f"Recall={general_metrics['recall']:.4f}\n")
                        output_file.write(f"F1 Score={general_metrics['f1_score']:.4f}\n")
                        output_file.write(f"AUC-ROC={general_metrics['auc_roc']:.4f}\n")
                        output_file.write(f"Confusion Matrix:\n{general_metrics['confusion_matrix']}\n")
                    else:
                        output_file.write("General model training failed or produced invalid metrics.\n")
                    output_file.write("\n")


                    if general_model is not None and len(combined_segments) > 0: # Ensure model was trained and data exists
                        general_model_state = general_model.state_dict()
                        del general_model, general_test_dataloader
                        gc.collect()
                        if torch.cuda.is_available(): torch.cuda.empty_cache()

                        # --- Step 2: Perform Personalization for this combination ---
                        # Pass the data already filtered and sliced for this combination
                        personalization_results = perform_personalization_pytorch(
                            general_model_state,
                            patients_in_combination_data_sliced, # Use the sliced data
                            model_type=current_model_type,
                            sensor_combination_name=combination_name
                        )

                        all_results[current_model_type][combination_name]['personalization_results'] = personalization_results # Store per-patient results

                        # --- Step 3: Summarize Personalized Model Performance ---
                        # Write per-patient summary to file
                        print_personalization_summary(personalization_results, output_file=output_file)
                        # Print per-patient summary to console too for monitoring
                        print_personalization_summary(personalization_results, output_file=None)

                        # Calculate and Write Average Personalized Model Performance
                        metrics_after_list = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'auc_roc': []}
                        count_valid_patients_pers = 0

                        for patient_id, results in personalization_results.items():
                            # Check if the 'after' personalization metrics are valid for this patient
                            if 'after' in results and 'accuracy' in results['after']: # Use .get is safer: results.get('after', {})
                                cm = results.get('after', {}).get('confusion_matrix', [[0,0],[0,0]]) # Safely get CM
                                # Check if confusion matrix indicates samples were evaluated
                                if len(cm) == 2 and len(cm[0]) == 2 and sum(sum(row) for row in cm) > 0: # Correct sum syntax
                                    count_valid_patients_pers += 1
                                    metrics_after_list['accuracy'].append(results['after']['accuracy'])
                                    metrics_after_list['precision'].append(results['after']['precision'])
                                    metrics_after_list['recall'].append(results['after']['recall'])
                                    metrics_after_list['f1_score'].append(results['after']['f1_score'])
                                    metrics_after_list['auc_roc'].append(results['after']['auc_roc'])


                        output_file.write("\n--- Personalized Model Performance (Average Across Patients) ---\n")
                        if count_valid_patients_pers > 0:
                            avg_metrics = {metric: np.mean(metrics_after_list[metric]) for metric in metrics_after_list}
                            all_results[current_model_type][combination_name]['avg_personalized_metrics'] = avg_metrics # Store for final comparison

                            output_file.write(f"Average Accuracy={avg_metrics['accuracy']:.4f} (across {count_valid_patients_pers} patients)\n")
                            output_file.write(f"Average Precision={avg_metrics['precision']:.4f}\n")
                            output_file.write(f"Average Recall={avg_metrics['recall']:.4f}\n")
                            output_file.write(f"Average F1 Score={avg_metrics['f1_score']:.4f}\n")
                            output_file.write(f"Average AUC-ROC={avg_metrics['auc_roc']:.4f}\n")

                        else:
                            output_file.write("No valid personalized patient results to average.\n")
                            all_results[current_model_type][combination_name]['avg_personalized_metrics'] = None # Store None

                        output_file.write("\n") # Add space

                    else:
                        # Handle case where general model training failed for this combination
                        print(f"General model training failed for {current_model_type} + {combination_name}. Skipping personalization.")
                        output_file.write(f"General model training failed for {current_model_type} + {combination_name}. Skipping personalization.\n\n")
                        all_results[current_model_type][combination_name]['personalization_results'] = {} # Store empty
                        all_results[current_model_type][combination_name]['avg_personalized_metrics'] = None # Store None


            # --- Final Overall Comparison Summary ---
            output_file.write(f"\n\n{'='*60}\n")
            output_file.write("FINAL OVERALL COMPARISON SUMMARY\n")
            output_file.write(f"{'='*60}\n\n")

            for model_type, results_by_combination in all_results.items():
                output_file.write(f"\n--- Summary for Model Type: {model_type} ---\n\n")

                output_file.write("General Model Performance Comparison:\n")
                output_file.write("Sensors    | Accuracy | Precision | Recall | F1 Score | AUC-ROC | Patients\n")
                output_file.write("----------------------------------------------------------------------------\n")
                # Sort combinations by name for consistent table output
                for combo_name in sorted(results_by_combination.keys()):
                    combo_results = results_by_combination[combo_name]
                    gen_metrics = combo_results.get('general_metrics', {}) # Use .get for safety
                    num_patients = combo_results.get('num_patients_in_combination', 0)
                    if 'accuracy' in gen_metrics:
                         output_file.write(f"{combo_name:<10} | {gen_metrics['accuracy']:.4f} | {gen_metrics['precision']:.4f} | {gen_metrics['recall']:.4f} | {gen_metrics['f1_score']:.4f} | {gen_metrics['auc_roc']:.4f} | {num_patients}\n")
                    else:
                         output_file.write(f"{combo_name:<10} | N/A      | N/A       | N/A    | N/A      | N/A     | {num_patients}\n")
                output_file.write("----------------------------------------------------------------------------\n\n")

                output_file.write("Personalized Model Performance Comparison (Average Across Patients):\n")
                output_file.write("Sensors    | Avg Acc  | Avg Prec  | Avg Rec  | Avg F1   | Avg AUC\n")
                output_file.write("--------------------------------------------------------------------\n")
                 # Sort combinations by name for consistent table output
                for combo_name in sorted(results_by_combination.keys()):
                     combo_results = results_by_combination[combo_name]
                     avg_metrics = combo_results.get('avg_personalized_metrics', None) # Use .get for safety
                     if avg_metrics is not None:
                         output_file.write(f"{combo_name:<10} | {avg_metrics['accuracy']:.4f} | {avg_metrics['precision']:.4f} | {avg_metrics['recall']:.4f} | {avg_metrics['f1_score']:.4f} | {avg_metrics['auc_roc']:.4f}\n")
                     else:
                         output_file.write(f"{combo_name:<10} | N/A      | N/A       | N/A      | N/A      | N/A\n")
                output_file.write("--------------------------------------------------------------------\n\n")


        print(f"\nAll results saved to {output_filename}")