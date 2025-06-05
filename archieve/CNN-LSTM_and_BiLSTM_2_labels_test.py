import datetime
import gc  # Garbage collection
import glob
import logging
import math  # For calculating output length
import os
import random  # For reproducibility
import time
import uuid

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
from tqdm.auto import tqdm  # Import tqdm for progress bars

# --- Logging Configuration ---
# Create a unique log filename with date and UUID
log_filename = f"log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}.log"
log_filepath = os.path.join('.', log_filename) # Save log file in the current directory

logging.basicConfig(level=logging.INFO, # Set lowest level to capture INFO and above
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_filepath), # Log to file
                        logging.StreamHandler() # Log to console
                    ])

# --- Configuration ---
DATA_ROOT_DIR = 'F:\\data_9' # Replace with the actual path to your 'data_9' folder
OUTPUT_DIR = 'processed_data_pytorch_test' # Directory to save intermediate processed files/models
SEGMENT_DURATION_SECONDS = 30
PRE_ICTAL_WINDOW_MINUTES = 30
# ======================================================================================================================================
# TODO CHANGE TO 5,10,15,20,25,30,35,40,45,50,55,60 MIN
# ======================================================================================================================================
INTERICTAL_BUFFER_MINUTES = 180 # Time buffer around seizures to define clear interictal state
SENSORS_TO_USE = ['HR', 'EDA', 'TEMP', 'ACC'] # Exclude BVP as requested
SAMPLING_FREQ_HZ = 1 # Target sampling frequency for synchronization (adjust based on actual data or paper details)

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
BATCH_SIZE = 32 # Batch size for DataLoaders
PERSONALIZATION_BATCH_SIZE = 16 # Smaller batch size for personalization fine-tuning

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
# logging.info(f"Using device: {DEVICE}")

# Helper function to get model class
def get_model_class(model_type):
    if model_type == 'CNN-LSTM':
        return CNN_LSTM
    elif model_type == 'CNN-BiLSTM':
        return CNN_BiLSTM
    else:
        raise ValueError(f"Unknown model type: {model_type}")
# --- Data Loading and Preprocessing (Adapted) ---

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
            logging.warning(f"No mapping found for sensor '{sensor_name}'. Skipping.")
            continue

        attr_folder = sensor_name
        attr_name_part = sensor_mapping[sensor_name]

        parquet_files = sorted(glob.glob(os.path.join(patient_dir, f'Empatica-{attr_folder}', f'{os.path.basename(patient_dir)}_Empatica-{attr_folder}_{attr_name_part}_segment_*.parquet')))

        if not parquet_files:
            logging.warning(f"No Parquet files found for Patient in {patient_dir}, Attribute {attr_folder}")
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
                    logging.warning(f"Parquet file {file_path} does not have expected 'time' and 'data' columns. Skipping.")

            except Exception as e:
                logging.error(f"Error reading Parquet file {file_path}: {e}")

        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            combined_df = combined_df.sort_values(by='timestamp').reset_index(drop=True)
            attribute_data[sensor_name.lower()] = combined_df

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
                          Returns None if the file is not found or has incorrect columns.
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
            logging.error(f"Annotation file {annotation_file} does not have expected 'start_time' and 'end_time' columns.")
            return None
    except FileNotFoundError:
        logging.warning(f"Annotation file not found at {annotation_file}")
        return pd.DataFrame(columns=['start_time', 'end_time'])
    except Exception as e:
        logging.error(f"Error reading annotation file {annotation_file}: {e}")
        return None


def synchronize_and_merge_data(sensor_data_dict, target_freq_hz=1):
    """
    Synchronizes sensor data from different sensors to a common time index
    by resampling, merges them, applies Robust Scaling, and handles NaNs.

    Args:
        sensor_data_dict (dict): Dictionary with sensor names as keys and
                                 DataFrames (with 'timestamp' and data column) as values.
        target_freq_hz (int): The target frequency in Hz for resampling.

    Returns:
        pandas.DataFrame: A single DataFrame with a datetime index containing
                          all synchronized and scaled sensor data. Returns None if input is empty.
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

    if merged_df.empty or merged_df.isnull().values.any():
        logging.warning("Merged DataFrame is empty or still contains NaNs after interpolation/filling. Skipping scaling.")
        # If critical NaNs remain, scaling might fail or be misleading.
        # Depending on tolerance, one might drop NaNs here or return None.
        # For now, proceed but the scaling might be affected.
        pass


    # Apply Robust Scaling
    scaler = RobustScaler()
    data_cols = merged_df.columns
    # Only scale if the DataFrame is not empty and has columns
    if not merged_df.empty and len(data_cols) > 0:
        merged_df[data_cols] = scaler.fit_transform(merged_df[data_cols])
    elif not merged_df.empty and len(data_cols) == 0:
         logging.warning("Merged DataFrame is not empty but has no data columns to scale.")


    return merged_df

def create_labeled_segments(synced_df, annotations_df, segment_duration_sec, pre_ictal_window_min, interictal_buffer_min, target_freq_hz):
    """
    Creates segments from synchronized data and labels them
    as pre-ictal (1) or interictal (0) based on seizure annotations. Samples interictal
    segments to attempt class balance.

    Args:
        synced_df (pandas.DataFrame): DataFrame with synchronized sensor data (datetime index).
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
    if synced_df is None or annotations_df is None or synced_df.empty or len(synced_df.columns) == 0:
        logging.warning("Synced data is empty, has no columns, or annotations are missing. Cannot create segments.")
        num_features = len(synced_df.columns) if synced_df is not None and not synced_df.empty else len(SENSORS_TO_USE)
        segment_length_steps = int(segment_duration_sec * target_freq_hz)
        return np.array([]).reshape(0, segment_length_steps, num_features), np.array([])

    segments = []
    labels = []
    segment_length_steps = int(segment_duration_sec * target_freq_hz)
    step_size = segment_length_steps # Use non-overlapping segments

    data_start_time = synced_df.index.min()
    data_end_time = synced_df.index.max()
    logging.info(f"Data range: {data_start_time} to {data_end_time}")

    # Helper to check overlap (inclusive of boundaries)
    def check_overlap(seg_start, seg_end, windows):
        for win_start, win_end in windows:
            # Check if intervals overlap OR touch at endpoints
            # A overlaps B if max(A_start, B_start) <= min(A_end, B_end)
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
            # logging.info(f" Seizure {i+1}: Ictal window: {seizure_start} to {seizure_end}") # Optional: too verbose

    # 2. Pre-ictal Windows
    pre_ictal_windows = []
    logging.info(f"Defining pre-ictal windows ({pre_ictal_window_min} mins before seizure onset)...")
    for i, seizure in annotations_df.iterrows():
        seizure_start = seizure['start_time']
        pre_ictal_start_uncapped = seizure_start - pd.Timedelta(minutes=pre_ictal_window_min)
        pre_ictal_end = seizure_start - pd.Timedelta(seconds=1)

        # Cap the pre-ictal start at the beginning of the available data
        pre_ictal_start = max(data_start_time, pre_ictal_start_uncapped)

        # Ensure the capped window is valid AND overlaps with the data range
        if pre_ictal_start < pre_ictal_end and max(pre_ictal_start, data_start_time) <= min(pre_ictal_end, data_end_time): # Use <= for inclusive overlap check
             # Ensure pre-ictal window does *not* overlap with the seizure itself
            if not check_overlap(pre_ictal_start, pre_ictal_end, seizure_windows):
                pre_ictal_windows.append((pre_ictal_start, pre_ictal_end))
                logging.info(f" Seizure {i+1}: Pre-ictal window added: {pre_ictal_start} to {pre_ictal_end}")
            else:
                logging.info(f" Seizure {i+1}: Calculated pre-ictal window ({pre_ictal_start} to {pre_ictal_end}) overlaps with seizure window. Skipped.")
        else:
            logging.info(f" Seizure {i+1}: Calculated pre-ictal window ({pre_ictal_start_uncapped} to {pre_ictal_end}) or capped window ({pre_ictal_start} to {pre_ictal_end}) is invalid or outside data range. Skipped.")


    # 3. Interictal Exclusion Windows (Seizure + Buffer)
    # These define areas NOT suitable for clean interictal samples
    interictal_exclusion_windows = []
    buffer_timedelta = pd.Timedelta(minutes=interictal_buffer_min)
    logging.info(f"Defining interictal exclusion windows ({interictal_buffer_min} mins buffer around seizures)...")
    for _, seizure in annotations_df.iterrows():
        exclusion_start = seizure['start_time'] - buffer_timedelta
        exclusion_end = seizure['end_time'] + buffer_timedelta
        interictal_exclusion_windows.append((exclusion_start, exclusion_end))


    # --- Create Segments and Assign Labels ---

    logging.info(f"Creating segments (len={segment_duration_sec}s, step={segment_duration_sec}s) from {len(synced_df)} total steps...")
    segments_skipped_ictal = 0
    segments_skipped_interictal_buffer = 0 # Segments in buffer BUT NOT pre-ictal
    segments_labeled_preictal = 0
    segments_labeled_interictal = 0
    segments_total_candidates = 0 # Count segments before any skipping

    for i in tqdm(range(0, len(synced_df) - segment_length_steps + 1, step_size), desc="Segmenting Data", leave=False):
        segment_df = synced_df.iloc[i : i + segment_length_steps]
        if len(segment_df) != segment_length_steps: # Should not happen with this range, but good check
            continue

        segments_total_candidates += 1 # This segment is a full-length candidate

        segment_start_time = segment_df.index[0]
        segment_end_time = segment_df.index[-1]

        # Check for overlap with actual seizure (ictal) windows - SKIP
        if check_overlap(segment_start_time, segment_end_time, seizure_windows):
            segments_skipped_ictal += 1
            continue # Skip this segment


        # Check for overlap with pre-ictal windows - LABEL 1 (PRIORITIZE)
        if check_overlap(segment_start_time, segment_end_time, pre_ictal_windows):
            segments.append(segment_df.values)
            labels.append(1)
            segments_labeled_preictal += 1
            continue # Labeled pre-ictal, move to next segment


        # Check for overlap with interictal exclusion buffer - SKIP (don't label as 0 if in buffer)
        # This check ONLY happens if the segment was NOT ictal and NOT pre-ictal
        if check_overlap(segment_start_time, segment_end_time, interictal_exclusion_windows):
            segments_skipped_interictal_buffer += 1
            continue # Skip this segment


        # If none of the above, label as Interictal (0)
        segments.append(segment_df.values)
        labels.append(0)
        segments_labeled_interictal += 1


    segments = np.array(segments)
    labels = np.array(labels)

    logging.info(f"Finished segmentation. Total full-length candidate segments: {segments_total_candidates}")
    logging.info(f" Segments skipped (ictal): {segments_skipped_ictal}")
    logging.info(f" Segments skipped (interictal buffer, not pre-ictal): {segments_skipped_interictal_buffer}")
    logging.info(f" Total segments included for labeling (Pre-ictal + Interictal): {len(segments)}")
    logging.info(f" Segments labeled Pre-ictal: {np.sum(labels)}, Interictal: {len(labels) - np.sum(labels)}")

    # Simple class balancing: Undersample majority class
    pre_ictal_indices = np.where(labels == 1)[0]
    interictal_indices = np.where(labels == 0)[0]

    min_count = min(len(pre_ictal_indices), len(interictal_indices))

    if min_count == 0:
        logging.warning("One class has zero samples after segmentation. Cannot balance.")
        num_features = segments.shape[2] if segments.shape[0] > 0 else len(SENSORS_TO_USE)
        segment_length_steps = int(segment_duration_sec * target_freq_hz)
        # Return empty with correct dimensions based on expected shape
        return np.array([]).reshape(0, segment_length_steps, num_features), np.array([])


    if len(pre_ictal_indices) > min_count or len(interictal_indices) > min_count:
        logging.info(f"Balancing classes: Reducing majority class to {min_count} samples.")
        balanced_indices_pre = np.random.choice(pre_ictal_indices, min_count, replace=False)
        balanced_indices_inter = np.random.choice(interictal_indices, min_count, replace=False)
        balanced_indices = np.concatenate([balanced_indices_pre, balanced_indices_inter])
        np.random.shuffle(balanced_indices)

        segments = segments[balanced_indices]
        labels = labels[balanced_indices]
        logging.info(f"After balancing: Total segments: {len(segments)}, Pre-ictal: {np.sum(labels)}, Interictal: {len(labels) - np.sum(labels)}")

    return segments, labels


def prepare_patient_data(patient_folder):
    """
    Loads, synchronizes, scales, and creates labeled segments for a single patient.

    Args:
        patient_folder (str): Path to the patient's data directory.

    Returns:
        tuple: (patient_id, segments, labels) or None if data processing fails.
    """
    patient_id = os.path.basename(patient_folder)
    logging.info(f"Processing data for patient: {patient_id}")

    sensor_data_dict = load_sensor_data_for_patient(patient_folder, SENSORS_TO_USE)
    if not sensor_data_dict:
        logging.warning(f"Skipping patient {patient_id}: Could not load sensor data.")
        return None

    annotations_df = load_seizure_annotations(patient_folder)

    synced_df = synchronize_and_merge_data(sensor_data_dict, target_freq_hz=SAMPLING_FREQ_HZ)
    if synced_df is None or synced_df.empty:
        logging.warning(f"Skipping patient {patient_id}: Could not synchronize or merge sensor data.")
        return None

    segments, labels = create_labeled_segments(
        synced_df,
        annotations_df,
        segment_duration_sec=SEGMENT_DURATION_SECONDS,
        pre_ictal_window_min=PRE_ICTAL_WINDOW_MINUTES,
        interictal_buffer_min=INTERICTAL_BUFFER_MINUTES,
        target_freq_hz=SAMPLING_FREQ_HZ
    )

    if len(segments) == 0:
         logging.warning(f"Skipping patient {patient_id}: No valid segments created.")
         return None

    return patient_id, segments, labels

# --- PyTorch Dataset ---

class SeizureDataset(Dataset):
    def __init__(self, segments, labels):
        """
        Args:
            segments (np.ndarray): Segments array (n_samples, seq_len, n_features).
            labels (np.ndarray): Labels array (n_samples,).
        """
        # PyTorch Conv1d expects (batch_size, channels, sequence_length)
        # Our segments are (n_samples, sequence_length, n_features)
        # Transpose to (n_samples, n_features, sequence_length)
        # Handle empty array case
        if segments.shape[0] == 0:
            self.segments = torch.empty(0, segments.shape[2], segments.shape[1], dtype=torch.float32)
            self.labels = torch.empty(0, 1, dtype=torch.float32)
        else:
            self.segments = torch.tensor(segments, dtype=torch.float32).permute(0, 2, 1)
            self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1) # BCELoss expects target shape (N, 1)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.segments[idx], self.labels[idx]

# --- PyTorch Model Definitions ---

class CNN_LSTM(nn.Module):
    def __init__(self, input_channels, seq_len):
        super(CNN_LSTM, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, CONV_FILTERS[0], kernel_size=CONV_KERNEL_SIZE, padding=CONV_KERNEL_SIZE // 2), # Use integer division for padding
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
        # Need a dummy tensor to pass through layers to get shape
        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, seq_len)
            dummy_output = self.conv_layers(dummy_input)
            self.lstm_input_features = dummy_output.shape[1] # This is CONV_FILTERS[-1]
            self.lstm_input_seq_len = dummy_output.shape[2] # This is the reduced sequence length

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=self.lstm_input_features,
                            hidden_size=LSTM_UNITS,
                            batch_first=True) # Set batch_first=True for convenience

        # Dense Layers
        self.dense_layers = nn.Sequential(
            nn.Linear(LSTM_UNITS, DENSE_UNITS),
            nn.Sigmoid(),
            nn.Linear(DENSE_UNITS, 1),
            nn.Sigmoid() # Output layer for binary classification
        )

    def forward(self, x):
        # x shape: (batch_size, channels, seq_len)

        # Pass through CNN layers
        cnn_out = self.conv_layers(x) # shape: (batch_size, filters, reduced_seq_len)

        # Permute for LSTM (batch_size, reduced_seq_len, filters)
        lstm_in = cnn_out.permute(0, 2, 1)

        # Pass through LSTM layer
        # lstm_out shape: (batch_size, reduced_seq_len, LSTM_UNITS)
        lstm_out, _ = self.lstm(lstm_in)

        # Take the output of the *last* time step for classification.
        last_timestep_out = lstm_out[:, -1, :] # shape: (batch_size, LSTM_UNITS)

        # Pass through Dense layers
        output = self.dense_layers(last_timestep_out) # shape: (batch_size, 1)

        return output


class CNN_BiLSTM(nn.Module):
    def __init__(self, input_channels, seq_len):
        super(CNN_BiLSTM, self).__init__()

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
        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, seq_len)
            dummy_output = self.conv_layers(dummy_input)
            self.lstm_input_features = dummy_output.shape[1] # This is CONV_FILTERS[-1]
            self.lstm_input_seq_len = dummy_output.shape[2] # This is the reduced sequence length


        # Bidirectional LSTM Layer
        self.bilstm = nn.LSTM(input_size=self.lstm_input_features,
                                hidden_size=LSTM_UNITS,
                                batch_first=True,
                                bidirectional=True)

        # Dense Layers
        # BiLSTM output size is hidden_size * 2
        self.dense_layers = nn.Sequential(
            nn.Linear(LSTM_UNITS * 2, DENSE_UNITS),
            nn.Sigmoid(),
            nn.Linear(DENSE_UNITS, 1),
            nn.Sigmoid() # Output layer
        )

    def forward(self, x):
        # x shape: (batch_size, channels, seq_len)

        # Pass through CNN layers
        cnn_out = self.conv_layers(x) # shape: (batch_size, filters, reduced_seq_len)

        # Permute for LSTM (batch_size, reduced_seq_len, filters)
        lstm_in = cnn_out.permute(0, 2, 1)

        # Pass through BiLSTM layer
        # bilstm_out shape: (batch_size, reduced_seq_len, LSTM_UNITS * 2)
        bilstm_out, _ = self.bilstm(lstm_in)

        # Take the output of the last time step for classification
        last_timestep_out = bilstm_out[:, -1, :] # shape: (batch_size, LSTM_UNITS * 2)

        # Pass through Dense layers
        output = self.dense_layers(last_timestep_out) # shape: (batch_size, 1)

        return output

# --- PyTorch Training and Evaluation ---

def train_one_epoch(model, dataloader, criterion, optimizer, device, class_weights=None):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    all_labels = []
    all_predictions = []

    # Wrap dataloader with tqdm for batch progress
    dataloader_tqdm = tqdm(dataloader, desc="Batch", leave=False)

    for inputs, labels in dataloader_tqdm:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs) # Outputs are probabilities

        if class_weights is not None:
            # Calculate weights per sample based on labels
            weight_tensor = torch.zeros_like(labels)
            weight_tensor[labels == 0] = class_weights[0] # Weight for class 0
            weight_tensor[labels == 1] = class_weights[1] # Weight for class 1
            loss = criterion(outputs, labels) # standard BCELoss
            loss = (loss * weight_tensor).mean() # Apply weights and take mean
        else:
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        predicted = (outputs > 0.5).float()
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

        # Update batch progress bar with current loss
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

    # Wrap dataloader with tqdm for batch progress during evaluation
    dataloader_tqdm = tqdm(dataloader, desc="Evaluating Batch", leave=False)

    with torch.no_grad():
        for inputs, labels in dataloader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs) # Outputs are probabilities

            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            predicted = (outputs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())

            # Update batch progress bar with current loss
            dataloader_tqdm.set_postfix(loss=loss.item())


    epoch_loss = running_loss / len(dataloader.dataset)

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)

    if len(all_labels) == 0:
        return {
            'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
            'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]
        }


    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    try:
        auc_roc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc_roc = 0.0
        if len(np.unique(all_labels)) < 2:
                logging.warning("Only one class present in evaluation set, AUC-ROC is undefined.")
                pass


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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=0.0001)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    # Wrap epoch loop with tqdm
    epoch_tqdm = tqdm(range(epochs), desc=desc, leave=True)

    for epoch in epoch_tqdm:
        start_time = time.time()

        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, DEVICE, class_weights)

        val_metrics = evaluate_pytorch_model(model, val_dataloader, criterion, DEVICE)
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']

        end_time = time.time()

        # Update epoch progress bar description
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
            # logging.info(f"EarlyStopping counter: {epochs_without_improvement} out of 5") # Log inside tqdm desc instead

        if epochs_without_improvement >= 5:
            logging.info(f"Early stopping triggered at epoch {epoch+1}.")
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
        # logging.info(f"Restored best model weights (Val Loss: {best_val_loss:.4f})") # Log inside tqdm completion or outside

    return model


def train_general_model_pytorch(all_patient_data, model_type='CNN-BiLSTM'):
    """
    Trains a general model on combined data from all patients using PyTorch.
    """
    logging.info("\n--- Training General Model (PyTorch) ---")

    if not all_patient_data:
        logging.warning("No patient data available for general training.")
        return None, {}, None

    all_segments = np.concatenate([data[1] for data in all_patient_data], axis=0)
    all_labels = np.concatenate([data[2] for data in all_patient_data], axis=0)

    logging.info(f"Combined data shape: {all_segments.shape}, Labels shape: {all_labels.shape}")
    logging.info(f"Combined data: Pre-ictal: {np.sum(all_labels)}, Interictal: {len(all_labels) - np.sum(all_labels)}")


    if len(np.unique(all_labels)) < 2:
        logging.warning("Combined data has only one class. Cannot train general model.")
        return None, {}, None

    classes = np.unique(all_labels)
    if len(classes) == 2:
        class_weights_np = class_weight.compute_class_weight(
            'balanced', classes=classes, y=all_labels
        )
        class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights_np)}
        logging.info(f"Computed general class weights: {class_weight_dict}")
    else:
        class_weights_dict = None
        logging.warning("Cannot compute class weights for general model (less than 2 classes).")


    X_train_general, X_temp, y_train_general, y_temp = train_test_split(
        all_segments, all_labels, test_size=0.4, random_state=SEED, stratify=all_labels
    )
    X_val_general, X_test_general, y_val_general, y_test_general = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )

    logging.info(f"General Training data shape: {X_train_general.shape}, Labels shape: {y_train_general.shape}")
    logging.info(f"General Validation data shape: {X_val_general.shape}, Labels shape: {y_val_general.shape}")
    logging.info(f"General Test data shape: {X_test_general.shape}, Labels shape: {y_test_general.shape}")

    train_dataset = SeizureDataset(X_train_general, y_train_general)
    val_dataset = SeizureDataset(X_val_general, y_val_general)
    test_dataset = SeizureDataset(X_test_general, y_test_general)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() // 2 or 1, persistent_workers=True if os.cpu_count() > 1 else False)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 2 or 1, persistent_workers=True if os.cpu_count() > 1 else False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 2 or 1, persistent_workers=True if os.cpu_count() > 1 else False)


    input_channels = all_segments.shape[2]
    seq_len = all_segments.shape[1]

    if model_type == 'CNN-LSTM':
        general_model = CNN_LSTM(input_channels, seq_len).to(DEVICE)
    elif model_type == 'CNN-BiLSTM':
        general_model = CNN_BiLSTM(input_channels, seq_len).to(DEVICE)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'CNN-LSTM' or 'CNN-BiLSTM'.")

    general_model_path = os.path.join(OUTPUT_DIR, 'general_best_model.pth')
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    general_model = train_pytorch_model(
        general_model,
        train_dataloader,
        val_dataloader,
        epochs=GENERAL_MODEL_EPOCHS,
        learning_rate=GENERAL_MODEL_LR,
        class_weights=class_weight_dict,
        save_best_model_path=general_model_path,
        desc="Training General Model"
    )

    logging.info("\nEvaluating general model on the combined test set...")
    general_metrics = evaluate_pytorch_model(general_model, test_dataloader, nn.BCELoss(), DEVICE)
    logging.info(f"General Model Metrics: Accuracy={general_metrics['accuracy']:.4f}, "
            f"Precision={general_metrics['precision']:.4f}, Recall={general_metrics['recall']:.4f}, "
            f"F1 Score={general_metrics['f1_score']:.4f}, AUC-ROC={general_metrics['auc_roc']:.4f}")
    logging.info(f"Confusion Matrix:\n{general_metrics['confusion_matrix']}")

    # Clean up memory
    del train_dataloader, val_dataloader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    # Return test dataloader for consistent input to personalization evaluation
    return general_model, general_metrics, test_dataloader


def perform_personalization_pytorch(general_model_state_dict, all_patient_data, model_type='CNN-BiLSTM'):
    """
    Performs personalization for each patient using transfer learning (fine-tuning) in PyTorch.
    """
    logging.info("\n--- Performing Personalization (PyTorch) ---")

    if general_model_state_dict is None:
        logging.warning("General model state dict is missing. Cannot perform personalization.")
        return {}

    personalization_results = {}

    # Wrap patient loop with tqdm
    patient_tqdm = tqdm(all_patient_data, desc="Personalizing Patients", leave=True)

    for patient_id, patient_segments, patient_labels in patient_tqdm:

        # Update patient progress bar description
        patient_tqdm.set_description(f"Personalizing Patient {patient_id}")


        if len(patient_segments) == 0:
            logging.warning(f"No data for patient {patient_id}, skipping personalization.")
            # Create dummy empty datasets/dataloaders for evaluation function to handle
            num_features = patient_segments.shape[2] if patient_segments.shape[0]>0 else len(SENSORS_TO_USE)
            segment_length_steps = patient_segments.shape[1] if patient_segments.shape[0]>0 else int(SEGMENT_DURATION_SECONDS*SAMPLING_FREQ_HZ)
            empty_dataset = SeizureDataset(np.array([]).reshape(0, segment_length_steps, num_features), np.array([]))
            empty_dataloader = DataLoader(empty_dataset, batch_size=BATCH_SIZE)

            # dummy_model = CNN_BiLSTM(num_features, segment_length_steps).to(DEVICE) # Assuming CNN-BiLSTM structure
            dummy_model_class = get_model_class(model_type)
            dummy_model = dummy_model_class(num_features, segment_length_steps).to(DEVICE)
            personalization_results[patient_id] = {"before": evaluate_pytorch_model(dummy_model, empty_dataloader, nn.BCELoss(), DEVICE),
                                                "after": evaluate_pytorch_model(dummy_model, empty_dataloader, nn.BCELoss(), DEVICE)}
            del dummy_model, empty_dataset, empty_dataloader
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue


        if len(np.unique(patient_labels)) < 2:
            logging.warning(f"Skipping personalization for patient {patient_id}: Only one class present in data.")
            pat_dataset = SeizureDataset(patient_segments, patient_labels)
            pat_dataloader = DataLoader(pat_dataset, batch_size=BATCH_SIZE)
            input_channels = patient_segments.shape[2]
            seq_len = patient_segments.shape[1]
            general_model_instance = get_model_class(model_type)(input_channels, seq_len).to(DEVICE)
            try:
                general_model_instance.load_state_dict(general_model_state_dict)
                metrics = evaluate_pytorch_model(general_model_instance, pat_dataloader, nn.BCELoss(), DEVICE)
            except Exception as e:
                logging.error(f"Error evaluating general model for patient {patient_id} with single class: {e}")
                metrics = {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc_roc': 0.0, 'confusion_matrix': [[0,0],[0,0]]}


            personalization_results[patient_id] = {"before": metrics, "after": metrics}
            del pat_dataset, pat_dataloader, general_model_instance
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue


        X_train_pat, X_temp_pat, y_train_pat, y_temp_pat = train_test_split(
            patient_segments, patient_labels, test_size=0.4, random_state=SEED, stratify=patient_labels
        )
        X_val_pat, X_test_pat, y_val_pat, y_test_pat = train_test_split(
            X_temp_pat, y_temp_pat, test_size=0.5, random_state=SEED, stratify=y_temp_pat
        )

        logging.info(f"Patient {patient_id} - Personalization Train shape: {X_train_pat.shape}, Val shape: {X_val_pat.shape}, Test shape: {X_test_pat.shape}")

        train_dataset_pat = SeizureDataset(X_train_pat, y_train_pat)
        val_dataset_pat = SeizureDataset(X_val_pat, y_val_pat)
        test_dataset_pat = SeizureDataset(X_test_pat, y_test_pat)

        train_dataloader_pat = DataLoader(train_dataset_pat, batch_size=PERSONALIZATION_BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() // 4 or 1, persistent_workers=True if os.cpu_count() > 1 else False)
        val_dataloader_pat = DataLoader(val_dataset_pat, batch_size=PERSONALIZATION_BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 4 or 1, persistent_workers=True if os.cpu_count() > 1 else False)
        test_dataloader_pat = DataLoader(test_dataset_pat, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 4 or 1, persistent_workers=True if os.cpu_count() > 1 else False)


        # Evaluate the general model on this patient's test data (Before Personalization)
        logging.info(f"Evaluating general model on patient {patient_id}'s test data (Before Personalization)...")
        input_channels = patient_segments.shape[2]
        seq_len = patient_segments.shape[1]
        # general_model_instance_eval = CNN_BiLSTM(input_channels, seq_len).to(DEVICE) # Assuming CNN-BiLSTM
        general_model_instance_eval = get_model_class(model_type)(input_channels, seq_len).to(DEVICE) # Used get_model_class
        general_model_instance_eval.load_state_dict(general_model_state_dict)

        metrics_before = evaluate_pytorch_model(general_model_instance_eval, test_dataloader_pat, nn.BCELoss(), DEVICE)
        logging.info(f"Before Personalization: Accuracy={metrics_before['accuracy']:.4f}, Precision={metrics_before['precision']:.4f}, Recall={metrics_before['recall']:.4f}, F1={metrics_before['f1_score']:.4f}, AUC={metrics_before['auc_roc']:.4f}")
        del general_model_instance_eval
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


        # Create a new model instance for personalization fine-tuning
        # personalized_model = CNN_BiLSTM(input_channels, seq_len).to(DEVICE) # Assuming CNN-BiLSTM
        personalized_model = get_model_class(model_type)(input_channels, seq_len).to(DEVICE) # Used get_model_class
        personalized_model.load_state_dict(general_model_state_dict)

        classes_pat = np.unique(y_train_pat)
        if len(classes_pat) == 2:
            class_weights_pat_np = class_weight.compute_class_weight(
                'balanced', classes=classes_pat, y=y_train_pat
            )
            class_weights_pat_dict = {cls: weight for cls, weight in zip(classes_pat, class_weights_pat_np)}
            logging.info(f"Computed patient {patient_id} class weights: {class_weights_pat_dict}")
        else:
            class_weights_pat_dict = None


        # Fine-tune the model on the patient's training data
        logging.info(f"Fine-tuning model on patient {patient_id}'s training data...")
        personalized_model_path = os.path.join(OUTPUT_DIR, f'{patient_id}_personalized_best_model_{model_type}.pth')
        personalized_model = train_pytorch_model(
            personalized_model,
            train_dataloader_pat,
            val_dataloader_pat,
            epochs=PERSONALIZATION_EPOCHS,
            learning_rate=PERSONALIZATION_LR,
            class_weights=class_weights_pat_dict,
            save_best_model_path=personalized_model_path,
            desc=f"Fine-tuning {patient_id}" # Specific description for patient
        )

        # Evaluate the personalized model on this patient's test data (After Personalization)
        logging.info(f"Evaluating personalized model on patient {patient_id}'s test data (After Personalization)...")
        metrics_after = evaluate_pytorch_model(personalized_model, test_dataloader_pat, nn.BCELoss(), DEVICE)
        logging.info(f"After Personalization: Accuracy={metrics_after['accuracy']:.4f}, Precision={metrics_after['precision']:.4f}, Recall={metrics_after['recall']:.4f}, F1={metrics_after['f1_score']:.4f}, AUC={metrics_after['auc_roc']:.4f}")
        logging.info(f"Confusion Matrix:\n{metrics_after['confusion_matrix']}")

        personalization_results[patient_id] = {
            "before": metrics_before,
            "after": metrics_after
        }

        # Clean up memory for the current patient's data/model
        del train_dataset_pat, val_dataset_pat, test_dataset_pat
        del train_dataloader_pat, val_dataloader_pat, test_dataloader_pat
        del personalized_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    return personalization_results



def print_personalization_summary(personalization_results, output_file=None):
    """ Prints a summary table of personalization results to console or file. """
    # Determine where to print (console or file)
    def print_func(*args, **kwargs):
        if output_file:
            logging.info(' '.join(map(str, args)), extra={'file': output_file})
        else:
            logging.info(' '.join(map(str, args)))

    logging.info("\n--- Personalized Model Performance (Per Patient Summary) ---")
    logging.info("Patient ID | Accuracy Before | Accuracy After | Change")
    logging.info("-----------------------------------------------------")
    total_change = 0
    count = 0
    for patient_id, results in personalization_results.items():
        # Ensure metrics are valid before printing/counting
        if 'accuracy' in results['before'] and 'accuracy' in results['after']:
            acc_before = results['before']['accuracy']
            acc_after = results['after']['accuracy']
            change = acc_after - acc_before
            logging.info(f"{patient_id:<10} | {acc_before:.4f}        | {acc_after:.4f}       | {change:.4f}")

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
                # else: logging.debug(f"--- Debug: Patient {patient_id} skipped average calculation due to empty test set/CM")


        else:
            logging.warning(f"{patient_id:<10} | N/A             | N/A            | N/A") # Indicate missing data


    logging.info("-----------------------------------------------------")
    if count > 0:
        average_change = total_change / count
        logging.info(f"Average Accuracy Improvement (across {count} patients with valid data): {average_change:.4f}")
    else:
        logging.warning("No valid personalized patient results to summarize average improvement.")


# --- Main Execution ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_patient_folders = [f.path for f in os.scandir(DATA_ROOT_DIR) if f.is_dir() and f.name.startswith('MSEL_')]

    if not all_patient_folders:
        logging.warning(f"No patient directories starting with 'MSEL_' found in {DATA_ROOT_DIR}.")
    else:
        logging.info(f"Found {len(all_patient_folders)} patient directories.")

    processed_patient_data = []
    # Data processing needs to happen only once for all models
    for patient_folder in tqdm(all_patient_folders, desc="Processing Patient Data"):
        patient_data = prepare_patient_data(patient_folder)
        if patient_data:
            processed_patient_data.append(patient_data)
            # Optional: Save processed data per patient (large files)
            patient_id, segments, labels = patient_data
            np.save(os.path.join(OUTPUT_DIR, f'{patient_id}_segments.npy'), segments)
            np.save(os.path.join(OUTPUT_DIR, f'{patient_id}_labels.npy'), labels)


    if not processed_patient_data:
        logging.warning("No valid patient data was processed. Exiting.")
    else:
        logging.info(f"\nSuccessfully processed data for {len(processed_patient_data)} patients.")

        # --- Define which models to run ---
        model_types_to_run = ['CNN-LSTM', 'CNN-BiLSTM'] # <--- List of models you want to run

        all_general_metrics = {}
        all_personalization_results = {} # Stores per-patient results
        avg_personalized_metrics = {} # Stores calculated averages for the final summary

        # --- Define the output file ---
        output_filename = os.path.join(OUTPUT_DIR, f'seizure_prediction_results_{time.strftime("%Y%m%d_%H%M%S")}.txt')

        # Use 'with' to ensure the file is closed automatically
        with open(output_filename, 'w') as output_file:

            # Write initial information to the file
            output_file.write(f"Seizure Prediction Results\n")
            output_file.write(f"Run Date: {time.ctime()}\n")
            output_file.write(f"Data Directory: {DATA_ROOT_DIR}\n")
            output_file.write(f"Processed Patients: {len(processed_patient_data)}\n")
            output_file.write(f"Models Run: {model_types_to_run}\n")
            output_file.write("=" * 50 + "\n\n")


            # # --- Loop through each model type and run the entire process ---
            # for current_model_type in model_types_to_run: # <--- The loop
            #     logging.info(f"\n{'='*30}")
            #     logging.info(f"Running process for Model Type: {current_model_type}")
            #     logging.info(f"{'='*30}\n")

            #     output_file.write(f"\n{'='*30}\n")
            #     output_file.write(f"RESULTS FOR MODEL TYPE: {current_model_type}\n")
            #     output_file.write(f"{'='*30}\n\n")


            #     # --- Step 1: Train the General Model ---
            #     general_model, general_metrics, general_test_dataloader = train_general_model_pytorch(
            #         processed_patient_data,
            #         model_type=current_model_type
            #     )

            #     all_general_metrics[current_model_type] = general_metrics

            #     # Write General Model Performance to file
            #     output_file.write("--- General Model Performance ---\n")
            #     if 'accuracy' in general_metrics: # Check if metrics are valid (training might fail)
            #         output_file.write(f"Accuracy={general_metrics['accuracy']:.4f}\n")
            #         output_file.write(f"Precision={general_metrics['precision']:.4f}\n")
            #         output_file.write(f"Recall={general_metrics['recall']:.4f}\n")
            #         output_file.write(f"F1 Score={general_metrics['f1_score']:.4f}\n")
            #         output_file.write(f"AUC-ROC={general_metrics['auc_roc']:.4f}\n")
            #         output_file.write(f"Confusion Matrix:\n{general_metrics['confusion_matrix']}\n")
            #     else:
            #         output_file.write("General model training failed or produced invalid metrics.\n")
            #     output_file.write("\n") # Add space


            #     if general_model is not None:
            #         # Get the state_dict for personalization later
            #         general_model_state = general_model.state_dict()

            #         # Clean up the general model instance to free up memory
            #         del general_model, general_test_dataloader
            #         gc.collect()
            #         if torch.cuda.is_available():
            #             torch.cuda.empty_cache()

            #         # --- Step 2: Perform Personalization ---
            #         personalization_results = perform_personalization_pytorch(
            #             general_model_state,
            #             processed_patient_data,
            #             model_type=current_model_type
            #         )

            #         all_personalization_results[current_model_type] = personalization_results # Store per-patient results

            #         # --- Step 3: Summarize Personalized Model Performance ---
            #         # Print per-patient summary to console AND write to file
            #         print_personalization_summary(personalization_results, output_file=output_file)
            #         print_personalization_summary(personalization_results, output_file=None) # Print to console too

            #         # Calculate and Write Average Personalized Model Performance
            #         metrics_after_list = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'auc_roc': []}
            #         count_valid_patients = 0

            #         for patient_id, results in personalization_results.items():
            #             # Check if the 'after' personalization metrics are valid for this patient
            #             if 'accuracy' in results.get('after', {}): # Use .get to handle missing 'after' key
            #                 cm = results['after'].get('confusion_matrix', [[0,0],[0,0]])
            #                 # Check if confusion matrix indicates samples were evaluated
            #                 # This line already seems correct in the code you provided for this block?
            #                 # Just double check you didn't revert it somewhere.
            #                 if len(cm) == 2 and len(cm[0]) == 2 and sum(sum(row) for row in cm) > 0:
            #                     count_valid_patients += 1
            #                     # Append metrics to lists for averaging
            #                     metrics_after_list['accuracy'].append(results['after']['accuracy'])
            #                     metrics_after_list['precision'].append(results['after']['precision'])
            #                     metrics_after_list['recall'].append(results['after']['recall'])
            #                     metrics_after_list['f1_score'].append(results['after']['f1_score'])
            #                     metrics_after_list['auc_roc'].append(results['after']['auc_roc'])
            #                 # else: logging.debug(f"--- Debug: Patient {patient_id} metrics skipped for average due to empty test set/CM")
            #             # else: logging.debug(f"--- Debug: Patient {patient_id} results['after'] is missing or invalid")


            #         output_file.write("\n--- Personalized Model Performance (Average Across Patients) ---\n")
            #         if count_valid_patients > 0:
            #             avg_metrics = {metric: np.mean(metrics_after_list[metric]) for metric in metrics_after_list}
            #             avg_personalized_metrics[current_model_type] = avg_metrics # Store for final comparison summary later

            #             output_file.write(f"Average Accuracy={avg_metrics['accuracy']:.4f} (across {count_valid_patients} patients)\n")
            #             output_file.write(f"Average Precision={avg_metrics['precision']:.4f}\n")
            #             output_file.write(f"Average Recall={avg_metrics['recall']:.4f}\n")
            #             output_file.write(f"Average F1 Score={avg_metrics['f1_score']:.4f}\n")
            #             output_file.write(f"Average AUC-ROC={avg_metrics['auc_roc']:.4f}\n")

            #         else:
            #             output_file.write("No valid personalized patient results to average.\n")
            #             avg_personalized_metrics[current_model_type] = None # Store None if no valid averages

            #         output_file.write("\n") # Add space before next model or final summary


            #     else:
            #         # Handle case where general model training failed
            #         logging.error(f"General model training failed for {current_model_type}. Skipping personalization.")
            #         output_file.write(f"General model training failed for {current_model_type}. Skipping personalization.\n\n")
            #         all_personalization_results[current_model_type] = {} # Store empty results
            #         avg_personalized_metrics[current_model_type] = None # Store None if no valid averages


            # --- Final comparison summary after all runs ---
            output_file.write(f"\n\n{'='*50}\n")
            output_file.write("OVERALL COMPARISON SUMMARY ACROSS MODEL TYPES\n")
            output_file.write(f"{'='*50}\n\n")

            output_file.write("General Model Performance Comparison:\n")
            output_file.write("Model Type | Accuracy | Precision | Recall | F1 Score | AUC-ROC\n")
            output_file.write("---------------------------------------------------------------\n")
            for model_type, metrics in all_general_metrics.items():
                if 'accuracy' in metrics: # Check if metrics are valid
                    output_file.write(f"{model_type:<10} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1_score']:.4f} | {metrics['auc_roc']:.4f}\n")
                else:
                    output_file.write(f"{model_type:<10} | N/A      | N/A       | N/A    | N/A      | N/A\n")
            output_file.write("---------------------------------------------------------------\n\n")


            output_file.write("Personalized Model Performance Comparison (Average Across Patients):\n")
            output_file.write("Model Type | Avg Accuracy | Avg Precision | Avg Recall | Avg F1 Score | Avg AUC-ROC\n")
            output_file.write("---------------------------------------------------------------------\n")
            for model_type, avg_metrics in avg_personalized_metrics.items():
                if avg_metrics is not None: # Check if average metrics were calculated
                    output_file.write(f"{model_type:<10} | {avg_metrics['accuracy']:.4f} | {avg_metrics['precision']:.4f} | {avg_metrics['recall']:.4f} | {avg_metrics['f1_score']:.4f} | {avg_metrics['auc_roc']:.4f}\n")
                else:
                    output_file.write(f"{model_type:<10} | N/A          | N/A           | N/A        | N/A          | N/A\n")
            output_file.write("---------------------------------------------------------------------\n\n")


        logging.info(f"\nResults saved to {output_filename}")