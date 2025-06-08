import concurrent.futures
import gc
import glob
import itertools
import logging
import math
import os
import pickle
import random
import sys
import time
from collections import OrderedDict

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
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# --- Configuration ---
# REMOVED: DATA_ROOT_DIR = "F:\\data_9" # No longer needed, data is pre-processed externally

# Directory to save results files and models
OUTPUT_DIR = "processed_data_pytorch_adaptive_pre_post_buffer_lovo_personalization_v2_enhanced"

# Ensure the base output directory exists early
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Feature Flags ---
RUN_ALL_MODEL_TYPES = True
ENABLE_ADAPTIVE_SENSORS = False
ENABLE_TUNABLE_HYPERPARAMETERS = True
ENABLE_PERSONALIZATION = False

# REMOVED: REUSE_PROCESSED_DATA = True # No longer needed, data is *always* loaded from external cache

# --- Data Processing Parameters (These values will now reflect what your external pipeline used) ---
# It's good practice to keep these as constants if they are fixed by your external pipeline,
# as they define the input shape expected by the models.
SEGMENT_DURATION_SECONDS = 30 # # Keep this value consistent with external preprocessing
# If your external preprocessing used a specific sampling frequency, hardcode it here
# This will be used for calculating sequence length for model initialization
EXTERNAL_PREPROCESS_SAMPLING_FREQ_HZ = 1 # # Assume 1Hz based on previous TUNABLE_SAMPLING_FREQ_HZ[0]

# REMOVED all TUNABLE_PRE_ICTAL_WINDOW_MINUTES etc.
# REMOVED FIXED_DATA_PROCESSING_CONFIG

# Define base sensors (ensure these are the possible columns after sync/scaling from your external pipeline)
BASE_SENSORS = ['HR', 'EDA', 'TEMP', 'ACC'] #

# --- Tunable Hyperparameters ---
# These lists define the values to iterate through if ENABLE_TUNABLE_HYPERPARAMETERS is True.
# If ENABLE_TUNABLE_HYPERPARAMETERS is False, only the first value from each list is used.
TUNABLE_CONV_FILTERS = [
    [16,32,64],
    [32, 64, 128],
    [64, 128, 256],
    [16,32,64,128],
    [32, 64, 128, 256],
    [16, 32, 64, 128, 256],
]

TUNABLE_CONV_KERNEL_SIZE = [
    10
]

TUNABLE_POOL_SIZE = [2]

TUNABLE_LSTM_UNITS = [
    64,
    128
]

TUNABLE_GRU_UNITS = [
    64
]
TUNABLE_DENSE_UNITS = [
    32,
    64,
    128,
]

TUNABLE_GENERAL_MODEL_EPOCHS = [
    200
]
TUNABLE_PERSONALIZATION_EPOCHS = [30]

TUNABLE_GENERAL_MODEL_LR = [0.00001,0.00005,0.0001,
                            0.0005,0.001,0.005]

TUNABLE_PERSONALIZATION_LR = [
    0.0001,
]
TUNABLE_BATCH_SIZE = [
    64
]
TUNABLE_PERSONALIZATION_BATCH_SIZE = [16]

TUNABLE_DROPOUT_RATE = [0]

TUNABLE_WEIGHT_DECAY_GENERAL = [
    0,
]

TUNABLE_WEIGHT_DECAY_PERSONALIZATION = [
    0
]

TUNABLE_TRANSFORMER_NHEAD = [
    8
]
TUNABLE_TRANSFORMER_NLAYERS = [
    4
]
TUNABLE_TRANSFORMER_DIM_FEEDFORWARD = [
    256
]

TUNABLE_TCN_NUM_CHANNELS = [
    [32, 32, 32],
    [64, 64, 64, 64]
]
TUNABLE_TCN_KERNEL_SIZE = [
    3,
    5
]

TUNABLE_DENSENET_GROWTH_RATE = [
    16,
    32,
]
TUNABLE_DENSENET_BLOCK_CONFIG = [
    (6, 12, 24),
]
TUNABLE_DENSENET_BN_SIZE = [
    4,
    8,
]

TUNABLE_RESNET_BLOCK_TYPE = ['BasicBlock']
TUNABLE_RESNET_LAYERS = [[2, 2, 2, 2], [3, 4, 6, 3]]
TUNABLE_RESNET_LSTM_HIDDEN_SIZE = [64, 128]
TUNABLE_RESNET_LSTM_NUM_LAYERS = [1, 2]
TUNABLE_RESNET_LSTM_DROPOUT = [0.2, 0.3]

MODEL_TYPES_TO_RUN = ["CNN-GRU"]

ALL_SENSOR_COMBINATIONS = []
for i in range(1, len(BASE_SENSORS) + 1):
    for combo in itertools.combinations(BASE_SENSORS, i):
        ALL_SENSOR_COMBINATIONS.append(list(combo))

# REMOVED: MAX_PATIENTS_TO_INCLUDE = 10 # Handled by external preprocessing
# REMOVED: PATIENTS_TO_EXCLUDE = [] # Handled by external preprocessing

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# REMOVED: get_data_config_identifier function and other data loading/preprocessing functions

# --- PyTorch Dataset ---
class SeizureDataset(Dataset):
    def __init__(self, segments, labels, seq_len, num_features):
        if segments.shape[0] == 0:
            self.segments = torch.empty(0, num_features, seq_len, dtype=torch.float32)
            self.labels = torch.empty(0, 1, dtype=torch.float32)
        else:
            if segments.ndim == 2:
                segments = segments[:, :, np.newaxis]
                if num_features != 1:
                    logging.warning(
                        f"Warning: Segments ndim=2 but expected num_features={num_features}. Assuming 1 feature."
                    )
                    num_features = 1
            elif segments.ndim < 2:
                logging.warning(
                    f"Warning: Segments array has unexpected ndim={segments.ndim}. Cannot create dataset."
                )
                self.segments = torch.empty(
                    0, num_features, seq_len, dtype=torch.float32
                )
                self.labels = torch.empty(0, 1, dtype=torch.float32)
                return

            if segments.shape[2] != num_features:
                logging.warning(
                    f"Warning: Segment features ({segments.shape[2]}) mismatch expected features ({num_features}). Cannot create dataset."
                )
                self.segments = torch.empty(
                    0, num_features, seq_len, dtype=torch.float32
                )
                self.labels = torch.empty(0, 1, dtype=torch.float32)
                return

            if segments.shape[1] != seq_len:
                logging.warning(
                    f"Warning: Segment length ({segments.shape[1]}) mismatch expected length ({seq_len}). Cannot create dataset."
                )
                self.segments = torch.empty(
                    0, num_features, seq_len, dtype=torch.float32
                )
                self.labels = torch.empty(0, 1, dtype=torch.float32)
                return

            self.segments = torch.tensor(segments, dtype=torch.float32).permute(
                0, 2, 1
            )
            self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(
                1
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.segments[idx], self.labels[idx]

# --- PyTorch Model Definitions (Keep all model definitions as they are) ---
class LSTM_Only(nn.Module):
    # ... (same as original) ...
    pass

class CNN_LSTM(nn.Module):
    # ... (same as original) ...
    pass

class CNN_BiLSTM(nn.Module):
    # ... (same as original) ...
    pass

class CNN_GRU(nn.Module):
    # ... (same as original) ...
    pass

class CNN_Transformer(nn.Module):
    # ... (same as original) ...
    pass

class CNN_TCN(nn.Module):
    # ... (same as original) ...
    pass

class DenseNet_LSTM(nn.Module):
    # ... (same as original) ...
    pass

class DenseNet_BiLSTM(nn.Module):
    # ... (same as original) ...
    pass

class ResNet_LSTM(nn.Module):
    # ... (same as original) ...
    pass

class ResNet_BiLSTM(nn.Module):
    # ... (same as original) ...
    pass

# ===================== CNN-TCN Helper Functions ===========================
class TemporalBlock(nn.Module):
    # ... (same as original) ...
    pass

class Chomp1d(nn.Module):
    # ... (same as original) ...
    pass

class TCN(nn.Module):
    # ... (same as original) ...
    pass

# ===================== DenseNet Helper Functions and Blocks ===========================
class DenseLayer(nn.Module):
    # ... (same as original) ...
    pass

class DenseBlock(nn.Module):
    # ... (same as original) ...
    pass

class TransitionLayer(nn.Module):
    # ... (same as original) ...
    pass

class DenseNet(nn.Module):
    # ... (same as original) ...
    pass

# ===================== ResNet Helper Functions and Blocks ===========================
class BasicBlock1d(nn.Module):
    # ... (same as original) ...
    pass

class ResNet1d(nn.Module):
    # ... (same as original) ...
    pass

def get_model_class(model_type):
    # ... (same as original) ...
    pass

# --- PyTorch Training and Evaluation (Keep all as they are) ---
def calculate_metrics(all_labels, all_predictions, all_probs):
    # ... (same as original) ...
    pass

def train_one_epoch(model, train_dataloader, criterion, optimizer, device, class_weights=None):
    # ... (same as original) ...
    pass

def evaluate_pytorch_model(model, dataloader, criterion, device):
    # ... (same as original) ...
    pass

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
    # ... (same as original) ...
    pass

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
    # ... (same as original, but ensure EXTERNAL_PREPROCESS_SAMPLING_FREQ_HZ is used where appropriate for seq_len calculation) ...
    logging.info(f"--- Training LOPO General Model (Excluding {excluded_patient_id}) for {model_type} ---")

    lopo_segments_raw = []
    lopo_labels_raw = []

    # Collect data from all patients EXCEPT the excluded one
    for patient_data_tuple in all_processed_patient_data:
        patient_id, segments_all_sensors, labels, found_sensors = patient_data_tuple
        if patient_id != excluded_patient_id:
            if (
                len(segments_all_sensors) > 0
                and len(sensor_combination_indices) > 0
                and segments_all_sensors.shape[2] == len(BASE_SENSORS)
            ):
                segments_sliced = segments_all_sensors[:, :, sensor_combination_indices]
                lopo_segments_raw.append(segments_sliced)
                lopo_labels_raw.append(labels)

    if not lopo_segments_raw:
        logging.warning(
            f"Warning: No data available from other patients for LOPO general training (Excluding {excluded_patient_id})."
        )
        return None, None

    lopo_segments_combined = np.concatenate(lopo_segments_raw, axis=0)
    lopo_labels_combined = np.concatenate(lopo_labels_raw, axis=0)

    if len(lopo_segments_combined) < 3 or len(np.unique(lopo_labels_combined)) < 2:
        logging.warning(
            f"Warning: Insufficient data ({len(lopo_segments_combined)} samples) or only one class ({np.unique(lopo_labels_combined)}) for LOPO general training split (Excluding {excluded_patient_id}). Skipping training."
        )
        return None, None

    try:
        X_train_lopo, X_temp_lopo, y_train_lopo, y_temp_lopo = train_test_split(
            lopo_segments_combined,
            lopo_labels_combined,
            test_size=0.4,
            random_state=SEED,
            stratify=lopo_labels_combined,
        )
        X_val_lopo, X_test_lopo, y_val_lopo, y_test_lopo = train_test_split(
            X_temp_lopo,
            y_temp_lopo,
            test_size=0.5,
            random_state=SEED,
            stratify=y_temp_lopo,
        )

        num_samples_train_lopo = X_train_lopo.shape[0]
        seq_len_train_lopo = X_train_lopo.shape[1]
        num_features_lopo = X_train_lopo.shape[2]

        num_samples_val_lopo = X_val_lopo.shape[0]
        seq_len_val_lopo = X_val_lopo.shape[1]

        num_samples_test_lopo = X_test_lopo.shape[0]
        seq_len_test_lopo = X_test_lopo.shape[1]


        if num_samples_train_lopo > 0 and num_samples_val_lopo > 0 and num_samples_test_lopo > 0:
            X_train_lopo_reshaped = X_train_lopo.reshape(-1, num_features_lopo)
            X_val_lopo_reshaped = X_val_lopo.reshape(-1, num_features_lopo)
            X_test_lopo_reshaped = X_test_lopo.reshape(-1, num_features_lopo)

            scaler_lopo = RobustScaler()
            scaler_lopo.fit(X_train_lopo_reshaped)

            X_train_lopo_scaled = scaler_lopo.transform(X_train_lopo_reshaped)
            X_val_lopo_scaled = scaler_lopo.transform(X_val_lopo_reshaped)
            X_test_lopo_scaled = scaler_lopo.transform(X_test_lopo_reshaped)

            X_train_lopo = X_train_lopo_scaled.reshape(num_samples_train_lopo, seq_len_train_lopo, num_features_lopo)
            X_val_lopo = X_val_lopo_scaled.reshape(num_samples_val_lopo, seq_len_val_lopo, num_features_lopo)
            X_test_lopo = X_test_lopo_scaled.reshape(num_samples_test_lopo, seq_len_test_lopo, num_features_lopo)

            logging.info(f"Applied RobustScaler to LOPO General data splits (Excluding {excluded_patient_id}, {model_type}, {current_hp_combo_str}).")
        else:
            logging.warning(f"One or more LOPO General data splits are empty after splitting. Skipping RobustScaler. (Excluding {excluded_patient_id}, {model_type}, {current_hp_combo_str})")

    except ValueError as e:
        logging.warning(
            f"Warning: LOPO data split failed for patient {excluded_patient_id}: {e}. This might happen with very few samples or imbalanced small datasets leading to single-class splits. Skipping training."
        )
        return None, None
    except Exception as e:
        logging.error(
            f"An unexpected error occurred during LOPO general data split for patient {excluded_patient_id}: {e}. Skipping training."
        )
        return None, None

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
        )
        return None, None

    # Calculate expected seq_len and num_features for the dataset
    # This should now explicitly use the EXTERNAL_PREPROCESS_SAMPLING_FREQ_HZ
    expected_seq_len = int(SEGMENT_DURATION_SECONDS * EXTERNAL_PREPROCESS_SAMPLING_FREQ_HZ) #
    expected_num_features = len(BASE_SENSORS) # Assumes BASE_SENSORS reflect external data features

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
    )

    num_workers = 0
    persistent_workers = False

    general_train_batch_size = general_hyperparameters["batch_size"]
    general_learning_rate = general_hyperparameters["learning_rate"]
    general_epochs = general_hyperparameters["epochs"]

    train_batch_size = general_train_batch_size
    if len(lopo_train_dataset) > 0:
        train_batch_size = max(1, min(train_batch_size, len(lopo_train_dataset)))
    val_batch_size = general_train_batch_size
    if len(lopo_val_dataset) > 0:
        val_batch_size = max(1, min(val_batch_size, len(lopo_val_dataset)))
    test_batch_size = general_train_batch_size
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
    )

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
            )

    input_channels = lopo_segments_combined.shape[2]
    seq_len = lopo_segments_combined.shape[1]
    ModelClass = get_model_class(model_type)

    try:
        # Model instantiation logic (same as original, uses model_hyperparameters)
        if model_type in ["CNN-LSTM", "CNN-BiLSTM"]:
            lstm_units = model_hyperparameters["lstm_units"]
            lopo_general_model = ModelClass(
                input_channels=input_channels,
                seq_len=seq_len,
                conv_filters=model_hyperparameters["conv_filters"],
                conv_kernel_size=model_hyperparameters["conv_kernel_size"],
                pool_size=model_hyperparameters["pool_size"],
                lstm_units=lstm_units,
                dense_units=model_hyperparameters["dense_units"],
                dropout_rate=model_hyperparameters["dropout_rate"],
            ).to(device)
        elif model_type == "CNN-GRU":
            gru_units = model_hyperparameters["gru_units"]
            lopo_general_model = ModelClass(
                input_channels=input_channels,
                seq_len=seq_len,
                conv_filters=model_hyperparameters["conv_filters"],
                conv_kernel_size=model_hyperparameters["conv_kernel_size"],
                pool_size=model_hyperparameters["pool_size"],
                gru_units=gru_units,
                dense_units=model_hyperparameters["dense_units"],
                dropout_rate=model_hyperparameters["dropout_rate"],
            ).to(device)
        elif model_type == "CNN-Transformer":
            transformer_nhead = model_hyperparameters["transformer_nhead"]
            transformer_nlayers = model_hyperparameters["transformer_nlayers"]
            transformer_dim_feedforward = model_hyperparameters["transformer_dim_feedforward"]

            lopo_general_model = ModelClass(
                input_channels=input_channels,
                seq_len=seq_len,
                conv_filters=model_hyperparameters["conv_filters"],
                conv_kernel_size=model_hyperparameters["conv_kernel_size"],
                pool_size=model_hyperparameters["pool_size"],
                transformer_nhead=transformer_nhead,
                transformer_nlayers=transformer_nlayers,
                transformer_dim_feedforward=transformer_dim_feedforward,
                dense_units=model_hyperparameters["dense_units"],
                dropout_rate=model_hyperparameters["dropout_rate"],
            ).to(device)
        elif model_type == "CNN-TCN":
            tcn_num_channels = model_hyperparameters["tcn_num_channels"]
            tcn_kernel_size = model_hyperparameters["tcn_kernel_size"]

            lopo_general_model = ModelClass_og( # Note: ModelClass_og needs to be defined or replaced with ModelClass
                input_channels=input_channels,
                seq_len=seq_len,
                conv_filters=model_hyperparameters["conv_filters"],
                conv_kernel_size=model_hyperparameters["conv_kernel_size"],
                pool_size=model_hyperparameters["pool_size"],
                tcn_num_channels=tcn_num_channels,
                tcn_kernel_size=tcn_kernel_size,
                dense_units=model_hyperparameters["dense_units"],
                dropout_rate=model_hyperparameters["dropout_rate"],
            ).to(device)
        elif model_type == "DenseNet-LSTM":
            lopo_general_model = ModelClass(
                input_channels=input_channels,
                seq_len=seq_len,
                densenet_growth_rate=model_hyperparameters["densenet_growth_rate"],
                densenet_block_config=model_hyperparameters["densenet_block_config"],
                densenet_bn_size=model_hyperparameters["densenet_bn_size"],
                densenet_pool_size=model_hyperparameters["pool_size"],
                densenet_kernel_size=model_hyperparameters["conv_kernel_size"],
                lstm_units=model_hyperparameters["lstm_units"],
                dense_units=model_hyperparameters["dense_units"],
                dropout_rate=model_hyperparameters["dropout_rate"],
            ).to(device)
        elif model_type == "DenseNet-BiLSTM":
            lopo_general_model = ModelClass(
                input_channels=input_channels,
                seq_len=seq_len,
                densenet_growth_rate=model_hyperparameters["densenet_growth_rate"],
                densenet_block_config=model_hyperparameters["densenet_block_config"],
                densenet_bn_size=model_hyperparameters["densenet_bn_size"],
                densenet_pool_size=model_hyperparameters["pool_size"],
                densenet_kernel_size=model_hyperparameters["conv_kernel_size"],
                lstm_units=model_hyperparameters["lstm_units"],
                dense_units=model_hyperparameters["dense_units"],
                dropout_rate=model_hyperparameters["dropout_rate"],
            ).to(device)
        elif model_type == "ResNet-LSTM":
            resnet_block_type = model_hyperparameters["resnet_block_type"]
            resnet_layers = model_hyperparameters["resnet_layers"]
            lstm_hidden_size = model_hyperparameters["lstm_hidden_size"]
            lstm_num_layers = model_hyperparameters["lstm_num_layers"]
            lstm_dropout = model_hyperparameters["lstm_dropout"]

            lopo_general_model = ModelClass(
                input_channels=input_channels,
                resnet_block_type=resnet_block_type,
                resnet_layers=resnet_layers,
                lstm_hidden_size=lstm_hidden_size,
                lstm_num_layers=lstm_num_layers,
                lstm_dropout=lstm_dropout,
                num_classes=1
            ).to(device)
        elif model_type == "ResNet-BiLSTM":
            resnet_block_type = model_hyperparameters["resnet_block_type"]
            resnet_layers = model_hyperparameters["resnet_layers"]
            lstm_hidden_size = model_hyperparameters["lstm_hidden_size"]
            lstm_num_layers = model_hyperparameters["lstm_num_layers"]
            lstm_dropout = model_hyperparameters["lstm_dropout"]

            lopo_general_model = ModelClass(
                input_channels=input_channels,
                resnet_block_type=resnet_block_type,
                resnet_layers=resnet_layers,
                lstm_hidden_size=lstm_hidden_size,
                lstm_num_layers=lstm_num_layers,
                lstm_dropout=lstm_dropout,
                num_classes=1
            ).to(device)
        elif model_type == "LSTM":
            lstm_units = model_hyperparameters["lstm_units"]
            lopo_general_model = ModelClass(
                input_features=input_channels,
                seq_len=seq_len,
                lstm_units=lstm_units,
                dense_units=model_hyperparameters["dense_units"],
                dropout_rate=model_hyperparameters["dropout_rate"],
            ).to(device)
        else:
            raise ValueError(f"Unknown model type for instantiation: {model_type}")

    except (ValueError, Exception) as e:
        logging.error(
            f"Error instantiating LOPO general model for {excluded_patient_id} ({model_type}, {current_hp_combo_str}): {e}. Skipping training."
        )
        del (
            lopo_train_dataloader,
            lopo_val_dataloader,
            lopo_test_dataloader,
        )
        del (
            lopo_train_dataset,
            lopo_val_dataset,
            lopo_test_dataset,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None, None

    lopo_model_save_dir = os.path.join(
        OUTPUT_DIR, model_type, "lopo_general"
    )
    lopo_model_save_path = os.path.join(
        lopo_model_save_dir, f"excl_{excluded_patient_id}.pth"
    )

    logging.info(
        f"Starting LOPO General Model training (Excluding {excluded_patient_id}) for {model_type} ({current_hp_combo_str})..."
    )
    lopo_general_model, lopo_general_metrics = train_pytorch_model(
        lopo_general_model,
        lopo_train_dataloader,
        lopo_val_dataloader,
        lopo_test_dataloader,
        epochs=general_epochs,
        learning_rate=general_learning_rate,
        class_weights=class_weights_lopo_dict,
        save_best_model_path=lopo_model_save_path,
        desc=f"Training LOPO General (Excl {excluded_patient_id})",
        device=device,
        weight_decay=general_hyperparameters["weight_decay"],
    )

    lopo_model_plot_dir = os.path.join(OUTPUT_DIR, model_type, "lopo_general", excluded_patient_id, "plots")
    os.makedirs(lopo_model_plot_dir, exist_ok=True)

    if 'history' in lopo_general_metrics:
        plot_training_history(
            lopo_general_metrics['history'],
            f'LOPO General Model (Excl {excluded_patient_id}, {model_type}, HP Combo {current_hp_combo_str})',
            lopo_model_plot_dir,
            f'excl_{excluded_patient_id}_lopo_general'
        )

    lopo_general_model_state = lopo_general_model.state_dict()
    del (
        lopo_general_model,
        lopo_train_dataloader,
        lopo_val_dataloader,
        lopo_test_dataloader,
    )
    del lopo_train_dataset, lopo_val_dataset, lopo_test_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return lopo_general_model_state, lopo_general_metrics

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
    run_specific_output_dir # <--- NEW PARAMETER: This is the folder like '20250601_092220/CNN-GRU/HR_EDA_TEMP_ACC/hp_combo_1'
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
    logging.info(f"Starting personalization for patient {current_patient_id} ({model_type}, {combo_name}, {current_hp_combo_str})")

    lopo_general_model_state_dict, lopo_general_metrics = train_lopo_general_model(
        all_processed_patient_data,
        current_patient_id,
        model_type,
        sensor_combination_indices,
        model_hyperparameters,
        general_hyperparameters,
        current_hp_combo_str,
        device,
    )

    if lopo_general_model_state_dict is None:
        logging.warning(
            f"Skipping personalization for patient {current_patient_id} ({model_type}, {combo_name}, {current_hp_combo_str}): LOPO general model training failed."
        )
        return (current_patient_id, None)

    if (
        len(current_patient_segments_all_sensors) > 0
        and len(sensor_combination_indices) > 0 # Use the length of the actual indices list
    ):
        if current_patient_segments_all_sensors.shape[2] == len(BASE_SENSORS):
            current_patient_segments_sliced = current_patient_segments_all_sensors[
                :, :, sensor_combination_indices
            ]
        else:
            logging.error(
                f"Error: Patient {current_patient_id} segments_all_sensors has unexpected feature count ({current_patient_segments_all_sensors.shape[2]}). Expected {len(BASE_SENSORS)}. Skipping."
            )
            return (current_patient_id, None)
    else:
        logging.warning(
            f"Skipping patient {current_patient_id} ({model_type}, {combo_name}, {current_hp_combo_str}): No segments or no features after slicing."
        )
        return (current_patient_id, None)

    if (
        len(current_patient_segments_sliced) < 3
        or len(np.unique(current_patient_labels)) < 2
    ):
        logging.warning(
            f"Warning: Insufficient data ({len(current_patient_segments_sliced)} samples) or only one class ({np.unique(current_patient_labels)}) for patient {current_patient_id} personalization splits ({model_type}, {combo_name}, {current_hp_combo_str}). Skipping."
        )
        return (current_patient_id, None)

    try:
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
            stratify=y_temp_pat,
        )

        num_samples_train_pat = X_train_pat.shape[0]
        seq_len_train_pat = X_train_pat.shape[1]
        num_features_pat = X_train_pat.shape[2]

        num_samples_val_pat = X_val_pat.shape[0]
        seq_len_val_pat = X_val_pat.shape[1]

        num_samples_test_pat = X_test_pat.shape[0]
        seq_len_test_pat = X_test_pat.shape[1]

        if num_samples_train_pat > 0 and num_samples_val_pat > 0 and num_samples_test_pat > 0:
            X_train_pat_reshaped = X_train_pat.reshape(-1, num_features_pat)
            X_val_pat_reshaped = X_val_pat.reshape(-1, num_features_pat)
            X_test_pat_reshaped = X_test_pat.reshape(-1, num_features_pat)

            scaler_pat = RobustScaler()
            scaler_pat.fit(X_train_pat_reshaped)

            X_train_pat_scaled = scaler_pat.transform(X_train_pat_reshaped)
            X_val_pat_scaled = scaler_pat.transform(X_val_pat_reshaped)
            X_test_pat_scaled = scaler_pat.transform(X_test_pat_reshaped)

            X_train_pat = X_train_pat_scaled.reshape(num_samples_train_pat, seq_len_train_pat, num_features_pat)
            X_val_pat = X_val_pat_scaled.reshape(num_samples_val_pat, seq_len_val_pat, num_features_pat)
            X_test_pat = X_test_pat_scaled.reshape(num_samples_test_pat, seq_len_test_pat, num_features_pat)

            logging.info(f"Applied RobustScaler to patient {current_patient_id}'s personalization data splits ({model_type}, {combo_name}, {current_hp_combo_str}).")
        else:
            logging.warning(f"One or more personalization data splits for patient {current_patient_id} are empty after splitting. Skipping RobustScaler. ({model_type}, {combo_name}, {current_hp_combo_str})")

    except ValueError as e:
        logging.warning(
            f"Warning: Patient {current_patient_id} data split failed ({model_type}, {combo_name}, {current_hp_combo_str}): {e}. This might happen with very few samples or imbalanced small datasets leading to single-class splits. Skipping personalization."
        )
        return (current_patient_id, None)
    except Exception as e:
        logging.error(
            f"An unexpected error occurred during patient {current_patient_id} data split ({model_type}, {combo_name}, {current_hp_combo_str}): {e}. Skipping personalization."
        )
        return (current_patient_id, None)

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
        )
        return (current_patient_id, None)

    # Use the hardcoded external preprocessing sampling frequency
    # This ensures consistency with how your external data was created
    expected_seq_len_sliced = int(SEGMENT_DURATION_SECONDS * EXTERNAL_PREPROCESS_SAMPLING_FREQ_HZ)
    expected_num_features_sliced = len(sensor_combination_indices) # This uses the current number of selected sensors

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

    num_workers_pat = 0
    persistent_workers_pat = False

    personalization_train_batch_size = personalization_hyperparameters["batch_size"]
    personalization_val_batch_size = personalization_hyperparameters["batch_size"]
    personalized_test_batch_size = general_hyperparameters["batch_size"]

    if len(train_dataset_pat) > 0:
        personalization_train_batch_size = max(
            1, min(personalization_train_batch_size, len(train_dataset_pat))
        )
    if len(val_dataset_pat) > 0:
        personalization_val_batch_size = max(
            1, min(personalization_val_batch_size, len(val_dataset_pat))
        )
    if len(test_dataset_pat) > 0:
        personalized_test_batch_size = max(
            1, min(personalized_test_batch_size, len(test_dataset_pat))
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
        )
    except Exception as e:
        logging.error(
            f"Error creating patient {current_patient_id} dataloaders ({model_type}, {combo_name}, {current_hp_combo_str}): {e}. Skipping personalization."
        )
        del train_dataset_pat, val_dataset_pat, test_dataset_pat
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (current_patient_id, None)

    logging.info(f"Evaluating LOPO general model on patient {current_patient_id}'s test data (Before Personalization)...")
    ModelClass = get_model_class(model_type)

    try:
        # Model instantiation for evaluation (same as original, uses model_hyperparameters)
        if model_type in ["CNN-LSTM", "CNN-BiLSTM"]:
            lstm_units = model_hyperparameters["lstm_units"]
            lopo_general_model_instance_eval = ModelClass(
                input_channels=expected_num_features_sliced,
                seq_len=expected_seq_len_sliced,
                conv_filters=model_hyperparameters["conv_filters"],
                conv_kernel_size=model_hyperparameters["conv_kernel_size"],
                pool_size=model_hyperparameters["pool_size"],
                lstm_units=lstm_units,
                dense_units=model_hyperparameters["dense_units"],
                dropout_rate=model_hyperparameters["dropout_rate"],
            ).to(device)
        elif model_type == "CNN-GRU":
            gru_units = model_hyperparameters["gru_units"]
            lopo_general_model_instance_eval = ModelClass(
                input_channels=expected_num_features_sliced,
                seq_len=expected_seq_len_sliced,
                conv_filters=model_hyperparameters["conv_filters"],
                conv_kernel_size=model_hyperparameters["conv_kernel_size"],
                pool_size=model_hyperparameters["pool_size"],
                gru_units=gru_units,
                dense_units=model_hyperparameters["dense_units"],
                dropout_rate=model_hyperparameters["dropout_rate"],
            ).to(device)
        elif model_type == "CNN-Transformer":
            transformer_nhead = model_hyperparameters["transformer_nhead"]
            transformer_nlayers = model_hyperparameters["transformer_nlayers"]
            transformer_dim_feedforward = model_hyperparameters["transformer_dim_feedforward"]

            lopo_general_model_instance_eval = ModelClass(
                input_channels=expected_num_features_sliced,
                seq_len=expected_seq_len_sliced,
                conv_filters=model_hyperparameters["conv_filters"],
                conv_kernel_size=model_hyperparameters["conv_kernel_size"],
                pool_size=model_hyperparameters["pool_size"],
                transformer_nhead=transformer_nhead,
                transformer_nlayers=transformer_nlayers,
                transformer_dim_feedforward=transformer_dim_feedforward,
                dense_units=model_hyperparameters["dense_units"],
                dropout_rate=model_hyperparameters["dropout_rate"],
            ).to(device)
        elif model_type == "CNN-TCN":
            tcn_num_channels = model_hyperparameters["tcn_num_channels"]
            tcn_kernel_size = model_hyperparameters["tcn_kernel_size"]

            lopo_general_model_instance_eval = ModelClass(
                input_channels=expected_num_features_sliced,
                seq_len=expected_seq_len_sliced,
                conv_filters=model_hyperparameters["conv_filters"],
                conv_kernel_size=model_hyperparameters["conv_kernel_size"],
                pool_size=model_hyperparameters["pool_size"],
                tcn_num_channels=tcn_num_channels,
                tcn_kernel_size=tcn_kernel_size,
                dense_units=model_hyperparameters["dense_units"],
                dropout_rate=model_hyperparameters["dropout_rate"],
            ).to(device)
        elif model_type == "DenseNet-LSTM":
            lopo_general_model_instance_eval = ModelClass(
                input_channels=expected_num_features_sliced,
                seq_len=expected_seq_len_sliced,
                densenet_growth_rate=model_hyperparameters["densenet_growth_rate"],
                densenet_block_config=model_hyperparameters["densenet_block_config"],
                densenet_bn_size=model_hyperparameters["densenet_bn_size"],
                densenet_pool_size=model_hyperparameters["pool_size"],
                densenet_kernel_size=model_hyperparameters["conv_kernel_size"],
                lstm_units=model_hyperparameters["lstm_units"],
                dense_units=model_hyperparameters["dense_units"],
                dropout_rate=model_hyperparameters["dropout_rate"],
            ).to(device)
        elif model_type == "DenseNet-BiLSTM":
            lopo_general_model_instance_eval = ModelClass(
                input_channels=expected_num_features_sliced,
                seq_len=expected_seq_len_sliced,
                densenet_growth_rate=model_hyperparameters["densenet_growth_rate"],
                densenet_block_config=model_hyperparameters["densenet_block_config"],
                densenet_bn_size=model_hyperparameters["densenet_bn_size"],
                densenet_pool_size=model_hyperparameters["pool_size"],
                densenet_kernel_size=model_hyperparameters["conv_kernel_size"],
                lstm_units=model_hyperparameters["lstm_units"],
                dense_units=model_hyperparameters["dense_units"],
                dropout_rate=model_hyperparameters["dropout_rate"],
            ).to(device)
        elif model_type == "ResNet-LSTM":
            resnet_block_type = model_hyperparameters["resnet_block_type"]
            resnet_layers = model_hyperparameters["resnet_layers"]
            lstm_hidden_size = model_hyperparameters["lstm_hidden_size"]
            lstm_num_layers = model_hyperparameters["lstm_num_layers"]
            lstm_dropout = model_hyperparameters["lstm_dropout"]

            lopo_general_model_instance_eval = ModelClass(
                input_channels=expected_num_features_sliced,
                resnet_block_type=resnet_block_type,
                resnet_layers=resnet_layers,
                lstm_hidden_size=lstm_hidden_size,
                lstm_num_layers=lstm_num_layers,
                lstm_dropout=lstm_dropout,
                num_classes=1
            ).to(device)
        elif model_type == "ResNet-BiLSTM":
            resnet_block_type = model_hyperparameters["resnet_block_type"]
            resnet_layers = model_hyperparameters["resnet_layers"]
            lstm_hidden_size = model_hyperparameters["lstm_hidden_size"]
            lstm_num_layers = model_hyperparameters["lstm_num_layers"]
            lstm_dropout = model_hyperparameters["lstm_dropout"]

            lopo_general_model_instance_eval = ModelClass(
                input_channels=expected_num_features_sliced,
                resnet_block_type=resnet_block_type,
                resnet_layers=resnet_layers,
                lstm_hidden_size=lstm_hidden_size,
                lstm_num_layers=lstm_num_layers,
                lstm_dropout=lstm_dropout,
                num_classes=1
            ).to(device)
        elif model_type == "LSTM":
            lstm_units = model_hyperparameters["lstm_units"]
            lopo_general_model_instance_eval = ModelClass(
                input_features=expected_num_features_sliced,
                seq_len=expected_seq_len_sliced,
                lstm_units=lstm_units,
                dense_units=model_hyperparameters["dense_units"],
                dropout_rate=model_hyperparameters["dropout_rate"],
            ).to(device)
        else:
            raise ValueError(f"Unknown model type for evaluation instantiation: {model_type}")

        lopo_general_model_instance_eval.load_state_dict(lopo_general_model_state_dict)
    except (ValueError, RuntimeError, Exception) as e:
        logging.error(
            f"Error instantiating or loading LOPO general model state for evaluation for patient {current_patient_id} ({model_type}, {combo_name}, {current_hp_combo_str}): {e}. Skipping patient."
        )
        del train_dataloader_pat, val_dataloader_pat, test_dataloader_pat
        del train_dataset_pat, val_dataset_pat, test_dataset_pat
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
                "after": { # This 'after' block remains as a placeholder for full return
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
        )

    metrics_before = evaluate_pytorch_model(
        lopo_general_model_instance_eval, test_dataloader_pat, nn.BCELoss(), device
    )
    logging.info(
        f"Patient {current_patient_id} ({model_type}, {combo_name}, {current_hp_combo_str}) - Before Personalization Metrics: Acc={metrics_before['accuracy']:.4f}, Prec={metrics_before['precision']:.4f}, Rec={metrics_before['recall']:.4f}, F1={metrics_before['f1_score']:.4f}, AUC={metrics_before['auc_roc']:.4f}"
    )

    # <--- FIRST INSTANCE: Plot directory for BEFORE Personalization (Corrected) ---
    plot_dir_pers_before = os.path.join(
        run_specific_output_dir, # Use the run_specific_output_dir passed from main
        "personalized", # Subfolder for all personalization results
        current_patient_id, # Specific patient's folder
        "plots_before_personalization" # Specific subfolder for before plots
    )
    os.makedirs(plot_dir_pers_before, exist_ok=True) # Ensure this directory exists

    if 'all_probs' in metrics_before and 'all_labels' in metrics_before:
        plot_auc_roc(
            metrics_before['all_probs'],
            metrics_before['all_labels'],
            f'Before Personalization AUC-ROC (Patient {current_patient_id}, {combo_name})',
            os.path.join(plot_dir_pers_before, 'before_personalization_auc_roc.png') # CHANGED
        )
        plot_probability_distribution(
            metrics_before['all_probs'],
            metrics_before['all_labels'],
            f'Before Personalization Probability Distribution (Patient {current_patient_id}, {combo_name})',
            os.path.join(plot_dir_pers_before, 'before_personalization_prob_dist.png') # CHANGED
        )
    else:
        logging.warning(f"Skipping Before Personalization AUC-ROC/ProbDist plot for Patient {current_patient_id}: 'all_probs' or 'all_labels' not found in metrics.")

    plot_confusion_matrix(
        metrics_before.get('confusion_matrix', [[0,0],[0,0]]),
        ['Interictal (0)', 'Pre-ictal (1)'],
        f'Before Personalization Confusion Matrix (Patient {current_patient_id}, {combo_name})',
        os.path.join(plot_dir_pers_before, 'before_personalization_confusion_matrix.png') # CHANGED
    )
    # --- END FIRST INSTANCE ---

    del lopo_general_model_instance_eval
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        # Model instantiation for fine-tuning (same as original, uses model_hyperparameters)
        if model_type in ["CNN-LSTM", "CNN-BiLSTM"]:
            lstm_units = model_hyperparameters["lstm_units"]

            personalized_model = ModelClass(
                input_channels=expected_num_features_sliced,
                seq_len=expected_seq_len_sliced,
                conv_filters=model_hyperparameters["conv_filters"],
                conv_kernel_size=model_hyperparameters["conv_kernel_size"],
                pool_size=model_hyperparameters["pool_size"],
                lstm_units=lstm_units,
                dense_units=model_hyperparameters["dense_units"],
                dropout_rate=model_hyperparameters["dropout_rate"],
            ).to(device)
        elif model_type == "CNN-GRU":
            gru_units = model_hyperparameters["gru_units"]

            personalized_model = ModelClass(
                input_channels=expected_num_features_sliced,
                seq_len=expected_seq_len_sliced,
                conv_filters=model_hyperparameters["conv_filters"],
                conv_kernel_size=model_hyperparameters["conv_kernel_size"],
                pool_size=model_hyperparameters["pool_size"],
                gru_units=gru_units,
                dense_units=model_hyperparameters["dense_units"],
                dropout_rate=model_hyperparameters["dropout_rate"],
            ).to(device)
        elif model_type == "CNN-Transformer":
            transformer_nhead = model_hyperparameters["transformer_nhead"]
            transformer_nlayers = model_hyperparameters["transformer_nlayers"]
            transformer_dim_feedforward = model_hyperparameters["transformer_dim_feedforward"]

            personalized_model = ModelClass(
                input_channels=expected_num_features_sliced,
                seq_len=expected_seq_len_sliced,
                conv_filters=model_hyperparameters["conv_filters"],
                conv_kernel_size=model_hyperparameters["conv_kernel_size"],
                pool_size=model_hyperparameters["pool_size"],
                transformer_nhead=transformer_nhead,
                transformer_nlayers=transformer_nlayers,
                transformer_dim_feedforward=transformer_dim_feedforward,
                dense_units=model_hyperparameters["dense_units"],
                dropout_rate=model_hyperparameters["dropout_rate"],
            ).to(device)
        elif model_type == "CNN-TCN":
            tcn_num_channels = model_hyperparameters["tcn_num_channels"]
            tcn_kernel_size = model_hyperparameters["tcn_kernel_size"]

            personalized_model = ModelClass( # Changed from ModelClass_og
                input_channels=expected_num_features_sliced,
                seq_len=expected_seq_len_sliced,
                conv_filters=model_hyperparameters["conv_filters"],
                conv_kernel_size=model_hyperparameters["conv_kernel_size"],
                pool_size=model_hyperparameters["pool_size"],
                tcn_num_channels=tcn_num_channels,
                tcn_kernel_size=tcn_kernel_size,
                dense_units=model_hyperparameters["dense_units"],
                dropout_rate=model_hyperparameters["dropout_rate"],
            ).to(device)
        elif model_type == "DenseNet-LSTM":
            personalized_model = ModelClass(
                input_channels=expected_num_features_sliced,
                seq_len=expected_seq_len_sliced,
                densenet_growth_rate=model_hyperparameters["densenet_growth_rate"],
                densenet_block_config=model_hyperparameters["densenet_block_config"],
                densenet_bn_size=model_hyperparameters["densenet_bn_size"],
                densenet_pool_size=model_hyperparameters["pool_size"],
                densenet_kernel_size=model_hyperparameters["conv_kernel_size"],
                lstm_units=model_hyperparameters["lstm_units"],
                dense_units=model_hyperparameters["dense_units"],
                dropout_rate=model_hyperparameters["dropout_rate"],
            ).to(device)
        elif model_type == "DenseNet-BiLSTM":
            personalized_model = ModelClass(
                input_channels=expected_num_features_sliced,
                seq_len=expected_seq_len_sliced,
                densenet_growth_rate=model_hyperparameters["densenet_growth_rate"],
                densenet_block_config=model_hyperparameters["densenet_block_config"],
                densenet_bn_size=model_hyperparameters["densenet_bn_size"],
                densenet_pool_size=model_hyperparameters["pool_size"],
                densenet_kernel_size=model_hyperparameters["conv_kernel_size"],
                lstm_units=model_hyperparameters["lstm_units"],
                dense_units=model_hyperparameters["dense_units"],
                dropout_rate=model_hyperparameters["dropout_rate"],
            ).to(device)
        elif model_type == "ResNet-LSTM":
            resnet_block_type = model_hyperparameters["resnet_block_type"]
            resnet_layers = model_hyperparameters["resnet_layers"]
            lstm_hidden_size = model_hyperparameters["lstm_hidden_size"]
            lstm_num_layers = model_hyperparameters["lstm_num_layers"]
            lstm_dropout = model_hyperparameters["lstm_dropout"]

            personalized_model = ModelClass(
                input_channels=expected_num_features_sliced,
                resnet_block_type=resnet_block_type,
                resnet_layers=resnet_layers,
                lstm_hidden_size=lstm_hidden_size,
                lstm_num_layers=lstm_num_layers,
                lstm_dropout=lstm_dropout,
                num_classes=1
            ).to(device)
        elif model_type == "ResNet-BiLSTM":
            resnet_block_type = model_hyperparameters["resnet_block_type"]
            resnet_layers = model_hyperparameters["resnet_layers"]
            lstm_hidden_size = model_hyperparameters["lstm_hidden_size"]
            lstm_num_layers = model_hyperparameters["lstm_num_layers"]
            lstm_dropout = model_hyperparameters["lstm_dropout"]

            personalized_model = ModelClass(
                input_channels=expected_num_features_sliced,
                resnet_block_type=resnet_block_type,
                resnet_layers=resnet_layers,
                lstm_hidden_size=lstm_hidden_size,
                lstm_num_layers=lstm_num_layers,
                lstm_dropout=lstm_dropout,
                num_classes=1
            ).to(device)
        elif model_type == "LSTM":
            lstm_units = model_hyperparameters["lstm_units"]

            personalized_model = ModelClass(
                input_features=expected_num_features_sliced,
                seq_len=expected_seq_len_sliced,
                lstm_units=lstm_units,
                dense_units=model_hyperparameters["dense_units"],
                dropout_rate=model_hyperparameters["dropout_rate"],
            ).to(device)
        else:
            raise ValueError(f"Unknown model type for personalization instantiation: {model_type}")

        personalized_model.load_state_dict(lopo_general_model_state_dict)
    except (ValueError, RuntimeError, Exception) as e:
        logging.error(
            f"Error instantiating or loading LOPO general model state for fine-tuning for patient {current_patient_id} ({model_type}, {combo_name}, {current_hp_combo_str}): {e}. Skipping patient."
        )
        del train_dataloader_pat, val_dataloader_pat, test_dataloader_pat
        del train_dataset_pat, val_dataset_pat, test_dataset_pat
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
        )

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

    # This is the base directory for personalized model file and "after" plots
    # It must be within the run_specific_output_dir.
    personalized_model_and_plots_base_dir = os.path.join(
        run_specific_output_dir, # This is the main run's folder like '.../20250601_092220/CNN-GRU/HR_EDA_TEMP_ACC/hp_combo_1'
        "personalized", # Subfolder for all personalization results within this run
        current_patient_id # Specific patient's folder (e.g., 'patient_01')
    )
    os.makedirs(personalized_model_and_plots_base_dir, exist_ok=True) # Ensure this base directory for personalized patient outputs exists


    if len(train_dataset_pat) > 0:
        logging.info(f"Starting fine-tuning for patient {current_patient_id}...")

        # Personalized model save path (for the model fine-tuned for this patient)
        # Uses the newly defined `personalized_model_and_plots_base_dir`
        personalized_model_save_path = os.path.join(
            personalized_model_and_plots_base_dir,
            f"patient_{current_patient_id}.pth"
        )

        personalized_model, personalized_metrics = train_pytorch_model(
            personalized_model,
            train_dataloader_pat,
            val_dataloader_pat,
            test_dataloader_pat,
            epochs=personalization_hyperparameters["epochs"],
            learning_rate=personalization_hyperparameters["learning_rate"],
            class_weights=class_weights_pat_dict,
            save_best_model_path=personalized_model_save_path,
            desc=f"Fine-tuning {current_patient_id}",
            device=device,
            weight_decay=personalization_hyperparameters["weight_decay"],
        )

        # Plot directory for *after personalization* plots
        # Also uses the newly defined `personalized_model_and_plots_base_dir`
        plot_dir_pers_after = os.path.join(
            personalized_model_and_plots_base_dir,
            "plots_after_personalization" # Specific subfolder for after plots
        )
        os.makedirs(plot_dir_pers_after, exist_ok=True) # Ensure plots directory exists

        if 'history' in personalized_metrics:
            plot_training_history(
            personalized_metrics['history'],
            f'Personalized Model (Patient {current_patient_id}, {combo_name})',
            plot_dir_pers_after, # CHANGED
            f'patient_{current_patient_id}_personalized'
        )
    else:
        logging.warning(
            f"Warning: No fine-tuning data for patient {current_patient_id} ({model_type}, {combo_name}, {current_hp_combo_str}). Skipping fine-tuning."
        )
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
        }

    metrics_after = personalized_metrics["test"]

    logging.info(
        f"Patient {current_patient_id} ({model_type}, {combo_name}, {current_hp_combo_str}) - After Personalization Metrics: Acc={metrics_after['accuracy']:.4f}, Prec={metrics_after['precision']:.4f}, Rec={metrics_after['recall']:.4f}, F1={metrics_after['f1_score']:.4f}, AUC={metrics_after['auc_roc']:.4f}"
    )

    # <--- SECOND INSTANCE: Plotting for AFTER Personalization (Corrected) ---
    if 'all_probs' in metrics_after and 'all_labels' in metrics_after:
        plot_auc_roc(
            metrics_after['all_probs'],
            metrics_after['all_labels'],
            f'After Personalization AUC-ROC (Patient {current_patient_id}, {combo_name})',
            os.path.join(plot_dir_pers_after, 'after_personalization_auc_roc.png') # CHANGED
        )
        plot_probability_distribution(
            metrics_after['all_probs'],
            metrics_after['all_labels'],
            f'After Personalization Probability Distribution (Patient {current_patient_id}, {combo_name})',
            os.path.join(plot_dir_pers_after, 'after_personalization_prob_dist.png') # CHANGED
        )
    else:
        logging.warning(f"Skipping After Personalization AUC-ROC plot for Patient {current_patient_id}: 'all_probs' or 'all_labels' not found in metrics.")

    plot_confusion_matrix(
        metrics_after.get('confusion_matrix', [[0,0],[0,0]]),
        ['Interictal (0)', 'Pre-ictal (1)'],
        f'After Personalization Confusion Matrix (Patient {current_patient_id}, {combo_name})',
        os.path.join(plot_dir_pers_after, 'after_personalization_confusion_matrix.png') # CHANGED
    )
    # --- END SECOND INSTANCE ---

    del train_dataset_pat, val_dataset_pat, test_dataset_pat
    del train_dataloader_pat, val_dataloader_pat, test_dataloader_pat
    del personalized_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        current_patient_id,
        {
            "before": metrics_before,
            "after": metrics_after,
            "lopo_general_metrics": lopo_general_metrics,
            "personalized_metrics_all_sets": personalized_metrics,
        },
    )


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
    # ... (same as original, but ensure EXTERNAL_PREPROCESS_SAMPLING_FREQ_HZ is used for expected_seq_len_sliced) ...
    combination_name = "_".join([s.lower() for s in sensor_combination]).upper()

    logging.info(
        f"--- Performing Personalization ({model_type}) for {combination_name} with HP: {current_hp_combo_str} using LOPO (Parallel) ---"
    )

    if not all_processed_patient_data:
        logging.warning(
            "No patient data available for personalization with LOPO."
        )
        return {}

    personalization_results = {}
    ModelClass = get_model_class(model_type)

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
        )
        return {}
    except Exception as e:
        logging.error(
            f"An unexpected error occurred getting sensor indices for combination {sensor_combination}: {e}"
        )
        return {}

    # Use the hardcoded external preprocessing sampling frequency
    # This ensures consistency with how your external data was created
    expected_seq_len_sliced = int(SEGMENT_DURATION_SECONDS * EXTERNAL_PREPROCESS_SAMPLING_FREQ_HZ) #
    expected_num_features_sliced = len(sensor_combination_indices)

    (
        patients_suitable_for_combination,
        sensor_combination_indices,
    ) = get_patients_and_indices_for_combination(
        all_processed_patient_data,
        sensor_combination,
    )

    if not patients_suitable_for_combination:
        logging.warning(
            f"Skipping personalization for {model_type} + {combination_name} with HP: {current_hp_combo_str}: No suitable patients found."
        )
        return {}

    logging.info(
        f"Initiating parallel personalization for {len(patients_suitable_for_combination)} suitable patients for combination: {combination_name} with HP: {current_hp_combo_str}."
    )

    personalized_model_save_dir_base = os.path.join(
        OUTPUT_DIR, model_type, combination_name, "personalized"
    )
    plot_dir_pers_base = os.path.join(
        OUTPUT_DIR, model_type, combination_name, "personalized", "plots"
    )
    try:
        os.makedirs(personalized_model_save_dir_base, exist_ok=True)
        os.makedirs(plot_dir_pers_base, exist_ok=True)
        logging.info(f"Created base personalization directories for HP: {current_hp_combo_str}, Model: {model_type}, Sensors: {combination_name}")
    except Exception as e:
        logging.error(f"Error creating base personalization directories for HP: {current_hp_combo_str}, Model: {model_type}, Sensors: {combination_name}: {e}. Skipping personalization for this combo.")
        return {}

    max_workers = 2
    max_workers = min(max_workers, len(patients_suitable_for_combination))
    max_workers = (
        max(1, max_workers) if len(patients_suitable_for_combination) > 0 else 0
    )

    futures = []
    if max_workers > 0:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            for patient_data_tuple in tqdm(
                patients_suitable_for_combination,
                desc=f"Submitting patient tasks ({model_type}, {combination_name}, {current_hp_combo_str})",
                leave=False,
            ):
                future = executor.submit(
                    process_single_patient_personalization,
                    patient_data_tuple,
                    all_processed_patient_data,
                    model_type,
                    sensor_combination,
                    sensor_combination_indices,
                    general_hyperparameters,
                    personalization_hyperparameters,
                    model_hyperparameters,
                    expected_seq_len_sliced, # This is derived from SEGMENT_DURATION_SECONDS * EXTERNAL_PREPROCESS_SAMPLING_FREQ_HZ
                    expected_num_features_sliced, # This is derived from len(sensor_combination_indices)
                    current_hp_combo_str,
                    device_name,
                )
                futures.append(future)

            personalization_results_list = []
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"Collecting patient results ({model_type}, {combination_name}, {current_hp_combo_str})",
            ):
                try:
                    patient_id, patient_results = future.result()
                    if patient_results is not None:
                        personalization_results_list.append(
                            (patient_id, patient_results)
                        )
                    else:
                        logging.warning(
                            f"Personalization failed or skipped for patient {patient_id} in a parallel process ({model_type}, {combination_name}, {current_hp_combo_str})."
                        )

                except Exception as exc:
                    logging.error(
                        f"A patient processing generated an exception: {exc} ({model_type}, {combination_name}, {current_hp_combo_str})"
                    )

        personalization_results = {
            patient_id: results for patient_id, results in personalization_results_list
        }

        logging.info(
            f"Finished parallel personalization for combination: {combination_name} with HP: {current_hp_combo_str}. Processed {len(personalization_results)} patients successfully."
        )
    else:
        logging.warning(
            f"No workers available for parallel processing for combination: {combination_name} with HP: {current_hp_combo_str}. Skipping."
        )
        personalization_results = {}

    return personalization_results

def get_patients_and_indices_for_combination(
    all_processed_patient_data, sensor_combination
):
    # ... (same as original) ...
    combination_name = "_".join(sensor_combination).upper()

    logging.info(f"Checking patients for sensor combination: {combination_name}")

    patients_suitable_for_combination = []

    try:
        sensor_combination_indices = [BASE_SENSORS.index(s) for s in sensor_combination]
        sensor_combination_indices.sort()
    except ValueError as e:
        logging.error(
            f"Error: Sensor '{e}' in combination {sensor_combination} not found in BASE_SENSORS {BASE_SENSORS}. Cannot process this combination."
        )
        return [], []
    except Exception as e:
        logging.error(
            f"An unexpected error occurred getting sensor indices for combination {sensor_combination}: {e}"
        )
        return [], []

    for patient_data_tuple in all_processed_patient_data:
        patient_id, segments_all_sensors, labels, found_sensors = patient_data_tuple
        if all(s in found_sensors for s in sensor_combination):
            if (
                segments_all_sensors.shape[2] == len(BASE_SENSORS)
                and len(segments_all_sensors) > 0
                and len(np.unique(labels)) > 1
            ):
                patients_suitable_for_combination.append(
                    patient_data_tuple
                )

    if not patients_suitable_for_combination:
        logging.warning(
            f"No patients found with all sensors for combination: {combination_name}. Skipping this combination."
        )
        return [], []

    return patients_suitable_for_combination, sensor_combination_indices

def format_metrics_for_summary(metrics_dict, prefix=""):
    # ... (same as original) ...
    pass

def print_personalization_summary(personalization_results, output_file=None):
    # ... (same as original) ...
    pass

def plot_auc_roc(all_probs, all_labels, title, save_path):
    # ... (same as original) ...
    pass

def plot_confusion_matrix(cm, classes, title, save_path):
    # ... (same as original) ...
    pass

def plot_training_history(history, title_prefix, save_dir, filename_suffix):
    # ... (same as original) ...
    pass

def plot_probability_distribution(all_probs, all_labels, title, save_path):
    # ... (same as original) ...
    pass

# --- Main Execution ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(
        OUTPUT_DIR, f"seizure_prediction_results_{timestamp_str}_v3enhanced.log"
    )
    summary_output_filename = os.path.join(
        OUTPUT_DIR, f"seizure_prediction_summary_{timestamp_str}_v3evnhanced.txt"
    )

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler(sys.stdout),
            ],
        )

    logging.info("--- Seizure Prediction Run Started ---")
    logging.info(f"Run Date: {time.ctime()}")
    logging.info(f"Output Directory: {OUTPUT_DIR}")
    logging.info(f"Using device: {DEVICE}")
    logging.info(f"Run All Model Types: {RUN_ALL_MODEL_TYPES}")
    logging.info(f"Adaptive Sensor Testing Enabled: {ENABLE_ADAPTIVE_SENSORS}")
    logging.info(f"Tunable Hyperparameters Enabled: {ENABLE_TUNABLE_HYPERPARAMETERS}")
    logging.info(f"Personalization Enabled: {ENABLE_PERSONALIZATION}")
    logging.info(f"Base Sensors: {BASE_SENSORS}")
    logging.info(f"Segment Duration (seconds): {SEGMENT_DURATION_SECONDS}")
    logging.info(f"External Preprocessing Sampling Frequency (Hz): {EXTERNAL_PREPROCESS_SAMPLING_FREQ_HZ}")

    # --- START AMENDMENT: Load pre-processed data from external pipeline ---
    # Define the path to your externally generated .pkl file
    # You MUST change this path to point to your actual file
    EXTERNAL_PROCESSED_DATA_FILE = "path/to/your/externally_preprocessed_patient_data.pkl" #
    # Example: EXTERNAL_PROCESSED_DATA_FILE = "C:\\Users\\YourUser\\Desktop\\my_preprocessed_data.pkl"

    all_processed_patient_data = [] #
    num_patients_in_run = 0 #

    logging.info(f"Attempting to load externally pre-processed data from: {EXTERNAL_PROCESSED_DATA_FILE}") #
    try:
        with open(EXTERNAL_PROCESSED_DATA_FILE, 'rb') as f: #
            all_processed_patient_data = pickle.load(f) #
        num_patients_in_run = len(all_processed_patient_data) #
        logging.info(f"Successfully loaded {num_patients_in_run} patients from external data.") #
        # Verify the structure of the loaded data for the first patient (optional, but highly recommended)
        if num_patients_in_run > 0: #
            first_patient_data = all_processed_patient_data[0] #
            if not (isinstance(first_patient_data, tuple) and len(first_patient_data) == 4 and
                    isinstance(first_patient_data[0], str) and # patient_id
                    isinstance(first_patient_data[1], np.ndarray) and # segments
                    isinstance(first_patient_data[2], np.ndarray) and # labels
                    isinstance(first_patient_data[3], list) and # found_sensors
                    first_patient_data[1].ndim == 3 and # (N, L, F)
                    first_patient_data[2].ndim == 1): # (N,)
                logging.error("Loaded external data has an unexpected format for the first patient. Expected (patient_id, segments(N,L,F), labels(N,), found_sensors). Please check your external preprocessing output.") #
                sys.exit(1) #
            logging.info(f"First patient data format verified: segments shape {first_patient_data[1].shape}, labels shape {first_patient_data[2].shape}.") #
            logging.info(f"Ensure BASE_SENSORS list ({BASE_SENSORS}) matches the feature order of your external data (shape F={first_patient_data[1].shape[2]}).") #
            logging.info(f"Ensure SEGMENT_DURATION_SECONDS ({SEGMENT_DURATION_SECONDS}s) * EXTERNAL_PREPROCESS_SAMPLING_FREQ_HZ ({EXTERNAL_PREPROCESS_SAMPLING_FREQ_HZ}Hz) = Sequence Length ({first_patient_data[1].shape[1]}).") #


    except FileNotFoundError: #
        logging.error(f"Error: External processed data file not found at {EXTERNAL_PROCESSED_DATA_FILE}. Please ensure the path is correct. Exiting.") #
        sys.exit(1) #
    except Exception as e: #
        logging.error(f"Error loading external processed data: {e}. Exiting.") #
        sys.exit(1) #

    if not all_processed_patient_data or num_patients_in_run == 0: #
        logging.error("No patient data available after attempting to load external processed data. Exiting.") #
        sys.exit(1) #
    # --- END AMENDMENT ---

    # --- Prepare Hyperparameter Combinations ---
    hp_combinations = []
    base_hp_param_lists = {
        "conv_filters": TUNABLE_CONV_FILTERS,
        "conv_kernel_size": TUNABLE_CONV_KERNEL_SIZE,
        "pool_size": TUNABLE_POOL_SIZE,
        "lstm_units": TUNABLE_LSTM_UNITS,
        "gru_units": TUNABLE_GRU_UNITS,
        "transformer_nhead":TUNABLE_TRANSFORMER_NHEAD,
        "transformer_nlayers":TUNABLE_TRANSFORMER_NLAYERS,
        "transformer_dim_feedforward":TUNABLE_TRANSFORMER_DIM_FEEDFORWARD,
        "dense_units": TUNABLE_DENSE_UNITS,
        "tcn_num_channels":TUNABLE_TCN_NUM_CHANNELS,
        "tcn_kernel_size":TUNABLE_TCN_KERNEL_SIZE,
        "densenet_growth_rate":TUNABLE_DENSENET_GROWTH_RATE,
        "densenet_block_config":TUNABLE_DENSENET_BLOCK_CONFIG,
        "densenet_bn_size":TUNABLE_DENSENET_BN_SIZE,
        "resnet_block_type":TUNABLE_RESNET_BLOCK_TYPE,
        "resnet_layers":TUNABLE_RESNET_LAYERS,
        "resnet_lstm_hidden_size":TUNABLE_RESNET_LSTM_HIDDEN_SIZE,
        "resnet_lstm_num_layers":TUNABLE_RESNET_LSTM_NUM_LAYERS,
        "resnet_lstm_dropout":TUNABLE_RESNET_LSTM_DROPOUT,

        "general_model_epochs": TUNABLE_GENERAL_MODEL_EPOCHS,
        "personalization_epochs": TUNABLE_PERSONALIZATION_EPOCHS,
        "general_model_lr": TUNABLE_GENERAL_MODEL_LR,
        "personalization_lr": TUNABLE_PERSONALIZATION_LR,
        "batch_size": TUNABLE_BATCH_SIZE,
        "personalization_batch_size": TUNABLE_PERSONALIZATION_BATCH_SIZE,
        "dropout_rate": TUNABLE_DROPOUT_RATE,
        "general_model_weight_decay": TUNABLE_WEIGHT_DECAY_GENERAL,
        "personalization_weight_decay": TUNABLE_WEIGHT_DECAY_PERSONALIZATION,
        # REMOVED: Data processing HPs are no longer 'tunable' or 'fixed' in this script
        # as they are determined by the external preprocessing.
    }

    if ENABLE_TUNABLE_HYPERPARAMETERS:
        keys, values = zip(*base_hp_param_lists.items())
        for bundle in itertools.product(*values):
            hp_combinations.append(dict(zip(keys, bundle)))
    else:
        single_combo = {}
        for key, value_list in base_hp_param_lists.items():
            single_combo[key] = value_list[0]
        hp_combinations.append(single_combo)

    logging.info(
        f"Prepared {len(hp_combinations)} hyperparameter combination(s) to test."
    )

    # --- Outer loop for Hyperparameter Combinations ---
    all_results = {}
    start_time_overall = time.time()

    for hp_idx, current_hp_combo in enumerate(tqdm(hp_combinations, desc="Overall HP Combinations")):
        hp_combo_desc_parts = []
        # Construct current_hp_combo_str using actual model HPs, as data HPs are now fixed
        for k in ["conv_filters", "lstm_units", "batch_size"]:
            if k in current_hp_combo:
                value_str = str(current_hp_combo[k]).replace('[', '').replace(']', '').replace(', ', '-')
                hp_combo_desc_parts.append(f"{k}-{value_str}")

        current_hp_combo_str = f"HP_Combo_{hp_idx}_" + "_".join(hp_combo_desc_parts)

        logging.info(f"{'='*80}")
        logging.info(
            f"STARTING RUN FOR HYPERPARAMETER COMBINATION {hp_idx+1}/{len(hp_combinations)}"
        )
        logging.info(f"Parameters: {OrderedDict(sorted(current_hp_combo.items()))}")
        logging.info(f"{'='*80}")

        all_results[current_hp_combo_str] = {}

        # Extract current HP values for clarity and passing
        # Data processing HPs are no longer extracted from current_hp_combo,
        # but are implicitly handled by the loaded data and EXTERNAL_PREPROCESS_SAMPLING_FREQ_HZ.
        current_conv_filters = current_hp_combo["conv_filters"]
        current_conv_kernel_size = current_hp_combo["conv_kernel_size"]
        current_pool_size = current_hp_combo["pool_size"]
        current_lstm_units = current_hp_combo["lstm_units"]
        current_gru_units = current_hp_combo["gru_units"]
        current_transformer_nhead = current_hp_combo["transformer_nhead"]
        current_transformer_nlayers = current_hp_combo["transformer_nlayers"]
        current_transformer_dim_feedforward = current_hp_combo["transformer_dim_feedforward"]
        current_tcn_num_channels = current_hp_combo["tcn_num_channels"]
        current_tcn_kernel_size = current_hp_combo["tcn_kernel_size"]
        current_densenet_growth_rate = current_hp_combo["densenet_growth_rate"]
        current_densenet_block_config = current_hp_combo["densenet_block_config"]
        current_densenet_bn_size = current_hp_combo["densenet_bn_size"]
        current_resnet_block_type = current_hp_combo["resnet_block_type"]
        current_resnet_layers = current_hp_combo["resnet_layers"]
        current_resnet_lstm_hidden_size = current_hp_combo["resnet_lstm_hidden_size"]
        current_resnet_lstm_num_layers = current_hp_combo["resnet_lstm_num_layers"]
        current_resnet_lstm_dropout = current_hp_combo["resnet_lstm_dropout"]
        current_dense_units = current_hp_combo["dense_units"]
        current_general_model_epochs = current_hp_combo["general_model_epochs"]
        current_personalization_epochs = current_hp_combo["personalization_epochs"]
        current_general_model_lr = current_hp_combo["general_model_lr"]
        current_personalization_lr = current_hp_combo["personalization_lr"]
        current_batch_size = current_hp_combo["batch_size"]
        current_personalization_batch_size = current_hp_combo["personalization_batch_size"]
        current_dropout_rate= current_hp_combo["dropout_rate"]
        current_general_model_weight_decay = current_hp_combo["general_model_weight_decay"]
        current_personalization_weight_decay = current_hp_combo["personalization_weight_decay"]

        model_hyperparameters = {
            "conv_filters": current_conv_filters,
            "conv_kernel_size": current_conv_kernel_size,
            "pool_size": current_pool_size,
            "lstm_units": current_lstm_units,
            "gru_units":current_gru_units,
            "transformer_nhead":current_transformer_nhead,
            "transformer_nlayers":current_transformer_nlayers,
            "transformer_dim_feedforward":current_transformer_dim_feedforward,
            "tcn_num_channels":current_tcn_num_channels,
            "tcn_kernel_size":current_tcn_kernel_size,
            "densenet_growth_rate": current_densenet_growth_rate,
            "densenet_block_config": current_densenet_block_config,
            "densenet_bn_size": current_densenet_bn_size,
            "dense_units": current_dense_units,
            "resnet_block_type":current_resnet_block_type,
            "resnet_layers":current_resnet_layers,
            "resnet_lstm_hidden_size":current_resnet_lstm_hidden_size,
            "resnet_lstm_num_layers":current_resnet_lstm_num_layers,
            "resnet_lstm_dropout":current_resnet_lstm_dropout,
            # REMOVED "sampling_freq_hz" as it's now EXTERNAL_PREPROCESS_SAMPLING_FREQ_HZ
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

        models_to_run = (
            MODEL_TYPES_TO_RUN if RUN_ALL_MODEL_TYPES else [MODEL_TYPES_TO_RUN[0]]
        )

        sensor_combinations_to_run = (
            ALL_SENSOR_COMBINATIONS if ENABLE_ADAPTIVE_SENSORS else [list(BASE_SENSORS)]
        )

        for current_model_type in models_to_run:
            all_results[current_hp_combo_str][
                current_model_type
            ] = {}

            for current_combination in sensor_combinations_to_run:
                combination_name = "_".join(
                    current_combination
                ).upper()
                all_results[current_hp_combo_str][current_model_type][
                    combination_name
                ] = {}

                logging.info(f"{'='*40}")
                logging.info(
                    f"RUNNING: Model {current_model_type} + Sensors {combination_name} with HP: {current_hp_combo_str}"
                )
                logging.info(f"{'='*40}")

                (
                    patients_suitable_for_combination,
                    sensor_combination_indices,
                ) = get_patients_and_indices_for_combination(
                    all_processed_patient_data,
                    current_combination,
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
                    continue

                all_results[current_hp_combo_str][current_model_type][combination_name][
                    "num_suitable_patients"
                ] = len(patients_suitable_for_combination)
                logging.info(
                    f"Proceeding with {len(patients_suitable_for_combination)} suitable patients for this run."
                )

                logging.info(f"{'--'*30}")
                logging.info("PHASE 1: TRAINING & EVALUATING OVERALL GENERAL MODEL")
                logging.info(f"{'--'*30}")

                overall_general_segments_raw = []
                overall_general_labels_raw = []

                for patient_data_tuple in patients_suitable_for_combination:
                    patient_id, segments_all_sensors, labels, found_sensors = patient_data_tuple

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

                if not overall_general_segments_raw:
                    logging.warning(
                        f"No segments available for Overall General Model training ({current_model_type}, {combination_name}, {current_hp_combo_str}). Skipping."
                    )
                    overall_general_results_by_combo_model_run = {
                        "metrics": {},
                        "num_suitable_patients": len(patients_suitable_for_combination),
                    }
                    overall_general_model_state = None
                else:
                    overall_general_segments_combined = np.concatenate(
                        overall_general_segments_raw, axis=0
                    )
                    overall_general_labels_combined = np.concatenate(
                        overall_general_labels_raw, axis=0
                    )

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
                        overall_general_model_state = None
                    else:
                        logging.info(
                            f"Overall General Combined data shape ({current_model_type}, {combination_name}, {current_hp_combo_str}): {overall_general_segments_combined.shape}"
                        )
                        logging.info(
                            f"Overall General Combined labels shape ({current_model_type}, {combination_name}, {current_hp_combo_str}): {overall_general_labels_combined.shape}"
                        )

                        try:
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
                                stratify=y_temp_og,
                            )

                            num_samples_train = X_train_og.shape[0]
                            seq_len_train = X_train_og.shape[1]
                            num_features = X_train_og.shape[2]

                            num_samples_val = X_val_og.shape[0]
                            seq_len_val = X_val_og.shape[1]

                            num_samples_test = X_test_og.shape[0]
                            seq_len_test = X_test_og.shape[1]

                            if num_samples_train > 0 and num_samples_val > 0 and num_samples_test > 0:
                                X_train_reshaped = X_train_og.reshape(-1, num_features)
                                X_val_reshaped = X_val_og.reshape(-1, num_features)
                                X_test_reshaped = X_test_og.reshape(-1, num_features)

                                scaler = RobustScaler()
                                scaler.fit(X_train_reshaped)

                                X_train_scaled = scaler.transform(X_train_reshaped)
                                X_val_scaled = scaler.transform(X_val_reshaped)
                                X_test_scaled = scaler.transform(X_test_reshaped)

                                X_train_og = X_train_scaled.reshape(num_samples_train, seq_len_train, num_features)
                                X_val_og = X_val_scaled.reshape(num_samples_val, seq_len_val, num_features)
                                X_test_og = X_test_scaled.reshape(num_samples_test, seq_len_test, num_features)

                                logging.info(f"Applied RobustScaler to Overall General data splits ({current_model_type}, {combination_name}, {current_hp_combo_str}).")
                            else:
                                logging.warning(f"One or more Overall General data splits are empty after splitting. Skipping RobustScaler. ({current_model_type}, {combination_name}, {current_hp_combo_str})")

                        except ValueError as e:
                            logging.warning(
                                f"Warning: Overall General Model data split failed ({current_model_type}, {combination_name}, {current_hp_combo_str}): {e}. This might happen with very few samples or imbalanced small datasets leading to single-class splits. Skipping training."
                            )
                            overall_general_model_metrics = {}
                            overall_general_model_state = None
                        except Exception as e:
                            logging.error(
                                f"An unexpected error occurred during Overall General Model data split ({current_model_type}, {combination_name}, {current_hp_combo_str}): {e}. Skipping training."
                            )
                            overall_general_model_metrics = {}
                            overall_general_model_state = None

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

                            num_workers_og = 0
                            persistent_workers_og = False

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

                            class_weights_og_tensor = None
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
                                    logging.warning(f"Training set for Overall General Model has only one class ({classes_og}). Using uniform weights.")
                                    class_weights_og_tensor = torch.tensor([1.0, 1.0], dtype=torch.float32).to(DEVICE)
                            else:
                                logging.warning(f"Overall General training set is empty. Cannot compute class weights. Using uniform weights.")
                                class_weights_og_tensor = torch.tensor([1.0, 1.0], dtype=torch.float32).to(DEVICE)

                            ModelClass_og = get_model_class(current_model_type)

                            try:
                                if current_model_type in ["CNN-LSTM", "CNN-BiLSTM"]:
                                    overall_general_model = ModelClass_og(
                                        input_channels=input_channels_og,
                                        seq_len=seq_len_og,
                                        conv_filters=current_conv_filters,
                                        conv_kernel_size=current_conv_kernel_size,
                                        pool_size=current_pool_size,
                                        lstm_units=current_lstm_units,
                                        dense_units=current_dense_units,
                                        dropout_rate=current_dropout_rate,
                                    ).to(DEVICE)
                                elif current_model_type == "CNN-GRU":
                                    overall_general_model = ModelClass_og(
                                        input_channels=input_channels_og,
                                        seq_len=seq_len_og,
                                        conv_filters=current_conv_filters,
                                        conv_kernel_size=current_conv_kernel_size,
                                        pool_size=current_pool_size,
                                        gru_units=current_gru_units,
                                        dense_units=current_dense_units,
                                        dropout_rate=current_dropout_rate,
                                    )
                                elif current_model_type == "CNN-Transformer":
                                    overall_general_model = ModelClass_og(
                                        input_channels=input_channels_og,
                                        seq_len=seq_len_og,
                                        conv_filters=current_conv_filters,
                                        conv_kernel_size=current_conv_kernel_size,
                                        pool_size=current_pool_size,
                                        transformer_nhead=current_transformer_nhead,
                                        transformer_nlayers=current_transformer_nlayers,
                                        transformer_dim_feedforward=current_transformer_dim_feedforward,
                                        dense_units=current_dense_units,
                                        dropout_rate=current_dropout_rate,
                                    )
                                elif current_model_type == "CNN-TCN":
                                    overall_general_model = ModelClass_og(
                                        input_channels=input_channels_og,
                                        seq_len=seq_len_og,
                                        conv_filters=current_conv_filters,
                                        conv_kernel_size=current_conv_kernel_size,
                                        pool_size=current_pool_size,
                                        tcn_num_channels=current_tcn_num_channels,
                                        tcn_kernel_size=current_tcn_kernel_size,
                                        dense_units=current_dense_units,
                                        dropout_rate=current_dropout_rate,
                                    )
                                elif current_model_type == "DenseNet-LSTM":
                                    overall_general_model = ModelClass_og(
                                        input_channels=input_channels_og,
                                        seq_len=seq_len_og,
                                        densenet_growth_rate=model_hyperparameters["densenet_growth_rate"],
                                        densenet_block_config=model_hyperparameters["densenet_block_config"],
                                        densenet_bn_size=model_hyperparameters["densenet_bn_size"],
                                        densenet_pool_size=model_hyperparameters["pool_size"],
                                        densenet_kernel_size=model_hyperparameters["conv_kernel_size"],
                                        lstm_units=current_lstm_units,
                                        dense_units=current_dense_units,
                                        dropout_rate=current_dropout_rate,
                                    )
                                elif current_model_type == "DenseNet-BiLSTM":
                                    overall_general_model = ModelClass_og(
                                        input_channels=input_channels_og,
                                        seq_len=seq_len_og,
                                        densenet_growth_rate=model_hyperparameters["densenet_growth_rate"],
                                        densenet_block_config=model_hyperparameters["densenet_block_config"],
                                        densenet_bn_size=model_hyperparameters["densenet_bn_size"],
                                        densenet_pool_size=model_hyperparameters["pool_size"],
                                        densenet_kernel_size=model_hyperparameters["conv_kernel_size"],
                                        lstm_units=current_lstm_units,
                                        dense_units=current_dense_units,
                                        dropout_rate=current_dropout_rate,
                                    )
                                elif current_model_type == "ResNet-LSTM":
                                    overall_general_model = ModelClass_og(
                                        input_channels=input_channels_og,
                                        resnet_block_type=current_resnet_block_type,
                                        resnet_layers=current_resnet_layers,
                                        lstm_hidden_size=current_resnet_lstm_hidden_size,
                                        lstm_num_layers=current_resnet_lstm_num_layers,
                                        lstm_dropout=current_resnet_lstm_dropout,
                                        num_classes=1
                                    )
                                elif current_model_type == "ResNet-BiLSTM":
                                    overall_general_model = ModelClass_og(
                                        input_channels=input_channels_og,
                                        resnet_block_type=current_resnet_block_type,
                                        resnet_layers=current_resnet_layers,
                                        lstm_hidden_size=current_resnet_lstm_hidden_size,
                                        lstm_num_layers=current_resnet_lstm_num_layers,
                                        lstm_dropout=current_resnet_lstm_dropout,
                                        num_classes=1
                                    )
                                elif current_model_type == "LSTM":
                                    overall_general_model = ModelClass_og(
                                        input_features=input_channels_og,
                                        seq_len=seq_len_og,
                                        lstm_units=current_lstm_units,
                                        dense_units=current_dense_units,
                                        dropout_rate=current_dropout_rate,
                                    )
                                else:
                                    raise ValueError(f"Unknown model type for instantiation: {current_model_type}")

                            except (ValueError, Exception) as e:
                                logging.error(
                                    f"Error instantiating Overall General Model ({current_model_type}, {combination_name}, {current_hp_combo_str}): {e}. Skipping training."
                                )
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
                                overall_general_model_state = None

                            if "overall_general_model" in locals() and overall_general_model is not None:

                                logging.info(
                                    f"Starting Overall General Model training ({current_model_type}, {combination_name}, {current_hp_combo_str})..."
                                )

                                overall_general_model_save_dir = os.path.join(
                                    OUTPUT_DIR,
                                    timestamp_str,
                                    current_model_type,
                                    combination_name,
                                )
                                overall_general_model_save_path = os.path.join(
                                    overall_general_model_save_dir,
                                    f"hp_combo_{hp_idx+1}",
                                    f"overall_general_model.pth",
                                )
                                plot_dir_og = os.path.join(overall_general_model_save_dir, f"hp_combo_{hp_idx+1}", 'plots')

                                try:
                                    os.makedirs(os.path.dirname(overall_general_model_save_path), exist_ok=True)
                                    os.makedirs(plot_dir_og, exist_ok=True)
                                    logging.info(f"Created output directories for HP: {current_hp_combo_str}, Model: {current_model_type}, Sensors: {combination_name}")
                                except Exception as e:
                                    logging.error(f"Error creating output directories for HP: {current_hp_combo_str}, Model: {current_model_type}, Sensors: {combination_name}: {e}. Skipping this run.")
                                    all_results[current_hp_combo_str][current_model_type][combination_name]['status'] = 'Directory Creation Failed'
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
                                    del overall_general_model
                                    gc.collect()
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    continue

                                criterion_og = nn.BCELoss()

                                (
                                    overall_general_model,
                                    overall_general_metrics,
                                ) = train_pytorch_model(
                                    overall_general_model,
                                    overall_general_train_dataloader,
                                    overall_general_val_dataloader,
                                    overall_general_test_dataloader,
                                    epochs=current_general_model_epochs,
                                    learning_rate=current_general_model_lr,
                                    class_weights=class_weights_og_tensor,
                                    save_best_model_path=overall_general_model_save_path,
                                    desc=f"Training Overall General ({current_model_type}, {combination_name}, HP {hp_idx+1})",
                                    device=DEVICE,
                                    weight_decay=current_general_model_weight_decay,
                                )

                                if 'history' in overall_general_metrics:
                                    plot_training_history(
                                        overall_general_metrics['history'],
                                        f'Overall General Model ({current_model_type}, {combination_name}, HP {hp_idx+1})',
                                        plot_dir_og,
                                        f'overall_general_hp_{hp_idx+1}'
                                    )

                                final_train_loss_from_history = overall_general_metrics['history']['train_loss'][-1] if overall_general_metrics['history']['train_loss'] else 0.0

                                logging.info(
                                    f"Overall General Model Training Metrics ({current_model_type}, {combination_name}, HP {hp_idx+1}) - Train (Final Eval): Acc={overall_general_metrics['train']['accuracy']:.4f}, Prec={overall_general_metrics['train']['precision']:.4f}, Rec={overall_general_metrics['train']['recall']:.4f}, F1={overall_general_metrics['train']['f1_score']:.4f}, AUC={overall_general_metrics['train']['auc_roc']:.4f}"
                                )
                                logging.info(
                                    f"Overall General Model Training Metrics ({current_model_type}, {combination_name}, HP {hp_idx+1}) - Train Loss (Final Epoch): {final_train_loss_from_history:.4f}"
                                )
                                logging.info(
                                    f"Overall General Model Validation Metrics ({current_model_type}, {combination_name}, HP {hp_idx+1}) - Val: {format_metrics_for_summary(overall_general_metrics['val'])}"
                                )
                                logging.info(
                                    f"Overall General Model Testing Metrics ({current_model_type}, {combination_name}, HP {hp_idx+1}) - Test: {format_metrics_for_summary(overall_general_metrics['test'])}"
                                )

                                overall_general_test_metrics_data = overall_general_metrics['test']
                                overall_general_test_probs = overall_general_test_metrics_data.get('all_probs', [])
                                overall_general_test_labels = overall_general_test_metrics_data.get('all_labels', [])
                                overall_general_test_cm = overall_general_test_metrics_data.get('confusion_matrix', [[0,0],[0,0]])

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

                                plot_confusion_matrix(
                                    overall_general_test_cm,
                                    ['Interictal (0)', 'Pre-ictal (1)'],
                                    f'Overall General Model Confusion Matrix ({current_model_type},  {timestamp_str}, {combination_name}, HP {hp_idx+1})',
                                    os.path.join(plot_dir_og, f'overall_general_hp_{hp_idx+1}_confusion_matrix.png')
                                )

                                overall_general_results_by_combo_model_run = {
                                    'metrics': overall_general_metrics,
                                    'num_suitable_patients': len(patients_suitable_for_combination)
                                }

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
                                overall_general_results_by_combo_model_run = {
                                    "metrics": {},
                                    "num_suitable_patients": len(
                                        patients_suitable_for_combination
                                    ),
                                }

                all_results[current_hp_combo_str][current_model_type][combination_name][
                    "overall_general"
                ] = overall_general_results_by_combo_model_run

                if ENABLE_PERSONALIZATION:
                    logging.info(f"{'--'*30}")
                    logging.info("PHASE 2: PER-PATIENT PERSONALIZATION (using LOPO)")
                    logging.info(f"{'--'*30}")

                    personalization_results = perform_personalization_pytorch_lopo(
                        all_processed_patient_data,
                        current_model_type,
                        current_combination,
                        general_hyperparameters,
                        personalization_hyperparameters,
                        model_hyperparameters,
                        current_hp_combo_str,
                        DEVICE.type,
                    )

                    all_results[current_hp_combo_str][current_model_type][
                        combination_name
                    ]["personalization"] = {
                        "personalization_results": personalization_results,
                        "num_suitable_patients": len(
                            patients_suitable_for_combination
                        ),
                    }

                    with open(
                        summary_output_filename, "a"
                    ) as summary_file:
                        summary_file.write(f"\n\n{'#'*60}\n")
                        summary_file.write(
                            f"PERSONALIZATION RESULTS FOR HP: {current_hp_combo_str}, MODEL: {current_model_type}, SENSORS: {combination_name}\n"
                        )
                        summary_file.write(
                            f"Hyperparameters: {OrderedDict(sorted(current_hp_combo.items()))}\n"
                        )
                        summary_file.write(f"{'#'*60}\n\n")
                        print_personalization_summary(
                            personalization_results, output_file=summary_file
                        )

                    print_personalization_summary(
                        personalization_results, output_file=None
                    )

                    metrics_after_list = {
                        "accuracy": [],
                        "precision": [],
                        "recall": [],
                        "f1_score": [],
                        "auc_roc": [],
                        "sensitivity": [],
                        "specificity": [],
                    }
                    count_valid_patients_pers = 0

                    for patient_id, results in personalization_results.items():
                        if isinstance(
                            results.get("after"), dict
                        ) and "accuracy" in results.get("after", {}):
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
                                if len(cm_after) == 2 and len(cm_after[0]) == 2:
                                    tn, fp, fn, tp = cm_after[0][0], cm_after[0][1], cm_after[1][0], cm_after[1][1]
                                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                                    metrics_after_list["sensitivity"].append(sensitivity)
                                    metrics_after_list["specificity"].append(specificity)
                                else:
                                    metrics_after_list["sensitivity"].append(0.0)
                                    metrics_after_list["specificity"].append(0.0)

                    with open(
                        summary_output_filename, "a"
                    ) as summary_file:
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
                            ] = avg_metrics

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
                            ] = None

                        summary_file.write("\n")

                else:
                    logging.info(
                        f"Personalization (Phase 2) is disabled. Skipping for {current_model_type} + {combination_name} with HP: {current_hp_combo_str}."
                    )
                    all_results[current_hp_combo_str][current_model_type][
                        combination_name
                    ][
                        "personalization"
                    ] = None


    logging.info(f"\n\n{'='*80}\n")
    logging.info("GENERATING FINAL SUMMARY TABLE...")
    logging.info(f"{'='*80}\n")

    try:
        with open(summary_output_filename, "w") as summary_file:
            summary_file.write(f"Experiment Summary - {timestamp_str}\n")
            summary_file.write(f"Total execution time: {time.time() - start_time_overall:.2f} seconds\n\n")

            summary_file.write("--- Feature Flags ---\n")
            summary_file.write(f"RUN_ALL_MODEL_TYPES: {RUN_ALL_MODEL_TYPES}\n")
            summary_file.write(f"ENABLE_ADAPTIVE_SENSORS: {ENABLE_ADAPTIVE_SENSORS}\n")
            summary_file.write(f"ENABLE_TUNABLE_HYPERPARAMETERS: {ENABLE_TUNABLE_HYPERPARAMETERS}\n")
            summary_file.write(f"ENABLE_PERSONALIZATION: {ENABLE_PERSONALIZATION}\n")
            # REMOVED: REUSE_PROCESSED_DATA logging

            summary_file.write("\n")

            # Logging about external data source
            summary_file.write("--- External Data Source Configuration ---\n")
            summary_file.write(f"  Pre-processed data loaded from: {EXTERNAL_PROCESSED_DATA_FILE}\n") #
            summary_file.write(f"  Number of patients loaded: {num_patients_in_run}\n") #
            summary_file.write(f"  Segment Duration: {SEGMENT_DURATION_SECONDS} seconds\n") #
            summary_file.write(f"  Effective Sampling Frequency: {EXTERNAL_PREPROCESS_SAMPLING_FREQ_HZ} Hz\n") #
            summary_file.write("\n")

            summary_file.write("--- Tunable Hyperparameters Settings ---\n")
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
                "TUNABLE_GRU_UNITS": TUNABLE_GRU_UNITS,
                "TUNABLE_TRANSFORMER_NHEAD": TUNABLE_TRANSFORMER_NHEAD,
                "TUNABLE_TRANSFORMER_NLAYERS": TUNABLE_TRANSFORMER_NLAYERS,
                "TUNABLE_TRANSFORMER_DIM_FEEDFORWARD": TUNABLE_TRANSFORMER_DIM_FEEDFORWARD,
                "TUNABLE_TCN_NUM_CHANNELS": TUNABLE_TCN_NUM_CHANNELS,
                "TUNABLE_TCN_KERNEL_SIZE": TUNABLE_TCN_KERNEL_SIZE,
                "TUNABLE_DENSENET_GROWTH_RATE": TUNABLE_DENSENET_GROWTH_RATE,
                "TUNABLE_DENSENET_BLOCK_CONFIG": TUNABLE_DENSENET_BLOCK_CONFIG,
                "TUNABLE_DENSENET_BN_SIZE": TUNABLE_DENSENET_BN_SIZE,
                "TUNABLE_RESNET_BLOCK_TYPE": TUNABLE_RESNET_BLOCK_TYPE,
                "TUNABLE_RESNET_LAYERS": TUNABLE_RESNET_LAYERS,
                "TUNABLE_RESNET_LSTM_HIDDEN_SIZE": TUNABLE_RESNET_LSTM_HIDDEN_SIZE,
                "TUNABLE_RESNET_LSTM_NUM_LAYERS": TUNABLE_RESNET_LSTM_NUM_LAYERS,
                "TUNABLE_RESNET_LSTM_DROPOUT": TUNABLE_RESNET_LSTM_DROPOUT,
            }

            for param_name, values in tunable_hp_for_summary.items():
                summary_file.write(f"  {param_name}: {values}\n")
            summary_file.write("\n")

            summary_file.write(f"MODEL_TYPES_TO_RUN: {MODEL_TYPES_TO_RUN}\n")
            if ENABLE_ADAPTIVE_SENSORS:
                summary_file.write(f"ALL_SENSOR_COMBINATIONS ({len(ALL_SENSOR_COMBINATIONS)} total):\n")
                for combo in ALL_SENSOR_COMBINATIONS:
                    summary_file.write(f"  - {'+'.join(combo)}\n")
            else:
                summary_file.write(f"BASE_SENSORS: {BASE_SENSORS}\n")
            summary_file.write("\n")


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

            for hp_combo_str in sorted(all_results.keys(), key=lambda x: int(x.split('_')[2])):
                hp_results = all_results[hp_combo_str]
                hp_combo_idx = int(hp_combo_str.split('_')[2]) + 1

                for model_type in sorted(hp_results.keys()):
                    model_results = hp_results[model_type]

                    for combo_name in sorted(model_results.keys()):
                        combo_results = model_results[combo_name]
                        num_suitable_patients = combo_results.get(
                            "num_suitable_patients", 0
                        )

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

                        # Provide default values for sensitivity and specificity if not present
                        val_sensitivity = overall_general_val_metrics.get('sensitivity', 0.0)
                        val_specificity = overall_general_val_metrics.get('specificity', 0.0)
                        test_sensitivity = overall_general_test_metrics.get('sensitivity', 0.0)
                        test_specificity = overall_general_test_metrics.get('specificity', 0.0)

                        overall_general_val_metrics_str = (
                            f"{overall_general_val_metrics.get('accuracy', 0.0):.2f} | {overall_general_val_metrics.get('precision', 0.0):.2f} | "
                            f"{overall_general_val_metrics.get('recall', 0.0):.2f} | {overall_general_val_metrics.get('f1_score', 0.0):.2f} | "
                            f"{overall_general_val_metrics.get('auc_roc', 0.0):.2f} | {val_sensitivity:.2f} | "
                            f"{val_specificity:.2f}"
                        )
                        overall_general_test_metrics_str = (
                            f"{overall_general_test_metrics.get('accuracy', 0.0):.2f} | {overall_general_test_metrics.get('precision', 0.0):.2f} | "
                            f"{overall_general_test_metrics.get('recall', 0.0):.2f} | {overall_general_test_metrics.get('f1_score', 0.0):.2f} | "
                            f"{overall_general_test_metrics.get('auc_roc', 0.0):.2f} | {test_sensitivity:.2f} | "
                            f"{test_specificity:.2f}"
                        )


                        personalization_data = combo_results.get(
                            "personalization", None
                        )
                        if personalization_data is not None:
                            avg_personalized_metrics = personalization_data.get(
                                "avg_personalized_metrics", None
                            )
                            if avg_personalized_metrics:
                                # Provide default values for sensitivity and specificity if not present
                                avg_pers_sensitivity = avg_personalized_metrics.get('sensitivity', 0.0)
                                avg_pers_specificity = avg_personalized_metrics.get('specificity', 0.0)

                                avg_personalized_metrics_str = (
                                    f"{avg_personalized_metrics.get('accuracy', 0.0):.2f} | {avg_personalized_metrics.get('precision', 0.0):.2f} | "
                                    f"{avg_personalized_metrics.get('recall', 0.0):.2f} | {avg_personalized_metrics.get('f1_score', 0.0):.2f} | "
                                    f"{avg_personalized_metrics.get('auc_roc', 0.0):.2f} | {avg_pers_sensitivity:.2f} | "
                                    f"{avg_pers_specificity:.2f}"
                                )

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
                                avg_personalized_metrics_str = "N/A    | N/A  | N/A  | N/A  | N/A  | N/A  | N/A" # To match column width, 7 fields
                                avg_change_combo_str = "N/A"
                        else:
                            avg_personalized_metrics_str = "N/A    | N/A  | N/A  | N/A  | N/A  | N/A  | N/A"
                            avg_change_combo_str = "N/A"


                        summary_file.write(
                            f"  {combo_name:<10} | {num_suitable_patients:<8} | {overall_general_val_metrics_str:<85} | {overall_general_test_metrics_str:<85} | {avg_personalized_metrics_str:<85} | {avg_change_combo_str}\n"
                        )

                    summary_file.write(
                        "  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n\n"
                    )
    except Exception as e:
        logging.error(f"An error occurred while writing the final summary file: {e}")

    logging.info("--- All Runs Complete ---")
    logging.info(f"Results saved in the '{OUTPUT_DIR}' directory.")
    logging.info(f"Log file: {log_filename}")
    logging.info(f"Summary file: {summary_output_filename}")