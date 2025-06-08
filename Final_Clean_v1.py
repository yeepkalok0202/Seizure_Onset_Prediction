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
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
# Import Transformer modules
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# =========================================================== TRAINING CONFIG ===========================================================

OUTPUT_DIR = "final_v1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# You MUST change this path to point to your actual file
EXTERNAL_PROCESSED_DATA_FILE = "final_v1/cached_processed_data/processed_patient_data_n10_piw_30_pieb_180_pib_180_sf_1_wait.pkl" #

# --- FEATURE FLAGS ---

# Set to True to run all model types in MODEL_TYPES_TO_RUN; False to run only the first one
RUN_ALL_MODEL_TYPES = True
# Set to True to run all sensor combinations; False to run only the full BASE_SENSORS set
ENABLE_ADAPTIVE_SENSORS = False
# Set to True to iterate through all combinations of TUNABLE_ hyperparameters; False to use only the first value from each list
ENABLE_TUNABLE_HYPERPARAMETERS = False
# Set to True to run Phase 2 (Personalization/LOPO); False to only run Phase 1 (Overall General Model)
ENABLE_PERSONALIZATION = False

# --- FEATURE FLAGS ---


# --- DATA PROCESSING PARAMEDDTERS --- (NEED TO CHANGE HERE TO BE ALIGNED WITH THE CACHED DATA)
SEGMENT_DURATION_SECONDS = 30 # Keep this value consistent with external preprocessing
EXTERNAL_PREPROCESS_SAMPLING_FREQ_HZ = 1 # Keep this value consistent with external preprocessing

# Define base sensors (ensure these are the possible columns after sync/scaling from your external pipeline)
BASE_SENSORS = ['HR', 'EDA', 'TEMP', 'ACC']

# ---TUNABLE HYPERPARAMETERS FOR CNN-LSTM/BiLSTM ---

# These lists define the values to iterate through if ENABLE_TUNABLE_HYPERPARAMETERS is True.
# If ENABLE_TUNABLE_HYPERPARAMETERS is False, only the first value from each list is used.
TUNABLE_CONV_FILTERS = [    
    [128, 256]
]

TUNABLE_CONV_KERNEL_SIZE = [
    5,
] 

TUNABLE_POOL_SIZE = [2]

TUNABLE_LSTM_UNITS = [
    128
]

TUNABLE_GRU_UNITS = [
    128
]
TUNABLE_DENSE_UNITS = [

    128,
]

TUNABLE_GENERAL_MODEL_EPOCHS = [
    200
]  # Used for Overall General Model and LOPO General Model

TUNABLE_PERSONALIZATION_EPOCHS = [
    200
    ]

# the sweetspot macam between 0.00001 to 0.0001
TUNABLE_GENERAL_MODEL_LR = [
    0.0001,
]  # Used for Overall General Model and LOPO General Model

TUNABLE_PERSONALIZATION_LR = [
    0.0001,
]  # Used for fine-tuning

TUNABLE_BATCH_SIZE = [
    64
]  
# Used for Overall General Model and LOPO General Model Train/Val/Test
TUNABLE_PERSONALIZATION_BATCH_SIZE = [
    64
]  # Used for personalization Train/Val

# ---TUNABLE HYPERPARAMETERS FOR CNN-LSTM/BiLSTM ---

# --- TUNABLE REGULARIZATION PARAMETERS ---

# 0.1,0.2 no overfit & decent underfit, can try 0.01,0.05
TUNABLE_DROPOUT_RATE = [0]

TUNABLE_WEIGHT_DECAY_GENERAL = [
    0
]  # For the Overall General Model and LOPO General Model

TUNABLE_WEIGHT_DECAY_PERSONALIZATION = [
    0
]  # For personalization

# --- TUNABLE REGULARIZATION PARAMETERS ---

# ---TUNABLE HYPERPARAMETERS FOR TRANSFORMER ---

TUNABLE_TRANSFORMER_NHEAD = [
    8 # Explore more attention heads
]
TUNABLE_TRANSFORMER_NLAYERS = [
    4 # Explore more layers
]
TUNABLE_TRANSFORMER_DIM_FEEDFORWARD = [
    256 # Explore larger feedforward dimensions
]
# New: Add TUNABLE_TRANSFORMER_D_MODEL
# Choose values that are compatible (divisible by) your nhead values.
# For example, if nhead is 8, d_model could be 16, 32, 64, etc.
TUNABLE_TRANSFORMER_D_MODEL = [
    32, # Example: 32 is divisible by 8 and 4
]

# ---TUNABLE HYPERPARAMETERS FOR TRANSFORMER ---

# --- TUNABLE HYPERPARAMETERS FOR DENSENET --- (Need Changes, will refer later)

TUNABLE_DENSENET_GROWTH_RATE = [
    16, # Smaller growth rate
    32, # Common growth rate
]
TUNABLE_DENSENET_BLOCK_CONFIG = [
    # (6, 12, 24, 16), # DenseNet-121 like config, IT DOENST WORK the seq length become <1
    (6, 12, 24), # Shorter config, THIS WORK the seq length become
    # (6, 12, 24, 16, 32), # Longer config, IT DOENST WORK the seq length become <1
]
TUNABLE_DENSENET_BN_SIZE = [
    4, # Common bottleneck size multiplier
    8, # Larger bottleneck size multiplier
]
# Note: DenseNet also uses kernel_size and pool_size, which we will reuse from TUNABLE_CONV_KERNEL_SIZE and TUNABLE_POOL_SIZE

# --- TUNABLE HYPERPARAMETERS FOR DENSENET --- 

# --- TUNABLE HYPERPARAMETER FOR RESNET ---

TUNABLE_RESNET_BLOCK_TYPE = ['BasicBlock'] # Only BasicBlock1d implemented here
TUNABLE_RESNET_LAYERS = [
    [2, 2, 2, 2]  # Use ResNet18, less capacity than [3, 4, 6, 3]
]

TUNABLE_RESNET_LSTM_HIDDEN_SIZE = [
    128
]

TUNABLE_RESNET_LSTM_NUM_LAYERS = [
    1  # Reduce to 1 to limit capacity
]

TUNABLE_RESNET_LSTM_DROPOUT = [
    0.3  # Higher dropout to regularize RNN
]


# --- TUNABLE HYPERPARAMETER FOR RESNET ---

# --- MODEL TYPES TO RUN ---
MODEL_TYPES_TO_RUN = ["CNN-LSTM"]  # Example: ['CNN-LSTM', 'CNN-BiLSTM', "CNN-GRU", "DenseNet-LSTM", "DenseNet-BiLSTM", "ResNet-LSTM", "ResNet-BiLSTM"]
# Example Benchmark with NonHybrid : ["LSTM", "BiLSTM", "CNN","GRU","Transformer"]
# =========================================================== TRAINING CONFIG ===========================================================


# =========================================================== GENERAL CONFIG ===========================================================
# --- Sensor Combinations ---
# Generate all combinations of 1 to len(BASE_SENSORS) sensors from BASE_SENSORS
ALL_SENSOR_COMBINATIONS = []
for i in range(1, len(BASE_SENSORS) + 1):
    for combo in itertools.combinations(BASE_SENSORS, i):
        ALL_SENSOR_COMBINATIONS.append(list(combo))
        

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================================================== GENERAL CONFIG ===========================================================

# =========================================================== PYTORCH DATASET ===========================================================

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

# =========================================================== PYTORCH DATASET ===========================================================

# =========================================================== PARAMETERS ===============================================================

def get_relevant_hyperparameters(model_type, all_model_hyperparams):
    """
    Filters the main hyperparameter dictionary to return only those
    relevant to the specified model_type.
    """
    relevant_params = {}
    
    # Common parameters for many models
    if "dense_units" in all_model_hyperparams:
        relevant_params["dense_units"] = all_model_hyperparams["dense_units"]
    if "dropout_rate" in all_model_hyperparams:
        relevant_params["dropout_rate"] = all_model_hyperparams["dropout_rate"]

    # CNN-specific parameters
    if "CNN" in model_type:
        relevant_params["conv_filters"] = all_model_hyperparams["conv_filters"]
        relevant_params["conv_kernel_size"] = all_model_hyperparams["conv_kernel_size"]
        relevant_params["pool_size"] = all_model_hyperparams["pool_size"]

    # LSTM/BiLSTM specific parameters
    if "LSTM" in model_type:
        # For hybrid ResNet models
        if "ResNet" in model_type:
             relevant_params["lstm_hidden_size"] = all_model_hyperparams["resnet_lstm_hidden_size"]
             relevant_params["lstm_num_layers"] = all_model_hyperparams["resnet_lstm_num_layers"]
             relevant_params["lstm_dropout"] = all_model_hyperparams["resnet_lstm_dropout"]
        # For all other LSTM models
        else:
             relevant_params["lstm_units"] = all_model_hyperparams["lstm_units"]
    
    # GRU specific parameters
    if "GRU" in model_type:
        relevant_params["gru_units"] = all_model_hyperparams["gru_units"]

    # Transformer specific parameters
    if "Transformer" in model_type:
        relevant_params["transformer_nhead"] = all_model_hyperparams["transformer_nhead"]
        relevant_params["transformer_nlayers"] = all_model_hyperparams["transformer_nlayers"]
        relevant_params["transformer_dim_feedforward"] = all_model_hyperparams["transformer_dim_feedforward"]
        relevant_params["transformer_d_model"] = all_model_hyperparams["transformer_d_model"]
        
    # DenseNet specific parameters
    if "DenseNet" in model_type:
        relevant_params["densenet_growth_rate"] = all_model_hyperparams["densenet_growth_rate"]
        relevant_params["densenet_block_config"] = all_model_hyperparams["densenet_block_config"]
        relevant_params["densenet_bn_size"] = all_model_hyperparams["densenet_bn_size"]
        # DenseNet reuses these from the CNN section
        relevant_params["pool_size"] = all_model_hyperparams["pool_size"]
        relevant_params["conv_kernel_size"] = all_model_hyperparams["conv_kernel_size"]

    # ResNet specific parameters
    if "ResNet" in model_type:
        relevant_params["resnet_block_type"] = all_model_hyperparams["resnet_block_type"]
        relevant_params["resnet_layers"] = all_model_hyperparams["resnet_layers"]

    return relevant_params
# =========================================================== PARAMETERS ===============================================================

# =========================================================== PYTORCH MODELS ===========================================================
class BiLSTM_Only(nn.Module):
    def __init__(
        self,
        input_features,
        seq_len,
        lstm_units,  # Note: This will be units per direction
        dense_units,
        dropout_rate=0.5,
    ):
        super(BiLSTM_Only, self).__init__()
        self.input_features = input_features
        self.seq_len = seq_len
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate

        if input_features <= 0:
            input_features = 1
        if seq_len <= 0:
            seq_len = 1

        self.bilstm = nn.LSTM(
            input_size=input_features,
            hidden_size=lstm_units,
            batch_first=True,
            bidirectional=True  # Key change for BiLSTM
        )
        self.bilstm_dropout = nn.Dropout(self.dropout_rate)

        # The input to the dense layer will be lstm_units * 2 because of bidirectionality
        self.dense_layers = nn.Sequential(
            nn.Linear(lstm_units * 2, dense_units),
            nn.ReLU(),
            nn.Linear(dense_units, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (batch_size, input_features, seq_len) from DataLoader
        # BiLSTM expects (batch_size, seq_len, input_features)
        bilstm_in = x.permute(0, 2, 1)  # Permute to (batch_size, seq_len, input_features)

        bilstm_out, _ = self.bilstm(bilstm_in)  # bilstm_out shape: (batch_size, seq_len, lstm_units * 2)
        
        bilstm_out = self.bilstm_dropout(bilstm_out)

        # Take the output from the last timestep for classification
        # The output of the BiLSTM for the last time step already concatenates
        # the forward LSTM's last hidden state and the backward LSTM's first hidden state (which is effectively its "last" from the sequence perspective).
        
        # last_timestep_out = bilstm_out[:, -1, :]  # shape: (batch_size, lstm_units * 2)

        # output = self.dense_layers(last_timestep_out)
        pooled = torch.mean(bilstm_out, dim=1)  # (batch_size, lstm_units * 2)
        output = self.dense_layers(pooled)
        return output

class CNN_Only(nn.Module):
    def __init__(
        self,
        input_channels,
        seq_len,
        conv_filters,
        conv_kernel_size,
        pool_size,
        dense_units,
        dropout_rate=0.5,
    ):
        super(CNN_Only, self).__init__()
        self.input_channels = input_channels
        self.seq_len = seq_len
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.pool_size = pool_size
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate

        if input_channels <= 0:
            input_channels = 1
        if seq_len <= 0:
            seq_len = 1
        if not conv_filters:
            conv_filters = [32]

        conv_layers_list = []
        in_channels = input_channels
        current_seq_len = seq_len

        for i, out_channels in enumerate(conv_filters):
            kernel_size = max(1, conv_kernel_size)
            pool_s = max(1, pool_size) # Renamed to avoid conflict with module
            padding = kernel_size // 2

            conv_layers_list.append(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=kernel_size, padding=padding
                )
            )
            conv_layers_list.append(nn.BatchNorm1d(out_channels))
            conv_layers_list.append(nn.ReLU())
            conv_layers_list.append(nn.MaxPool1d(pool_s))
            conv_layers_list.append(nn.Dropout(self.dropout_rate))
            in_channels = out_channels
            current_seq_len = math.floor(
                (current_seq_len + 2 * padding - 1 * (kernel_size - 1) - 1) / 1 + 1
            )
            current_seq_len = math.floor(
                (current_seq_len + 2 * 0 - 1 * (pool_s - 1) - 1) / pool_s + 1
            )

        self.conv_layers = nn.Sequential(*conv_layers_list)

        try:
            dummy_input = torch.randn(
                1, self.input_channels, self.seq_len, dtype=torch.float32
            )
            dummy_output = self.conv_layers(dummy_input)
            self.flattened_features = dummy_output.shape[1] * dummy_output.shape[2]
            if dummy_output.shape[2] <= 0:
                raise ValueError(
                    f"Calculated sequence length after CNN is zero or negative ({dummy_output.shape[2]})."
                )
        except Exception as e:
            logging.error(
                f"Error calculating CNN output size for CNN_Only: {e}"
            )
            raise e

        self.dense_layers = nn.Sequential(
            nn.Linear(self.flattened_features, dense_units),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(dense_units, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        cnn_out = self.conv_layers(x)
        if cnn_out.shape[2] == 0: # Handle collapsed sequence
             return torch.tensor([[0.5]] * x.size(0), device=x.device, dtype=x.dtype)
        flattened_out = cnn_out.flatten(start_dim=1)
        output = self.dense_layers(flattened_out)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #.transpose(0, 1) no, for batch_first=True, shape is (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Transformer_Only(nn.Module):
    def __init__(
        self,
        input_features, # Number of input features per timestep
        seq_len,
        d_model, # Dimension of the transformer model (must be same as input_features or use a projection)
        transformer_nhead,
        transformer_nlayers,
        transformer_dim_feedforward,
        dense_units,
        dropout_rate=0.5,
    ):
        super(Transformer_Only, self).__init__()
        self.input_features = input_features
        self.seq_len = seq_len
        self.d_model = d_model # Often, d_model is set to input_features directly or via a projection

        # Optional: Input projection if input_features != d_model
        if input_features != d_model:
            self.input_projection = nn.Linear(input_features, d_model)
        else:
            self.input_projection = nn.Identity()

        self.pos_encoder = PositionalEncoding(d_model, dropout_rate, max_len=seq_len + 1) # max_len needs to be >= seq_len
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=transformer_nlayers)
        self.dense_layers = nn.Sequential(
            nn.Linear(d_model, dense_units), # Input from the last timestep's features
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_units, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (batch_size, input_channels/features, seq_len) from DataLoader
        # Transformer expects (batch_size, seq_len, d_model)
        x = x.permute(0, 2, 1) # (batch_size, seq_len, input_features)

        x = self.input_projection(x) # Project to d_model if necessary
        x = self.pos_encoder(x)
        transformer_out = self.transformer_encoder(x) # (batch_size, seq_len, d_model)

        # Use the output of the last sequence element for classification
        last_timestep_out = transformer_out[:, -1, :] # (batch_size, d_model)
        output = self.dense_layers(last_timestep_out)
        return output

class GRU_Only(nn.Module):
    def __init__(
        self,
        input_features,
        seq_len,
        gru_units,
        dense_units,
        dropout_rate=0.5,
    ):
        super(GRU_Only, self).__init__()
        self.input_features = input_features
        self.seq_len = seq_len
        self.gru_units = gru_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate

        if input_features <= 0:
            input_features = 1
        if seq_len <= 0:
            seq_len = 1

        self.gru = nn.GRU(
            input_size=input_features,
            hidden_size=gru_units,
            batch_first=True,
        )
        self.gru_dropout = nn.Dropout(self.dropout_rate)
        self.dense_layers = nn.Sequential(
            nn.Linear(gru_units, dense_units),
            nn.ReLU(),
            nn.Linear(dense_units, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (batch_size, input_features, seq_len)
        gru_in = x.permute(0, 2, 1) # Permute to (batch_size, seq_len, input_features)
        gru_out, _ = self.gru(gru_in) # gru_out shape: (batch_size, seq_len, gru_units)
        gru_out = self.gru_dropout(gru_out)
        mean_output = torch.mean(gru_out, dim=1)
        output = self.dense_layers(mean_output)
        return output
    
class LSTM_Only(nn.Module):
    def __init__(
        self,
        input_features, # Renamed from input_channels for clarity in LSTM context
        seq_len,
        lstm_units,
        dense_units,
        dropout_rate=0.5,
    ):
        super(LSTM_Only, self).__init__()
        self.input_features = input_features
        self.seq_len = seq_len
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate

        # Ensure input_features and seq_len are valid
        if input_features <= 0:
            input_features = 1
        if seq_len <= 0:
            seq_len = 1

        # LSTM layer(s)
        # LSTM expects input shape (batch_size, seq_len, input_features)
        # Our dataset provides (batch_size, input_features, seq_len), so we need to permute in forward pass
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=lstm_units,
            batch_first=True, # Expects (batch, seq, features)
        )
        # Add Dropout after LSTM layer
        self.lstm_dropout = nn.Dropout(self.dropout_rate)

        # Dense layers
        # The input to this layer will be the mean of the LSTM outputs, which has a size of lstm_units.
        self.dense_layers = nn.Sequential(
            nn.Linear(lstm_units, dense_units), # Input size is LSTM hidden size
            nn.ReLU(),
            nn.Linear(dense_units, 1), # Output size 1 for binary classification
            nn.Sigmoid(), # Output probability between 0 and 1
        )

    def forward(self, x):
        # x shape: (batch_size, input_features, seq_len) from the DataLoader/Dataset
        # LSTM expects (batch_size, seq_len, input_features)
        lstm_in = x.permute(0, 2, 1) # Permute to (batch_size, seq_len, input_features)

        lstm_out, _ = self.lstm(lstm_in) # lstm_out shape: (batch_size, seq_len, lstm_units)

        lstm_out = self.lstm_dropout(lstm_out) # Apply dropout

        # --- MODIFICATION ---
        # Instead of taking the last timestep, take the mean of the outputs across the sequence dimension.
        # This aggregates information from all time steps.
        mean_output = torch.mean(lstm_out, dim=1) # shape: (batch_size, lstm_units)

        output = self.dense_layers(mean_output) # shape: (batch_size, 1)
        return output
    
# Mean Version
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
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.pool_size = pool_size
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate

        if input_channels <= 0:
            input_channels = 1
        if seq_len <= 0:
            seq_len = 1
        if not conv_filters:
            conv_filters = [32]

        conv_layers_list = []
        in_channels = input_channels

        for i, out_channels in enumerate(conv_filters):
            kernel_size = max(1, conv_kernel_size)
            current_pool_size = max(1, self.pool_size)
            padding = kernel_size // 2

            conv_layers_list.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            )
            conv_layers_list.append(nn.BatchNorm1d(out_channels))
            conv_layers_list.append(nn.ReLU())
            conv_layers_list.append(nn.MaxPool1d(current_pool_size))
            conv_layers_list.append(nn.Dropout(self.dropout_rate))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers_list)

        try:
            dummy_input = torch.randn(1, self.input_channels, self.seq_len, dtype=torch.float32)
            dummy_output = self.conv_layers(dummy_input)
            self.lstm_input_features = dummy_output.shape[1]
            self.lstm_input_seq_len = dummy_output.shape[2]

            if self.lstm_input_seq_len <= 0:
                raise ValueError(
                    f"Calculated LSTM input sequence length is zero or negative ({self.lstm_input_seq_len}). "
                    f"Check CNN/Pooling parameters relative to segment length ({self.seq_len})."
                )
        except Exception as e:
            logging.error(
                f"Error calculating layer output size during model init for {self.__class__.__name__} "
                f"with input_channels={self.input_channels}, seq_len={self.seq_len}: {e}"
            )
            raise e

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_features,
            hidden_size=lstm_units,
            batch_first=True,
        )
        self.lstm_dropout = nn.Dropout(self.dropout_rate)

        # Use mean output: shape (batch_size, lstm_units)
        self.dense_layers = nn.Sequential(
            nn.Linear(lstm_units, dense_units),
            nn.ReLU(),
            nn.Linear(dense_units, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        cnn_out = self.conv_layers(x)
        if cnn_out.shape[2] == 0:
            return torch.tensor([[0.5]] * x.size(0), device=x.device)

        lstm_in = cnn_out.permute(0, 2, 1)  # shape: (batch_size, seq_len, features)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out = self.lstm_dropout(lstm_out)

        mean_output = torch.mean(lstm_out, dim=1)  # shape: (batch_size, lstm_units)
        output = self.dense_layers(mean_output)
        return output

# Mean Version
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
        pooled = torch.mean(bilstm_out, dim=1)  # (batch, 2*hidden)
        output = self.dense_layers(pooled)
        return output

class CNN_GRU(nn.Module):
    def __init__(
        self,
        input_channels,
        seq_len,
        conv_filters,
        conv_kernel_size,
        pool_size,
        gru_units, # Changed from lstm_units to gru_units for clarity
        dense_units,
        dropout_rate=0.5,
    ):
        super(CNN_GRU, self).__init__()
        self.input_channels = input_channels
        self.seq_len = seq_len
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.pool_size = pool_size
        self.gru_units = gru_units # Store gru_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate

        if input_channels <= 0:
            input_channels = 1
        if seq_len <= 0:
            seq_len = 1
        if not conv_filters:
            conv_filters = [32]

        conv_layers_list = []
        in_channels = input_channels
        current_seq_len = seq_len

        # Dynamically build conv layers based on conv_filters list
        for i, out_channels in enumerate(conv_filters):
            kernel_size = max(1, conv_kernel_size)
            pool_size = max(1, pool_size)

            padding = kernel_size // 2

            conv_layers_list.append(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=kernel_size, padding=padding
                )
            )
            conv_layers_list.append(nn.BatchNorm1d(out_channels))
            conv_layers_list.append(nn.ReLU())
            conv_layers_list.append(nn.MaxPool1d(pool_size))
            conv_layers_list.append(nn.Dropout(self.dropout_rate))

            in_channels = out_channels

            # Recalculate sequence length after conv and pool
            current_seq_len = math.floor(
                (current_seq_len + 2 * padding - 1 * (kernel_size - 1) - 1) / 1 + 1
            )
            current_seq_len = math.floor(
                (current_seq_len + 2 * 0 - 1 * (pool_size - 1) - 1) / pool_size + 1
            )

        self.conv_layers = nn.Sequential(*conv_layers_list)

        # Calculate the output sequence length after CNN layers dynamically
        try:
            dummy_input = torch.randn(
                1, self.input_channels, self.seq_len, dtype=torch.float32
            )
            dummy_output = self.conv_layers(dummy_input)
            self.gru_input_features = dummy_output.shape[1]
            self.gru_input_seq_len = dummy_output.shape[2]

            if self.gru_input_seq_len <= 0:
                raise ValueError(
                    f"Calculated GRU input sequence length is zero or negative ({self.gru_input_seq_len}). Check CNN/Pooling parameters relative to segment length ({self.seq_len})."
                )

        except Exception as e:
            logging.error(
                f"Error calculating layer output size during model init for {self.__class__.__name__} with input_channels={self.input_channels}, seq_len={self.seq_len}: {e}"
            )
            raise e

        # GRU layer
        # GRU expects input shape (batch_size, seq_len, input_features)
        self.gru = nn.GRU(
            input_size=self.gru_input_features,
            hidden_size=gru_units, # Use gru_units argument
            batch_first=True,
        )
        # Add Dropout after GRU layer
        self.gru_dropout = nn.Dropout(self.dropout_rate)

        # Dense layers
        self.dense_layers = nn.Sequential(
            nn.Linear(
                gru_units, dense_units # Input size is GRU hidden size
            ),
            nn.ReLU(),
            nn.Linear(dense_units, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (batch_size, channels, seq_len)
        cnn_out = self.conv_layers(x)  # shape: (batch_size, filters, reduced_seq_len)
        if cnn_out.shape[2] == 0:
            return torch.tensor(
                [[0.5]] * x.size(0), device=x.device
            )

        gru_in = cnn_out.permute(
            0, 2, 1
        )  # shape: (batch_size, reduced_seq_len, filters)
        gru_out, _ = self.gru(
            gru_in
        )  # shape: (batch_size, reduced_seq_len, GRU_UNITS)
        gru_out = self.gru_dropout(gru_out) # Apply dropout
        pooled = torch.mean(gru_out, dim=1)  # (batch, 2*hidden)
        output = self.dense_layers(pooled)
        return output

class DenseNet_LSTM(nn.Module):
    def __init__(
        self,
        input_channels,
        seq_len, # Initial sequence length
        densenet_growth_rate,
        densenet_block_config,
        densenet_bn_size,
        densenet_pool_size,
        densenet_kernel_size,
        lstm_units,
        dense_units,
        dropout_rate=0.5,
    ):
        super(DenseNet_LSTM, self).__init__()
        self.input_channels = input_channels
        self.seq_len = seq_len
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate

        # Ensure input_channels and seq_len are valid
        if input_channels <= 0:
            input_channels = 1
        if seq_len <= 0:
            seq_len = 1

        # DenseNet part
        self.densenet = DenseNet(
            input_channels=input_channels,
            growth_rate=densenet_growth_rate,
            block_config=densenet_block_config,
            bn_size=densenet_bn_size,
            pool_size=densenet_pool_size,
            kernel_size=densenet_kernel_size,
            dropout_rate=dropout_rate # Pass dropout to DenseNet
        )

        # Calculate the output shape after DenseNet layers dynamically
        try:
            # Create a dummy tensor on the CPU for shape calculation
            dummy_input = torch.randn(
                1, self.input_channels, self.seq_len, dtype=torch.float32
            )
            dummy_output = self.densenet(dummy_input)
            self.lstm_input_features = dummy_output.shape[
                1
            ]  # Features dimension after DenseNet
            self.lstm_input_seq_len = dummy_output.shape[
                2
            ]  # Sequence length dimension after DenseNet/Pooling

            # Check if the output sequence length is valid for LSTM
            if self.lstm_input_seq_len <= 0:
                raise ValueError(
                    f"Calculated LSTM input sequence length after DenseNet is zero or negative ({self.lstm_input_seq_len}). Check DenseNet/Pooling parameters relative to segment length ({self.seq_len})."
                )
            elif self.lstm_input_seq_len == 1:
                 logging.warning(f"Warning: Calculated LSTM input sequence length after DenseNet is 1 ({self.lstm_input_seq_len}). LSTM may not be effective with sequence length 1. Consider adjusting DenseNet/Pooling parameters or segment length.")


        except Exception as e:
            logging.error(
                f"Error calculating layer output size during model init for {self.__class__.__name__} with input_channels={self.input_channels}, seq_len={self.seq_len}: {e}"
            )
            raise e

        # LSTM layer(s)
        # LSTM expects input shape (batch_size, seq_len, input_features)
        # Our DenseNet output is (batch_size, features, seq_len), so we need to permute in forward pass
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_features,
            hidden_size=lstm_units,
            batch_first=True, # Expects (batch, seq, features)
        )
        self.lstm_dropout = nn.Dropout(self.dropout_rate)

        # Linear layer to handle case where sequence length becomes 1 after DenseNet
        # This layer will map the flattened DenseNet output to the expected input size of the dense layers
        self.flattened_linear = nn.Linear(self.lstm_input_features, lstm_units) # Map DenseNet features to LSTM hidden size

        # Dense layers
        self.dense_layers = nn.Sequential(
            nn.Linear(lstm_units, dense_units), # Input size is LSTM hidden size OR output of flattened_linear
            nn.ReLU(),
            nn.Linear(dense_units, 1), # Output size 1 for binary classification
            nn.Sigmoid(), # Output probability between 0 and 1
        )

    def forward(self, x):
        # x shape: (batch_size, input_channels, seq_len)
        densenet_out = self.densenet(x) # shape: (batch_size, densenet_features, reduced_seq_len)
        logging.debug(f"DenseNet_LSTM: densenet_out shape: {densenet_out.shape}") # Added logging

        # Handle potential empty output after DenseNet if seq_len collapsed to 0
        if densenet_out.shape[2] == 0:
            logging.warning("DenseNet_LSTM: DenseNet output sequence length is 0. Returning 0.5 probability.") # Added logging
            return torch.tensor(
                [[0.5]] * x.size(0), device=x.device
            )

        # Check if sequence length is 1 after DenseNet
        if densenet_out.shape[2] == 1:
            logging.debug("DenseNet_LSTM: DenseNet output sequence length is 1. Bypassing LSTM.") # Added logging
            # Flatten the output and pass through the linear layer
            flattened_out = densenet_out.squeeze(2) # Remove the seq_len dimension (batch_size, densenet_features)
            linear_out = self.flattened_linear(flattened_out) # (batch_size, lstm_units)
            # Pass directly to dense layers
            output = self.dense_layers(linear_out) # shape: (batch_size, 1)
        else:
            logging.debug(f"DenseNet_LSTM: DenseNet output sequence length > 1 ({densenet_out.shape[2]}). Proceeding with LSTM.") # Added logging
            # Permute for LSTM input: (batch_size, reduced_seq_len, densenet_features)
            lstm_in = densenet_out.permute(0, 2, 1)
            logging.debug(f"DenseNet_LSTM: lstm_in shape after permute: {lstm_in.shape}") # Added logging

            # Explicitly check sequence length before passing to LSTM
            if lstm_in.shape[1] <= 1:
                 logging.warning(f"DenseNet_LSTM: Sequence length for LSTM input is <= 1 ({lstm_in.shape[1]}) in else block. This should not happen if the outer check worked. Bypassing LSTM.")
                 # Fallback to the flattened linear layer
                 flattened_out = densenet_out.squeeze(2) # (batch_size, densenet_features)
                 linear_out = self.flattened_linear(flattened_out) # (batch_size, lstm_units)
                 output = self.dense_layers(linear_out)
            else:
                # Proceed with LSTM as normal
                lstm_out, _ = self.lstm(lstm_in) # lstm_out shape: (batch_size, reduced_seq_len, lstm_units)
                lstm_out = self.lstm_dropout(lstm_out) # Apply dropout
                logging.debug(f"DenseNet_LSTM: lstm_out shape: {lstm_out.shape}") # Added logging

                # Take the output from the last timestep for classification
                last_timestep_out = lstm_out[:, -1, :] # shape: (batch_size, lstm_units)
                logging.debug(f"DenseNet_LSTM: last_timestep_out shape: {last_timestep_out.shape}") # Added logging

                output = self.dense_layers(last_timestep_out) # shape: (batch_size, 1)
                logging.debug(f"DenseNet_LSTM: output shape: {output.shape}") # Added logging


        return output

class DenseNet_BiLSTM(nn.Module):
    def __init__(
        self,
        input_channels,
        seq_len, # Initial sequence length
        densenet_growth_rate,
        densenet_block_config,
        densenet_bn_size,
        densenet_pool_size,
        densenet_kernel_size,
        lstm_units, # Note: BiLSTM output size is 2 * lstm_units
        dense_units,
        dropout_rate=0.5,
    ):
        super(DenseNet_BiLSTM, self).__init__()
        self.input_channels = input_channels
        self.seq_len = seq_len
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate

        # Ensure input_channels and seq_len are valid
        if input_channels <= 0:
            input_channels = 1
        if seq_len <= 0:
            seq_len = 1

        # DenseNet part
        self.densenet = DenseNet(
            input_channels=input_channels,
            growth_rate=densenet_growth_rate,
            block_config=densenet_block_config,
            bn_size=densenet_bn_size,
            pool_size=densenet_pool_size,
            kernel_size=densenet_kernel_size,
            dropout_rate=dropout_rate # Pass dropout to DenseNet
        )

        # Calculate the output shape after DenseNet layers dynamically
        try:
            # Create a dummy tensor on the CPU for shape calculation
            dummy_input = torch.randn(
                1, self.input_channels, self.seq_len, dtype=torch.float32
            )
            dummy_output = self.densenet(dummy_input)
            self.lstm_input_features = dummy_output.shape[
                1
            ]  # Features dimension after DenseNet
            self.lstm_input_seq_len = dummy_output.shape[
                2
            ]  # Sequence length dimension after DenseNet/Pooling

            # Check if the output sequence length is valid for BiLSTM
            if self.lstm_input_seq_len <= 0:
                raise ValueError(
                    f"Calculated BiLSTM input sequence length after DenseNet is zero or negative ({self.lstm_input_seq_len}). Check DenseNet/Pooling parameters relative to segment length ({self.seq_len})."
                )
            elif self.lstm_input_seq_len == 1:
                 logging.warning(f"Warning: Calculated BiLSTM input sequence length after DenseNet is 1 ({self.lstm_input_seq_len}). BiLSTM may not be effective with sequence length 1. Consider adjusting DenseNet/Pooling parameters or segment length.")


        except Exception as e:
            logging.error(
                f"Error calculating layer output size during model init for {self.__class__.__name__} with input_channels={self.input_channels}, seq_len={self.seq_len}: {e}"
            )
            raise e

        # BiLSTM layer(s)
        # BiLSTM expects input shape (batch_size, seq_len, input_features)
        # Our DenseNet output is (batch_size, features, seq_len), so we need to permute in forward pass
        self.bilstm = nn.LSTM(
            input_size=self.lstm_input_features,
            hidden_size=lstm_units, # Note: This is the hidden size *per direction*
            batch_first=True, # Expects (batch, seq, features)
            bidirectional=True # Make it bidirectional
        )
        self.bilstm_dropout = nn.Dropout(self.dropout_rate)

        # Linear layer to handle case where sequence length becomes 1 after DenseNet
        # This layer will map the flattened DenseNet output to the expected input size of the dense layers
        self.flattened_linear = nn.Linear(self.lstm_input_features, lstm_units * 2) # Map DenseNet features to BiLSTM output size

        # Dense layers
        # Input size to dense layer is 2 * lstm_units because of bidirectionality OR output of flattened_linear
        self.dense_layers = nn.Sequential(
            nn.Linear(lstm_units * 2, dense_units),
            nn.ReLU(),
            nn.Linear(dense_units, 1), # Output size 1 for binary classification
            nn.Sigmoid(), # Output probability between 0 and 1
        )

    def forward(self, x):
        # x shape: (batch_size, input_channels, seq_len)
        densenet_out = self.densenet(x) # shape: (batch_size, densenet_features, reduced_seq_len)
        logging.debug(f"DenseNet_BiLSTM: densenet_out shape: {densenet_out.shape}") # Added logging

        # Handle potential empty output after DenseNet if seq_len collapsed to 0
        if densenet_out.shape[2] == 0:
            logging.warning("DenseNet_BiLSTM: DenseNet output sequence length is 0. Returning 0.5 probability.") # Added logging
            return torch.tensor(
                [[0.5]] * x.size(0), device=x.device
            )

        # Check if sequence length is 1 after DenseNet
        if densenet_out.shape[2] == 1:
            logging.debug("DenseNet_BiLSTM: DenseNet output sequence length is 1. Bypassing BiLSTM.") # Added logging
            # Flatten the output and pass through the linear layer
            flattened_out = densenet_out.squeeze(2) # Remove the seq_len dimension (batch_size, densenet_features)
            linear_out = self.flattened_linear(flattened_out) # (batch_size, lstm_units * 2)
            # Pass directly to dense layers
            output = self.dense_layers(linear_out) # shape: (batch_size, 1)
        else:
            logging.debug(f"DenseNet_BiLSTM: DenseNet output sequence length > 1 ({densenet_out.shape[2]}). Proceeding with BiLSTM.") # Added logging
            # Permute for BiLSTM input: (batch_size, reduced_seq_len, densenet_features)
            lstm_in = densenet_out.permute(0, 2, 1)
            logging.debug(f"DenseNet_BiLSTM: lstm_in shape after permute: {lstm_in.shape}") # Added logging

            # Explicitly check sequence length before passing to BiLSTM
            if lstm_in.shape[1] <= 1:
                 logging.warning(f"DenseNet_BiLSTM: Sequence length for BiLSTM input is <= 1 ({lstm_in.shape[1]}) in else block. This should not happen if the outer check worked. Bypassing BiLSTM.")
                 # Fallback to the flattened linear layer
                 flattened_out = densenet_out.squeeze(2) # (batch_size, densenet_features)
                 linear_out = self.flattened_linear(flattened_out) # (batch_size, lstm_units * 2)
                 output = self.dense_layers(linear_out)
            else:
                # Proceed with BiLSTM as normal
                bilstm_out, _ = self.bilstm(lstm_in) # bilstm_out shape: (batch_size, reduced_seq_len, lstm_units * 2)
                bilstm_out = self.bilstm_dropout(bilstm_out) # Apply dropout
                logging.debug(f"DenseNet_BiLSTM: bilstm_out shape: {bilstm_out.shape}") # Added logging

                # Take the output from the last timestep for classification
                last_timestep_out = bilstm_out[:, -1, :] # shape: (batch_size, lstm_units * 2)
                logging.debug(f"DenseNet_BiLSTM: last_timestep_out shape: {last_timestep_out.shape}") # Added logging

                output = self.dense_layers(last_timestep_out) # shape: (batch_size, 1)
                logging.debug(f"DenseNet_BiLSTM: output shape: {output.shape}") # Added logging

        return output

class ResNet_LSTM(nn.Module):
    """
    Combines a 1D ResNet backbone with an LSTM layer for time series classification.
    """
    def __init__(self, input_channels, resnet_block_type, resnet_layers, lstm_hidden_size, lstm_num_layers, lstm_dropout, num_classes):
        """
        Args:
            input_channels (int): Number of input features (sensors).
            resnet_block_type (str): Type of ResNet block ('BasicBlock' or 'Bottleneck').
                                     Only 'BasicBlock' is implemented here.
            resnet_layers (list): Number of blocks in each ResNet layer (e.g., [2, 2, 2, 2]).
            lstm_hidden_size (int): Number of hidden units in the LSTM layer.
            lstm_num_layers (int): Number of stacked LSTM layers.
            lstm_dropout (float): Dropout rate for the LSTM layer(s).
            num_classes (int): Number of output classes (e.g., 1 for binary classification).
        """
        super(ResNet_LSTM, self).__init__()

        # Choose the ResNet block type
        if resnet_block_type == 'BasicBlock':
            block = BasicBlock1d
        else:
            raise ValueError(f"Unknown ResNet block type: {resnet_block_type}. Only 'BasicBlock' is supported.")

        # Instantiate the 1D ResNet backbone
        # It takes input shape (batch_size, num_features, sequence_length)
        self.resnet_backbone = ResNet1d(block, resnet_layers, input_channels)

        # Determine the input size for the LSTM layer dynamically
        # after passing data through the ResNet backbone.
        # We need the number of output channels from the last ResNet layer.
        # Create a dummy input to trace the shape.
        # The sequence length after ResNet depends on the input sequence length
        # and the strides/pooling in the ResNet layers.
        # We need a representative dummy sequence length.
        # Using a fixed dummy length like 100 is a common approach,
        # but the actual seq_len from your data processing should be used if possible.
        # A more robust way is to calculate it based on the input seq_len and ResNet strides/kernels.
        # For simplicity here, we'll rely on the ResNet1d forward pass output shape.
        # The `train_pytorch_model` and `process_single_patient_personalization` functions
        # will determine the actual `seq_len` and `input_channels` from the data batches
        # and pass them during model instantiation.

        # The ResNet1d output shape is (batch_size, final_resnet_channels, effective_sequence_length)
        # The LSTM expects input shape (batch_size, sequence_length, input_size)
        # So, the LSTM input_size will be the `final_resnet_channels`.

        # The number of output channels from the last ResNet layer (layer4) is 512 * block.expansion
        resnet_output_channels = 512 * block.expansion

        self.lstm = nn.LSTM(
            input_size=resnet_output_channels, # Input size is the number of channels from the last ResNet layer
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0, # Apply dropout between layers if num_layers > 1
            batch_first=True, # Expects input shape (batch_size, seq_len, features)
        )

        # Final fully connected layer for classification
        # It takes the output of the last timestep of the LSTM
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid() # Sigmoid for binary classification output (0-1 probability)


    def forward(self, x):
        """
        Forward pass for the ResNet-LSTM model.
        Input x shape: (batch_size, num_features, sequence_length) - from DataLoader
        """
        # The ResNet1d backbone expects input shape (batch_size, num_features, sequence_length)
        # Our DataLoader provides (batch_size, num_features, sequence_length), which matches.
        resnet_out = self.resnet_backbone(x) # Shape: (batch_size, final_resnet_channels, effective_sequence_length)

        # Permute ResNet output for LSTM input
        # LSTM expects (batch_size, sequence_length, input_size)
        lstm_in = resnet_out.permute(0, 2, 1) # Shape: (batch_size, effective_sequence_length, final_resnet_channels)

        # Apply LSTM
        # lstm_out shape: (batch_size, effective_sequence_length, lstm_hidden_size)
        # _ is the hidden state (h_n, c_n), which we don't need for sequence-to-one prediction
        lstm_out, _ = self.lstm(lstm_in)

        # Take the output from the last timestep for classification
        # shape: (batch_size, lstm_hidden_size)
        last_timestep_out = lstm_out[:, -1, :]

        # Apply final fully connected layer and Sigmoid
        output = self.fc(last_timestep_out) # shape: (batch_size, num_classes)
        output = self.sigmoid(output) # Apply Sigmoid

        return output

class ResNet_BiLSTM(nn.Module):
    """
    Combines a 1D ResNet backbone with a Bidirectional LSTM layer for time series classification.
    """
    def __init__(self, input_channels, resnet_block_type, resnet_layers, lstm_hidden_size, lstm_num_layers, lstm_dropout, num_classes):
        """
        Args:
            input_channels (int): Number of input features (sensors).
            resnet_block_type (str): Type of ResNet block ('BasicBlock' or 'Bottleneck').
                                     Only 'BasicBlock' is implemented here.
            resnet_layers (list): Number of blocks in each ResNet layer (e.g., [2, 2, 2, 2]).
            lstm_hidden_size (int): Number of hidden units *per direction* in the BiLSTM layer.
            lstm_num_layers (int): Number of stacked BiLSTM layers.
            lstm_dropout (float): Dropout rate for the BiLSTM layer(s).
            num_classes (int): Number of output classes (e.g., 1 for binary classification).
        """
        super(ResNet_BiLSTM, self).__init__()

        # Choose the ResNet block type
        if resnet_block_type == 'BasicBlock':
            block = BasicBlock1d
        else:
            raise ValueError(f"Unknown ResNet block type: {resnet_block_type}. Only 'BasicBlock' is supported.")

        # Instantiate the 1D ResNet backbone
        self.resnet_backbone = ResNet1d(block, resnet_layers, input_channels)

        # Determine the input size for the BiLSTM layer dynamically
        # The ResNet1d output shape is (batch_size, final_resnet_channels, effective_sequence_length)
        # The BiLSTM expects input shape (batch_size, sequence_length, input_size)
        # So, the BiLSTM input_size will be the `final_resnet_channels`.
        resnet_output_channels = 512 * block.expansion

        self.bilstm = nn.LSTM(
            input_size=resnet_output_channels, # Input size is the number of channels from the last ResNet layer
            hidden_size=lstm_hidden_size, # Hidden size *per direction*
            num_layers=lstm_num_layers,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0, # Apply dropout between layers if num_layers > 1
            batch_first=True, # Expects input shape (batch_size, seq_len, features)
            bidirectional=True # Make it bidirectional
        )

        # Final fully connected layer for classification
        # It takes the output of the last timestep of the BiLSTM.
        # The output size of a BiLSTM is 2 * hidden_size (concatenated forward and backward outputs).
        self.fc = nn.Linear(lstm_hidden_size * 2, num_classes) # Multiply by 2 for bidirectional output
        self.sigmoid = nn.Sigmoid() # Sigmoid for binary classification output (0-1 probability)


    def forward(self, x):
        """
        Forward pass for the ResNet-BiLSTM model.
        Input x shape: (batch_size, num_features, sequence_length) - from DataLoader
        """
        # The ResNet1d backbone expects input shape (batch_size, num_features, sequence_length)
        # Our DataLoader provides (batch_size, num_features, sequence_length), which matches.
        resnet_out = self.resnet_backbone(x) # Shape: (batch_size, final_resnet_channels, effective_sequence_length)

        # Permute ResNet output for BiLSTM input
        # BiLSTM expects (batch_size, sequence_length, input_size)
        lstm_in = resnet_out.permute(0, 2, 1) # Shape: (batch_size, effective_sequence_length, final_resnet_channels)

        # Apply BiLSTM
        # bilstm_out shape: (batch_size, effective_sequence_length, lstm_hidden_size * 2)
        # _ is the hidden state (h_n, c_n), which we don't need for sequence-to-one prediction
        bilstm_out, _ = self.bilstm(lstm_in)

        # Take the output from the last timestep for classification
        # shape: (batch_size, lstm_hidden_size * 2)
        last_timestep_out = bilstm_out[:, -1, :]

        # Apply final fully connected layer and Sigmoid
        output = self.fc(last_timestep_out) # shape: (batch_size, num_classes)
        output = self.sigmoid(output) # Apply Sigmoid

        return output

# ===================== DenseNet Helper Functions and Blocks ===========================
# Based on the DenseNet architecture adapted for 1D data
class DenseLayer(nn.Module):
    """
    A single Dense Layer within a Dense Block.
    Consists of BatchNorm, ReLU, Conv1d.
    """
    def __init__(self, num_input_features, growth_rate, bn_size, kernel_size, padding, dropout):
        super(DenseLayer, self).__init__()
        # Ensure kernel_size is odd for padding calculation
        if kernel_size % 2 == 0:
             kernel_size += 1
             padding = kernel_size // 2 # Recalculate padding

        self.norm1 = nn.BatchNorm1d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        # Bottleneck layer (optional, controlled by bn_size)
        # Conv1d expects input (batch, channels, seq_len)
        self.conv1 = nn.Conv1d(num_input_features, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False) # 1x1 Conv for bottleneck

        self.norm2 = nn.BatchNorm1d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        # Main convolution
        self.conv2 = nn.Conv1d(bn_size * growth_rate, growth_rate,
                               kernel_size=kernel_size, stride=1, padding=padding, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x is a list of tensors from previous layers in the block
        # Concatenate along the channel dimension (dim=1)
        bottleneck_input = torch.cat(x, 1)

        out = self.norm1(bottleneck_input)
        out = self.relu1(out)
        out = self.conv1(out) # Bottleneck convolution

        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(out) # Main convolution

        out = self.dropout(out)

        # The output of a DenseLayer is the output of its last convolution.
        # This output will be concatenated with previous layer outputs in the DenseBlock.
        return out

class DenseBlock(nn.Module):
    """
    A Dense Block consisting of multiple Dense Layers.
    Features are concatenated from previous layers.
    """
    def __init__(self, num_input_features, num_layers, bn_size, growth_rate, kernel_size, dropout):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            # Input features to the i-th layer is the initial features + features from all i-1 previous layers
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate,
                               bn_size, kernel_size, kernel_size // 2, dropout)
            self.layers.append(layer)

    def forward(self, init_features):
        # init_features is the output from the previous block or the initial input
        features = [init_features]
        for layer in self.layers:
            # Pass the current list of features to the layer
            new_features = layer(features)
            # Append the new features to the list for the next layer
            features.append(new_features)
        # The output of the DenseBlock is the concatenation of all layer outputs (including initial features)
        return torch.cat(features, 1)

class TransitionLayer(nn.Module):
    """
    A Transition Layer between Dense Blocks.
    Consists of BatchNorm, ReLU, 1x1 Conv, and Average Pooling.
    Reduces spatial dimension and number of channels.
    """
    def __init__(self, num_input_features, num_output_features, pool_size):
        super(TransitionLayer, self).__init__()
        self.norm = nn.BatchNorm1d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        # 1x1 Convolution to reduce number of feature maps
        self.conv = nn.Conv1d(num_input_features, num_output_features,
                              kernel_size=1, stride=1, bias=False)
        # Average Pooling to reduce spatial dimension
        self.pool = nn.AvgPool1d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x):
        out = self.norm(x)
        out = self.relu(out)
        out = self.conv(out)
        out = self.pool(out)
        return out

class DenseNet(nn.Module):
    """
    The DenseNet part of the model, adapted for 1D data.
    Consists of an initial convolution/pooling, Dense Blocks, and Transition Layers.
    """
    def __init__(self, input_channels, growth_rate=32, block_config=(6, 12, 24, 16),
                 bn_size=4, pool_size=2, kernel_size=10, dropout_rate=0.5):
        """
        Args:
            input_channels (int): Number of input features (sensors).
            growth_rate (int): How many filters each layer adds (k).
            block_config (tuple): Number of layers in each Dense Block.
            bn_size (int): Multiplier for bottleneck layer filters.
            pool_size (int): Kernel size and stride for pooling layers.
            kernel_size (int): Kernel size for convolutions within Dense Layers.
            dropout_rate (float): Dropout rate.
        """
        super(DenseNet, self).__init__()

        # Initial Convolution and Pooling
        # Adjust initial convolution kernel size and stride as needed for your data
        # Using a larger kernel initially might be beneficial for time series
        initial_kernel_size = 2 * kernel_size # Example: Larger initial kernel
        initial_padding = initial_kernel_size // 2
        initial_out_channels = 2 * growth_rate # Example: Initial features before first block

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv1d(input_channels, initial_out_channels, kernel_size=initial_kernel_size,
                                stride=1, padding=initial_padding, bias=False)),
            ('norm0', nn.BatchNorm1d(initial_out_channels)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)),
            ('dropout0', nn.Dropout(dropout_rate)), # Add dropout after initial pooling
        ]))

        # Dense Blocks and Transition Layers
        num_features = initial_out_channels
        for i, num_layers in enumerate(block_config):
            # Add Dense Block
            block = DenseBlock(num_features, num_layers=num_layers,
                               bn_size=bn_size, growth_rate=growth_rate,
                               kernel_size=kernel_size, dropout=dropout_rate)
            self.features.add_module(f'denseblock{i + 1}', block)
            num_features = num_features + num_layers * growth_rate

            # Add Transition Layer after all but the last Dense Block
            if i != len(block_config) - 1:
                trans = TransitionLayer(num_features, num_features // 2, pool_size) # Reduce channels by half
                self.features.add_module(f'transition{i + 1}', trans)
                num_features = num_features // 2

        # Final BatchNorm before the classifier/RNN
        self.features.add_module('norm5', nn.BatchNorm1d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        # No final pooling or dropout here, as it feeds into RNN

        # The number of output features from DenseNet is `num_features` calculated above
        self.num_features = num_features

    def forward(self, x):
        # x shape: (batch_size, channels, seq_len)
        out = self.features(x) # shape: (batch_size, num_features, reduced_seq_len)
        return out

# ===================== DenseNet Helper Functions and Blocks ===========================

# ===================== ResNet Helper Functions and Blocks ===========================

class BasicBlock1d(nn.Module):
    """
    A basic block for a 1D ResNet.
    Consists of two 1D convolutional layers with Batch Normalization and ReLU activation,
    and a residual connection.
    """
    expansion = 1 # Expansion factor for the output channels

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first convolution. Used for spatial downsampling.
            downsample (nn.Module, optional): A module to downsample the input
                                              for the residual connection if needed.
        """
        super(BasicBlock1d, self).__init__()
        # First convolution layer
        # kernel_size=3, padding=1 maintains sequence length if stride=1
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels) # Batch Normalization
        self.relu = nn.ReLU(inplace=True) # ReLU activation
        # Second convolution layer
        self.conv2 = nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels * self.expansion) # Batch Normalization

        self.downsample = downsample # Downsampling module for residual connection
        self.stride = stride # Store stride

    def forward(self, x):
        """
        Forward pass for the BasicBlock1d.
        Input x shape: (batch_size, channels, sequence_length)
        """
        identity = x # Store input for residual connection

        out = self.conv1(x) # Apply first convolution
        out = self.bn1(out) # Apply Batch Normalization
        out = self.relu(out) # Apply ReLU activation

        out = self.conv2(out) # Apply second convolution
        out = self.bn2(out) # Apply Batch Normalization

        # Apply downsample to identity if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity # Add residual connection
        out = self.relu(out) # Apply final ReLU activation

        return out

class ResNet1d(nn.Module):
    """
    A 1D ResNet backbone for feature extraction from time series data.
    Adapts the standard ResNet architecture to 1D convolutions.
    """
    def __init__(self, block, layers, input_channels):
        """
        Args:
            block (nn.Module): The basic block type (e.g., BasicBlock1d).
            layers (list): A list specifying the number of blocks in each layer.
                           e.g., [2, 2, 2, 2] for ResNet18-like structure.
            input_channels (int): Number of input features (channels) for the first layer.
        """
        super(ResNet1d, self).__init__()
        # Initial number of channels after the first convolution
        self.in_channels = 64

        # Initial convolution and pooling layers
        # Input: (batch_size, num_features, sequence_length)
        # kernel_size=7, stride=2, padding=3 reduces sequence length significantly
        self.conv1 = nn.Conv1d(input_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        # Max pooling to further reduce sequence length
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet layers (groups of basic blocks)
        # Each layer potentially reduces sequence length by setting stride=2 in the first block
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Initialize weights using Kaiming Normal initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        Helper function to create a ResNet layer.
        A layer consists of multiple basic blocks.
        """
        downsample = None
        # Determine if downsampling is needed for the residual connection
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        # Add the first block in the layer (may have stride > 1 for downsampling)
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        # Update input channels for subsequent blocks in this layer
        self.in_channels = out_channels * block.expansion
        # Add remaining blocks in the layer (stride=1)
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass for the ResNet1d backbone.
        Input x shape: (batch_size, num_features, sequence_length)
        Output shape: (batch_size, final_resnet_channels, effective_sequence_length)
        """
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # The output is the feature map sequence after the last ResNet layer.
        # This output will be fed into the LSTM/BiLSTM.
        return x

# ===================== ResNet Helper Functions and Blocks ===========================

# ===================== RETURN MODEL TYPES ===========================
def get_model_class(model_type):
    # Original Hybrid Models
    if model_type == "CNN-LSTM":
        return CNN_LSTM
    elif model_type == "CNN-BiLSTM":
        return CNN_BiLSTM
    elif model_type == "CNN-GRU":
        return CNN_GRU
    elif model_type == "DenseNet-LSTM":
        return DenseNet_LSTM
    elif model_type == "DenseNet-BiLSTM":
        return DenseNet_BiLSTM
    elif model_type == "ResNet-LSTM":
        return ResNet_LSTM
    elif model_type == "ResNet-BiLSTM":
        return ResNet_BiLSTM
    # "Only" Non-Hybrid Models
    elif model_type == "LSTM":  # Maps to LSTM_Only
        return LSTM_Only
    elif model_type == "BiLSTM": # Maps to BiLSTM_Only
        return BiLSTM_Only
    elif model_type == "GRU":   # Maps to GRU_Only
        return GRU_Only
    elif model_type == "CNN":   # Maps to CNN_Only
        return CNN_Only
    elif model_type == "Transformer":  # Maps to Transformer_Only
        return Transformer_Only
    else:
        raise ValueError(f"Unknown model type: {model_type}")
# ===================== RETURN MODEL TYPES ===========================

# =========================================================== PYTORCH MODELS ===========================================================

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
    # Add sensitivity and specificity calculation
    # cm is [[TN, FP], [FN, TP]] where True_Label is row, Predicted_Label is column
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Recall of positive class
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # Recall of negative class
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_roc": auc_roc,
        "confusion_matrix": cm,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def train_one_epoch(
    model, train_dataloader, criterion, optimizer, device, class_weights=None
):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    # Define criterion - Use BCELoss as models now output probabilities
    criterion = nn.BCELoss()

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
            all_probs.extend(outputs.cpu().numpy().flatten().tolist())
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
    desc="Training",
    device=torch.device("cpu"),
    weight_decay=0.0,
):
    """Main training loop with validation, early stopping, LR scheduling, and returns metrics for all sets."""
    # Ensure the model is on the correct device BEFORE initializing the optimizer
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=0.00001
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
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= 20:
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

# --- LOPO General Model Training ---
def train_lopo_general_model(
    all_processed_patient_data,
    excluded_patient_id,
    model_type,
    sensor_combination_indices,
    model_hyperparameters,
    general_hyperparameters,
    current_hp_combo_str,
    run_specific_output_dir,
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
    logging.info(f"--- Training LOPO General Model (Excluding {excluded_patient_id}) for {model_type} ---")

    lopo_segments_raw = []
    lopo_labels_raw = []
    
    lopo_model_save_dir = os.path.join(
        run_specific_output_dir, "lopo_general"
    )
    os.makedirs(lopo_model_save_dir, exist_ok=True)
    
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

            scaler_lopo = MinMaxScaler()
            scaler_lopo.fit(X_train_lopo_reshaped)
            
            X_train_lopo_scaled = scaler_lopo.transform(X_train_lopo_reshaped)
            X_val_lopo_scaled = scaler_lopo.transform(X_val_lopo_reshaped)
            X_test_lopo_scaled = scaler_lopo.transform(X_test_lopo_reshaped)

            X_train_lopo = X_train_lopo_scaled.reshape(num_samples_train_lopo, seq_len_train_lopo, num_features_lopo)
            X_val_lopo = X_val_lopo_scaled.reshape(num_samples_val_lopo, seq_len_val_lopo, num_features_lopo)
            X_test_lopo = X_test_lopo_scaled.reshape(num_samples_test_lopo, seq_len_test_lopo, num_features_lopo)

            logging.info(f"Applied MinMaxScaler to LOPO General data splits (Excluding {excluded_patient_id}, {model_type}, {current_hp_combo_str}).")
        else:
            logging.warning(f"One or more LOPO General data splits are empty after splitting. Skipping MinMaxScaler. (Excluding {excluded_patient_id}, {model_type}, {current_hp_combo_str})")

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
    lopo_general_model = None
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
        elif model_type == "BiLSTM":
            lstm_units = model_hyperparameters["lstm_units"]
            lopo_general_model = ModelClass(
                input_features=input_channels,
                seq_len=seq_len,
                lstm_units=lstm_units,
                dense_units=model_hyperparameters["dense_units"],
                dropout_rate=model_hyperparameters["dropout_rate"],
            ).to(device)
        elif model_type == "CNN":
            lopo_general_model = ModelClass(
                input_channels=input_channels,
                seq_len=seq_len,
                conv_filters=model_hyperparameters["conv_filters"],
                conv_kernel_size=model_hyperparameters["conv_kernel_size"],
                pool_size=model_hyperparameters["pool_size"],
                dense_units=model_hyperparameters["dense_units"],
                dropout_rate=model_hyperparameters["dropout_rate"],
            ).to(device)
        elif model_type == "Transformer": # Or "Transformer_Only", ensure this matches your MODEL_TYPES_TO_RUN string
            current_nhead = model_hyperparameters["transformer_nhead"]
            # This 'transformer_d_model' MUST come from the model_hyperparameters dict
            current_d_model = model_hyperparameters["transformer_d_model"]

            if current_d_model % current_nhead != 0:
                logging.error(
                    f"LOPO General Model (Excluding {excluded_patient_id}): Skipping Transformer HP combination: "
                    f"d_model ({current_d_model}) is not divisible by nhead ({current_nhead}). "
                    f"HP_Combo_Str: {current_hp_combo_str}"
                )
                lopo_general_model = None # Critical: This will cause training to be skipped for this LOPO model
            else:
                lopo_general_model = ModelClass(
                    input_features=input_channels, # Actual features from current LOPO data
                    seq_len=seq_len,
                    d_model=current_d_model,      # Tunable d_model from HPs
                    transformer_nhead=current_nhead,
                    transformer_nlayers=model_hyperparameters["transformer_nlayers"],
                    transformer_dim_feedforward=model_hyperparameters["transformer_dim_feedforward"],
                    dense_units=model_hyperparameters["dense_units"],
                    dropout_rate=model_hyperparameters["dropout_rate"],
                ).to(device)
        elif model_type == "GRU":
            gru_units = model_hyperparameters["gru_units"] # Make sure TUNABLE_GRU_UNITS is defined
            lopo_general_model = ModelClass(
                input_features=input_channels, # GRU_Only takes input_features
                seq_len=seq_len,
                gru_units=gru_units,
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
        desc=f"Training LOPO General (Excl {excluded_patient_id})",
        device=device,
        weight_decay=general_hyperparameters["weight_decay"],
    )
    relevant_hp = get_relevant_hyperparameters(model_type, model_hyperparameters)

    lopo_general_bundle = {
        'model_state_dict': lopo_general_model.state_dict(),
        'scaler': scaler_lopo,
        'hyperparameters': {
            'model_hyperparameters': relevant_hp,
            'general_hyperparameters': general_hyperparameters
        },
        'model_type':model_type
    }
        # Save the complete bundle
    bundle_save_path = os.path.join(lopo_model_save_dir, f'excl_{excluded_patient_id}_inference_bundle.pkl')
    with open(bundle_save_path, 'wb') as f:
        pickle.dump(lopo_general_bundle, f)
    logging.info(f"Saved LOPO General Inference Bundle for excluded patient {excluded_patient_id} to {bundle_save_path}")
    
    lopo_model_plot_dir = os.path.join(lopo_model_save_dir, excluded_patient_id, "plots")
    os.makedirs(lopo_model_plot_dir, exist_ok=True)

    if 'history' in lopo_general_metrics:
        plot_training_history(
            lopo_general_metrics['history'],
            f'LOPO General Model (Excl {excluded_patient_id}, {model_type}, HP Combo {current_hp_combo_str})',
            lopo_model_plot_dir,
            f'excl_{excluded_patient_id}_lopo_general'
        )

    lopo_general_bundle_to_return = lopo_general_bundle.copy()
    del lopo_general_bundle_to_return['model_state_dict'] 
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

    return lopo_general_bundle, lopo_general_metrics

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
    run_specific_output_dir 
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

    personalized_model_and_plots_base_dir = os.path.join(
        run_specific_output_dir,
        "personalized",
        current_patient_id
    )
    os.makedirs(personalized_model_and_plots_base_dir, exist_ok=True)
    
    lopo_general_bundle, lopo_general_metrics = train_lopo_general_model(
        all_processed_patient_data,
        current_patient_id,
        model_type,
        sensor_combination_indices,
        model_hyperparameters,
        general_hyperparameters,
        current_hp_combo_str,
        run_specific_output_dir, 
        device,
    )

    if lopo_general_bundle is None:
        logging.warning(
            f"Skipping personalization for patient {current_patient_id} ({model_type}, {combo_name}, {current_hp_combo_str}): LOPO general model training failed."
        )
        return (current_patient_id, None)
    lopo_general_model_state_dict = lopo_general_bundle['model_state_dict']
    
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

            scaler_pat = MinMaxScaler()
            scaler_pat.fit(X_train_pat_reshaped)

            X_train_pat_scaled = scaler_pat.transform(X_train_pat_reshaped)
            X_val_pat_scaled = scaler_pat.transform(X_val_pat_reshaped)
            X_test_pat_scaled = scaler_pat.transform(X_test_pat_reshaped)

            X_train_pat = X_train_pat_scaled.reshape(num_samples_train_pat, seq_len_train_pat, num_features_pat)
            X_val_pat = X_val_pat_scaled.reshape(num_samples_val_pat, seq_len_val_pat, num_features_pat)
            X_test_pat = X_test_pat_scaled.reshape(num_samples_test_pat, seq_len_test_pat, num_features_pat)

            logging.info(f"Applied MinMaxScaler to patient {current_patient_id}'s personalization data splits ({model_type}, {combo_name}, {current_hp_combo_str}).")
        else:
            logging.warning(f"One or more personalization data splits for patient {current_patient_id} are empty after splitting. Skipping MinMaxScaler. ({model_type}, {combo_name}, {current_hp_combo_str})")

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
        elif model_type == "BiLSTM":
            lstm_units = model_hyperparameters["lstm_units"]
            lopo_general_model_instance_eval = ModelClass(
                input_features=expected_num_features_sliced,
                seq_len=expected_seq_len_sliced,
                lstm_units=lstm_units,
                dense_units=model_hyperparameters["dense_units"],
                dropout_rate=model_hyperparameters["dropout_rate"],
            ).to(device)
        elif model_type == "CNN":
            lopo_general_model_instance_eval = ModelClass(
                input_channels=expected_num_features_sliced,
                seq_len=expected_seq_len_sliced,
                conv_filters=model_hyperparameters["conv_filters"],
                conv_kernel_size=model_hyperparameters["conv_kernel_size"],
                pool_size=model_hyperparameters["pool_size"],
                dense_units=model_hyperparameters["dense_units"],
                dropout_rate=model_hyperparameters["dropout_rate"],
            ).to(device)
        elif model_type == "Transformer":
            current_nhead = model_hyperparameters["transformer_nhead"]
            current_d_model = model_hyperparameters["transformer_d_model"]

            if current_d_model % current_nhead != 0:
                logging.error(
                    f"Skipping Transformer HP combination: d_model ({current_d_model}) is not divisible by nhead ({current_nhead}). "
                    f"HP_Combo: {current_hp_combo_str}"
                )
                # Set overall_general_model to None or handle appropriately to skip training
                lopo_general_model_instance_eval = None # This will cause the subsequent training steps to be skipped
                # Or you might want to 'continue' the loop for the next model_type/sensor_combo if this check is earlier
            else:
                lopo_general_model_instance_eval = ModelClass(
                    input_features=expected_num_features_sliced,
                    seq_len=expected_seq_len_sliced,
                    d_model=current_d_model,
                    transformer_nhead=current_nhead,
                    transformer_nlayers=model_hyperparameters["transformer_nlayers"],
                    transformer_dim_feedforward=model_hyperparameters["transformer_dim_feedforward"],
                    dense_units=model_hyperparameters["dense_units"],
                    dropout_rate=model_hyperparameters["dropout_rate"],
                ).to(device)
        elif model_type == "GRU":
            gru_units = model_hyperparameters["gru_units"] # Make sure TUNABLE_GRU_UNITS is defined
            lopo_general_model_instance_eval = ModelClass(
                input_features=expected_num_features_sliced, # GRU_Only takes input_features
                seq_len=expected_seq_len_sliced,
                gru_units=gru_units,
                dense_units=model_hyperparameters["dense_units"],
                dropout_rate=model_hyperparameters["dropout_rate"],
            ).to(device)
        else:
            raise ValueError(f"Unknown model type for instantiation: {model_type}")

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

    # <--- FIRST INSTANCE: Plot directory for BEFORE Personalization  ---
    plot_dir_pers_before = os.path.join(
        run_specific_output_dir, # Use the run_specific_output_dir passed from main
        "personalized", # Subfolder for all personalization results
        current_patient_id, # Specific patient's folder
        "plots_before_personalization" # Specific subfolder for before plots
    )
    os.makedirs(plot_dir_pers_before, exist_ok=True) # Ensure this directory exists
    
    if lopo_general_metrics and 'history' in lopo_general_metrics and lopo_general_metrics['history']:
        logging.info(f"Plotting training history of LOPO General Model for patient {current_patient_id} (before personalization evaluation)...")
        plot_training_history(
            lopo_general_metrics['history'],
            title_prefix=f'LOPO Gen. Model (used for Pt {current_patient_id} Before Pers.) - {combo_name}, HP {current_hp_combo_str.split("_")[2]}', # Adjusted title
            save_dir=plot_dir_pers_before, # Save to the "plots_before_personalization" directory
            filename_suffix=f'lopo_general_train_hist_for_{current_patient_id}' # Unique filename suffix
        )
    else:
        logging.warning(f"No training history found for LOPO general model to plot for patient {current_patient_id} (before personalization).")
        
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
        elif model_type == "BiLSTM":
            lstm_units = model_hyperparameters["lstm_units"]
            personalized_model = ModelClass(
                input_features=expected_num_features_sliced,
                seq_len=expected_seq_len_sliced,
                lstm_units=lstm_units,
                dense_units=model_hyperparameters["dense_units"],
                dropout_rate=model_hyperparameters["dropout_rate"],
            ).to(device)
        elif model_type == "CNN":
            personalized_model = ModelClass(
                input_channels=expected_num_features_sliced,
                seq_len=expected_seq_len_sliced,
                conv_filters=model_hyperparameters["conv_filters"],
                conv_kernel_size=model_hyperparameters["conv_kernel_size"],
                pool_size=model_hyperparameters["pool_size"],
                dense_units=model_hyperparameters["dense_units"],
                dropout_rate=model_hyperparameters["dropout_rate"],
            ).to(device)
        elif model_type == "Transformer":
            current_nhead = model_hyperparameters["transformer_nhead"]
            current_d_model = model_hyperparameters["transformer_d_model"]

            if current_d_model % current_nhead != 0:
                logging.error(
                    f"Skipping Transformer HP combination: d_model ({current_d_model}) is not divisible by nhead ({current_nhead}). "
                    f"HP_Combo: {current_hp_combo_str}"
                )
                # Set overall_general_model to None or handle appropriately to skip training
                personalized_model = None # This will cause the subsequent training steps to be skipped
                # Or you might want to 'continue' the loop for the next model_type/sensor_combo if this check is earlier
            else:
                personalized_model = ModelClass(
                    input_features=expected_num_features_sliced,
                    seq_len=expected_seq_len_sliced,
                    d_model=current_d_model,
                    transformer_nhead=current_nhead,
                    transformer_nlayers=model_hyperparameters["transformer_nlayers"],
                    transformer_dim_feedforward=model_hyperparameters["transformer_dim_feedforward"],
                    dense_units=model_hyperparameters["dense_units"],
                    dropout_rate=model_hyperparameters["dropout_rate"],
                ).to(device)
        elif model_type == "GRU":
            gru_units = model_hyperparameters["gru_units"] # Make sure TUNABLE_GRU_UNITS is defined
            personalized_model = ModelClass(
                input_features=expected_num_features_sliced, # GRU_Only takes input_features
                seq_len=expected_seq_len_sliced,
                gru_units=gru_units,
                dense_units=model_hyperparameters["dense_units"],
                dropout_rate=model_hyperparameters["dropout_rate"],
            ).to(device)
        else:
            raise ValueError(f"Unknown model type for instantiation: {model_type}")
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

    if len(train_dataset_pat) > 0:
        logging.info(f"Starting fine-tuning for patient {current_patient_id}...")

        personalized_model, personalized_metrics = train_pytorch_model(
            personalized_model,
            train_dataloader_pat,
            val_dataloader_pat,
            test_dataloader_pat,
            epochs=personalization_hyperparameters["epochs"],
            learning_rate=personalization_hyperparameters["learning_rate"],
            class_weights=class_weights_pat_dict,
            desc=f"Fine-tuning {current_patient_id}",
            device=device,
            weight_decay=personalization_hyperparameters["weight_decay"],
        )
        
        relevant_hp = get_relevant_hyperparameters(model_type, model_hyperparameters)
        # Create the final personalized inference bundle
        personalized_bundle = {
            'model_state_dict': personalized_model.state_dict(),
            'scaler': scaler_pat, # The patient-specific scaler
            'hyperparameters': {
                'model_hyperparameters': relevant_hp,
                'general_hyperparameters': general_hyperparameters,
                'personalization_hyperparameters': personalization_hyperparameters
            },
            'model_type':model_type
        }
                # Save the complete personalized bundle
        bundle_save_path = os.path.join(personalized_model_and_plots_base_dir, f'patient_{current_patient_id}_inference_bundle.pkl')
        with open(bundle_save_path, 'wb') as f:
            pickle.dump(personalized_bundle, f)
        logging.info(f"Saved Personalized Inference Bundle for patient {current_patient_id} to {bundle_save_path}")
        
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
    run_specific_output_dir 
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
        run_specific_output_dir, "personalized"
    )
    plot_dir_pers_base = os.path.join(
        personalized_model_save_dir_base, "plots"
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
                    run_specific_output_dir
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

# Need to fix this
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

# --- Main Execution ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    TRAINING_OUTPUT_BASE_DIR = os.path.join(OUTPUT_DIR, "training_data")
    os.makedirs(TRAINING_OUTPUT_BASE_DIR, exist_ok=True)

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    current_run_timestamp_dir = os.path.join(TRAINING_OUTPUT_BASE_DIR, timestamp_str)
    os.makedirs(current_run_timestamp_dir, exist_ok=True)
    
    log_filename = os.path.join(
        current_run_timestamp_dir, f"seizure_prediction_results_{timestamp_str}_v3enhanced.log"
    )
    summary_output_filename = os.path.join(
        current_run_timestamp_dir, f"seizure_prediction_summary_{timestamp_str}_v3evnhanced.txt"
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
        "transformer_d_model": TUNABLE_TRANSFORMER_D_MODEL,
        "dense_units": TUNABLE_DENSE_UNITS,
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
        current_transformer_d_model = current_hp_combo["transformer_d_model"]
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
            "transformer_d_model": current_transformer_d_model,
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
            logging.info(f"Re-seeding RNGs with SEED={SEED} for model type: {current_model_type}")
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)
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

                run_specific_output_dir = os.path.join(
                    current_run_timestamp_dir,
                    current_model_type,
                    combination_name,
                    f"hp_combo_{hp_idx+1}" # Using hp_idx from the outer loop
                )
                os.makedirs(run_specific_output_dir, exist_ok=True)
                logging.info(f"Created output directory for this run: {run_specific_output_dir}")
                
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

                                scaler = MinMaxScaler()
                                scaler.fit(X_train_reshaped)

                                X_train_scaled = scaler.transform(X_train_reshaped)
                                X_val_scaled = scaler.transform(X_val_reshaped)
                                X_test_scaled = scaler.transform(X_test_reshaped)

                                X_train_og = X_train_scaled.reshape(num_samples_train, seq_len_train, num_features)
                                X_val_og = X_val_scaled.reshape(num_samples_val, seq_len_val, num_features)
                                X_test_og = X_test_scaled.reshape(num_samples_test, seq_len_test, num_features)

                                logging.info(f"Applied MinMaxScaler to Overall General data splits ({current_model_type}, {combination_name}, {current_hp_combo_str}).")
                            else:
                                logging.warning(f"One or more Overall General data splits are empty after splitting. Skipping MinMaxScaler. ({current_model_type}, {combination_name}, {current_hp_combo_str})")

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
                                elif current_model_type == "BiLSTM":
                                    overall_general_model = ModelClass_og(
                                        input_features=input_channels_og,
                                        seq_len=seq_len_og,
                                        lstm_units=current_lstm_units,
                                        dense_units=current_dense_units,
                                        dropout_rate=current_dropout_rate,
                                    ) 
                                elif current_model_type == "CNN":
                                    overall_general_model = ModelClass_og(
                                        input_channels=input_channels_og,
                                        seq_len=seq_len_og,
                                        conv_filters=model_hyperparameters["conv_filters"],
                                        conv_kernel_size=model_hyperparameters["conv_kernel_size"],
                                        pool_size=model_hyperparameters["pool_size"],
                                        dense_units=model_hyperparameters["dense_units"],
                                        dropout_rate=model_hyperparameters["dropout_rate"],
                                    )
                                elif current_model_type == "Transformer":
                                    current_nhead = model_hyperparameters["transformer_nhead"]
                                    current_d_model = model_hyperparameters["transformer_d_model"]

                                    if current_d_model % current_nhead != 0:
                                        logging.error(
                                            f"Skipping Transformer HP combination: d_model ({current_d_model}) is not divisible by nhead ({current_nhead}). "
                                            f"HP_Combo: {current_hp_combo_str}"
                                        )
                                        # Set overall_general_model to None or handle appropriately to skip training
                                        overall_general_model = None # This will cause the subsequent training steps to be skipped
                                        # Or you might want to 'continue' the loop for the next model_type/sensor_combo if this check is earlier
                                    else:
                                        overall_general_model = ModelClass_og(
                                            input_features=input_channels_og,
                                            seq_len=seq_len_og,
                                            d_model=current_d_model,
                                            transformer_nhead=current_nhead,
                                            transformer_nlayers=model_hyperparameters["transformer_nlayers"],
                                            transformer_dim_feedforward=model_hyperparameters["transformer_dim_feedforward"],
                                            dense_units=model_hyperparameters["dense_units"],
                                            dropout_rate=model_hyperparameters["dropout_rate"],
                                        )
                                elif current_model_type == "GRU":
                                    gru_units = model_hyperparameters["gru_units"] # Make sure TUNABLE_GRU_UNITS is defined
                                    overall_general_model = ModelClass_og(
                                        input_features=input_channels_og, # GRU_Only takes input_features
                                        seq_len=seq_len_og,
                                        gru_units=gru_units,
                                        dense_units=model_hyperparameters["dense_units"],
                                        dropout_rate=model_hyperparameters["dropout_rate"],
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
                                overall_general_model_save_path = os.path.join(
                                    run_specific_output_dir,
                                    "overall_general_model.pth",
                                )
                                plot_dir_og = os.path.join(run_specific_output_dir, 'plots')
                                os.makedirs(plot_dir_og, exist_ok=True) # Ensure plots subdir exists
                                
                                try:
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
                                    desc=f"Training Overall General ({current_model_type}, {combination_name}, HP {hp_idx+1})",
                                    device=DEVICE,
                                    weight_decay=current_general_model_weight_decay,
                                )
                                
                                relevant_hp = get_relevant_hyperparameters(current_model_type, model_hyperparameters)                                
                                overall_general_bundle = {
                                    'model_state_dict': overall_general_model.state_dict(),
                                    'scaler': scaler,
                                    'hyperparameters': {
                                        'model_hyperparameters': relevant_hp,
                                        'general_hyperparameters': general_hyperparameters
                                    },
                                    'model_type':current_model_type
                                }

                                # Save the complete bundle
                                bundle_save_path = os.path.join(run_specific_output_dir, 'overall_general_inference_bundle.pkl')
                                with open(bundle_save_path, 'wb') as f:
                                    pickle.dump(overall_general_bundle, f)
                                logging.info(f"Saved Overall General Inference Bundle to {bundle_save_path}")
                                
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
                        run_specific_output_dir
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
            # --- CHANGED HEADER ---
            summary_file.write(
                "  HP Combo | Model Type | Sensors  | Patients | Overall General Model (Validation)                    | Overall General Model (Test)                      | Average Personalized Model (Test)                 | Avg Personalization Change\n"
            )
            summary_file.write(
                "  Idx      |            |          | Suitable | Acc  | Prec | Rec  | F1   | AUC  | Sens | Spec | Acc  | Prec | Rec  | F1   | AUC  | Sens | Spec | Acc  | Prec | Rec  | F1   | AUC  | Sens | Spec | Acc Change\n" # Added 'Acc Change' for consistency
            )
            summary_file.write(
                "  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n" # Adjusted length
            )
            # --- END CHANGED HEADER ---

            for hp_combo_str in sorted(all_results.keys(), key=lambda x: int(x.split('_')[2])):
                hp_results = all_results[hp_combo_str]
                hp_combo_idx = int(hp_combo_str.split('_')[2]) + 1 # Extract actual index

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

                        # Fetch metrics for printing (now includes Sens/Spec from calculate_metrics)
                        val_acc = overall_general_val_metrics.get('accuracy', 0.0)
                        val_prec = overall_general_val_metrics.get('precision', 0.0)
                        val_rec = overall_general_val_metrics.get('recall', 0.0)
                        val_f1 = overall_general_val_metrics.get('f1_score', 0.0)
                        val_auc = overall_general_val_metrics.get('auc_roc', 0.0)
                        val_sens = overall_general_val_metrics.get('sensitivity', 0.0) # NEW
                        val_spec = overall_general_val_metrics.get('specificity', 0.0) # NEW

                        test_acc = overall_general_test_metrics.get('accuracy', 0.0)
                        test_prec = overall_general_test_metrics.get('precision', 0.0)
                        test_rec = overall_general_test_metrics.get('recall', 0.0)
                        test_f1 = overall_general_test_metrics.get('f1_score', 0.0)
                        test_auc = overall_general_test_metrics.get('auc_roc', 0.0)
                        test_sens = overall_general_test_metrics.get('sensitivity', 0.0) # NEW
                        test_spec = overall_general_test_metrics.get('specificity', 0.0) # NEW


                        overall_general_val_metrics_str = (
                            f"{val_acc:.2f} | {val_prec:.2f} | {val_rec:.2f} | {val_f1:.2f} | {val_auc:.2f} | {val_sens:.2f} | {val_spec:.2f}" # CHANGED format
                        )
                        overall_general_test_metrics_str = (
                            f"{test_acc:.2f} | {test_prec:.2f} | {test_rec:.2f} | {test_f1:.2f} | {test_auc:.2f} | {test_sens:.2f} | {test_spec:.2f}" # CHANGED format
                        )


                        personalization_data = combo_results.get(
                            "personalization", None
                        )
                        if personalization_data is not None:
                            avg_personalized_metrics = personalization_data.get(
                                "avg_personalized_metrics", None
                            )
                            if avg_personalized_metrics:
                                avg_pers_acc = avg_personalized_metrics.get('accuracy', 0.0)
                                avg_pers_prec = avg_personalized_metrics.get('precision', 0.0)
                                avg_pers_rec = avg_personalized_metrics.get('recall', 0.0)
                                avg_pers_f1 = avg_personalized_metrics.get('f1_score', 0.0)
                                avg_pers_auc = avg_personalized_metrics.get('auc_roc', 0.0)
                                avg_pers_sens = avg_personalized_metrics.get('sensitivity', 0.0) # NEW
                                avg_pers_spec = avg_personalized_metrics.get('specificity', 0.0) # NEW

                                avg_personalized_metrics_str = (
                                    f"{avg_pers_acc:.2f} | {avg_pers_prec:.2f} | {avg_pers_rec:.2f} | {avg_pers_f1:.2f} | {avg_pers_auc:.2f} | {avg_pers_sens:.2f} | {avg_pers_spec:.2f}" # CHANGED format
                                )

                                total_change_combo = 0
                                count_valid_patients_combo_change = 0
                                # Recalculate Avg Personalization Change here, directly from patient results
                                for patient_id, pers_results in personalization_data.get(
                                    "personalization_results", {}
                                ).items():
                                    if isinstance(
                                        pers_results.get("before"), dict
                                    ) and isinstance(pers_results.get("after"), dict):
                                        # Make sure 'after' test metrics exist and are not empty, and 'before' accuracy exists
                                        if (
                                            pers_results["after"].get("confusion_matrix", [[0,0],[0,0]]) and
                                            sum(sum(row) for row in pers_results["after"].get("confusion_matrix", [[0,0],[0,0]])) > 0 and
                                            'accuracy' in pers_results["before"] and 'accuracy' in pers_results["after"]
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
                            f"  {hp_combo_idx:<8} | {model_type:<10} | {combo_name:<8} | {num_suitable_patients:<8} | {overall_general_val_metrics_str:<53} | {overall_general_test_metrics_str:<53} | {avg_personalized_metrics_str:<53} | {avg_change_combo_str}\n" # CHANGED COLUMN SPACING
                        )

                    summary_file.write(
                        "  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n\n" # Adjusted length
                    )
    except Exception as e:
        logging.error(f"An error occurred while writing the final summary file: {e}")

    logging.info("--- All Runs Complete ---")
    logging.info(f"Results saved in the '{OUTPUT_DIR}' directory.")
    logging.info(f"Log file: {log_filename}")
    logging.info(f"Summary file: {summary_output_filename}")