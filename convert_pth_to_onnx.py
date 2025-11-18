import logging
import math

import torch
import torch.nn as nn
import torch.onnx


# --- PASTE YOUR CNN_LSTM MODEL CLASS DEFINITION HERE ---
# (Make sure this class definition is in the script)
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

# ----------------------------------------------------
# --- Main Conversion Script ---
# ----------------------------------------------------
if __name__ == "__main__":

    # --- 1. DEFINE YOUR FILE PATHS ---
    PTH_FILE_PATH = 'overall_general_model.pth' 
    ONNX_OUTPUT_PATH = 'model.onnx' # The file we will create

    # --- 2. SET YOUR MODEL'S HYPERPARAMETERS ---
    # ⬇️ *** IMPORTANT: These values MUST match the model you trained ***
    MODEL_INPUT_CHANNELS = 4
    MODEL_SEQ_LEN = 30
    MODEL_CONV_FILTERS = [128, 256]
    MODEL_CONV_KERNEL_SIZE = 5
    MODEL_POOL_SIZE = 2
    MODEL_LSTM_UNITS = 128
    MODEL_DENSE_UNITS = 128
    MODEL_DROPOUT_RATE = 0.0 

    print("--- Starting PyTorch (.pth) to ONNX (.onnx) Conversion ---")

    # --- Step 1.1: Instantiate the model ---
    print("[1/5] Initializing model architecture...")
    model = CNN_LSTM(
        input_channels=MODEL_INPUT_CHANNELS,
        seq_len=MODEL_SEQ_LEN,
        conv_filters=MODEL_CONV_FILTERS,
        conv_kernel_size=MODEL_CONV_KERNEL_SIZE,
        pool_size=MODEL_POOL_SIZE,
        lstm_units=MODEL_LSTM_UNITS,
        dense_units=MODEL_DENSE_UNITS,
        dropout_rate=MODEL_DROPOUT_RATE
    )

    # --- Step 1.2: Load your trained weights (.pth) ---
    print(f"[2/5] Loading weights from '{PTH_FILE_PATH}'...")
    model.load_state_dict(torch.load(PTH_FILE_PATH, map_location=torch.device('cpu')))

    # --- Step 1.3: Set model to evaluation mode ---
    print("[3/5] Setting model to evaluation mode (model.eval())...")
    model.eval()
    
    # --- Step 1.4: Create a dummy input tensor ---
    # Shape must be (batch_size, channels, seq_len)
    print(f"[4/5] Creating dummy input tensor of shape (1, {MODEL_INPUT_CHANNELS}, {MODEL_SEQ_LEN})...")
    dummy_input = torch.randn(1, MODEL_INPUT_CHANNELS, MODEL_SEQ_LEN)

    # --- Step 1.5: Export to ONNX ---
    print(f"[5/5] Exporting model to '{ONNX_OUTPUT_PATH}'...")
    torch.onnx.export(
        model,               # The model to export
        dummy_input,         # A sample input to trace the model
        ONNX_OUTPUT_PATH,    # Where to save the file
        input_names=['input'],   # Name for the input node
        output_names=['output'],  # Name for the output node
        opset_version=11,    # A stable ONNX version
        dynamic_axes={'input': {0: 'batch_size'},  # Allows variable batch size
                    'output': {0: 'batch_size'}}
    )
    
    print(f"\n✅ SUCCESS! Model saved as '{ONNX_OUTPUT_PATH}'")