import logging
import math
import sys

import torch
import torch.nn as nn
from executorch.exir import to_edge
# --- NEW ExecuTorch Imports ---
# Make sure you have executorch installed: pip install executorch
from torch.export import export

# --- PASTE YOUR ENTIRE MODEL CLASS HERE ---
# (Copied exactly from your example)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

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
        lstm_in = cnn_out.permute(0, 2, 1)  # shape: (batch_size, seq_len, features)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out = self.lstm_dropout(lstm_out)
        mean_output = torch.mean(lstm_out, dim=1)  # shape: (batch_size, lstm_units)
        output = self.dense_layers(mean_output)
        return output

# --- END OF MODEL CLASS ---


if __name__ == "__main__":
    
    # --- 1. DEFINE YOUR FILE PATHS ---
    PTH_FILE_PATH = 'overall_general_model.pth' 
    
    # This is the output file for react-native-executorch
    PTE_OUTPUT_PATH = 'model.pte'


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


    print(f"--- Starting PyTorch to ExecuTorch (.pte) Conversion ---")

    try:
        # --- Step 1: Instantiate the model ---
        print(f"[1/7] Initializing model with {MODEL_INPUT_CHANNELS} channels and {MODEL_SEQ_LEN} seq_len...")
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
        print("     Model architecture initialized.")

        # --- Step 2: Load your trained weights (.pth) ---
        print(f"[2/7] Loading weights from '{PTH_FILE_PATH}'...")
        model.load_state_dict(torch.load(PTH_FILE_PATH, map_location=torch.device('cpu')))
        print("     Weights loaded successfully.")

        # --- Step 3: Set model to evaluation mode ---
        print("[3/7] Setting model to evaluation mode (model.eval())...")
        model.eval()
        print("     Model in eval mode.")
        
        # --- Step 4: Create an example input (REQUIRED for ExecuTorch) ---
        # This is used to "trace" the model's operations.
        # Shape must be (batch_size, channels, seq_len)
        print(f"[4/7] Creating example input tensor...")
        example_input = (torch.randn(1, MODEL_INPUT_CHANNELS, MODEL_SEQ_LEN),)
        print(f"     Example input shape: (1, {MODEL_INPUT_CHANNELS}, {MODEL_SEQ_LEN})")


        # --- Step 5: Export the model ---
        print("[5/7] Exporting model with torch.export.export()...")
        edge_program = export(model, example_input)
        print("     Model exported.")

        # --- Step 6: Convert to ExecuTorch Edge dialect ---
        print("[6/7] Converting to ExecuTorch Edge dialect...")
        execu_program = to_edge(edge_program)
        print("     Model converted to Edge.")

        # --- Step 7: Save the final .pte file ---
        print(f"[7/7] Saving ExecuTorch model to '{PTE_OUTPUT_PATH}'...")
        with open(PTE_OUTPUT_PATH, "wb") as f:
            f.write(execu_program.to_executorch().buffer)
        print("     Done.")

        print(f"\n✅ SUCCESS!")
        print(f"Your model has been converted and saved as '{PTE_OUTPUT_PATH}'")
        print("You can now copy this file into your React Native project.")

    except FileNotFoundError:
        print(f"\n❌ ERROR: File not found.")
        print(f"Could not find the model file at: {PTH_FILE_PATH}")
        print("Please make sure the path is correct and the file exists.")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n❌ ERROR during weight loading or model instantiation:")
        print(e)
        print("\nThis *often* means the hyperparameters in section 2 of this script")
        print("do NOT match the hyperparameters of the model saved in your .pth file.")
        print("Please double-check all 'MODEL_' variables.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred:")
        print(e)
        sys.exit(1)