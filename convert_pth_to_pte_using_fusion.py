import logging
import math
import sys

import torch
# --- NEW IMPORT FOR FUSING ---
import torch.ao.quantization
import torch.nn as nn
from executorch.exir import to_edge
from torch.export import export


# --- PASTE YOUR *ORIGINAL* CNN_LSTM MODEL CLASS HERE ---
# (The one with BatchNorm1d and Dropout in it)
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

            # --- This is the original structure ---
            conv_layers_list.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            )
            conv_layers_list.append(nn.BatchNorm1d(out_channels)) # <-- The culprit
            conv_layers_list.append(nn.ReLU())
            conv_layers_list.append(nn.MaxPool1d(current_pool_size))
            conv_layers_list.append(nn.Dropout(self.dropout_rate)) # <-- This is ok
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
        # --- REMOVED THE IF STATEMENT (as we did before) ---
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
    PTE_OUTPUT_PATH = 'fusion_CNN_LSTM.pte' # The final, working model

    # --- 2. SET YOUR MODEL'S HYPERPARAMETERS ---
    # ⬇️ *** These MUST match your .pth file ***
    MODEL_INPUT_CHANNELS = 4
    MODEL_SEQ_LEN = 30
    MODEL_CONV_FILTERS = [128, 256]
    MODEL_CONV_KERNEL_SIZE = 5
    MODEL_POOL_SIZE = 2
    MODEL_LSTM_UNITS = 128
    MODEL_DENSE_UNITS = 128
    MODEL_DROPOUT_RATE = 0.0 # This value doesn't matter for eval

    print(f"--- Starting PyTorch to ExecuTorch (with Fusing) ---")

    try:
        # --- Step 1: Instantiate the *original* model ---
        print(f"[1/8] Initializing ORIGINAL model architecture...")
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
        print("      Model architecture initialized.")

        # --- Step 2: Load your trained weights (.pth) ---
        print(f"[2/8] Loading weights from '{PTH_FILE_PATH}'...")
        model.load_state_dict(torch.load(PTH_FILE_PATH, map_location=torch.device('cpu')))
        print("      Weights loaded successfully.")

        # --- Step 3: Set model to evaluation mode ---
        print("[3/8] Setting model to evaluation mode (model.eval())...")
        model.eval()
        print("      Model in eval mode.")
        
        # --- Step 4 (NEW): Fuse Conv1d + BatchNorm1d ---
        print("[4/8] Fusing Conv1d and BatchNorm1d layers...")
        
        # Find the pairs to fuse: (Conv, BN)
        # In your model, they are ['0', '1'], ['5', '6'], ['10', '11'], etc.
        modules_to_fuse = []
        for i in range(len(model.conv_layers)):
            if isinstance(model.conv_layers[i], nn.Conv1d) and \
               i + 1 < len(model.conv_layers) and \
               isinstance(model.conv_layers[i+1], nn.BatchNorm1d):
                
                modules_to_fuse.append([str(i), str(i+1)])
        
        if not modules_to_fuse:
            print("      WARNING: Could not find any Conv1d/BatchNorm1d pairs to fuse.")
        else:
            print(f"      Found {len(modules_to_fuse)} pairs to fuse: {modules_to_fuse}")
            torch.ao.quantization.fuse_modules(model.conv_layers, modules_to_fuse, inplace=True)
            print("      Fusing complete.")
        
        # --- Step 5: Create an example input ---
        print(f"[5/8] Creating example input tensor...")
        example_input = (torch.randn(1, MODEL_INPUT_CHANNELS, MODEL_SEQ_LEN),)
        print(f"      Example input shape: (1, {MODEL_INPUT_CHANNELS}, {MODEL_SEQ_LEN})")

        # --- Step 6: Export the model ---
        print("[6/8] Exporting *fused* model with torch.export.export()...")
        edge_program = export(model, example_input)
        print("      Model exported.")

        # --- Step 7: Convert to ExecuTorch Edge dialect ---
        print("[7/8] Converting to ExecuTorch Edge dialect...")
        execu_program = to_edge(edge_program)
        print("      Model converted to Edge.")

        # --- Step 8: Save the final .pte file ---
        print(f"[8/8] Saving ExecuTorch model to '{PTE_OUTPUT_PATH}'...")
        with open(PTE_OUTPUT_PATH, "wb") as f:
            f.write(execu_program.to_executorch().buffer)
        print("      Done.")

        print(f"\n✅ SUCCESS!")
        print(f"Your model has been converted and saved as '{PTE_OUTPUT_PATH}'")
        print("This one should work!")

    except FileNotFoundError:
        print(f"\n❌ ERROR: File not found: {PTH_FILE_PATH}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n❌ ERROR during weight loading or model instantiation:")
        print(e)
        print("\n*** Double-check your hyperparameters in Step 2! ***")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred:")
        print(e)
        sys.exit(1)