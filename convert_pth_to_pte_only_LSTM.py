import logging
import math
import sys

import torch
import torch.nn as nn
from executorch.exir import to_edge
from torch.export import export


# --- NEW ExecuTorch Imports ---
# Make sure you have executorch i# (Replace your CNN_LSTM class for this test)
class TestModel_LSTM_Only(nn.Module):
    def __init__(
        self,
        input_channels,
        seq_len,
        conv_filters,
        conv_kernel_size,
        pool_size,
        lstm_units,
        dense_units,
        dropout_rate=0.0, # Dropout is ignored
    ):
        super(TestModel_LSTM_Only, self).__init__()
        self.input_channels = input_channels
        self.seq_len = seq_len
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.pool_size = pool_size
        self.lstm_units = lstm_units
        self.dense_units = dense_units

        if input_channels <= 0: input_channels = 1
        if seq_len <= 0: seq_len = 1
        if not conv_filters: conv_filters = [32]

        conv_layers_list = []
        in_channels = input_channels

        for i, out_channels in enumerate(conv_filters):
            kernel_size = max(1, conv_kernel_size)
            current_pool_size = max(1, self.pool_size)
            padding = kernel_size // 2

            conv_layers_list.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            )
            # --- BatchNorm1d REMOVED ---
            conv_layers_list.append(nn.ReLU())
            conv_layers_list.append(nn.MaxPool1d(current_pool_size))
            # --- Dropout REMOVED ---
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers_list)

        try:
            dummy_input = torch.randn(1, self.input_channels, self.seq_len, dtype=torch.float32)
            dummy_output = self.conv_layers(dummy_input)
            self.lstm_input_features = dummy_output.shape[1]
            self.lstm_input_seq_len = dummy_output.shape[2]
            if self.lstm_input_seq_len <= 0: raise ValueError("LSTM input seq len is zero")
        except Exception as e:
            logging.error(f"Error calculating layer output size: {e}")
            raise e

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_features,
            hidden_size=lstm_units,
            batch_first=True,
        )
        # --- lstm_dropout REMOVED ---

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
        # --- lstm_dropout call REMOVED ---
        mean_output = torch.mean(lstm_out, dim=1)  # shape: (batch_size, lstm_units)
        output = self.dense_layers(mean_output)
        return output
# --- END OF MODEL CLASS ---

# ... (keep all the imports)

if __name__ == "__main__":
    
    # --- 1. DEFINE YOUR FILE PATHS ---
    # Save it as a new test model
    PTE_OUTPUT_PATH = 'test_lstm_model.pte'
    # ⬇️ *** IMPORTANT: These values MUST match the model you trained ***
    MODEL_INPUT_CHANNELS = 4
    MODEL_SEQ_LEN = 30
    MODEL_CONV_FILTERS = [128, 256]
    MODEL_CONV_KERNEL_SIZE = 5
    MODEL_POOL_SIZE = 2
    MODEL_LSTM_UNITS = 128
    MODEL_DENSE_UNITS = 128
    MODEL_DROPOUT_RATE = 0.0 
    # --- 2. INSTANTIATE THE *SIMPLE* MODEL ---
    print(f"[1/7] Initializing SimpleModel...")
    model = TestModel_LSTM_Only(  # <--- USE THE NEW TEST CLASS
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
    
    # --- 3. NO .PTH FILE TO LOAD ---
    print(f"[2/7] Skipping weight loading for test...")
    
    # --- 4. SET TO EVAL MODE ---
    print("[3/7] Setting model to evaluation mode (model.eval())...")
    model.eval()
    print("      Model in eval mode.")
    
    # --- 5. CREATE EXAMPLE INPUT ---
    print(f"[4/7] Creating example input tensor...")
    # ⬇️ *** IMPORTANT: Input shape matches SimpleModel (1, 4, 30) ***
    example_input = (torch.randn(1, 4, 30),)
    print(f"      Example input shape: (1, 4, 30)")

    # --- 6. EXPORT ---
    print("[5/7] Exporting model with torch.export.export()...")
    edge_program = export(model, example_input)
    print("      Model exported.")

    # --- 7. CONVERT TO EDGE ---
    print("[6/7] Converting to ExecuTorch Edge dialect...")
    execu_program = to_edge(edge_program)
    print("      Model converted to Edge.")

    # --- 8. SAVE ---
    print(f"[7/7] Saving ExecuTorch model to '{PTE_OUTPUT_PATH}'...")
    with open(PTE_OUTPUT_PATH, "wb") as f:
        f.write(execu_program.to_executorch().buffer)
    print("      Done.")

    print(f"\n✅ SUCCESS!")
    print(f"Test model saved as '{PTE_OUTPUT_PATH}'")