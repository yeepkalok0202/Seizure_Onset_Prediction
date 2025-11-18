import logging
import math
import torch
import torch.nn as nn
import torch.utils.mobile_optimizer as mobile_optimizer
import sys # Added for exiting on error

# --- PASTE YOUR ENTIRE MODEL CLASS HERE ---
# (I have copied it exactly from your example)

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
        current_calc_seq_len = seq_len # For dummy pass calculation

        for i, out_channels in enumerate(conv_filters):
            kernel_size = max(1, conv_kernel_size)
            current_layer_pool_size = max(1, self.pool_size) 
            padding = kernel_size // 2

            conv_layers_list.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            )
            conv_layers_list.append(nn.BatchNorm1d(out_channels))
            conv_layers_list.append(nn.ReLU())
            conv_layers_list.append(nn.MaxPool1d(current_layer_pool_size))
            conv_layers_list.append(nn.Dropout(self.dropout_rate))
            in_channels = out_channels
            
            # Calculate sequence length after this conv-pool block
            current_calc_seq_len = math.floor((current_calc_seq_len + 2 * padding - kernel_size) / 1 + 1) # After Conv1d
            current_calc_seq_len = math.floor((current_calc_seq_len - current_layer_pool_size) / current_layer_pool_size + 1) # After MaxPool1d


        self.conv_layers = nn.Sequential(*conv_layers_list)

        try:
            # This dummy pass is critical for defining the LSTM input size
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
            input_size=self.lstm_input_features, # Features from CNN
            hidden_size=lstm_units,
            batch_first=True, # expects (batch, seq, feature)
        )
        self.lstm_dropout = nn.Dropout(self.dropout_rate)

        self.dense_layers = nn.Sequential(
            nn.Linear(lstm_units, dense_units),
            nn.ReLU(),
            nn.Linear(dense_units, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (batch_size, input_channels, seq_len)
        cnn_out = self.conv_layers(x)  # shape: (batch_size, last_cnn_filter_count, reduced_seq_len)
        
        if cnn_out.shape[2] == 0: 
            return torch.full((x.size(0), 1), 0.5, device=x.device, dtype=x.dtype)

        # Permute to (batch, seq_len_after_cnn, features_after_cnn) for LSTM
        lstm_in = cnn_out.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(lstm_in) 
        lstm_out = self.lstm_dropout(lstm_out)

        mean_output = torch.mean(lstm_out, dim=1)  # shape: (batch_size, lstm_units)
        
        output = self.dense_layers(mean_output) # shape: (batch_size, 1)
        return output

# --- END OF MODEL CLASS ---


if __name__ == "__main__":
    
    # --- 1. DEFINE YOUR FILE PATHS ---
    # ⬇️ *** IMPORTANT: Set this to the path of your trained .pth file ***
    PTH_FILE_PATH = 'overall_general_model.pth' 
    
    # ⬇️ This is the name of the output file you will use in React Native
    PTL_OUTPUT_PATH = 'ptl_version_CNN_LSTM.ptl'


    # --- 2. SET YOUR MODEL'S HYPERPARAMETERS ---
    # ⬇️ *** IMPORTANT: These values MUST match the model you trained ***
    # (Using the example values you provided)
    
    MODEL_INPUT_CHANNELS = 4
    MODEL_SEQ_LEN = 30
    MODEL_CONV_FILTERS = [128, 256]
    MODEL_CONV_KERNEL_SIZE = 5
    MODEL_POOL_SIZE = 2
    MODEL_LSTM_UNITS = 128
    MODEL_DENSE_UNITS = 128
    MODEL_DROPOUT_RATE = 0.0  # Set to 0.0 if you trained with 0.0, otherwise match it


    print(f"--- Starting PyTorch to PyTorch Lite Conversion ---")

    try:
        # --- Step 1: Instantiate the model ---
        # This re-creates the model architecture using your hyperparameters
        print(f"[1/6] Initializing model with {MODEL_INPUT_CHANNELS} channels and {MODEL_SEQ_LEN} seq_len...")
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
        # This loads your saved parameters into the model structure
        print(f"[2/6] Loading weights from '{PTH_FILE_PATH}'...")
        # Use map_location=torch.device('cpu') if your .pth was saved on a GPU
        # and you are converting on a machine without a GPU.
        model.load_state_dict(torch.load(PTH_FILE_PATH, map_location=torch.device('cpu')))
        print("     Weights loaded successfully.")

        # --- Step 3: Set model to evaluation mode ---
        # This is CRITICAL. It disables dropout and batch normalization updates.
        print("[3/6] Setting model to evaluation mode (model.eval())...")
        model.eval()
        print("     Model in eval mode.")

        # --- Step 4: "Script" the model using TorchScript ---
        # This converts your Python model code into a static, serializable graph
        print("[4/6] Scripting model with torch.jit.script()...")
        scripted_model = torch.jit.script(model)
        print("     Model scripted.")

        # --- Step 5: Optimize the model for mobile ---
        # This runs mobile-specific optimizations
        print("[5/6] Optimizing scripted model for mobile...")
        optimized_model = mobile_optimizer.optimize_for_mobile(scripted_model)
        print("     Model optimized.")

        # --- Step 6: Save the final .ptl file ---
        # This is the file you will copy to your React Native project
        print(f"[6/6] Saving optimized model to '{PTL_OUTPUT_PATH}'...")
        optimized_model.save(PTL_OUTPUT_PATH)
        print("     Done.")

        print(f"\n✅ SUCCESS!")
        print(f"Your model has been converted and saved as '{PTL_OUTPUT_PATH}'")
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