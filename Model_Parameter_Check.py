import logging
import math

import torch
import torch.nn as nn

# Configure basic logging in case the model's __init__ uses it for errors
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
            current_layer_pool_size = max(1, self.pool_size) # Use self.pool_size for consistency
            padding = kernel_size // 2

            conv_layers_list.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            )
            conv_layers_list.append(nn.BatchNorm1d(out_channels))
            conv_layers_list.append(nn.ReLU())
            conv_layers_list.append(nn.MaxPool1d(current_layer_pool_size))
            conv_layers_list.append(nn.Dropout(self.dropout_rate))
            in_channels = out_channels
            
            # Calculate sequence length after this conv-pool block for dummy pass
            current_calc_seq_len = math.floor((current_calc_seq_len + 2 * padding - kernel_size) / 1 + 1) # After Conv1d
            current_calc_seq_len = math.floor((current_calc_seq_len - current_layer_pool_size) / current_layer_pool_size + 1) # After MaxPool1d


        self.conv_layers = nn.Sequential(*conv_layers_list)

        try:
            dummy_input = torch.randn(1, self.input_channels, self.seq_len, dtype=torch.float32)
            dummy_output = self.conv_layers(dummy_input)
            self.lstm_input_features = dummy_output.shape[1] # Number of channels out of CNN
            self.lstm_input_seq_len = dummy_output.shape[2] # Sequence length out of CNN

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

        # Dense layers take LSTM hidden units as input because of mean pooling
        self.dense_layers = nn.Sequential(
            nn.Linear(lstm_units, dense_units),
            nn.ReLU(),
            nn.Linear(dense_units, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (batch_size, input_channels, seq_len)
        cnn_out = self.conv_layers(x)  # shape: (batch_size, last_cnn_filter_count, reduced_seq_len)
        
        if cnn_out.shape[2] == 0: # If sequence length becomes 0
            # Return a neutral prediction for each item in the batch
            return torch.full((x.size(0), 1), 0.5, device=x.device, dtype=x.dtype)

        # LSTM expects (batch, seq, feature)
        # cnn_out is (batch, features_after_cnn, seq_len_after_cnn)
        # So, permute to (batch, seq_len_after_cnn, features_after_cnn)
        lstm_in = cnn_out.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(lstm_in) # lstm_out shape: (batch_size, reduced_seq_len, lstm_units)
        lstm_out = self.lstm_dropout(lstm_out)

        # Apply mean pooling over the sequence dimension
        mean_output = torch.mean(lstm_out, dim=1)  # shape: (batch_size, lstm_units)
        
        output = self.dense_layers(mean_output) # shape: (batch_size, 1)
        return output

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # --- Define Example Hyperparameters for CNN_LSTM (Mean Version) ---
    # These should reflect a typical configuration you use or intend for on-device.
    # You can adjust these values to see how parameters change.

    # Taken from your `v5_test2.py` script's first options in TUNABLE_ lists
    # and general config
    example_input_channels = 4  # Assuming 4 base sensors ['HR', 'EDA', 'TEMP', 'ACC']
    example_seq_len = 30        # SEGMENT_DURATION_SECONDS * EXTERNAL_PREPROCESS_SAMPLING_FREQ_HZ = 30 * 1
    
    example_conv_filters = [128, 256] # First option from TUNABLE_CONV_FILTERS
    example_conv_kernel_size = 5      # First option from TUNABLE_CONV_KERNEL_SIZE
    example_pool_size = 2             # First option from TUNABLE_POOL_SIZE
    example_lstm_units = 128          # First option from TUNABLE_LSTM_UNITS
    example_dense_units = 128         # First option from TUNABLE_DENSE_UNITS
    example_dropout_rate = 0.0        # First option from TUNABLE_DROPOUT_RATE

    print(f"--- Checking Parameters for CNN_LSTM (Mean Version) ---")
    print(f"Configuration:")
    print(f"  Input Channels: {example_input_channels}")
    print(f"  Sequence Length: {example_seq_len}")
    print(f"  CNN Filters: {example_conv_filters}")
    print(f"  CNN Kernel Size: {example_conv_kernel_size}")
    print(f"  CNN Pool Size: {example_pool_size}")
    print(f"  LSTM Units: {example_lstm_units}")
    print(f"  Dense Units: {example_dense_units}")
    print(f"  Dropout Rate: {example_dropout_rate}\n")

    # Instantiate the model
    try:
        cnn_lstm_model = CNN_LSTM(
            input_channels=example_input_channels,
            seq_len=example_seq_len,
            conv_filters=example_conv_filters,
            conv_kernel_size=example_conv_kernel_size,
            pool_size=example_pool_size,
            lstm_units=example_lstm_units,
            dense_units=example_dense_units,
            dropout_rate=example_dropout_rate
        )

        # Count and print trainable parameters
        total_params = count_parameters(cnn_lstm_model)
        print(f"Total trainable parameters in CNN_LSTM (Mean Version): {total_params:,}")

        # Estimate model size (assuming float32, 4 bytes per parameter)
        model_size_bytes = total_params * 4
        model_size_kb = model_size_bytes / 1024
        model_size_mb = model_size_kb / 1024
        print(f"Estimated model size (float32 parameters):")
        print(f"  {model_size_bytes:,} bytes")
        print(f"  {model_size_kb:,.2f} KB")
        print(f"  {model_size_mb:,.2f} MB")

        # Optional: Print detailed parameter shapes
        # print("\nDetailed parameters:")
        # for name, parameter in cnn_lstm_model.named_parameters():
        #     if parameter.requires_grad:
        #         print(f"  {name}: {list(parameter.shape)} -> {parameter.numel():,} params")

    except ValueError as ve:
        print(f"\nERROR during model instantiation or parameter calculation: {ve}")
        print("This often happens if the sequence length becomes too small after CNN/pooling layers.")
        print("Please check your CNN configuration (filters, kernel size, pool size) relative to the input sequence length.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")