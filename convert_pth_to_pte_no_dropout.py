import logging
import math
import pickle
import sys

import torch
import torch.ao.quantization
import torch.nn as nn
from executorch.exir import to_edge
from torch.export import export


# --- PASTE YOUR *ORIGINAL* CNN_LSTM MODEL CLASS HERE ---
# (The one with BatchNorm1d and Dropout)
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
            conv_layers_list.append(nn.Dropout(self.dropout_rate)) # <-- Problem op
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers_list)

        try:
            dummy_input = torch.randn(1, self.input_channels, self.seq_len, dtype=torch.float32)
            dummy_output = self.conv_layers(dummy_input)
            self.lstm_input_features = dummy_output.shape[1]
            self.lstm_input_seq_len = dummy_output.shape[2]
            if self.lstm_input_seq_len <= 0:
                raise ValueError("Calculated LSTM input sequence length is zero")
        except Exception as e:
            logging.error(f"Error calculating layer output size: {e}")
            raise e

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_features,
            hidden_size=lstm_units,
            batch_first=True,
        )
        self.lstm_dropout = nn.Dropout(self.dropout_rate) # <-- Problem op

        self.dense_layers = nn.Sequential(
            nn.Linear(lstm_units, dense_units),
            nn.ReLU(),
            nn.Linear(dense_units, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        cnn_out = self.conv_layers(x)
        lstm_in = cnn_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out = self.lstm_dropout(lstm_out)
        mean_output = torch.mean(lstm_out, dim=1)
        output = self.dense_layers(mean_output)
        return output
# --- END OF MODEL CLASS ---


# --- ⬇️ NEW HELPER FUNCTION ⬇️ ---
def replace_dropout_with_identity(module):
    """
    Recursively iterates through all modules and replaces
    nn.Dropout with nn.Identity.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Dropout):
            print(f"      Replacing {name} (Dropout) with Identity.")
            setattr(module, name, nn.Identity())
        else:
            replace_dropout_with_identity(child)
# --- ⬆️ NEW HELPER FUNCTION ⬆️ ---


if __name__ == "__main__":
    
    PTH_FILE_PATH = 'overall_general_model.pth' 
    PTE_OUTPUT_PATH = 'model_CNN_LSTM_no_DROPOUT.pte' # Saving as a new file

    MODEL_INPUT_CHANNELS = 4
    MODEL_SEQ_LEN = 30
    MODEL_CONV_FILTERS = [128, 256]
    MODEL_CONV_KERNEL_SIZE = 5
    MODEL_POOL_SIZE = 2
    MODEL_LSTM_UNITS = 128
    MODEL_DENSE_UNITS = 128
    MODEL_DROPOUT_RATE = 0.0
    INFERENCE_BUNDLE_PATH = "C:\\Users\\XxOx\\Desktop\\FYP1 Seizure\\final_v1\\training_data\\20250609_061959_allModels_persona_0.5\\CNN-LSTM\\HR_EDA_TEMP_ACC\\hp_combo_1\\personalized\\MSEL_01110\\patient_MSEL_01110_inference_bundle.pkl"

    print(f"--- Starting Final Conversion (Fusing + Dropout Removal) ---")
    import json

    with open("model_state_dict.json") as f:
        json_state = json.load(f)

    restored_state = {k: torch.tensor(v) for k, v in json_state.items()}
    try:
        # Step 1: Instantiate
        print(f"[1/9] Initializing ORIGINAL model architecture...")
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

        # Step 2: Load weights
        print(f"[2/9] Loading weights from '{PTH_FILE_PATH}'...")
        # model.load_state_dict(torch.load(PTH_FILE_PATH, map_location=torch.device('cpu')))
        model.load_state_dict(restored_state)
        print("      Weights loaded successfully.")

        # Step 3: Set to eval mode
        print("[3/9] Setting model to evaluation mode (model.eval())...")
        model.eval()
        print("      Model in eval mode.")
        
        # Step 4: Fuse Conv + BN
        print("[4/9] Fusing Conv1d and BatchNorm1d layers...")
        modules_to_fuse = []
        for i in range(len(model.conv_layers)):
            if isinstance(model.conv_layers[i], nn.Conv1d) and \
               i + 1 < len(model.conv_layers) and \
               isinstance(model.conv_layers[i+1], nn.BatchNorm1d):
                modules_to_fuse.append([str(i), str(i+1)])
        
        if modules_to_fuse:
            print(f"      Found {len(modules_to_fuse)} pairs to fuse: {modules_to_fuse}")
            torch.ao.quantization.fuse_modules(model.conv_layers, modules_to_fuse, inplace=True)
            print("      Fusing complete.")
        else:
            print("      No Conv/BN pairs found to fuse.")

        # --- ⬇️ STEP 5 (NEW): REPLACE DROPOUT ⬇️ ---
        print("[5/9] Replacing all nn.Dropout layers with nn.Identity...")
        replace_dropout_with_identity(model)
        print("      Dropout replacement complete.")
        # --- ⬆️ END OF NEW STEP ⬆️ ---

        # Step 6: Create example input
        print(f"[6/9] Creating example input tensor...")
        example_input = (torch.randn(1, MODEL_INPUT_CHANNELS, MODEL_SEQ_LEN),)
        print(f"      Example input shape: (1, {MODEL_INPUT_CHANNELS}, {MODEL_SEQ_LEN})")

        # Step 7: Export
        print("[7/9] Exporting *final* model with torch.export.export()...")
        edge_program = export(model, example_input)
        print("      Model exported.")

        # Step 8: Convert
        print("[8/9] Converting to ExecuTorch Edge dialect...")
        execu_program = to_edge(edge_program)
        print("      Model converted to Edge.")

        # Step 9: Save
        print(f"[9/9] Saving ExecuTorch model to '{PTE_OUTPUT_PATH}'...")
        with open(PTE_OUTPUT_PATH, "wb") as f:
            f.write(execu_program.to_executorch().buffer)
        print("      Done.")

        print(f"\n✅ SUCCESS!")
        print(f"Your model has been saved as '{PTE_OUTPUT_PATH}'")
        print("This graph is now clean of BatchNorm and Dropout. It should work.")

    except Exception as e:
        print(f"\n❌ An unexpected error occurred:")
        print(e)
        import traceback
        traceback.print_exc()
        sys.exit(1)