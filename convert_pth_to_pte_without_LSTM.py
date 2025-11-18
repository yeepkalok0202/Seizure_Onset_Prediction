import logging
import math
import sys

import torch
import torch.nn as nn
from executorch.exir import to_edge
from torch.export import export


# --- NEW ExecuTorch Imports ---
# Make sure you have executorch i# (Replace your CNN_LSTM class for this test)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # Input shape (1, 4, 30)
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        # After pool: (1, 8, 15)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 15, 10)
        self.fc2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.sigmoid(x)
# --- END OF MODEL CLASS ---

# ... (keep all the imports)

if __name__ == "__main__":
    
    # --- 1. DEFINE YOUR FILE PATHS ---
    # Save it as a new test model
    PTE_OUTPUT_PATH = 'simple_CNN.pte'

    # --- 2. INSTANTIATE THE *SIMPLE* MODEL ---
    print(f"[1/7] Initializing SimpleModel...")
    model = SimpleModel()
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