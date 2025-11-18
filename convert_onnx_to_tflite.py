import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import sys

# --- 1. DEFINE YOUR FILE PATHS ---
ONNX_FILE_PATH = 'model.onnx'       # Input from Step 1
TF_MODEL_DIR = 'tf_saved_model'   # Intermediate folder
TFLITE_OUTPUT_PATH = 'model.tflite' # Final output file

print("--- Starting ONNX (.onnx) to TFLite (.tflite) Conversion ---")

try:
    # --- Step 2.1: Load the ONNX model ---
    print(f"[1/4] Loading ONNX model from '{ONNX_FILE_PATH}'...")
    onnx_model = onnx.load(ONNX_FILE_PATH)
    onnx.checker.check_model(onnx_model) # Check if the model is valid
    print("     ONNX model loaded and checked.")

    # --- Step 2.2: Convert ONNX to TensorFlow SavedModel ---
    print(f"[2/4] Converting ONNX to TensorFlow SavedModel (in '{TF_MODEL_DIR}')...")
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(TF_MODEL_DIR)
    print("     TensorFlow SavedModel created.")

    # --- Step 2.3: Convert SavedModel to TFLite ---
    print(f"[3/4] Initializing TFLiteConverter...")
    converter = tf.lite.TFLiteConverter.from_saved_model(TF_MODEL_DIR)
    
    # Optional: Add optimizations
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    print("[4/4] Converting to TFLite format...")
    tflite_model = converter.convert()
    print("     Conversion complete.")

    # --- Step 2.4: Save the .tflite file ---
    with open(TFLITE_OUTPUT_PATH, 'wb') as f:
        f.write(tflite_model)

    print(f"\n✅ SUCCESS! Model saved as '{TFLITE_OUTPUT_PATH}'")
    print(f"You can now use this file in your React Native app.")

except Exception as e:
    print(f"\n❌ ERROR during conversion:")
    print(e)
    print("\nThis often happens if an operation in your PyTorch model")
    print("is not supported by the ONNX-TF or TFLite converter.")
    sys.exit(1)