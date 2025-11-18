import json
import os  # Import os to handle file paths
import pickle

import numpy as np

# --- Master Configuration (from your details) ---
config = {
    'DATA_EMITTER_URL': 'http://127.0.0.1:5000/data',
    'TARGET_SAMPLING_HZ': 1,
    'SEGMENT_SECONDS': 30,
    'BASE_SENSORS': ['HR', 'EDA', 'TEMP', 'ACC'],
    'EMISSION_RATE_HZ': 1,
    'INFERENCE_BUNDLE_PATH': 'final_v1\\training_data\\20250609_061959_allModels_persona_0.5\\CNN-LSTM\\HR_EDA_TEMP_ACC\\hp_combo_1\\personalized\\MSEL_01110\\patient_MSEL_01110_inference_bundle.pkl'
}

# --- Output Path ---
JSON_OUT_PATH = 'config_for_mobile.json'

# --- Automatically calculate values ---
BUNDLE_PATH = config['INFERENCE_BUNDLE_PATH']
SEQUENCE_LENGTH = config['SEGMENT_SECONDS'] * config['TARGET_SAMPLING_HZ'] # 30 * 1 = 30
BASE_SENSORS = config['BASE_SENSORS'] # ['HR', 'EDA', 'TEMP', 'ACC']

print(f"Loading inference bundle from {BUNDLE_PATH}...")

# Check if the bundle file exists
if not os.path.exists(BUNDLE_PATH):
    print(f"\n--- FATAL ERROR ---")
    print(f"File not found at: {BUNDLE_PATH}")
    print("Please make sure the 'INFERENCE_BUNDLE_PATH' in your config is correct.")
else:
    try:
        # 1. Load your .pkl file
        with open(BUNDLE_PATH, 'rb') as f:
            bundle = pickle.load(f)

        # 2. Extract the data we need for the mobile app
        scaler = bundle['scaler']
        hyperparams = bundle['hyperparameters']
        model_type = bundle['model_type']

        # 3. Create the mobile-friendly config dictionary
        mobile_config = {
            'model_type': model_type,
            'sequence_length': SEQUENCE_LENGTH,
            'base_sensors': BASE_SENSORS,
            'scaler_data_min': scaler.data_min_.tolist(),  # e.g., [min_hr, min_eda, ...]
            'scaler_data_range': scaler.data_range_.tolist(), # e.g., [range_hr, range_eda, ...]
            'hyperparameters': hyperparams
        }

        # 4. Save the config as a JSON file
        with open(JSON_OUT_PATH, 'w') as f:
            json.dump(mobile_config, f, indent=2)

        print(f"\n--- SUCCESS ---")
        print(f"Successfully exported mobile config to: {JSON_OUT_PATH}")
        print(f"  - Sequence Length: {SEQUENCE_LENGTH}")
        print(f"  - Features: {', '.join(BASE_SENSORS)}")
        print("Next step: Add this JSON file and your 'model.pte' to your React Native project.")

    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"An error occurred: {e}")
        print("Please check your .pkl file is not corrupt and Python environment is correct.")