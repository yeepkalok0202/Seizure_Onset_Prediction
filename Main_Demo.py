# main_app.py

from multiprocessing import Process, Queue
import time

# Import the target functions from the other files
from Preprocessing_Daemon import preprocessing_process
from Seizure_Predictor import launch_predictor_ui

if __name__ == '__main__':
    # --- Master Configuration ---
    config = {
        'DATA_EMITTER_URL': 'http://127.0.0.1:5000/data',
        'TARGET_SAMPLING_HZ': 1,
        'SEGMENT_SECONDS': 30,
        'BASE_SENSORS': ['HR', 'EDA', 'TEMP', 'ACC'],
        'EMISSION_RATE_HZ': 1,
        'INFERENCE_BUNDLE_PATH': 'final_v1/training_data/20250608_172456/CNN-LSTM/HR_EDA_TEMP_ACC/hp_combo_1/overall_general_inference_bundle.pkl'
    }

    # 1. Create the shared queue that will pass the DataFrame
    data_queue = Queue(maxsize=10)

    # 2. Create the process for the preprocessor
    preprocessor = Process(target=preprocessing_process, args=(data_queue, config))

    # 3. Create the process for the predictor and UI
    predictor_ui = Process(target=launch_predictor_ui, args=(data_queue, config))

    print("[MainApp] Starting all processes...")
    preprocessor.start()
    predictor_ui.start()

    try:
        preprocessor.join()
        predictor_ui.join()
    except KeyboardInterrupt:
        print("\n[MainApp] Shutdown signal received. Terminating processes.")
        preprocessor.terminate()
        predictor_ui.terminate()