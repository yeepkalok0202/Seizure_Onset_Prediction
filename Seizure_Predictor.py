# predictor.py

import torch
import torch.nn as nn
import time
import pickle
import gradio as gr
import pandas as pd
from collections import deque
from multiprocessing import Process, Queue, Manager

from Final_Clean_v1 import get_model_class

# =================================================================================
# IMPORTANT: PASTE ALL YOUR MODEL CLASS DEFINITIONS AND get_model_class HERE
# =================================================================================

# (Your model classes and get_model_class function go here)


# =================================================================================
# The model prediction process (runs in the background)
# =================================================================================
def prediction_process(data_queue, shared_prediction, shared_history, shared_pickled_df, config):
    # (This function is unchanged)
    print("[Predictor] Process started. Loading model and scaler from bundle...")
    try:
        bundle = pickle.load(open(config['INFERENCE_BUNDLE_PATH'], 'rb'))
        model_hyperparams = bundle['hyperparameters']['model_hyperparameters']
        model_state_dict = bundle['model_state_dict']
        model_type = bundle['model_type']
        scaler = bundle['scaler']
        
        ModelClass = get_model_class(model_type)
        num_features = len(config['BASE_SENSORS'])
        seq_len = config['SEGMENT_SECONDS'] * config['TARGET_SAMPLING_HZ']
        
        model = ModelClass(input_channels=num_features, seq_len=seq_len, **model_hyperparams)
        model.load_state_dict(model_state_dict)
        model.eval()
        
        print(f"[Predictor] Model '{model_type}' and scaler loaded successfully.")
    except Exception as e:
        print(f"[Predictor] FATAL: Error loading bundle: {e}")
        return

    print("[Predictor] Waiting for DataFrames from the queue...")
    while True:
        try:
            processed_df = data_queue.get()
            shared_pickled_df.value = pickle.dumps(processed_df)

            scaled_segment = scaler.transform(processed_df.values)
            segment_tensor = torch.tensor(scaled_segment, dtype=torch.float32).permute(1, 0).unsqueeze(0)
            
            with torch.no_grad():
                output = model(segment_tensor)
                probability = torch.sigmoid(output).item()

            shared_prediction.value = probability
            shared_history.append(probability)
            
            print(f"[Predictor] Prediction: {probability:.4f}")
        except Exception as e:
            print(f"[Predictor] An error occurred during prediction: {e}")
            time.sleep(1)

# =================================================================================
# The main function that launches the Gradio UI
# =================================================================================
def launch_predictor_ui(data_queue, config):
    with Manager() as manager:
        shared_prediction = manager.Value('f', 0.0)
        shared_history = manager.list()
        shared_pickled_df = manager.Value('c', b'')
        
        plot_history = deque(maxlen=60)

        predictor_proc = Process(target=prediction_process, args=(data_queue, shared_prediction, shared_history, shared_pickled_df, config))
        predictor_proc.daemon = True
        predictor_proc.start()

        def stream_updates():
            while True:
                # --- Update Prediction Plot ---
                if shared_history:
                    for _ in range(len(shared_history)):
                        plot_history.append(shared_history.pop(0))
                latest_prob = shared_prediction.value
                prediction_label_text = f"Seizure Probability: {latest_prob:.2%}"
                prediction_plot_data = pd.DataFrame({
                    "Time (seconds ago)": range(-len(plot_history) + 1, 1),
                    "Probability": list(plot_history)
                })

                # --- Prepare data for the 4 sensor plots using real timestamps ---
                pickled_df = shared_pickled_df.value
                if pickled_df:
                    sensor_df = pickle.loads(pickled_df)
                    
                    # --- FIX: Turn the DataFrame's datetime index into a 'Timestamp' column ---
                    sensor_df.reset_index(inplace=True)
                    sensor_df.rename(columns={'index': 'Timestamp'}, inplace=True)
                    
                    # Create the data for each plot using the new 'Timestamp' column
                    hr_data = sensor_df[['Timestamp', 'HR']] if 'HR' in sensor_df.columns else pd.DataFrame({'Timestamp': [], 'HR': []})
                    eda_data = sensor_df[['Timestamp', 'EDA']] if 'EDA' in sensor_df.columns else pd.DataFrame({'Timestamp': [], 'EDA': []})
                    temp_data = sensor_df[['Timestamp', 'TEMP']] if 'TEMP' in sensor_df.columns else pd.DataFrame({'Timestamp': [], 'TEMP': []})
                    acc_data = sensor_df[['Timestamp', 'ACC']] if 'ACC' in sensor_df.columns else pd.DataFrame({'Timestamp': [], 'ACC': []})
                else:
                    # If there's no data yet, create empty DataFrames
                    hr_data = pd.DataFrame({'Timestamp': [], 'HR': []})
                    eda_data = pd.DataFrame({'Timestamp': [], 'EDA': []})
                    temp_data = pd.DataFrame({'Timestamp': [], 'TEMP': []})
                    acc_data = pd.DataFrame({'Timestamp': [], 'ACC': []})
                
                yield prediction_label_text, prediction_plot_data, hr_data, eda_data, temp_data, acc_data
                
                time.sleep(1)

        with gr.Blocks(theme=gr.themes.Base(), css=".gradio-container {background-color: #1a202c;}") as demo:
            gr.Markdown("# 🧠 Real-Time Seizure Prediction")
            with gr.Row():
                prediction_label = gr.Label(value="Waiting for first prediction...", label="Prediction")
            
            with gr.Tab("Prediction History"):
                prediction_plot = gr.LinePlot(
                    x="Time (seconds ago)", y="Probability",
                    y_lim=[0, 1], title="Prediction History (Last 60s)",
                    show_label=False, width=700, height=400
                )

            with gr.Tab("Live Sensor Data Segment (30s)"):
                with gr.Row():
                    # --- FIX: Update plots to use the 'Timestamp' column for the x-axis ---
                    hr_plot = gr.LinePlot(x="Timestamp", y="HR", title="Heart Rate", width=400, height=250)
                    eda_plot = gr.LinePlot(x="Timestamp", y="EDA", title="Electrodermal Activity", width=400, height=250)
                with gr.Row():
                    temp_plot = gr.LinePlot(x="Timestamp", y="TEMP", title="Temperature", width=400, height=250)
                    acc_plot = gr.LinePlot(x="Timestamp", y="ACC", title="Accelerometer", width=400, height=250)
            
            demo.load(
                fn=stream_updates, 
                inputs=None, 
                outputs=[prediction_label, prediction_plot, hr_plot, eda_plot, temp_plot, acc_plot]
            )

        print("[Gradio] Launching UI at http://127.0.0.1:7860 (or a similar address)...")
        demo.launch()