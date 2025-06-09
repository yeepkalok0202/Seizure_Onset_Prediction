import pickle
import time
from collections import deque
from multiprocessing import Manager, Process, Queue

import gradio as gr
import pandas as pd
import torch
import torch.nn as nn

from Final_Clean_v1 import get_model_class


# =================================================================================
# The model prediction process (runs in the background)
# =================================================================================
def prediction_process(data_queue, shared_prediction, shared_history, shared_pickled_df, shared_status, config):
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

    print("[Predictor] Waiting for data/status packets from the queue...")
    while True:
        try:
            packet = data_queue.get()
            
            if packet['status'] == 'OK':
                shared_status['status'] = 'OK'
                shared_status['missing'] = []

                processed_df = packet['data']
                shared_pickled_df.value = pickle.dumps(processed_df)
                scaled_segment = scaler.transform(processed_df.values)
                segment_tensor = torch.tensor(scaled_segment, dtype=torch.float32).permute(1, 0).unsqueeze(0)
                
                with torch.no_grad():
                    output = model(segment_tensor)
                    probability = torch.sigmoid(output).item()
                
                shared_prediction.value = probability
                shared_history.append(probability)
                print(f"[Predictor] Prediction: {probability:.4f}")

            elif packet['status'] == 'ERROR':
                missing = packet['missing_sensors']
                shared_status['status'] = 'ERROR'
                shared_status['missing'] = missing
                print(f"[Predictor] Received ERROR status for sensors: {missing}. Halting prediction.")
                shared_pickled_df.value = pickle.dumps(None)
                continue
            
        except Exception as e:
            print(f"[Predictor] An error occurred during prediction loop: {e}")
            time.sleep(1)

# =================================================================================
# The main function that launches the Gradio UI
# =================================================================================
def launch_predictor_ui(data_queue, config):
    with Manager() as manager:
        shared_prediction = manager.Value('f', 0.0)
        shared_history = manager.list()
        shared_pickled_df = manager.Value('c', b'')
        shared_status = manager.dict({'status': 'OK', 'missing': []})
        
        plot_history = deque(maxlen=60)

        predictor_proc = Process(target=prediction_process, args=(data_queue, shared_prediction, shared_history, shared_pickled_df, shared_status, config))
        predictor_proc.daemon = True
        predictor_proc.start()

        def stream_updates():
            while True:
                current_status = shared_status.get('status', 'OK')
                error_update = None
                if current_status == 'ERROR':
                    missing_list = shared_status.get('missing', [])
                    
                    # --- MODIFIED: Added margin styles to the error messages ---
                    if 'COMM_FAILURE' in missing_list:
                         error_text = f"""
                         <div style='margin-top: 20px; margin-bottom: 20px;'>
                            <h1 style='text-align: center; color: #f56565; margin-bottom: 5px;'>&#9888; DATA EMITTER NOT RESPONDING</h1>
                            <p style='text-align: center;'>Could not connect to the data source. Please ensure <strong>Mock_Data_Emitter.py</strong> is running.</p>
                         </div>
                         """
                    else:
                        error_text = f"""
                        <div style='margin-top: 20px; margin-bottom: 20px;'>
                            <h1 style='text-align: center; color: #f56565; margin-bottom: 5px;'>&#9888; SENSOR CONNECTION ISSUE</h1>
                            <p style='text-align: center;'>The following sensors are not responding: <strong>{', '.join(missing_list)}</strong>.<br>Please check the sensor hardware and restart the device.</p>
                        </div>
                        """
                    
                    error_update = gr.update(value=error_text, visible=True)
                else:
                    error_update = gr.update(visible=False)

                # Prediction Data
                if shared_history:
                    for _ in range(len(shared_history)):
                        plot_history.append(shared_history.pop(0))
                latest_prob = shared_prediction.value
                prediction_label_text = f"Seizure Probability: {latest_prob:.2%}"
                prediction_plot_data = pd.DataFrame({
                    "Time (seconds ago)": range(-len(plot_history) + 1, 1),
                    "Probability": list(plot_history)
                })

                # Sensor Data
                pickled_df = shared_pickled_df.value
                sensor_df = pickle.loads(pickled_df) if pickled_df else None
                
                if sensor_df is None:
                    hr_data, eda_data, temp_data, acc_data = (pd.DataFrame({'Time Step': [], 'HR': []}), 
                                                              pd.DataFrame({'Time Step': [], 'EDA': []}), 
                                                              pd.DataFrame({'Time Step': [], 'TEMP': []}), 
                                                              pd.DataFrame({'Time Step': [], 'ACC': []}))
                else:
                    sensor_df['Time Step'] = range(len(sensor_df))
                    hr_data = sensor_df[['Time Step', 'HR']]
                    eda_data = sensor_df[['Time Step', 'EDA']]
                    temp_data = sensor_df[['Time Step', 'TEMP']]
                    acc_data = sensor_df[['Time Step', 'ACC']]
                
                yield error_update, prediction_label_text, prediction_plot_data, hr_data, eda_data, temp_data, acc_data
                
                time.sleep(1)

        with gr.Blocks(theme=gr.themes.Base(), css=".gradio-container {background-color: #1a202c; color: white;}") as demo:
            gr.Markdown("# Real-Time Seizure Prediction")
            
            error_message_box = gr.Markdown(visible=False)

            with gr.Row():
                prediction_label = gr.Label(value="Waiting for first prediction...", label="Prediction", scale=1)
            
            prediction_plot = gr.LinePlot(
                x="Time (seconds ago)", y="Probability",
                y_lim=[0, 1], title="Prediction History (Last 60s)",
                show_label=False, height=300
            )

            gr.Markdown("--- \n ## Live Sensor Data Segment (30s)")
            with gr.Row():
                hr_plot = gr.LinePlot(x="Time Step", y="HR", title="Heart Rate", height=250)
                eda_plot = gr.LinePlot(x="Time Step", y="EDA", title="Electrodermal Activity", height=250)
            with gr.Row():
                temp_plot = gr.LinePlot(x="Time Step", y="TEMP", title="Temperature", height=250)
                acc_plot = gr.LinePlot(x="Time Step", y="ACC", title="Accelerometer", height=250)
            
            demo.load(
                fn=stream_updates, 
                inputs=None, 
                outputs=[error_message_box, prediction_label, prediction_plot, hr_plot, eda_plot, temp_plot, acc_plot]
            )

        print("[Gradio] Launching UI at http://127.0.0.1:7860 (or a similar address)...")
        demo.launch()