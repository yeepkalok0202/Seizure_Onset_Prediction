import pickle
import time
from collections import deque
from datetime import datetime, timezone
from multiprocessing import Manager, Process, Queue

import gradio as gr
import pandas as pd
import torch
import torch.nn as nn

from Final_Clean_v1 import get_model_class


# =================================================================================
# The model prediction process (runs in the background)
# =================================================================================
def prediction_process(data_queue, shared_prediction, shared_history, shared_pickled_df, shared_status, shared_alert_state, shared_alert_history, config):
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
                
                latest_timestamp = processed_df.index[-1]
                if latest_timestamp.tzinfo is None:
                    latest_timestamp = latest_timestamp.tz_localize("UTC")
                else:
                    latest_timestamp = latest_timestamp.tz_convert("UTC")                
                scaled_segment = scaler.transform(processed_df.values)
                segment_tensor = torch.tensor(scaled_segment, dtype=torch.float32).permute(1, 0).unsqueeze(0)
                
                with torch.no_grad():
                    output = model(segment_tensor)
                    probability = output.item()
                
                shared_prediction.value = probability
                shared_history.append((latest_timestamp.isoformat(), probability))
                
                print(f"[Predictor] Prediction: {probability:.4f}")
                print()
                # --- CHANGE 1: New, stateful alerting logic ---
                if probability >= 0.5:
                    # If this is the start of a new alert period (was not active before)
                    if not shared_alert_state['is_active']:
                        print("[Predictor] New alert period started.")
                        shared_alert_state['is_active'] = True
                        shared_alert_state['first_ts'] = latest_timestamp.isoformat()
                    
                    # Always update the latest timestamp during an active alert
                    shared_alert_state['latest_ts'] = latest_timestamp.isoformat()
                    
                    # Add every high-prob event to the detailed history table
                    alert_record = {
                        "Timestamp": latest_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'),
                        "Probability": f"{probability:.2%}",
                        "HR (avg)": f"{processed_df['HR'].mean():.1f}",
                        "EDA (avg)": f"{processed_df['EDA'].mean():.3f}",
                        "Temp (avg)": f"{processed_df['TEMP'].mean():.2f}",
                        "ACC (avg)": f"{processed_df['ACC'].mean():.2f}"
                    }
                    shared_alert_history.append(alert_record)
                else:
                    # If the probability drops, the alert period is no longer active
                    if shared_alert_state['is_active']:
                        print("[Predictor] Alert period ended.")
                        shared_alert_state['is_active'] = False

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
        # --- CHANGE 2: Use the new, more detailed alert state dictionary ---
        shared_alert_state = manager.dict({'first_ts': None, 'latest_ts': None, 'is_active': False})
        shared_alert_history = manager.list()
        
        plot_history = deque(maxlen=60)

        predictor_proc = Process(target=prediction_process, args=(
            data_queue, shared_prediction, shared_history, shared_pickled_df, 
            shared_status, shared_alert_state, shared_alert_history, config
        ))
        predictor_proc.daemon = True
        predictor_proc.start()

        def stream_updates():
            while True:
                # 1. --- System Status / Connection Error Update ---
                current_status = shared_status.get('status', 'OK')
                connection_error_update = gr.update(visible=False)
                if current_status == 'ERROR':
                    missing_list = shared_status.get('missing', [])
                    if 'COMM_FAILURE' in missing_list:
                         error_text = f"<div style='margin-top: 20px; margin-bottom: 20px;'><h1 style='text-align: center; color: #f56565;'>&#9888; DATA STREAM SERVICE NOT RESPONDING</h1><p style='text-align: center;'>Could not connect to the data source. Please ensure <strong>Data Stream Daemon</strong> is running.</p></div>"
                    else:
                         error_text = f"<div style='margin-top: 20px; margin-bottom: 20px;'><h1 style='text-align: center; color: #f56565;'>&#9888; SENSOR CONNECTION ISSUE</h1><p style='text-align: center;'>The following sensors are not responding: <strong>{', '.join(missing_list)}</strong>.<br>Please check the sensor hardware and restart the device.</p></div>"
                    connection_error_update = gr.update(value=error_text, visible=True)

                # --- CHANGE 3: Update the alert box logic to show first and latest detection times ---
                seizure_alert_update = gr.update(visible=False) 
                first_alert_ts_iso = shared_alert_state.get('first_ts')

                # Show the alert box if a high-risk event has ever occurred in the last alert period
                if first_alert_ts_iso:
                    latest_alert_ts_iso = shared_alert_state.get('latest_ts')
                    
                    first_time_str = pd.to_datetime(first_alert_ts_iso, utc=True).strftime('%Y-%m-%d %H:%M:%S UTC')
                    latest_time_str = pd.to_datetime(latest_alert_ts_iso, utc=True).strftime('%Y-%m-%d %H:%M:%S UTC')
                    
                    alert_text = f"""
                    <div style='margin-top: 15px; margin-bottom: 15px; border: 2px solid #e53e3e; border-radius: 10px; padding: 10px; background-color: #4A0404;'>
                        <h2 class='flashing-text' style='text-align: center; color: #fed7d7;'>&#9888; POSSIBLE SEIZURE DETECTED</h2>
                        <p style='text-align: center; color: white;'>A high probability event was first detected at: <strong>{first_time_str}</strong></p>
                        <p style='text-align: center; color: white;'>The latest detection was at: <strong>{latest_time_str}</strong></p>
                        <p style='text-align: center; color: white; margin-top: 10px;'>A seizure may occur within the next <strong>5 to 35 minutes</strong>. Please take necessary precautions.</p>
                    </div>
                    """
                    seizure_alert_update = gr.update(value=alert_text, visible=True)

                # 3. --- Prediction History Table Update ---
                history_df = pd.DataFrame(list(shared_alert_history))

                # 4. --- Risk Level and Prediction Graph Update ---
                if shared_history:
                    for _ in range(len(shared_history)):
                        plot_history.append(shared_history.pop(0))
                
                prediction_plot_data = pd.DataFrame(list(plot_history), columns=["Timestamp", "Probability"])
                if not prediction_plot_data.empty:
                    prediction_plot_data["Timestamp"] = pd.to_datetime(prediction_plot_data["Timestamp"], utc=True)

                latest_prob = shared_prediction.value
                
                if latest_prob >= 0.8:
                    risk_indicator = "🟠 High Risk:"
                    risk_message = "High seizure risk. Consider precautions."
                elif latest_prob >= 0.5:
                    risk_indicator = "🟡 Moderate Risk:"
                    risk_message = "Slightly elevated risk. Be mindful."
                else:
                    risk_indicator = "🟢 Low Risk:"
                    risk_message = "Your risk is currently low."
                prediction_label_text = f"{risk_indicator} {latest_prob:.2%} | {risk_message}"
                
                utc_time_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

                # 5. --- Sensor Plots Update ---
                pickled_df = shared_pickled_df.value
                sensor_df = pickle.loads(pickled_df) if pickled_df else None
                if sensor_df is None:
                    hr_data, eda_data, temp_data, acc_data = (pd.DataFrame({'Timestamp': [], 'HR': []}), pd.DataFrame({'Timestamp': [], 'EDA': []}), pd.DataFrame({'Timestamp': [], 'TEMP': []}), pd.DataFrame({'Timestamp': [], 'ACC': []}))
                else:
                    sensor_df.reset_index(inplace=True)
                    sensor_df.rename(columns={'index': 'Timestamp'}, inplace=True)
                    hr_data = sensor_df[['Timestamp', 'HR']]
                    eda_data = sensor_df[['Timestamp', 'EDA']]
                    temp_data = sensor_df[['Timestamp', 'TEMP']]
                    acc_data = sensor_df[['Timestamp', 'ACC']]
                
                yield (
                    connection_error_update, 
                    prediction_label_text,
                    prediction_plot_data, 
                    seizure_alert_update,
                    history_df,
                    hr_data, eda_data, temp_data, acc_data
                )
                
                time.sleep(1)

        css_styles = """
        .gradio-container {background-color: #1a202c; color: white;}
        @keyframes flashing {
            0% { opacity: 1; }
            50% { opacity: 0.2; }
            100% { opacity: 1; }
        }
        .flashing-text {
            animation: flashing 1.5s infinite;
        }
        """
        with gr.Blocks(theme=gr.themes.Base(), css=css_styles) as demo:
            gr.Markdown("# Real-Time Seizure Prediction")
            
            connection_error_box = gr.Markdown(visible=False)

            with gr.Row():
                prediction_label = gr.Label(value="Waiting for first prediction...", label="Risk Assessment", scale=2)

            prediction_plot = gr.LinePlot(x="Timestamp", y="Probability", y_lim=[0, 1], title="Prediction History (Last 60s)", show_label=False, height=300)
            
            seizure_alert_box = gr.Markdown(visible=False)
            
            history_table = gr.DataFrame(label="Seizure Alert History", interactive=False, headers=["Timestamp", "Probability", "HR (avg)", "EDA (avg)", "Temp (avg)", "ACC (avg)"])

            gr.Markdown("--- \n ## Live Sensor Data Segment (30s)")
            with gr.Row():
                hr_plot = gr.LinePlot(x="Timestamp", y="HR", title="Heart Rate", height=250)
                eda_plot = gr.LinePlot(x="Timestamp", y="EDA", title="Electrodermal Activity", height=250)
            with gr.Row():
                temp_plot = gr.LinePlot(x="Timestamp", y="TEMP", title="Temperature", height=250)
                acc_plot = gr.LinePlot(x="Timestamp", y="ACC", title="Accelerometer", height=250)
            
            demo.load(
                fn=stream_updates, 
                inputs=None, 
                outputs=[
                    connection_error_box, 
                    prediction_label,
                    prediction_plot, 
                    seizure_alert_box, 
                    history_table,
                    hr_plot, eda_plot, temp_plot, acc_plot
                ]
            )

        print("[Gradio] Launching UI at http://127.0.0.1:7860 (or a similar address)...")
        demo.launch()