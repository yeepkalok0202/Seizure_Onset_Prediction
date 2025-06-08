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
# The model prediction process (runs in the background)
# =================================================================================
def prediction_process(tensor_queue, shared_prediction, shared_history, config):
    """
    Gets tensors from the queue, runs inference, and updates shared variables for the UI.
    """
    print("[Predictor] Process started. Loading model from bundle...")
    try:
        # Load the entire inference bundle
        bundle = pickle.load(open(config['INFERENCE_BUNDLE_PATH'], 'rb'))
        
        # Isolate only the model_hyperparameters to build the model's architecture.
        model_hyperparams = bundle['hyperparameters']['model_hyperparameters']
        
        model_state_dict = bundle['model_state_dict']
        model_type = bundle['model_type']
        
        # Get the correct model class using your function
        ModelClass = get_model_class(model_type)
        
        num_features = len(config['BASE_SENSORS'])
        
        # --- THE FINAL FIX ---
        # Calculate seq_len from the main config, as it's a data shape property.
        seq_len = config['SEGMENT_SECONDS'] * config['TARGET_SAMPLING_HZ']
        
        # Create the model, now providing seq_len along with other arguments.
        model = ModelClass(
            input_channels=num_features,
            seq_len=seq_len, # Provide the missing argument
            **model_hyperparams
        )
        
        model.load_state_dict(model_state_dict)
        model.eval() # Set model to evaluation mode
        
        print(f"[Predictor] Model '{model_type}' loaded successfully.")
    except Exception as e:
        print(f"[Predictor] FATAL: Error loading model: {e}")
        return

    print("[Predictor] Waiting for processed tensors from the queue...")
    while True:
        try:
            segment_tensor = tensor_queue.get()
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
def launch_predictor_ui(tensor_queue, config):
    # (This function is unchanged)
    with Manager() as manager:
        shared_prediction = manager.Value('f', 0.0)
        shared_history = manager.list()
        plot_history = deque(maxlen=60)

        predictor_proc = Process(target=prediction_process, args=(tensor_queue, shared_prediction, shared_history, config))
        predictor_proc.daemon = True
        predictor_proc.start()

        def update_ui():
            if shared_history:
                for _ in range(len(shared_history)):
                    plot_history.append(shared_history.pop(0))
            
            latest_prob = shared_prediction.value
            label_text = f"Seizure Probability: {latest_prob:.2%}"
            
            history = list(plot_history)
            x_values = range(-len(history) + 1, 1)
            plot_data = pd.DataFrame({"Time (seconds ago)": x_values, "Probability": history})
            
            return label_text, plot_data

        with gr.Blocks(css=".gradio-container {background-color: #f0f2f6;}", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🧠 Real-Time Seizure Prediction")
            with gr.Row():
                prediction_label = gr.Label(value="Waiting for first prediction...", label="Prediction")
            with gr.Row():
                prediction_plot = gr.LinePlot(
                    x="Time (seconds ago)", y="Probability",
                    value=pd.DataFrame({"Time (seconds ago)": [], "Probability": []}),
                    y_lim=[0, 1], title="Prediction History (Last 60s)",
                    show_label=False, width=600, height=300
                )
            
            timer = gr.Timer(1)
            timer.tick(fn=update_ui, inputs=None, outputs=[prediction_label, prediction_plot])

        print("[Gradio] Launching UI at http://127.0.0.1:7860 (or a similar address)...")
        demo.launch()