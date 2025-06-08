import pandas as pd
import numpy as np
import requests
import time
import torch
import pickle

def preprocess_realtime_segment(raw_data_dict, scaler_object, config):
    if not raw_data_dict:
        return None

    df = pd.DataFrame.from_dict(raw_data_dict, orient='index')
    
    # --- FIX: Explicitly set the format to handle high-precision timestamps ---
    df.index = pd.to_datetime(df.index, format='ISO8601')

    df.replace(0, np.nan, inplace=True)
    df = df.reindex(columns=config['BASE_SENSORS'])

    rule = f"{int(1000/config['TARGET_SAMPLING_HZ'])}ms"
    resampled_df = df.asfreq(rule)

    interpolated_df = resampled_df.interpolate(method='time', limit_direction='both')
    final_df = interpolated_df.ffill().bfill()
    
    expected_rows = config['SEGMENT_SECONDS'] * config['TARGET_SAMPLING_HZ']
    if len(final_df) < expected_rows:
        return None
    
    final_df = final_df.iloc[-expected_rows:]
    
    processed_segment = final_df.values
    scaled_segment = scaler_object.transform(processed_segment)

    return scaled_segment

def preprocessing_process(tensor_queue, config):
    print("[Preprocessor] Process started. Loading scaler from bundle...")
    try:
        bundle = pickle.load(open(config['INFERENCE_BUNDLE_PATH'], 'rb'))
        scaler = bundle['scaler']
        print(f"[Preprocessor] Scaler loaded successfully.")
    except FileNotFoundError:
        print(f"[Preprocessor] FATAL: Inference bundle not found at {config['INFERENCE_BUNDLE_PATH']}")
        return
    except Exception as e:
        print(f"[Preprocessor] FATAL: Error loading scaler from bundle: {e}")
        return

    while True:
        try:
            response = requests.get(config['DATA_EMITTER_URL'], timeout=5)
            response.raise_for_status()
            raw_data = response.json()
            
            processed_segment_np = preprocess_realtime_segment(raw_data, scaler, config)

            if processed_segment_np is not None:
                segment_tensor = torch.tensor(processed_segment_np, dtype=torch.float32)
                segment_tensor = segment_tensor.permute(1, 0).unsqueeze(0)
                
                tensor_queue.put(segment_tensor)
                print(f"[Preprocessor] Processed a segment and put it on the queue.")

            time.sleep(1 / config['EMISSION_RATE_HZ'])
        except requests.exceptions.RequestException:
            time.sleep(2)
        except Exception as e:
            # We will print the error but continue the loop
            print(f"[Preprocessor] An error occurred: {e}")
            time.sleep(2)