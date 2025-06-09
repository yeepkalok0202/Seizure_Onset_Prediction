import pandas as pd
import numpy as np
import requests
import time
import pickle

def preprocess_realtime_segment(raw_data_dict, config):
    """
    This function prepares the clean DataFrame and also checks for missing sensors.
    - If successful, it returns the DataFrame.
    - If sensors are missing, it returns a list of the missing sensor names.
    """
    if not raw_data_dict:
        return [] # Return empty list if no data at all

    df = pd.DataFrame.from_dict(raw_data_dict, orient='index')
    df.index = pd.to_datetime(df.index, format='ISO8601')
    
    # --- MODIFIED: Check for missing sensors before processing ---
    # Reindex to ensure all expected sensor columns are present
    df = df.reindex(columns=config['BASE_SENSORS'])
    
    # Identify sensors that have no data points at all (are all NaN)
    missing_sensors = [sensor for sensor in config['BASE_SENSORS'] if df[sensor].isnull().all()]
    
    if missing_sensors:
        # If any sensor is completely missing, report it immediately
        return missing_sensors # --- MODIFIED ---

    # Continue with processing only if all sensors are present
    df.replace(0, np.nan, inplace=True)
    rule = f"{int(1000/config['TARGET_SAMPLING_HZ'])}ms"
    resampled_df = df.resample(rule).mean()

    interpolated_df = resampled_df.interpolate(method='time', limit_direction='both')
    final_df = interpolated_df.ffill().bfill()
    
    expected_rows = config['SEGMENT_SECONDS'] * config['TARGET_SAMPLING_HZ']
    if len(final_df) < expected_rows:
        return None # Not a critical error, just not enough data yet
    
    final_df = final_df.iloc[-expected_rows:]
    
    return final_df

def preprocessing_process(data_queue, config):
    """
    Main loop. Fetches data, checks for missing sensors, and puts a status
    dictionary onto the shared queue.
    """
    print("[Preprocessor] Process started.")
    
    while True:
        try:
            response = requests.get(config['DATA_EMITTER_URL'], timeout=5)
            response.raise_for_status()
            raw_data = response.json()
            
            # --- MODIFIED: Process data and check the result ---
            result = preprocess_realtime_segment(raw_data, config)

            if isinstance(result, pd.DataFrame):
                # SUCCESS: All sensors are present and data is processed
                status_packet = {'status': 'OK', 'data': result}
                data_queue.put(status_packet)
                print(f"[Preprocessor] Success. Sent processed DataFrame to queue.")

            elif isinstance(result, list) and result:
                # FAILURE: One or more sensors are missing
                status_packet = {'status': 'ERROR', 'missing_sensors': result}
                data_queue.put(status_packet)
                print(f"[Preprocessor] ERROR: Missing sensors detected: {result}. Sent error status to queue.")

            # If result is None, it means not enough data yet, so we just wait and retry
            
            time.sleep(1 / config['EMISSION_RATE_HZ'])
        except requests.exceptions.RequestException as e:
            print(f"[Preprocessor] An error occurred connecting to the emitter: {e}")
            # --- NEW: Send an error status if request fails ---
            error_packet = {'status': 'ERROR', 'missing_sensors': ['COMM_FAILURE']}
            data_queue.put(error_packet)
            time.sleep(2)
        except Exception as e:
            print(f"[Preprocessor] An unknown error occurred: {e}")
            time.sleep(2)