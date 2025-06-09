import pandas as pd
import numpy as np
import requests
import time
import pickle

def preprocess_realtime_segment(raw_data_dict, config):
    """
    This function now prepares the clean DataFrame but does NOT scale it.
    It returns the final DataFrame ready for plotting and final processing.
    """
    if not raw_data_dict:
        return None

    df = pd.DataFrame.from_dict(raw_data_dict, orient='index')
    df.index = pd.to_datetime(df.index, format='ISO8601')
    df.replace(0, np.nan, inplace=True)
    df = df.reindex(columns=config['BASE_SENSORS'])

    rule = f"{int(1000/config['TARGET_SAMPLING_HZ'])}ms"
    resampled_df = df.resample(rule).mean()

    interpolated_df = resampled_df.interpolate(method='time', limit_direction='both')
    final_df = interpolated_df.ffill().bfill()
    
    expected_rows = config['SEGMENT_SECONDS'] * config['TARGET_SAMPLING_HZ']
    if len(final_df) < expected_rows:
        return None
    
    final_df = final_df.iloc[-expected_rows:]
    
    # Return the clean DataFrame before scaling
    return final_df

def preprocessing_process(data_queue, config):
    """
    Main loop. Fetches data, gets the processed DataFrame,
    and puts the DataFrame onto the shared queue.
    """
    print("[Preprocessor] Process started.")
    
    while True:
        try:
            response = requests.get(config['DATA_EMITTER_URL'], timeout=5)
            response.raise_for_status()
            raw_data = response.json()
            
            # Get the processed DataFrame
            processed_df = preprocess_realtime_segment(raw_data, config)

            if processed_df is not None:
                # --- Put the DataFrame onto the queue ---
                data_queue.put(processed_df)
                print(f"[Preprocessor] Processed a segment and put DataFrame on the queue.")

            time.sleep(1 / config['EMISSION_RATE_HZ'])
        except Exception as e:
            print(f"[Preprocessor] An error occurred: {e}")
            time.sleep(2)