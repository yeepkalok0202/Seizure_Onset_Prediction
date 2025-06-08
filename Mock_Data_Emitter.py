import gc
import glob
import heapq
import logging
import os
import threading
import time
from collections import OrderedDict, deque

import pandas as pd
from flask import Flask, jsonify

# --- CONFIGURATION ---
PATIENT_DATA_DIR = 'F:\\benchmarking_4\\data\\MSEL_01676'
BASE_SENSORS = ['HR', 'EDA', 'TEMP', 'ACC']
SENSOR_MAPPING = {"HR": "HR", "EDA": "EDA", "TEMP": "TEMP", "ACC": "Acc Mag"}
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EMISSION_RATE_HZ = 1
app = Flask(__name__)
data_window = deque()
data_lock = threading.Lock()

# (The create_sensor_stream_generator and data_ingestion_thread functions are unchanged from the previous version)
def create_sensor_stream_generator(patient_dir, sensor_name):
    patient_id = os.path.basename(patient_dir)
    attr_folder = sensor_name
    attr_name_part = SENSOR_MAPPING[sensor_name]
    glob_pattern = os.path.join(
        patient_dir, f"Empatica-{attr_folder}",
        f"{patient_id}_Empatica-{attr_folder}_{attr_name_part}_segment_*.parquet",
    )
    files = sorted(glob.glob(glob_pattern))
    if not files:
        logging.warning(f"No files for {sensor_name}.")
        return
    logging.info(f"Initialized stream for {sensor_name} with {len(files)} files.")
    for file_path in files:
        try:
            chunk_df = pd.read_parquet(file_path)
            if "time" in chunk_df.columns and "data" in chunk_df.columns:
                chunk_df['data'] = chunk_df['data'] / 1_000_000_000
                chunk_df["timestamp"] = pd.to_datetime(chunk_df["time"] / 1000, unit="s", utc=True)
                chunk_df = chunk_df.rename(columns={"data": sensor_name})
                for row in chunk_df[["timestamp", sensor_name]].itertuples(index=False, name=None):
                    yield row
            del chunk_df
            gc.collect()
        except Exception as e:
            logging.error(f"Error in {sensor_name} generator for file {file_path}: {e}")
            continue

def data_ingestion_thread():
    sensor_streams = {s: create_sensor_stream_generator(PATIENT_DATA_DIR, s) for s in BASE_SENSORS}
    merged_stream_heap = []
    for sensor, stream_gen in sensor_streams.items():
        try:
            first_row = next(stream_gen)
            heapq.heappush(merged_stream_heap, (first_row[0], sensor, first_row[1]))
        except StopIteration:
            pass
    while merged_stream_heap:
        bundle_start_time = merged_stream_heap[0][0]
        bundle_end_time = bundle_start_time + pd.Timedelta(seconds=1)
        current_bundle = []
        while merged_stream_heap and merged_stream_heap[0][0] < bundle_end_time:
            timestamp, sensor, value = heapq.heappop(merged_stream_heap)
            current_bundle.append({"timestamp": timestamp.isoformat(), "sensor": sensor, "value": value})
            try:
                next_row = next(sensor_streams[sensor])
                heapq.heappush(merged_stream_heap, (next_row[0], sensor, next_row[1]))
            except StopIteration:
                pass
        if current_bundle:
            with data_lock:
                data_window.append(current_bundle)
                while len(data_window) > (30 * EMISSION_RATE_HZ):
                     data_window.popleft()
            logging.info(f"Emitted a bundle with {len(current_bundle)} data points.")
        time.sleep(1 / EMISSION_RATE_HZ)
    logging.info("All data streams have been fully processed.")


@app.route('/data', methods=['GET'])
def get_data():
    """
    Returns a dictionary of data points grouped by timestamp, ensuring all
    sensor keys are present for every timestamp.
    """
    with data_lock:
        flat_list = [item for bundle in data_window for item in bundle]
    
    reformatted_data = OrderedDict()
    for item in flat_list:
        timestamp_str = item['timestamp']
        sensor = item['sensor']
        value = item['value']
        
        # --- MODIFIED LOGIC ---
        # If the timestamp is not yet a key, initialize it with all sensors set to None.
        if timestamp_str not in reformatted_data:
            reformatted_data[timestamp_str] = {s: None for s in BASE_SENSORS}
        
        # Now, update the specific sensor that has a real value for this timestamp.
        reformatted_data[timestamp_str][sensor] = value
        
    return jsonify(reformatted_data)


if __name__ == '__main__':
    logging.info("--- Data Emitter (Timestamp Grouped w/ Nulls Mode) ---")
    ingestion_thread = threading.Thread(target=data_ingestion_thread, daemon=True)
    ingestion_thread.start()
    logging.info(f"Flask server starting... Access data at http://127.0.0.1:5000/data")
    app.run(host='0.0.0.0', port=5000)