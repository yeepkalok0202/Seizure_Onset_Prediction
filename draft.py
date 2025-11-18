import gc
import glob
import heapq
import io  # --- NEW: For reading from S3 in-memory
import logging
import os
import re
import threading
import time
from collections import OrderedDict, deque

import boto3  # --- NEW: AWS library
import pandas as pd
from flask import Flask, jsonify

# --- CONFIGURATION ---
BUCKET_NAME = 'fyp-seizure-data'  # <--- !! PUT YOUR S3 BUCKET NAME HERE
PATIENT_ID = 'MSEL_01110'         # <--- !! The patient folder in S3
BASE_SENSORS = ['HR', 'EDA', 'TEMP', 'ACC']
SENSOR_MAPPING = {"HR": "HR", "EDA": "EDA", "TEMP": "TEMP", "ACC": "Acc Mag"}
EMISSION_RATE_HZ = 1

# --- Logging Setup (Unchanged) ---
log_file = 'data_emitter.log'
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
if logger.hasHandlers():
    logger.handlers.clear()
file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
# --- End of Logging Setup ---

app = Flask(__name__)
data_window = deque()
latest_bundle = []  # --- NEW: Store just the latest bundle
data_lock = threading.Lock()

# --- S3 Client ---
# No keys needed! The IAM Role on the EC2 instance provides permission.
s3 = boto3.client('s3')

def create_sensor_stream_generator(patient_id, sensor_name):
    attr_folder = sensor_name
    attr_name_part = SENSOR_MAPPING[sensor_name]

    # 1. List files in S3
    s3_prefix = f"{patient_id}/Empatica-{attr_folder}/"
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=s3_prefix)

    files = []
    for page in pages:
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('.parquet') and f"_{attr_name_part}_segment_" in obj['Key']:
                files.append(obj['Key'])

    # 2. Sort the S3 file keys (paths)
    def natural_sort_key(s):
        match = re.search(r'(\d+)\.parquet$', s)
        return int(match.group(1)) if match else 0

    files = sorted(files, key=natural_sort_key)

    if not files:
        logging.warning(f"No files found in S3 for {sensor_name} at {BUCKET_NAME}/{s3_prefix}")
        return

    logging.info(f"Initialized S3 stream for {sensor_name} with {len(files)} files.")

    # 3. Read each file from S3
    for file_key in files:
        try:
            # Get the file object from S3
            obj = s3.get_object(Bucket=BUCKET_NAME, Key=file_key)
            # Read its content directly into pandas
            # io.BytesIO makes it behave like a file on disk
            chunk_df = pd.read_parquet(io.BytesIO(obj['Body'].read()))

            # --- Rest of your generator logic is unchanged ---
            if "time" in chunk_df.columns and "data" in chunk_df.columns:
                chunk_df['data'] = chunk_df['data'] / 1_000_000_000
                chunk_df["timestamp"] = pd.to_datetime(chunk_df["time"] / 1000, unit="s", utc=True)
                chunk_df = chunk_df.rename(columns={"data": sensor_name})
                for row in chunk_df[["timestamp", sensor_name]].itertuples(index=False, name=None):
                    yield row
            del chunk_df
            gc.collect()
        except Exception as e:
            logging.error(f"Error in {sensor_name} generator for S3 file {file_key}: {e}")
            continue

def data_ingestion_thread():
    sensor_streams = {s: create_sensor_stream_generator(PATIENT_ID, s) for s in BASE_SENSORS}

    # --- Heap setup (Unchanged) ---
    merged_stream_heap = []
    for sensor, stream_gen in sensor_streams.items():
        try:
            if stream_gen:
                first_row = next(stream_gen)
                heapq.heappush(merged_stream_heap, (first_row[0], sensor, first_row[1]))
        except StopIteration:
            pass

    while merged_stream_heap:
        
        # 1. Determine the next absolute integer-second boundary
        earliest_ts = merged_stream_heap[0][0]
        bundle_start_time = earliest_ts.floor("1s")
        bundle_end_time   = bundle_start_time + pd.Timedelta(seconds=1)

        logging.debug(f"Creating 1s bundle: [{bundle_start_time} → {bundle_end_time})")

        current_bundle = []

        # 2. Collect rows that fall into this exact aligned 1-second window
        while merged_stream_heap and bundle_start_time <= merged_stream_heap[0][0] < bundle_end_time:
            timestamp, sensor, value = heapq.heappop(merged_stream_heap)
            current_bundle.append({
                "timestamp": timestamp.isoformat(),
                "sensor": sensor,
                "value": value
            })
            try:
                next_row = next(sensor_streams[sensor])
                heapq.heappush(merged_stream_heap, (next_row[0], sensor, next_row[1]))
            except StopIteration:
                pass

        # 3. Save result (latest bundle + 30s rolling history)
        if current_bundle:
            present_sensors = {item["sensor"] for item in current_bundle}
            missing_sensors = set(BASE_SENSORS) - present_sensors

            if missing_sensors:
                logging.warning(
                    f"[{bundle_start_time.isoformat()}] Missing sensors: {sorted(missing_sensors)}"
                )

            with data_lock:
                global latest_bundle
                latest_bundle = current_bundle
                data_window.append(current_bundle)

                while len(data_window) > 30 * EMISSION_RATE_HZ:
                    data_window.popleft()

            logging.info(
                f"Emitted aligned second bundle {bundle_start_time}: "
                f"{len(current_bundle)} points"
            )
        # 4. Emit bundles at 1Hz
        time.sleep(1 / EMISSION_RATE_HZ)

    logging.info("All data streams have been fully processed.")

def format_bundle(bundle_list):
    """Helper function to pivot data."""
    reformatted_data = OrderedDict()
    for item in bundle_list:
        timestamp_str = item['timestamp']
        sensor = item['sensor']
        value = item['value']
        if timestamp_str not in reformatted_data:
            reformatted_data[timestamp_str] = {s: None for s in BASE_SENSORS}
        reformatted_data[timestamp_str][sensor] = value
    return reformatted_data

# --- NEW ENDPOINT (for 1-second polling) ---
@app.route('/data/latest', methods=['GET'])
def get_latest_data():
    """Returns only the most recent 1-second bundle."""
    with data_lock:
        bundle_to_send = list(latest_bundle)

    reformatted_data = format_bundle(bundle_to_send)
    logging.info(f"API: Sent /data/latest with {len(reformatted_data)} timestamps")
    return jsonify(reformatted_data)

# --- YOUR OLD ENDPOINT (for loading history) ---
@app.route('/data/history', methods=['GET'])
def get_data_history():
    """Returns the full 30-second history window."""
    with data_lock:
        flat_list = [item for bundle in data_window for item in bundle]

    reformatted_data = format_bundle(flat_list)
    logging.info(f"API: Sent /data/history with {len(reformatted_data)} timestamps")
    return jsonify(reformatted_data)

if __name__ == '__main__':
    logging.info("--- Data Emitter (S3 Version) ---")
    ingestion_thread = threading.Thread(target=data_ingestion_thread, daemon=True)
    ingestion_thread.start()
    logging.info("Flask server starting... Access data at http://0.0.0.0:5000/")
    app.run(host='0.0.0.0', port=5000)