import re

import matplotlib.pyplot as plt

log_file = "final_v1\\training_data\\20250610_195720\\seizure_prediction_results_20250610_195720_v3enhanced.log"

sensor_auc = {}  # To store {sensor_combo: auc_value}

with open(log_file, "r") as file:
    lines = file.readlines()

sensor_key = None

for line in lines:
    # Detect sensor combo
    match_sensor = re.search(r"Model CNN-LSTM \+ Sensors (.*?) with HP", line)
    if match_sensor:
        sensor_key = match_sensor.group(1)

    # Detect Test Metrics (with AUC-ROC)
    match_test = re.search(r"Test: .*?AUC-ROC: ([0-9.]+)", line)
    if match_test and sensor_key:
        auc = float(match_test.group(1))
        sensor_auc[sensor_key] = auc
        sensor_key = None  # Reset to avoid mismatch

# Plotting
plt.figure(figsize=(10, 6))
sensor_names = list(sensor_auc.keys())
auc_values = list(sensor_auc.values())

plt.bar(sensor_names, auc_values, color='skyblue')
plt.ylabel("Test AUC-ROC")
plt.title("Model AUC-ROC per Sensor Combination (Test Set)")
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1.0)

for i, val in enumerate(auc_values):
    plt.text(i, val + 0.01, f"{val:.3f}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()
