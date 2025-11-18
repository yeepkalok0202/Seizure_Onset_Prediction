import re
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd

# === CONFIG ===
log_file_path ="final_v1\\training_data\\20250612_023907\\seizure_prediction_results_20250612_023907_v3enhanced.log"  
# === REGEX PATTERN ===
pattern = re.compile(
    r"Overall General Model Testing Metrics \((.+?), (.+?), HP (\d+)\) - Test: "
    r"Loss: ([\d.]+), Acc: ([\d.]+), Prec: ([\d.]+), Rec: ([\d.]+), F1: ([\d.]+), "
    r"AUC-ROC: ([\d.]+), AUC-PR: ([\d.]+)"
)

# === PARSE LOG FILE ===
models_metrics = defaultdict(dict)

with open(log_file_path, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            model_name, input_set, hp, loss, acc, prec, rec, f1, auc_roc, auc_pr = match.groups()
            model_id = f"{model_name.strip()} (HP {hp.strip()})"
            models_metrics[model_id] = {
                "Loss": float(loss),
                "Accuracy": float(acc),
                "Precision": float(prec),
                "Recall": float(rec),
                "F1 Score": float(f1),
                "AUC-ROC": float(auc_roc),
                "AUC-PR": float(auc_pr)
            }

# === CONVERT TO DATAFRAME ===
df = pd.DataFrame(models_metrics).T  # Transpose to models as rows
metrics_list = df.columns.tolist()
model_ids = df.index.tolist()

# === PLOT ===
fig, ax = plt.subplots(figsize=(14, 9))

for metric in metrics_list:
    ax.plot(model_ids, df[metric], marker='o', label=metric)

ax.set_xticks(range(len(model_ids)))
ax.set_xticklabels(model_ids, rotation=30, ha='right')
ax.set_title("Test Set Metric Comparison Across General Models")
ax.set_ylabel("Metric Value")
ax.set_ylim(0, 1.05)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()

# === ADD TABLE ===
table_data = df.round(4).T  # Transpose for table
table = plt.table(
    cellText=table_data.values,
    rowLabels=table_data.index,
    colLabels=table_data.columns,
    cellLoc='center',
    loc='bottom',
    bbox=[0.0, -0.75, 1, 0.6]  # Pushed further down to avoid overlap
)

# === ADJUST SPACING ===
plt.subplots_adjust(left=0.1, bottom=0.45)  # More bottom space for X-axis and table
plt.tight_layout()
plt.show()

# === SAVE METRICS TO TEXT FILE ===
output_text_path = "model_test_metrics_summary.txt"

with open(output_text_path, "w") as f:
    for model_name, metrics in models_metrics.items():
        f.write(f"Model: {model_name}\n")
        for metric_name, value in metrics.items():
            f.write(f"  {metric_name}: {value:.4f}\n")
        f.write("\n")  # Blank line between models

print(f"\n✅ Text summary saved to: {output_text_path}")
