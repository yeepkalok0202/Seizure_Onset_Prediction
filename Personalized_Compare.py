import re

import matplotlib.pyplot as plt
import pandas as pd

# === File paths ===
logfile_path = "final_v1\\training_data\\20250612_044027\\seizure_prediction_results_20250612_044027_v3enhanced.log" # <- replace this
before_table_path = "before_metrics_table.png"
after_table_path = "after_metrics_table.png"

before_text_path = "before_metrics_friendly.txt"
after_text_path = "after_metrics_friendly.txt"

# Adjusted regex to exclude CM and capture Acc Change
metrics_pattern = re.compile(
    r"(MSEL_\d+)\s+\|\s+Test: Loss: ([\d.]+), Acc: ([\d.]+), Prec: ([\d.]+), Rec: ([\d.]+), "
    r"F1: ([\d.]+), AUC-ROC: ([\d.]+), AUC-PR: ([\d.]+), CM: \[\[.*?\]\] \|\s+"
    r"Test: Loss: ([\d.]+), Acc: ([\d.]+), Prec: ([\d.]+), Rec: ([\d.]+), "
    r"F1: ([\d.]+), AUC-ROC: ([\d.]+), AUC-PR: ([\d.]+), CM: \[\[.*?\]\] \|\s+([\d.]+)"
)


# === Data containers ===
before_data = []
after_data = []
acc_change_data = [] # To store the accuracy change

with open(logfile_path, "r") as file:
    for line in file:
        match = metrics_pattern.search(line)
        if match:
            patient_id = match.group(1)
            # Extract before and after metrics (7 metrics each)
            # groups() starts from index 1 for the first captured group.
            # group(1) is patient_id
            # group(2) to group(8) are before_metrics (7 values)
            # group(9) to group(15) are after_metrics (7 values)
            # group(16) is Acc Change
            before_metrics = list(map(float, match.groups()[1:8]))
            after_metrics = list(map(float, match.groups()[8:15]))
            acc_change = float(match.group(16))

            before_data.append([patient_id] + before_metrics)
            after_data.append([patient_id] + after_metrics)
            acc_change_data.append([patient_id, acc_change])


# === Columns ===
# Updated columns to reflect the 7 metrics being extracted
columns = ["Patient ID", "Loss", "Accuracy", "Precision", "Recall", "F1", "AUC-ROC", "AUC-PR"]

# === Create DataFrames ===
df_before = pd.DataFrame(before_data, columns=columns)
df_after = pd.DataFrame(after_data, columns=columns)
df_acc_change = pd.DataFrame(acc_change_data, columns=["Patient ID", "Accuracy Change"]) # DataFrame for accuracy change

# === Sort for better viewing ===
df_before.sort_values("Accuracy", ascending=False, inplace=True)
df_after.sort_values("Accuracy", ascending=False, inplace=True)
df_acc_change.sort_values("Accuracy Change", ascending=False, inplace=True) # Sort accuracy change as well


# === Friendly format writer ===
def write_friendly(df, out_path):
    with open(out_path, "w") as f:
        for _, row in df.iterrows():
            f.write(f"{row['Patient ID']}\n")
            f.write(f"  Loss      : {row['Loss']:.4f}\n")
            f.write(f"  Accuracy  : {row['Accuracy']:.4f}\n")
            f.write(f"  Precision : {row['Precision']:.4f}\n")
            f.write(f"  Recall    : {row['Recall']:.4f}\n")
            f.write(f"  F1 Score  : {row['F1']:.4f}\n")
            f.write(f"  AUC-ROC   : {row['AUC-ROC']:.4f}\n")
            f.write(f"  AUC-PR    : {row['AUC-PR']:.4f}\n")
            f.write("\n")

# === Save text files ===
write_friendly(df_before, before_text_path)
write_friendly(df_after, after_text_path)

# You can also save the accuracy change data if needed
with open("accuracy_change_friendly.txt", "w") as f:
    f.write("--- Accuracy Change Per Patient ---\n\n")
    for _, row in df_acc_change.iterrows():
        f.write(f"{row['Patient ID']}: {row['Accuracy Change']:.4f}\n")


# === Table plot ===
def render_mpl_table(data, filename, col_width=2.5, row_height=0.5, font_size=12):
    fig, ax = plt.subplots(figsize=(col_width * len(data.columns), row_height * len(data)))
    ax.axis("off")

    table = plt.table(
        cellText=data.values,
        colLabels=data.columns,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.2, 1.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# === Save table images ===
render_mpl_table(df_before, before_table_path)
render_mpl_table(df_after, after_table_path)
render_mpl_table(df_acc_change, "accuracy_change_table.png") # Save accuracy change table as well

print(f"✅ Saved: {before_text_path}, {before_table_path}")
print(f"✅ Saved: {after_text_path}, {after_table_path}")
print(f"✅ Saved: accuracy_change_friendly.txt, accuracy_change_table.png")