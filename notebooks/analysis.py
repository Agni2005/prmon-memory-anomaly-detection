import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# ==========================================
# Load Data
# ==========================================

s1 = pd.read_csv("../data/normal_part.csv", sep=r"\s+")
s2 = pd.read_csv("../data/anomaly_part.csv", sep=r"\s+")
s3 = pd.read_csv("../data/segment3_recovery.csv", sep=r"\s+")

df = pd.concat([s1, s2, s3], ignore_index=True)
df["t"] = range(len(df))

# Ground truth labels
df["is_anomaly"] = 0
df.loc[len(s1):len(s1)+len(s2)-1, "is_anomaly"] = 1

print("Total samples:", len(df))

# ==========================================
# Parameters
# ==========================================

window = 30
threshold = 3

# ==========================================
# 1️⃣ Rolling Z-Score (Adaptive Detector)
# ==========================================

df["rolling_mean"] = df["rss"].rolling(window=window).mean()
df["rolling_std"] = df["rss"].rolling(window=window).std()

df["z_rolling"] = (
    (df["rss"] - df["rolling_mean"]) /
    df["rolling_std"].replace(0, np.nan)
).fillna(0)

df["rolling_detected"] = (df["z_rolling"].abs() > threshold).astype(int)

print("Rolling detected:", df["rolling_detected"].sum())

# ==========================================
# 2️⃣ Frozen Baseline Z-Score (Fixed Reference)
# ==========================================

baseline_mean = s1["rss"].mean()
baseline_std = s1["rss"].std()

df["z_frozen"] = (df["rss"] - baseline_mean) / baseline_std
df["frozen_detected"] = (df["z_frozen"].abs() > threshold).astype(int)

print("Frozen detected:", df["frozen_detected"].sum())

# ==========================================
# Evaluation
# ==========================================

def evaluate(col, name):
    print(f"\n--- {name} ---")
    print("Precision:", precision_score(df["is_anomaly"], df[col], zero_division=0))
    print("Recall:", recall_score(df["is_anomaly"], df[col], zero_division=0))
    print("F1:", f1_score(df["is_anomaly"], df[col], zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(df["is_anomaly"], df[col]))

evaluate("rolling_detected", "Rolling Z-Score")
evaluate("frozen_detected", "Frozen Baseline")

# ==========================================
# Plot Settings
# ==========================================

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

# ==========================================
# Plot Rolling (Point-based)
# ==========================================

def plot_rolling():
    fig, ax = plt.subplots(figsize=(12,6), dpi=300)

    ax.plot(df["t"], df["rss"],
            color="black", linewidth=1.8, label="RSS")

    ax.axvspan(len(s1), len(s1)+len(s2),
               color="gray", alpha=0.2,
               label="Injected Anomaly")

    points = df[df["rolling_detected"] == 1]

    ax.scatter(points["t"], points["rss"],
               color="red", s=60,
               edgecolors="black",
               label="Detected Points",
               zorder=3)

    ax.set_ylim(df["rss"].min()*0.98,
                df["rss"].max()*1.02)

    ax.set_title("Rolling Z-Score Detection (Adaptive)")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("RSS (KB)")
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig("../data/rolling.png", dpi=300)
    plt.close()

# ==========================================
# Plot Frozen (Region-based)
# ==========================================

def plot_frozen():
    fig, ax = plt.subplots(figsize=(12,6), dpi=300)

    ax.plot(df["t"], df["rss"],
            color="black", linewidth=1.8, label="RSS")

    ax.axvspan(len(s1), len(s1)+len(s2),
               color="gray", alpha=0.2,
               label="Injected Anomaly")

    in_seg = False
    start = None
    first = True

    for i in range(len(df)):
        if df["frozen_detected"].iloc[i] == 1 and not in_seg:
            in_seg = True
            start = i
        elif df["frozen_detected"].iloc[i] == 0 and in_seg:
            in_seg = False
            ax.axvspan(start, i,
                       color="purple",
                       alpha=0.25,
                       label="Detected Region" if first else "")
            first = False

    if in_seg:
        ax.axvspan(start, len(df)-1,
                   color="purple",
                   alpha=0.25,
                   label="Detected Region" if first else "")

    ax.set_ylim(df["rss"].min()*0.98,
                df["rss"].max()*1.02)

    ax.set_title("Frozen Baseline Detection (Fixed Reference)")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("RSS (KB)")
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig("../data/frozen.png", dpi=300)
    plt.close()

# ==========================================
# Generate Plots
# ==========================================

plot_rolling()
plot_frozen()