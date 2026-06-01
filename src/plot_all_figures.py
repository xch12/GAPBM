#自动生成全部论文图，自动保存到 figures/

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULT_DIR = Path("../results")
FIG_DIR = Path("../figures")

FIG_DIR.mkdir(exist_ok=True)

# ==========================
# Figure 1
# Epsilon Analysis
# ==========================

df = pd.read_csv(
    RESULT_DIR / "epsilon_experiment.csv"
)

plt.figure(figsize=(8,6))

plt.plot(
    df["epsilon"],
    df["MAE"],
    marker="o"
)

plt.xlabel("Epsilon Max")
plt.ylabel("MAE")
plt.title("Impact of Privacy Budget")

plt.grid(True)

plt.savefig(
    FIG_DIR / "fig_epsilon.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================
# Figure 2
# Grid Analysis
# ==========================

df = pd.read_csv(
    RESULT_DIR / "grid_experiment.csv"
)

plt.figure(figsize=(8,6))

plt.plot(
    df["grid_size"],
    df["MAE"],
    marker="o"
)

plt.xlabel("Grid Size")
plt.ylabel("MAE")
plt.title("Impact of Grid Size")

plt.grid(True)

plt.savefig(
    FIG_DIR / "fig_grid.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================
# Figure 3
# Sensitive Ratio
# ==========================

df = pd.read_csv(
    RESULT_DIR / "ratio_experiment.csv"
)

plt.figure(figsize=(8,6))

plt.plot(
    df["ratio"],
    df["MAE"],
    marker="o"
)

plt.xlabel("Sensitive Region Ratio")
plt.ylabel("MAE")
plt.title("Impact of Sensitive Region Ratio")

plt.grid(True)

plt.savefig(
    FIG_DIR / "fig_ratio.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================
# Figure 4
# Trajectory Length
# ==========================

df = pd.read_csv(
    RESULT_DIR / "length_experiment.csv"
)

plt.figure(figsize=(8,6))

plt.plot(
    df["length"],
    df["MAE"],
    marker="o"
)

plt.xlabel("Trajectory Length")
plt.ylabel("MAE")
plt.title("Impact of Trajectory Length")

plt.grid(True)

plt.savefig(
    FIG_DIR / "fig_length.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================
# Figure 5
# Heatmap
# ==========================

grid_size = 10

heatmap = np.full(
    (grid_size, grid_size),
    0.2
)

center = grid_size // 2

heatmap[
    center-2:center+2,
    center-2:center+2
] = 0.9

plt.figure(figsize=(7,6))

plt.imshow(heatmap)

plt.colorbar()

plt.title(
    "Sensitive Region Heatmap"
)

plt.savefig(
    FIG_DIR / "fig_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ==========================
# Figure 6
# Dynamic Epsilon
# ==========================

epsilon_max = 5

sensitivity = np.random.choice(
    [0.2,0.9],
    200
)

epsilon = epsilon_max * (
    1 - sensitivity
)

plt.figure(figsize=(8,6))

plt.plot(epsilon)

plt.xlabel("Trajectory Point")

plt.ylabel("Epsilon")

plt.title(
    "Dynamic Privacy Budget Distribution"
)

plt.grid(True)

plt.savefig(
    FIG_DIR / "fig_dynamic_epsilon.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("All figures generated successfully.")