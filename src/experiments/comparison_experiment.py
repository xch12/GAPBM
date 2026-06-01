##Fixed DP vs GAPBM对比实验

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from trajectory import load_geolife_trajectory

from grid import get_grid_id
from grid import build_sensitivity_map

from privacy import add_laplace_noise

from evaluation import calculate_mae
from evaluation import calculate_rmse

# ====================================
# 1. 加载真实轨迹
# ====================================

file_path = "../../data/raw/Geolife Trajectories 1.3/Data/000/Trajectory/20081023025304.plt"

x, y = load_geolife_trajectory(file_path)

# ====================================
# 2. 参数设置
# ====================================

grid_size = 10

epsilon = 5

sensitivity_map = build_sensitivity_map(grid_size)

# ====================================
# 3. Fixed DP
# ====================================

print("\nRunning Fixed DP...")

start_time = time.time()

x_fixed = []
y_fixed = []

for i in range(len(x)):

    noisy_x = add_laplace_noise(
        x[i],
        epsilon
    )

    noisy_y = add_laplace_noise(
        y[i],
        epsilon
    )

    x_fixed.append(noisy_x)
    y_fixed.append(noisy_y)

x_fixed = np.array(x_fixed)
y_fixed = np.array(y_fixed)

fixed_mae = calculate_mae(
    x,
    y,
    x_fixed,
    y_fixed
)

fixed_rmse = calculate_rmse(
    x,
    y,
    x_fixed,
    y_fixed
)

fixed_runtime = time.time() - start_time

# ====================================
# 4. GAPBM
# ====================================

print("\nRunning GAPBM...")

start_time = time.time()

x_gapbm = []
y_gapbm = []

epsilon_list = []

for i in range(len(x)):

    gx, gy = get_grid_id(
        x[i],
        y[i],
        grid_size
    )

    sensitivity = sensitivity_map[(gx, gy)]

    dynamic_epsilon = epsilon * (1 - sensitivity)

    dynamic_epsilon = max(dynamic_epsilon, 0.1)

    epsilon_list.append(dynamic_epsilon)

    noisy_x = add_laplace_noise(
        x[i],
        dynamic_epsilon
    )

    noisy_y = add_laplace_noise(
        y[i],
        dynamic_epsilon
    )

    x_gapbm.append(noisy_x)
    y_gapbm.append(noisy_y)

x_gapbm = np.array(x_gapbm)
y_gapbm = np.array(y_gapbm)

gapbm_mae = calculate_mae(
    x,
    y,
    x_gapbm,
    y_gapbm
)

gapbm_rmse = calculate_rmse(
    x,
    y,
    x_gapbm,
    y_gapbm
)

gapbm_runtime = time.time() - start_time

# ====================================
# 5. 结果表
# ====================================

results = pd.DataFrame({

    "Method": ["Fixed DP", "GAPBM"],

    "MAE": [fixed_mae, gapbm_mae],

    "RMSE": [fixed_rmse, gapbm_rmse],

    "Runtime": [fixed_runtime, gapbm_runtime]

})

print("\nExperiment Results:")
print(results)

# ====================================
# 6. 保存CSV
# ====================================

results.to_csv(
    "../results/comparison_experiment.csv",
    index=False
)

# ====================================
# 7. MAE对比图
# ====================================

plt.figure(figsize=(6,5))

plt.bar(
    results["Method"],
    results["MAE"]
)

plt.ylabel("MAE")

plt.title("MAE Comparison")

plt.savefig(
    "../figures/comparison_mae.png"
)

plt.show()

# ====================================
# 8. RMSE对比图
# ====================================

plt.figure(figsize=(6,5))

plt.bar(
    results["Method"],
    results["RMSE"]
)

plt.ylabel("RMSE")

plt.title("RMSE Comparison")

plt.savefig(
    "../figures/comparison_rmse.png"
)

plt.show()

# ====================================
# 9. 轨迹对比图
# ====================================

plt.figure(figsize=(10,8))

plt.plot(
    x,
    y,
    label="Original",
    linewidth=2
)

plt.plot(
    x_fixed,
    y_fixed,
    label="Fixed DP",
    linestyle='--'
)

plt.plot(
    x_gapbm,
    y_gapbm,
    label="GAPBM",
    linestyle=':'
)

plt.xlabel("X")

plt.ylabel("Y")

plt.title("Trajectory Comparison")

plt.legend()

plt.grid(True)

plt.savefig(
    "../figures/trajectory_comparison.png"
)

plt.show()

print("\nAll figures saved successfully.")