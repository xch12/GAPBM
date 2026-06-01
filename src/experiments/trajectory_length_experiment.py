#规模扩展实验
#不同轨迹长度对：GAPBM稳定性；累积误差；隐私保护效果的影响。

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
# 1. 加载完整轨迹
# ====================================

file_path = "../../data/raw/Geolife Trajectories 1.3/Data/000/Trajectory/20081023025304.plt"

x_full, y_full = load_geolife_trajectory(file_path)

# ====================================
# 2. 参数
# ====================================

grid_size = 10

epsilon_max = 5

trajectory_lengths = [100, 500, 1000, 2000]

results = []

# ====================================
# 3. 敏感度地图
# ====================================

sensitivity_map = build_sensitivity_map(grid_size)

# ====================================
# 4. 自动实验
# ====================================

for length in trajectory_lengths:

    print(f"\nRunning Length={length}")

    # 防止越界
    if length > len(x_full):

        print("Skip: trajectory too short")

        continue

    x = x_full[:length]
    y = y_full[:length]

    start_time = time.time()

    x_private = []
    y_private = []

    # ====================================
    # GAPBM
    # ====================================

    for i in range(len(x)):

        gx, gy = get_grid_id(
            x[i],
            y[i],
            grid_size
        )

        sensitivity = sensitivity_map[(gx, gy)]

        epsilon = epsilon_max * (1 - sensitivity)

        epsilon = max(epsilon, 0.1)

        noisy_x = add_laplace_noise(
            x[i],
            epsilon
        )

        noisy_y = add_laplace_noise(
            y[i],
            epsilon
        )

        x_private.append(noisy_x)
        y_private.append(noisy_y)

    x_private = np.array(x_private)
    y_private = np.array(y_private)

    # ====================================
    # 指标
    # ====================================

    mae = calculate_mae(
        x,
        y,
        x_private,
        y_private
    )

    rmse = calculate_rmse(
        x,
        y,
        x_private,
        y_private
    )

    runtime = time.time() - start_time

    results.append({

        "Trajectory_Length": length,

        "MAE": mae,

        "RMSE": rmse,

        "Runtime": runtime

    })

    print("MAE:", mae)
    print("RMSE:", rmse)

# ====================================
# 5. 保存CSV
# ====================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "../results/trajectory_length_experiment.csv",
    index=False
)

print("\nExperiment Results:")
print(results_df)

# ====================================
# 6. MAE图
# ====================================

plt.figure(figsize=(8,6))

plt.plot(
    results_df["Trajectory_Length"],
    results_df["MAE"],
    marker='o'
)

plt.xlabel("Trajectory Length")

plt.ylabel("MAE")

plt.title("Effect of Trajectory Length on MAE")

plt.grid(True)

plt.savefig(
    "../figures/trajectory_length_mae.png"
)

plt.show()

# ====================================
# 7. RMSE图
# ====================================

plt.figure(figsize=(8,6))

plt.plot(
    results_df["Trajectory_Length"],
    results_df["RMSE"],
    marker='o'
)

plt.xlabel("Trajectory Length")

plt.ylabel("RMSE")

plt.title("Effect of Trajectory Length on RMSE")

plt.grid(True)

plt.savefig(
    "../figures/trajectory_length_rmse.png"
)

plt.show()

print("\nAll figures saved successfully.")