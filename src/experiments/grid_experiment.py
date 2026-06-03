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
# 2. 实验参数
# ====================================

grid_sizes = [5, 10, 20, 30]

epsilon_max = 5

results = []

# ====================================
# 3. 自动实验
# ====================================

for grid_size in grid_sizes:

    print(f"\nRunning Grid Size={grid_size}")

    start_time = time.time()

    sensitivity_map = build_sensitivity_map(grid_size)

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

        "grid_size": grid_size,

        "MAE": mae,

        "RMSE": rmse,

        "Runtime": runtime

    })

    print("MAE:", mae)
    print("RMSE:", rmse)

# ====================================
# 4. 保存CSV
# ====================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "../results/grid_experiment.csv",
    index=False
)

print("\nExperiment Results:")
print(results_df)

# ====================================
# 5. MAE图
# ====================================

plt.figure(figsize=(8,6))

plt.plot(
    results_df["grid_size"],
    results_df["MAE"],
    marker='o'
)

plt.xlabel("Grid Size")

plt.ylabel("MAE")

plt.title("Effect of Grid Size on MAE")

plt.grid(True)

plt.savefig(
    "../figures/grid_mae.png"
)

plt.show()

# ====================================
# 6. RMSE图
# ====================================

plt.figure(figsize=(8,6))

plt.plot(
    results_df["grid_size"],
    results_df["RMSE"],
    marker='o'
)

plt.xlabel("Grid Size")

plt.ylabel("RMSE")

plt.title("Effect of Grid Size on RMSE")

plt.grid(True)

plt.savefig(
    "../figures/grid_rmse.png"
)

plt.show()

print("\nAll figures saved successfully.")