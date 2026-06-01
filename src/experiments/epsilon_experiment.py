##专门实验文件
##每个实验一个文件


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
# 2. 网格参数
# ====================================

grid_size = 10

sensitivity_map = build_sensitivity_map(grid_size)

# ====================================
# 3. epsilon_max实验参数
# ====================================

epsilon_values = [0.5, 1, 3, 5, 10]

results = []

# ====================================
# 4. 自动实验
# ====================================

for epsilon_max in epsilon_values:

    print(f"\nRunning epsilon_max={epsilon_max}")

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

        # 动态epsilon
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
    # 计算指标
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

    # ====================================
    # 保存结果
    # ====================================

    results.append({

        "epsilon_max": epsilon_max,

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
    "../results/epsilon_experiment.csv",
    index=False
)

print("\nExperiment Results:")
print(results_df)

# ====================================
# 6. 画MAE图
# ====================================

plt.figure(figsize=(8,6))

plt.plot(
    results_df["epsilon_max"],
    results_df["MAE"],
    marker='o'
)

plt.xlabel("epsilon_max")

plt.ylabel("MAE")

plt.title("Effect of epsilon_max on MAE")

plt.grid(True)

plt.savefig(
    "../figures/epsilon_mae.png"
)

plt.show()

# ====================================
# 7. 画RMSE图
# ====================================

plt.figure(figsize=(8,6))

plt.plot(
    results_df["epsilon_max"],
    results_df["RMSE"],
    marker='o'
)

plt.xlabel("epsilon_max")

plt.ylabel("RMSE")

plt.title("Effect of epsilon_max on RMSE")

plt.grid(True)

plt.savefig(
    "../figures/epsilon_rmse.png"
)

plt.show()

print("\nAll figures saved successfully.")