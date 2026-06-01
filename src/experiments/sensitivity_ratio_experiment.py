import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from trajectory import load_geolife_trajectory
from grid import get_grid_id
from grid import build_sensitivity_map
from privacy import add_laplace_noise
from evaluation import calculate_mae, calculate_rmse

# =========================
# 1. 数据
# =========================

file_path = "../../data/raw/Geolife Trajectories 1.3/Data/000/Trajectory/20081023025304.plt"

x, y = load_geolife_trajectory(file_path)

# =========================
# 2. 参数
# =========================

grid_size = 10
epsilon_max = 5

ratios = [0.1, 0.3, 0.5]

results = []

# =========================
# 3. 实验循环
# =========================

for r in ratios:

    print(f"\nRunning Sensitivity Ratio = {r}")

    start_time = time.time()

    sensitivity_map = build_sensitivity_map(grid_size)

    # =========================
    # 动态生成敏感区域数量
    # =========================

    total_cells = grid_size * grid_size
    sensitive_num = int(total_cells * r)

    all_cells = [(i, j) for i in range(grid_size) for j in range(grid_size)]

    np.random.seed(42)
    sensitive_cells = np.random.choice(len(all_cells), sensitive_num, replace=False)

    # 重置敏感度
    for cell in all_cells:
        sensitivity_map[cell] = 0.2

    for idx in sensitive_cells:
        sensitivity_map[all_cells[idx]] = 0.9

    # =========================
    # GAPBM
    # =========================

    x_private = []
    y_private = []

    for i in range(len(x)):

        gx, gy = get_grid_id(x[i], y[i], grid_size)

        sensitivity = sensitivity_map[(gx, gy)]

        epsilon = epsilon_max * (1 - sensitivity)
        epsilon = max(epsilon, 0.1)

        noisy_x = add_laplace_noise(x[i], epsilon)
        noisy_y = add_laplace_noise(y[i], epsilon)

        x_private.append(noisy_x)
        y_private.append(noisy_y)

    x_private = np.array(x_private)
    y_private = np.array(y_private)

    # =========================
    # 指标
    # =========================

    mae = calculate_mae(x, y, x_private, y_private)
    rmse = calculate_rmse(x, y, x_private, y_private)

    runtime = time.time() - start_time

    results.append({
        "ratio": r,
        "MAE": mae,
        "RMSE": rmse,
        "Runtime": runtime
    })

    print("MAE:", mae)
    print("RMSE:", rmse)

# =========================
# 4. 保存结果
# =========================

df = pd.DataFrame(results)
df.to_csv("../results/sensitivity_ratio.csv", index=False)

# =========================
# 5. 画图
# =========================

plt.figure(figsize=(8,6))
plt.plot(df["ratio"], df["MAE"], marker='o')
plt.title("Sensitivity Ratio vs MAE")
plt.xlabel("Sensitive Ratio")
plt.ylabel("MAE")
plt.grid(True)
plt.savefig("../figures/sensitivity_mae.png")
plt.show()

plt.figure(figsize=(8,6))
plt.plot(df["ratio"], df["RMSE"], marker='o')
plt.title("Sensitivity Ratio vs RMSE")
plt.xlabel("Sensitive Ratio")
plt.ylabel("RMSE")
plt.grid(True)
plt.savefig("../figures/sensitivity_rmse.png")
plt.show()

print("\nDone")