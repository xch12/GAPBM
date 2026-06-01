##负责自动实验

import numpy as np
import pandas as pd
import time

from trajectory import load_geolife_trajectory

from grid import get_grid_id
from grid import build_sensitivity_map

from privacy import get_dynamic_epsilon
from privacy import add_laplace_noise

from evaluation import calculate_mae
from evaluation import calculate_rmse

# ====================================
# 1. 加载真实轨迹
# ====================================

file_path = "../data/raw/Geolife Trajectories 1.3/Data/000/Trajectory/20081023025304.plt"

x, y = load_geolife_trajectory(file_path)

# ====================================
# 2. 参数实验设置
# ====================================

epsilon_values = [1, 3, 5, 10]

results = []

# ====================================
# 3. 构建敏感度地图
# ====================================

sensitivity_map = build_sensitivity_map()

# ====================================
# 4. 自动实验
# ====================================

for epsilon_max in epsilon_values:

    print(f"\nRunning Experiment: epsilon_max={epsilon_max}")

    start_time = time.time()

    x_private = []
    y_private = []

    epsilon_list = []

    for i in range(len(x)):

        gx, gy = get_grid_id(x[i], y[i])

        sensitivity = sensitivity_map[(gx, gy)]

        epsilon = epsilon_max * (1 - sensitivity)

        epsilon = max(epsilon, 0.1)

        epsilon_list.append(epsilon)

        noisy_x = add_laplace_noise(x[i], epsilon)
        noisy_y = add_laplace_noise(y[i], epsilon)

        x_private.append(noisy_x)
        y_private.append(noisy_y)

    x_private = np.array(x_private)
    y_private = np.array(y_private)

    # ====================================
    # 5. 评估指标
    # ====================================

    mae = calculate_mae(x, y,
                        x_private, y_private)

    rmse = calculate_rmse(x, y,
                          x_private, y_private)

    runtime = time.time() - start_time

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("Runtime:", runtime)

    # ====================================
    # 6. 保存结果
    # ====================================

    results.append({

        "epsilon_max": epsilon_max,

        "MAE": mae,

        "RMSE": rmse,

        "Runtime": runtime

    })

# ====================================
# 7. 保存CSV
# ====================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "../results/epsilon_experiment.csv",
    index=False
)

print("\nAll experiments completed.")
print(results_df)