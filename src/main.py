import numpy as np

from trajectory import generate_trajectory
from grid import get_grid_id
from grid import build_sensitivity_map

from privacy import get_dynamic_epsilon
from privacy import add_laplace_noise

from src.evaluation.evaluation import calculate_mae
from src.evaluation.evaluation import calculate_rmse

from visualization import plot_trajectory

# =========================
# 1. 生成轨迹
# =========================

from trajectory import load_geolife_trajectory

file_path = "../data/raw/Geolife Trajectories 1.3/Data/000/Trajectory/20081023025304.plt"

x, y = load_geolife_trajectory(file_path)

# =========================
# 2. 构建敏感度地图
# =========================

sensitivity_map = build_sensitivity_map()

# =========================
# 3. GAPBM
# =========================

x_private = []
y_private = []

for i in range(len(x)):

    gx, gy = get_grid_id(x[i], y[i])

    sensitivity = sensitivity_map[(gx, gy)]

    epsilon = get_dynamic_epsilon(sensitivity)

    noisy_x = add_laplace_noise(x[i], epsilon)
    noisy_y = add_laplace_noise(y[i], epsilon)

    x_private.append(noisy_x)
    y_private.append(noisy_y)

x_private = np.array(x_private)
y_private = np.array(y_private)

# =========================
# 4. 计算指标
# =========================

mae = calculate_mae(x, y, x_private, y_private)

rmse = calculate_rmse(x, y, x_private, y_private)

print("MAE:", mae)

print("RMSE:", rmse)

# =========================
# 5. 可视化
# =========================

plot_trajectory(x, y, "Original Trajectory")

plot_trajectory(x_private,
                y_private,
                "GAPBM Protected Trajectory")