import matplotlib.pyplot as plt

from src.epsilon_visualization import epsilon_max
from trajectory_loader import load_geolife_trajectory
from fixed_dp import laplace_noise
from gapbm import gapbm_perturb

# ==========================
# 1. 读取GeoLife轨迹
# ==========================

file_path = r"../data/raw/Geolife Trajectories 1.3/Data/000/Trajectory/20081023025304.plt"

x, y = load_geolife_trajectory(file_path)

# 为了图更清晰
# 先取前1000个点

x = x[:1000]
y = y[:1000]

# ==========================
# 2. Fixed DP
# ==========================

epsilon = 3

x_fixed = [
    laplace_noise(v, epsilon)
    for v in x
]

y_fixed = [
    laplace_noise(v, epsilon)
    for v in y
]

# ==========================
# 3. GAPBM
# ==========================

x_gapbm, y_gapbm, epsilon_list = gapbm_perturb(
    x,
    y,
    epsilon
)

# ==========================
# 4. 绘制SCI风格三联图
# ==========================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)

# --------------------------
# (a) Original
# --------------------------

axes[0].plot(
    x,
    y,
    linewidth=1
)

axes[0].set_title(
    "(a) Original Trajectory"
)

axes[0].set_xlabel(
    "Longitude"
)

axes[0].set_ylabel(
    "Latitude"
)

axes[0].grid(True)

# --------------------------
# (b) Fixed DP
# --------------------------

axes[1].plot(
    x_fixed,
    y_fixed,
    linewidth=1
)

axes[1].set_title(
    "(b) Fixed DP"
)

axes[1].set_xlabel(
    "Longitude"
)

axes[1].set_ylabel(
    "Latitude"
)

axes[1].grid(True)

# --------------------------
# (c) GAPBM
# --------------------------

axes[2].plot(
    x_gapbm,
    y_gapbm,
    linewidth=1
)

axes[2].set_title(
    "(c) GAPBM"
)

axes[2].set_xlabel(
    "Longitude"
)

axes[2].set_ylabel(
    "Latitude"
)

axes[2].grid(True)

# --------------------------
# 自动调整布局
# --------------------------

plt.tight_layout()

# ==========================
# 保存
# ==========================

plt.savefig(
    "../figures/Fig4_Trajectory_Comparison.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()




# from trajectory_loader import load_geolife_trajectory
#
# from fixed_dp import laplace_noise
#
# from gapbm import gapbm_perturb
#
# import matplotlib.pyplot as plt
#
# # 读取数据
#
# file_path = r"../data/Geolife Trajectories 1.3/Data/000/Trajectory/20081023025304.plt"
#
# x, y = load_geolife_trajectory(
#     file_path
# )
#
# # =====================
# # Fixed DP
# # =====================
#
# epsilon = 3
#
# x_fixed = [
#     laplace_noise(v, epsilon)
#     for v in x
# ]
#
# y_fixed = [
#     laplace_noise(v, epsilon)
#     for v in y
# ]
#
# # =====================
# # GAPBM
# # =====================
#
# x_gapbm, y_gapbm = gapbm_perturb(
#     x,
#     y
# )
#
# # =====================
# # 绘图
# # =====================
#
# plt.figure(figsize=(10,8))
#
# plt.plot(
#     x,
#     y,
#     label="Original"
# )
#
# plt.plot(
#     x_fixed,
#     y_fixed,
#     label="Fixed DP"
# )
#
# plt.plot(
#     x_gapbm,
#     y_gapbm,
#     label="GAPBM"
# )
#
# plt.legend()
#
# plt.savefig(
#     "../figures/trajectory_compare.png",
#     dpi=300
# )
#
# plt.show()