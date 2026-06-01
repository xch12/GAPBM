#生成真实轨迹图

import matplotlib.pyplot as plt

from trajectory_loader import load_geolife_trajectory

file_path = r"../data/Geolife Trajectories 1.3/Data/000/Trajectory/20081023025304.plt"

x, y = load_geolife_trajectory(file_path)

plt.figure(figsize=(8,6))

plt.plot(
    x,
    y
)

plt.title(
    "Original Trajectory"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.savefig(
    "../figures/original_trajectory.png",
    dpi=300
)

plt.show()