#Figure 2 动态ε分布图

import numpy as np
import matplotlib.pyplot as plt

epsilon_max = 5

sensitivity = np.random.choice(
    [0.2,0.9],
    200
)

epsilon = epsilon_max * (
    1 - sensitivity
)

plt.figure(figsize=(10,6))

plt.plot(
    epsilon
)

plt.xlabel(
    "Trajectory Point"
)

plt.ylabel(
    "Privacy Budget"
)

plt.title(
    "Dynamic Privacy Budget Distribution"
)

plt.grid()

plt.savefig(
    "../figures/epsilon_distribution.png",
    dpi=300
)

plt.show()