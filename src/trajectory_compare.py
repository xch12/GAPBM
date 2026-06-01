#原始轨迹 vs Fixed DP vs GAPBM
from matplotlib import pyplot as plt

from src.experiments.comparison_experiment import x_gapbm, y_gapbm, y_fixed, x_fixed, x, y

plt.plot(
    x,
    y,
    label='Original'
)

plt.plot(
    x_fixed,
    y_fixed,
    label='Fixed DP'
)

plt.plot(
    x_gapbm,
    y_gapbm,
    label='GAPBM'
)

plt.legend()