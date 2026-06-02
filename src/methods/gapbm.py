#动态ε扰动
#敏感区域不再随机。而是：中心区域↓高敏感外围区域↓低敏感
import numpy as np

def gapbm_perturb(
        x,
        y,
        epsilon_max=5,
        sensitive_ratio=0.2
):

    xmin = min(x)
    xmax = max(x)

    ymin = min(y)
    ymax = max(y)

    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2

    radius_x = (
        xmax - xmin
    ) * sensitive_ratio

    radius_y = (
        ymax - ymin
    ) * sensitive_ratio

    x_gapbm = []
    y_gapbm = []

    epsilon_list = []

    for i in range(len(x)):

        # 判断是否位于敏感区域

        if (
            abs(x[i] - center_x)
            < radius_x
            and
            abs(y[i] - center_y)
            < radius_y
        ):

            sensitivity = 0.9

        else:

            sensitivity = 0.2

        epsilon = (
            epsilon_max
            *
            (1 - sensitivity)
        )

        epsilon = max(
            epsilon,
            0.1
        )

        epsilon_list.append(
            epsilon
        )

        x_gapbm.append(
            np.random.laplace(
                x[i],
                1/epsilon
            )
        )

        y_gapbm.append(
            np.random.laplace(
                y[i],
                1/epsilon
            )
        )

    return (
        x_gapbm,
        y_gapbm,
        epsilon_list
    )