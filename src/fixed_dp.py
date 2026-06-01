#生成 Fixed DP 轨迹


import numpy as np


def laplace_noise(value, epsilon):

    noise = np.random.laplace(
        0,
        1/epsilon
    )

    return value + noise

