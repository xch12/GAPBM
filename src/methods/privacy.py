##动态ε + 拉普拉斯
import numpy as np

epsilon_max = 5.0

def get_dynamic_epsilon(sensitivity):

    epsilon = epsilon_max * (1 - sensitivity)

    return max(epsilon, 0.1)

def add_laplace_noise(value, epsilon):

    noise = np.random.laplace(0, 1 / epsilon)

    return value + noise