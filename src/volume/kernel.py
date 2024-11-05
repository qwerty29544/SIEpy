import numpy as np


def helm3d_kernel(x, k=1.0):
    return np.exp(1j * x * k) / (4 * np.pi * x)
