import numpy as np


def wave3d(x,
           k_0=1.0,
           amplitude=np.array([1.0, 0.0, 0.0]),
           direction=np.array([1.0, 0.0, 0.0]),
           sign=-1):
    return amplitude * np.exp(sign * 1j * x.dot(k_0 * direction))
