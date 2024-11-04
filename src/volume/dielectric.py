import numpy as np
import numpy.typing as npt
from typing import Iterable


def step_dielectric_3d(grid: npt.NDArray[np.float64],
                       eps_real: Iterable = None,
                       eps_imag: Iterable = None,
                       x1_bounds: Iterable = None,
                       x2_bounds: Iterable = None,
                       x3_bounds: Iterable = None) -> npt.NDArray[np.complex128]:
    """
    grid - Тензор (n, 3, 3) для каждой точки пространства
    eps_real
    """
    if eps_real is None:
        eps_real = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    if eps_imag is None:
        eps_imag = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    if x1_bounds is None:
        x1_bounds = [-1.0, 0.5]
    if x2_bounds is None:
        x2_bounds = [-1.0, 1.0]
    if x3_bounds is None:
        x3_bounds = [-1.0, 1.0]

    eps = np.zeros((grid.shape[0], 3, 3)) + 0j
    indexes = ((grid[:, 0] >= x1_bounds[0]) * (grid[:, 0] <= x1_bounds[1]) *
               (grid[:, 1] >= x2_bounds[0]) * (grid[:, 1] <= x2_bounds[1]) *
               (grid[:, 2] >= x3_bounds[0]) * (grid[:, 2] <= x3_bounds[1]))
    eps[indexes, :, :] = np.array(eps_real) + 1j * np.array(eps_imag)
    return eps


def ellipsis_dielectric_3d(grid: npt.NDArray[np.float64],
                           eps_real: Iterable = None,
                           eps_imag: Iterable = None,
                           center: Iterable = None,
                           radius: Iterable = None) -> npt.NDArray[np.complex128]:
    if eps_real is None:
        eps_real = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    if eps_imag is None:
        eps_imag = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    if center is None:
        center = [0.0, 0.0, 0.0]
    if radius is None:
        radius = [1.0, 1.0, 1.0]
    eps = np.zeros((grid.shape[0], 3, 3)) + 0j
    indexes = (
        ((grid[:, 0] - center[0]) ** 2) / (radius[0] ** 2) +
        ((grid[:, 1] - center[1]) ** 2) / (radius[1] ** 2) +
        ((grid[:, 2] - center[2]) ** 2) / (radius[2] ** 2)
    ) <= 1
    eps[indexes, :, :] = np.array(eps_real) + 1j * np.array(eps_imag)
    return eps


def apply_dielectric_3d(grid,
                        eps_vec=None):
    if eps_vec is None:
        eps_vec = [
            {
                "type": "step",
                "eps_real": [[1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]],
                "eps_imag": [[0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]],
                "x1_bounds": [-1.0, 0.0],
                "x2_bounds": [-1.0, 1.0],
                "x3_bounds": [-1.0, 1.0]
            },
            {
                "type": "ellipsis",
                "eps_real": [[1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]],
                "eps_imag": [[0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]],
                "center": [0.0, 0.0, 0.0],
                "radius": [1.0, 1.0, 1.0]
            }
        ]

    eps = np.zeros((grid.shape[0], 3, 3)) + 0j
    for num, element in enumerate(eps_vec):
        if element.get("type") == "step":
            eps += step_dielectric_3d(grid=grid,
                                      eps_real=element.get('eps_real'),
                                      eps_imag=element.get('eps_imag'),
                                      x1_bounds=element.get('x1_bounds'),
                                      x2_bounds=element.get('x2_bounds'),
                                      x3_bounds=element.get('x3_bounds'))
        elif element.get("type") == "ellipsis":
            eps += ellipsis_dielectric_3d(grid=grid,
                                          eps_real=element.get('eps_real'),
                                          eps_imag=element.get('eps_imag'),
                                          center=element.get('center'),
                                          radius=element.get('radius'))
        else:
            continue
    return eps
