import numpy as np
import numpy.typing as npt
from typing import Iterable, Callable


def BiSGStab_EM3d(operator_array: npt.NDArray[np.complex128],
                  vec_array: npt.NDArray[np.complex128],
                  u0_array: npt.NDArray[np.complex128] = None,
                  rtol: float = 1e-8,
                  max_iter: int = 1000) -> [npt.NDArray[np.complex128], Iterable, Iterable]:
    """

    :param operator_array: известная матрица A оператора размера (N, N), состоящая из комплексных чисел
    :param vec_array: известный вектор f для правой части системы уравнений размера (N,), состоящий из комплексных чисел
    :param u0_array: начальное приближение u0 в качестве начальной точки итерационного метода размера (N,) со значением None по умолчанию. Если передан None, то создавать вектор нужного размера из единиц
    :param permutation_func: функция, реализующая умножение оператора на вектор
    :param rtol: относительная ошибка нормы разницы итераций поделенная на норму вектора f, обозначаемый rtol как критерий остановки алгоритма, по умолчанию равный 10е-8
    :param max_iter: максимальное количество итерационных шагов max_iter как критерий остановки алгоритма, по умолчанию равный 1000
    :return:
    """

    problem_shape = vec_array.shape
    if u0_array is None:
        u = np.ones(problem_shape, dtype=np.complex128)
    else:
        u = u0_array.astype(np.complex128).copy()

    r = vec_array - permutation_func(operator_array, u)
    r_hat = r.copy()
    rho_old = alpha = omega_old = 1 + 0j
    v = np.zeros(problem_shape, dtype=np.complex128)
    p = np.zeros(problem_shape, dtype=np.complex128)
    rel_error_list = []
    residual_list = []
    f_norm = np.linalg.norm(vec_array)

    for k in range(max_iter):
        rho_new = np.dot(r_hat.conj(), r)
        if rho_new == 0:
            print("Breakdown: rho_new == 0")
            break

        if k == 0:
            p = r.copy()
        else:
            beta = (rho_new / rho_old) * (alpha / omega_old)
            p = r + beta * (p - omega_old * v)

        v = A @ p

        sigma = np.dot(r_hat.conj(), v)
        if sigma == 0:
            print("Breakdown: sigma == 0")
            break

        alpha = rho_new / sigma

        s = r - alpha * v

        if np.linalg.norm(s) < rtol * f_norm:
            u = u + alpha * p
            # Compute relative error and residual
            rel_error = np.linalg.norm(u - u0) / f_norm
            rel_error_list.append(rel_error)
            residual = A @ u - f
            residual_list.append(residual)
            break

        t = A @ s
        t_norm = np.dot(t.conj(), t)
        if t_norm == 0:
            omega_new = 0
        else:
            omega_new = np.dot(t.conj(), s) / t_norm

        if omega_new == 0:
            print("Breakdown: omega_new == 0")
            break

        u_old = u.copy()
        u = u + alpha * p + omega_new * s
        r = s - omega_new * t

        # Compute relative error and residual
        rel_error = np.linalg.norm(u - u_old) / f_norm
        rel_error_list.append(rel_error)
        residual = A @ u - f
        residual_list.append(residual)

        if np.linalg.norm(r) < rtol * f_norm or rel_error < rtol:
            break

        rho_old = rho_new
        omega_old = omega_new

    return u, rel_error_list, residual_list

