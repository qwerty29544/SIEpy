import matplotlib.pyplot as plt
import numpy as np
from volume.dielectric import apply_dielectric_3d
from volume.kernel import helm3d_kernel
from volume.wave import wave3d
from special.fast_mul_mv import prep_fbbtmv, prep_fftbbtensor, fast_bbtmv_mul
from special.utils import save_np_file_txt
from special.hull2d import ConvexHull
from special.spectre_params import find_params
from special.visualisation import plot_cube_scatter
import os


def init_grid(x1_bound=None,
              x2_bound=None,
              x3_bound=None,
              discret=None):
    if x1_bound is None:
        x1_bound = [0, 1]
    if x2_bound is None:
        x2_bound = [0, 1]
    if x3_bound is None:
        x3_bound = [0, 1]
    if discret is None:
        discret = [10, 10, 10]

    x1_grid = np.linspace(x1_bound[0], x1_bound[1], discret[0] + 1)
    x2_grid = np.linspace(x2_bound[0], x2_bound[1], discret[1] + 1)
    x3_grid = np.linspace(x3_bound[0], x3_bound[1], discret[2] + 1)

    h1 = x1_grid[1] - x1_grid[0]
    h2 = x2_grid[1] - x2_grid[0]
    h3 = x3_grid[1] - x3_grid[0]
    volume = h1 * h2 * h3

    x1_grid = (x1_grid[1:] + x1_grid[:-1]) / 2
    x2_grid = (x2_grid[1:] + x2_grid[:-1]) / 2
    x3_grid = (x3_grid[1:] + x3_grid[:-1]) / 2

    grid = np.array(np.meshgrid(x2_grid, x1_grid, x3_grid, indexing='xy')).T.reshape(-1, 3)[:, [1, 0, 2]]
    return grid, volume


def init_operator(grid, volume, discret, distance, k, eps, kernel):
    dist_coef_1 = np.zeros(np.prod(discret)) + 0.0j
    dist_coef_1[1:] = (3 / (distance[1:] ** 2)) - ((3j * k) / distance[1:]) - (k ** 2)
    alpha = np.zeros((np.prod(discret), 3))
    alpha[1:, :] = (grid[0, :] - grid[1:, :]) / distance[1:].reshape(-1, 1)
    indexing_matrix = np.einsum('...kj,...ki->kij', alpha, alpha)
    part_1 = indexing_matrix * dist_coef_1.reshape(-1, 1, 1)

    dist_coef_2 = np.zeros(np.prod(discret)) + 0.0j
    dist_coef_2[1:] = (k ** 2) + ((1j * k) / distance[1:]) - (1 / (distance[1:] ** 2))
    part_2 = np.eye(3).reshape(-1, 3, 3) * dist_coef_2.reshape(-1, 1, 1)

    gr = np.zeros(np.prod(discret)) + 0.0j
    gr[1:] = kernel(distance[1:], k)
    result = gr.reshape(-1, 1, 1) * (part_1 + part_2) * volume.reshape(-1, 1)
    eps = eps.reshape(-1, 3, 3) - (np.eye(3).reshape(-1, 3, 3))
    result[0, :, :] = -(1 / 3) * np.eye(3)
    return result, gr, eps


def operator(operator_array, vector_array, eps, n, m, k):
    result = np.zeros((n * m * k, 3)) + 0.0j
    vec_array = np.einsum("...ijk,...ij->...ik", eps, vector_array)
    for col in range(3):
        res = prep_fbbtmv(prep_fft_arr=operator_array[:, col].reshape((2 * n, 2 * m, 2 * k)),
                          vec_arr=vec_array[:, col].reshape((n, m, k)),
                          n=n, m=m, k=k).reshape((-1,))
        result[:, 0] += res
    for col in range(3):
        result[:, 1] += prep_fbbtmv(prep_fft_arr=operator_array[:, 3 + col].reshape((2 * n, 2 * m, 2 * k)),
                                    vec_arr=vec_array[:, col].reshape((n, m, k)),
                                    n=n, m=m, k=k).reshape((-1,))
    for col in range(3):
        result[:, 2] += prep_fbbtmv(prep_fft_arr=operator_array[:, 6 + col].reshape((2 * n, 2 * m, 2 * k)),
                                    vec_arr=vec_array[:, col].reshape((n, m, k)),
                                    n=n, m=m, k=k).reshape((-1,))

    result = vector_array - result
    return result


def GIM_EM3dprep(prep_oper_array, vec_array, u0_array=None,
                 mu_0=None, n=None, m=None, k=None, rtol=1e-8, max_iter=1000):
    if n is None:
        n = 10
    if m is None:
        m = 10
    if k is None:
        k = 10
    if mu_0 is None:
        mu_0 = 1.0 + 0.0j
    if u0_array is None:
        u = np.ones((n * m * k, 3), dtype=np.complex128)
    else:
        u = u0_array.astype(np.complex128).copy()

    rel_error_list = []
    residual_list = []
    f_norm = np.linalg.norm(np.linalg.norm(vec_array, axis=1))

    for iter in range(max_iter):
        resid = operator(prep_oper_array, u, n, m, k) - vec_array
        u = u - resid / mu_0

        rel_error = np.linalg.norm(np.linalg.norm(u - u0_array, axis=1)) / f_norm
        rel_error_list.append(rel_error)
        residual = np.linalg.norm(np.linalg.norm(resid, axis=1))
        residual_list.append(residual)
        print(f"iter {iter},\t rel_error is {round(rel_error, 8)}, \t system residual is {round(residual, 8)}")
        if rel_error < rtol:
            break
        u0_array = u.copy()

    return u, rel_error_list, residual_list


def BiCGStab_EM3dprep(prep_oper_array, vec_array, u0_array=None, eps=None,
                      n=None, m=None, k=None,
                      rtol=1e-8, max_iter=1000):
    """

    :param prep_oper_array: известная матрица A оператора размера (N, N), состоящая из комплексных чисел
    :param vec_array: известный вектор f для правой части системы уравнений размера (N,), состоящий из комплексных чисел
    :param u0_array: начальное приближение u0 в качестве начальной точки итерационного метода размера (N,) со значением None по умолчанию. Если передан None, то создавать вектор нужного размера из единиц
    :param n:
    :param m:
    :param k:
    :param rtol: относительная ошибка нормы разницы итераций поделенная на норму вектора f, обозначаемый rtol как критерий остановки алгоритма, по умолчанию равный 10е-8
    :param max_iter: максимальное количество итерационных шагов max_iter как критерий остановки алгоритма, по умолчанию равный 1000
    :return:
    """

    if n is None:
        n = 10
    if m is None:
        m = 10
    if k is None:
        k = 10
    if u0_array is None:
        u = np.ones((n * m * k, 3), dtype=np.complex128)
    else:
        u = u0_array.astype(np.complex128).copy()

    r = vec_array - operator(prep_oper_array, u, eps, n, m, k)
    r_hat = r.copy()
    rho_old = alpha = omega_old = 1 + 0j
    v = np.zeros((n * m * k, 3), dtype=np.complex128)
    p = np.zeros((n * m * k, 3), dtype=np.complex128)
    rel_error_list = []
    residual_list = []
    f_norm = np.linalg.norm(np.linalg.norm(vec_array, axis=1))

    for iter in range(max_iter):
        rho_new = np.einsum('...ij,...ij', r_hat.conj(), r)
        if rho_new == 0:
            print("Breakdown: rho_new == 0")
            break

        if iter == 0:
            p = r.copy()
        else:
            beta = (rho_new / rho_old) * (alpha / omega_old)
            p = r + beta * (p - omega_old * v)

        v = operator(prep_oper_array, p, eps, n, m, k)
        sigma = np.einsum('...ij,...ij', r_hat.conj(), v)
        if sigma == 0:
            print("Breakdown: sigma == 0")
            break

        alpha = rho_new / sigma
        s = r - alpha * v
        t = operator(prep_oper_array, s, eps, n, m, k)
        t_norm = np.einsum('...ij,...ij', t.conj(), t)
        if t_norm == 0:
            omega_new = 0
        else:
            omega_new = np.einsum('...ij,...ij', t.conj(), s) / t_norm

        u0_array = u.copy()
        u = u + alpha * p + omega_new * s

        rel_error = np.linalg.norm(np.linalg.norm(u - u0_array, axis=1)) / f_norm
        rel_error_list.append(rel_error)
        residual = np.linalg.norm(np.linalg.norm(operator(prep_oper_array, u, eps, n, m, k) - vec_array, axis=1))
        residual_list.append(residual)
        print(f"iter {iter},\t rel_error is {round(rel_error, 8)}, \t system residual is {round(residual, 8)}")
        if rel_error < rtol:
            break

        r = s - omega_new * t
        rho_old = rho_new
        omega_old = omega_new
    return u, rel_error_list, residual_list


class Em3dProblem:
    def __init__(self):
        self.config = None  # Словарь с настройками эксперимента
        self.n = None  # Список из 3 элементов количества разбиений по каждой оси
        self.eps = None  # Список словарей с настройкой относительной диэлектрической проницаемости
        self.exp_no = None  # Название эксперимента
        self.path = None  # Путь до выводов программы
        self.amplitude = None  # Амплитуда колебаний
        self.k_0 = None  # Волновое число задачи
        self.length = None  # Список из 3 элементов длины исходной области решения
        self.center = None  # Центр исходной области решения
        self.direction = None  # Направление внешних колебаний

        self.grid = None  # Сетка дискретизации исходной области задачи
        self.volume = None  # Объем прямоугольного разбиения
        self.eps_on_grid = None  # Диэлектрическая проницаемость на сетке
        self.distance = None  # Массив расстояний от первой точки области до всех остальных

        self.in_Q = None    # Логический массив принадлежности области
        self.gr = None           # Коэффициенты Гельмгольца
        self.oper_coeffs = None  # Коэффициенты матрицы
        self.prep_coeffs = None  # Подготовленные для быстрого умноженмя коэффициенты
        self.f_values = None  # Значения внешней волны

        self.result = None  # Массив решения задачи
        self.resid = None   # Остатки невязки
        self.rel_err_list = None    # Остатки на итерациях

    def fit(self, fit_config=None):
        if fit_config is None:
            self.config = {
                "exp_no": "exp1",
                "path": "output/",
                "amplitude": [0.0, 1.0, 0.0],
                "k_0": 1.0,
                "n": [20, 30, 40],
                "length": [1.0, 1.0, 1.0],
                "center": [0.0, 0.0, 0.0],
                "direction": [1.0, 0.0, 0.0],
                "eps": [
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
            }
        else:
            self.config = fit_config

        self.n = self.config.get("n")
        self.eps = self.config.get("eps")
        self.exp_no = self.config.get("exp_no")
        self.path = self.config.get("path")
        self.amplitude = self.config.get("amplitude")
        self.k_0 = self.config.get("k_0")
        self.length = self.config.get("length")
        self.center = self.config.get("center")
        self.direction = np.array(self.config.get("direction"))
        self.direction = self.direction / np.linalg.norm(self.direction)

        self.path = os.path.join(self.path, self.exp_no)
        if not os.path.isdir(self.path):
            os.mkdir(self.path)

        bounds = np.array([(-np.array(self.length) / 2) + np.array(self.center),
                           (np.array(self.length) / 2) + np.array(self.center)])
        x1_bound = list(bounds[:, 0])
        x2_bound = list(bounds[:, 1])
        x3_bound = list(bounds[:, 2])
        self.grid, self.volume = init_grid(x1_bound=x1_bound,
                                           x2_bound=x2_bound,
                                           x3_bound=x3_bound,
                                           discret=self.n)

        self.eps_on_grid, self.in_Q = apply_dielectric_3d(self.grid, self.eps)

        self.distance = np.zeros(np.prod(self.n))
        self.distance[1:] = np.linalg.norm(self.grid[0, :] - self.grid[1:, :], axis=1)

        self.oper_coeffs, self.gr, self.eps_on_grid = init_operator(grid=self.grid,
                                                                  volume=self.volume,
                                                                  discret=self.n,
                                                                  distance=self.distance,
                                                                  k=self.k_0,
                                                                  eps=self.eps_on_grid,
                                                                  kernel=helm3d_kernel)
        self.oper_coeffs = self.oper_coeffs.reshape((-1, 9))

        self.prep_coeffs = np.zeros((8 * np.prod(self.n), 9)) + 0.0j
        for rowcol in range(9):
            self.prep_coeffs[:, rowcol] = prep_fftbbtensor(col_arr=self.oper_coeffs[:, rowcol],
                                                           row_arr=self.oper_coeffs[:, rowcol],
                                                           n=int(self.n[0]),
                                                           m=int(self.n[1]),
                                                           k=int(self.n[2])).reshape((8 * np.prod(self.n),))

        self.f_values = wave3d(x=self.grid,
                               k_0=float(self.k_0),
                               direction=np.array(self.direction),
                               amplitude=np.array(self.amplitude))



        print("fit completed")
        print(f"prep_coeffs shape is {self.prep_coeffs.shape}")
        print(f"wave shape is {self.f_values.shape}")
        return self

    def compute_BiCGStab(self, max_iter=None, save=True):
        if max_iter is None:
            max_iter = 100
        print("BiCGStab method iterations:")
        self.result, self.rel_err_list, self.resid = \
            BiCGStab_EM3dprep(prep_oper_array=self.prep_coeffs,
                              vec_array=self.f_values,
                              u0_array=np.ones((np.prod(self.n), 3)) + 0.0j,
                              eps=self.eps_on_grid,
                              n=self.n[0], m=self.n[1], k=self.n[2], max_iter=max_iter)
        if save:
            save_np_file_txt(np.array(self.result), os.path.join(self.path, "result_BiCGStab.txt"))
            save_np_file_txt(np.array(self.rel_err_list), os.path.join(self.path, "rel_err_list_BiCGStab.txt"))
            save_np_file_txt(np.array(self.resid), os.path.join(self.path, "resid_list_BiCGStab.txt"))
            print(f"Results of iterations saved in {self.path}")
        print("iterations is over")

    def compute_RCS(self, n_phi):
        epsilon = 1e-6
        angles = np.linspace(-np.pi+epsilon, np.pi-epsilon, n_phi)
        vectors = np.hstack([
            np.cos(angles).reshape((n_phi, 1)),
            np.zeros((n_phi, 1)),
            np.sin(angles).reshape((n_phi, 1)),
        ])
        s_q = np.einsum('...ijk,...il->...ij', self.eps_on_grid-np.eye(3).reshape(-1, 3, 3), self.result)
        cross_matrix_e_r = np.einsum("...ij,...kj->...ik", self.grid, vectors)
        waves = np.exp(1j * self.k_0 * cross_matrix_e_r) * (self.in_Q.reshape(-1, 1) * self.volume)
        result = np.einsum('...ij,...ik->jk', waves, s_q)
        result = (self.k_0**4) / (16 * np.pi**2) * np.power(np.linalg.norm(np.cross(vectors, result), 2, axis=1), 2)
        return result, angles

    def compute_GIM(self, mu_param=None, max_iter=None, save=True):
        if mu_param is None:
            mu_param = 1.0 + 0.0j
        if max_iter is None:
            max_iter = 100
        print("GIM method iterations:")
        self.result, self.rel_err_list, self.resid = \
            GIM_EM3dprep(prep_oper_array=self.prep_coeffs,
                         vec_array=self.f_values,
                         u0_array=np.ones((np.prod(self.n), 3)) + 0.0j,
                         mu_0=mu_param,
                         n=self.n[0], m=self.n[1], k=self.n[2], max_iter=max_iter)
        if save:
            save_np_file_txt(np.array(self.result), os.path.join(self.path, "result_GIM.txt"))
            save_np_file_txt(np.array(self.rel_err_list), os.path.join(self.path, "rel_err_list_GIM.txt"))
            save_np_file_txt(np.array(self.resid), os.path.join(self.path, "resid_list_GIM.txt"))
            print(f"Results of iterations saved in {self.path}")

        print("iterations is over")

    def compute_mu(self):
        result = np.sort(np.unique(self.eps_on_grid))
        mu_0, radius = find_params(result)
        return mu_0, radius

    def plot_cube(self):
        plot_cube_scatter(grid=self.grid,
                          points=np.linalg.norm(self.result, axis=1),
                          save_to=os.path.join(self.path, "wave_modulus.png"),
                          show=False)


if __name__ == "__main__":
    config = {
        "exp_no": "experiment_k_2.5",
        "path": "C:\\Users\\qwert\\PycharmProjects\\SIEpy\\output",
        "amplitude": [0.0, 1.0, 0.0],
        "k_0": 1.25,
        "n": [40, 40, 40],
        "length": [2.0, 2.0, 2.0],
        "center": [0.0, 0.0, 0.0],
        "direction": [1.0, 0.0, 0.0],
        "eps": [
            {
                "type": "step",
                "eps_real": [[1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]],
                "eps_imag": [[0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]],
                "x1_bound": [-5.0, 5.0],
                "x2_bound": [-5.0, 5.0],
                "x3_bound": [-5.0, 5.0]
            },
            {
                "type": "ellipsis",
                "eps_real": [[2.56, 0.0, 0.0],
                             [0.0, 2.56, 0.0],
                             [0.0, 0.0, 2.56]],
                "eps_imag": [[0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]],
                "center": [0.0, 0.0, 0.0],
                "radius": [1.0, 1.0, 1.0]
            }
        ]
    }
    test_obj = Em3dProblem()
    test_obj.fit(fit_config=config)
    print(test_obj.eps_on_grid)
    print(test_obj.distance)
    test_obj.compute_BiCGStab(max_iter=200)
    test_obj.plot_cube()
    rcs, theta = test_obj.compute_RCS(n_phi=300)

    plt.figure(figsize=(9, 6), dpi=100)
    plt.polar(theta, np.abs(rcs)/max(np.abs(rcs)), label="Abs")
    plt.legend()
    plt.grid()
    plt.show()

