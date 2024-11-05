import numpy as np
from volume.dielectric import apply_dielectric_3d
from volume.kernel import helm3d_kernel
from special.fast_mul_mv import prep_fbbtmv, prep_fftbbtensor


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
    dist_coef_1[1:] = 3 / distance[1:] ** 2 - 3j * k / distance[1:] - k ** 2
    alpha = np.zeros((np.prod(discret), 3))
    alpha[1:, :] = (grid[0, :] - grid[1:, :]) / distance[1:].reshape(-1, 1)
    indexing_matrix = np.einsum('...kj,...ki->kij', alpha, alpha)
    part_1 = indexing_matrix * dist_coef_1.reshape(-1, 1, 1)

    dist_coef_2 = np.zeros(np.prod(discret)) + 0.0j
    dist_coef_2[1:] = k ** 2 + 1j * k / distance[1:] - 1 / distance[1:] ** 2
    part_2 = np.eye(3).reshape(-1, 3, 3) * dist_coef_2.reshape(-1, 1, 1)

    gr = np.zeros(np.prod(discret)) + 0.0j
    gr[1:] = kernel(distance[1:], k)
    result = gr.reshape(-1, 1, 1) * (part_1 + part_2) * volume.reshape(-1, 1)
    eps = eps - np.eye(3).reshape(-1, 3, 3)
    result = np.einsum("...ij,...jk->...ik", result, eps)
    result[0, :] = -1/3 * np.eye(3)
    return result


def operator(operator_array, vector_array, n, m, k):
    result = np.zeros((n * m * k, 3)) + 0.0j
    operator_array = operator_array.reshape(-1, 9)
    result[:, 0] += 0
    return vector_array


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

        self.oper_coeffs = None  # Коэффициенты матрицы оператора
        self.prep_coeffs = None  # Подготовленные для быстрого умноженмя коэффициенты
        self.f_values = None     # Значения внешней волны

    def fit(self, fit_config=None):
        if fit_config is None:
            self.config = {
                "exp_no": "exp1",
                "path": "output/",
                "amplitude": [1.0, 0.0, 0.0],
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
        self.direction = self.config.get("direction")

        bounds = np.array([(-np.array(self.length) / 2) + np.array(self.center),
                           (np.array(self.length) / 2) + np.array(self.center)])
        x1_bound = list(bounds[:, 0])
        x2_bound = list(bounds[:, 1])
        x3_bound = list(bounds[:, 2])
        self.grid, self.volume = init_grid(x1_bound=x1_bound,
                                           x2_bound=x2_bound,
                                           x3_bound=x3_bound,
                                           discret=self.n)

        self.eps_on_grid = apply_dielectric_3d(self.grid, self.eps)

        self.distance = np.zeros(np.prod(self.n))
        self.distance[1:] = np.linalg.norm(self.grid[0, :] - self.grid[1:, :], axis=1)

        self.oper_coeffs = init_operator(grid=self.grid,
                                         volume=self.volume,
                                         discret=self.n,
                                         distance=self.distance,
                                         k=self.k_0,
                                         eps=self.eps_on_grid,
                                         kernel=helm3d_kernel).reshape((-1, 9))

        self.prep_coeffs = np.zeros((8*np.prod(self.n), 9)) + 0.0j
        for rowcol in range(9):
            self.prep_coeffs[:, rowcol] = prep_fftbbtensor(col_arr=self.oper_coeffs[:, rowcol],
                                                           row_arr=self.oper_coeffs[:, rowcol],
                                                           n=int(self.n[0]),
                                                           m=int(self.n[1]),
                                                           k=int(self.n[2])).reshape((8*np.prod(self.n),))

        print("fit completed")
        print(f"coeffs shape is: {self.oper_coeffs.shape}")
        return self


if __name__ == "__main__":
    config = {
        "exp_no": "exp1",
        "path": "output/",
        "amplitude": [1.0, 0.0, 0.0],
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
    test_obj = Em3dProblem()
    test_obj.fit(fit_config=config)