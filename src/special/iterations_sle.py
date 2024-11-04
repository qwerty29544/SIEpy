import numpy as np
import numpy.typing as npt
from typing import Iterable, Callable


def BiSGStab(operator_array: npt.NDArray[np.complex128],
             vec_array: npt.NDArray[np.complex128],
             u0_array: npt.NDArray[np.complex128] = None,
             permutation_function: Callable = np.matmul,
             rtol: float = 1e-8,
             max_iter: int = 1000) -> [npt.NDArray[np.complex128], Iterable, Iterable]:
    """

    :param operator_array: известная матрица A оператора размера (N, N), состоящая из комплексных чисел
    :param vec_array: известный вектор f для правой части системы уравнений размера (N,), состоящий из комплексных чисел
    :param u0_array: начальное приближение u0 в качестве начальной точки итерационного метода размера (N,) со значением None по умолчанию. Если передан None, то создавать вектор нужного размера из единиц
    :param permutation_function: функция, реализующая умножение оператора на вектор
    :param rtol: относительная ошибка нормы разницы итераций поделенная на норму вектора f, обозначаемый rtol как критерий остановки алгоритма, по умолчанию равный 10е-8
    :param max_iter: максимальное количество итерационных шагов max_iter как критерий остановки алгоритма, по умолчанию равный 1000
    :return:
    """

