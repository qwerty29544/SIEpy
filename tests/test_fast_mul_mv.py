import numpy as np
from src.special.fast_mul_mv import fast_tmv_mul, fast_btmv_mul


def test_row():
    def kernel(x):
        return np.exp(-1j * x) / (4 * np.pi * x)

    n = 100
    array_grid = np.linspace(0, 1, n + 1)
    array_grid = (array_grid[1:] + array_grid[:-1]) / 2

    array_coeffs = np.zeros(n) + 0j
    array_coeffs[1:] = kernel(np.abs(array_grid[1:] - array_grid[0]))

    matrix_coeffs = np.zeros((n, n)) + 0.0j
    for row in range(n - 1):
        for col in range(row + 1, n):
            matrix_coeffs[row, col] = kernel(np.abs(array_grid[row] - array_grid[col]))
            matrix_coeffs[col, row] = matrix_coeffs[row, col]

    vec = np.ones((n,))
    multiplication = fast_tmv_mul(array_coeffs, array_coeffs, vec, n)
    mult = matrix_coeffs @ vec
    assert np.prod(np.isclose(mult, multiplication)) > 0


def test_matrix():
    def kernel(x):
        return np.exp(-1j * x)/(4 * np.pi * x)

    n = 8
    m = 12
    x_grid = np.linspace(0, 1, n + 1)
    x_grid = (x_grid[1:] + x_grid[:-1]) / 2
    y_grid = np.linspace(0, 2, m + 1)
    y_grid = (y_grid[1:] + y_grid[:-1]) / 2
    grid = np.array(np.meshgrid(x_grid, y_grid)).T.reshape(-1, 2)

    matrix_toepl = np.zeros((n * m, )) + 0.0j
    matrix_toepl[1:] = kernel(x=np.linalg.norm(grid[0, :] - grid[1:, :], axis=1))

    matrix = np.zeros((n * m, n * m)) + 0.0j
    for row in range(n * m - 1):
        for col in range(row + 1, n * m):
            matrix[row, col] = kernel(x=np.linalg.norm(grid[row, :] - grid[col, :]))
            matrix[col, row] = matrix[row, col]

    x_vec = np.ones((n * m, )) + 1.0j

    result_fourier = fast_btmv_mul(col_arr=matrix_toepl.reshape((n, m)),
                                   row_arr=matrix_toepl.reshape((n, m)),
                                   vec_arr=x_vec.reshape((n, m)),
                                   n=n, m=m)

    result_fourier = result_fourier.reshape((n * m, ))
    result = matrix @ x_vec

    assert np.prod(np.isclose(result, result_fourier)) > 0


class LevelMult:
    def __init__(self):
        test_row()
        test_matrix()


if __name__ == "__main__":
    LevelMult()