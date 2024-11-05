import numpy as np
import numpy.typing as npt


def fast_tmv_mul(col_arr: npt.NDArray[np.complex128],
                 row_arr: npt.NDArray[np.complex128],
                 vec_arr: npt.NDArray[np.complex128],
                 n: int) -> npt.NDArray[np.complex128]:
    """
    One-level fast toeplitz matrix to vector multiplication

    :param col_arr:
    :param row_arr:
    :param vec_arr:
    :param n:
    :return:
    """
    circulant_row = np.zeros((2 * n,)) + 0.0j
    x_arr = np.zeros((2 * n,)) + 0.0j

    circulant_row[:n] = row_arr[:n]
    circulant_row[(n + 1):(2 * n)] = col_arr[n:0:-1]

    x_arr[:n] = vec_arr[:n]

    circulant_row = np.fft.fft(circulant_row) * np.fft.fft(x_arr)
    circulant_row = np.fft.ifft(circulant_row)[:n]

    return circulant_row


def fast_btmv_mul(col_arr: npt.NDArray[np.complex128],
                  row_arr: npt.NDArray[np.complex128],
                  vec_arr: npt.NDArray[np.complex128],
                  n: int, m: int) -> npt.NDArray[np.complex128]:
    """

    :param col_arr:
    :param row_arr:
    :param vec_arr:
    :param n:
    :param m:
    :return:
    """
    circulant_matrix = np.zeros((2 * n, 2 * m)) + 0.0j
    x_arr = np.zeros((2 * n, 2 * m)) + 0.0j

    circulant_matrix[:n, :m] = row_arr[:n, :m]
    circulant_matrix[n + 1:2 * n, :m] = col_arr[n:0:-1, :m]
    circulant_matrix[:n, m + 1:2 * m] = col_arr[:n, m:0:-1]
    circulant_matrix[n + 1:2 * n, m + 1:2 * m] = col_arr[n:0:-1, m:0:-1]

    x_arr[:n, :m] = vec_arr[:n, :m]

    circulant_matrix = np.fft.fft2(circulant_matrix) * np.fft.fft2(x_arr)
    circulant_matrix = np.fft.ifft2(circulant_matrix)

    return circulant_matrix[:n, :m]


def fast_bbtmv_mul(col_arr: npt.NDArray[np.complex128],
                   row_arr: npt.NDArray[np.complex128],
                   vec_arr: npt.NDArray[np.complex128],
                   n: int, m: int, k: int) -> npt.NDArray[np.complex128]:
    """

    :param col_arr:
    :param row_arr:
    :param vec_arr:
    :param n:
    :param m:
    :param k:
    :return:
    """
    circulant_tensor = np.zeros((2 * n, 2 * m, 2 * k)) + 0.0j
    x_arr = np.zeros((2 * n, 2 * m, 2 * k)) + 0.0j

    circulant_tensor[:n, :m, :k] = row_arr[:n, :m, :k]
    circulant_tensor[n + 1:2 * n, :m, :k] = col_arr[n:0:-1, :m, :k]
    circulant_tensor[:n, m + 1:2 * m, :k] = col_arr[:n, m:0:-1, :k]
    circulant_tensor[:n, :m, k + 1: 2 * k] = col_arr[:n, :m, k:0:-1]
    circulant_tensor[n + 1:2 * n, m + 1:2 * m, :k] = col_arr[n:0:-1, m:0:-1, :k]
    circulant_tensor[:n, m + 1:2 * m, k + 1:2 * k] = col_arr[:n, m:0:-1, k:0:-1]
    circulant_tensor[n + 1:2 * n, :m, k + 1:2 * k] = col_arr[n:0:-1, :m, k:0:-1]
    circulant_tensor[n + 1:2 * n, m + 1:2 * m, k + 1:2 * k] = col_arr[n:0:-1, m:0:-1, k:0:-1]

    x_arr[:n, :m, :k] = vec_arr[:n, :m, :k]

    circulant_tensor = np.fft.fftn(circulant_tensor) * np.fft.fftn(x_arr)
    circulant_tensor = np.fft.ifft(circulant_tensor)

    return circulant_tensor[:n, :m, :k]


def prep_fftbbtensor(col_arr: npt.NDArray[np.complex128],
                     row_arr: npt.NDArray[np.complex128],
                     n: int, m: int, k: int) -> npt.NDArray[np.complex128]:
    """

    """
    circulant_tensor = np.zeros((2 * n, 2 * m, 2 * k)) + 0.0j
    col_arr = col_arr.reshape((n, m, k))
    row_arr = row_arr.reshape((n, m, k))
    circulant_tensor[:n, :m, :k] = row_arr[:n, :m, :k]
    circulant_tensor[n + 1:2 * n, :m, :k] = col_arr[n:0:-1, :m, :k]
    circulant_tensor[:n, m + 1:2 * m, :k] = col_arr[:n, m:0:-1, :k]
    circulant_tensor[:n, :m, k + 1: 2 * k] = col_arr[:n, :m, k:0:-1]
    circulant_tensor[n + 1:2 * n, m + 1:2 * m, :k] = col_arr[n:0:-1, m:0:-1, :k]
    circulant_tensor[:n, m + 1:2 * m, k + 1:2 * k] = col_arr[:n, m:0:-1, k:0:-1]
    circulant_tensor[n + 1:2 * n, :m, k + 1:2 * k] = col_arr[n:0:-1, :m, k:0:-1]
    circulant_tensor[n + 1:2 * n, m + 1:2 * m, k + 1:2 * k] = col_arr[n:0:-1, m:0:-1, k:0:-1]
    circulant_tensor = np.fft.fftn(circulant_tensor)
    return circulant_tensor


def prep_fbbtmv(prep_fft_arr: npt.NDArray[np.complex128],
                vec_arr: npt.NDArray[np.complex128],
                n: int, m: int, k: int) -> npt.NDArray[np.complex128]:
    """

    """
    x_arr = np.zeros((2 * n, 2 * m, 2 * k)) + 0.0j
    x_arr[:n, :m, :k] = vec_arr[:n, :m, :k]
    prep_fft_arr = np.fft.ifftn(prep_fft_arr * np.fft.fftn(x_arr))
    return prep_fft_arr[:n, :m, :k]
