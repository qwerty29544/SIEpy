import numpy as np
import numpy.typing as npt
import math
import numba as nb


@nb.njit(fastmath=True)
def mu_2points(lambda_1: complex,
               lambda_2: complex) -> complex:
    """
    Finds a center of a circle, around \( \lambda_1 \) and \( \lambda_2 \) complex points

    :param lambda_1:
    :param lambda_2:
    :return mu: complex value of circle center on a complex plane
    """
    a = (lambda_1 + lambda_2) / 2
    numerator = 1j * (lambda_1.imag * np.conj(lambda_2)) * (lambda_2 - lambda_1)
    denominator = 2 * (np.abs(lambda_1 * np.conj(lambda_2)) + np.real(lambda_1 * np.conj(lambda_2)))
    mu = a + numerator / denominator
    return mu


@nb.njit(fastmath=True)
def radius_2points(lambda_1: complex,
                   lambda_2: complex) -> float:
    """

    :param lambda_1:
    :param lambda_2:
    :return:
    """
    numerator = np.abs(lambda_1 - lambda_2) ** 2 * np.abs(np.conj(lambda_1) * lambda_2)
    denominator = 2 * (np.abs(np.conj(lambda_1) * lambda_2) + np.real(np.conj(lambda_1) * lambda_2))
    radius = np.sqrt(np.real(numerator / denominator))
    return radius


@nb.njit(fastmath=True)
def mu_3points(lambda_1: complex,
               lambda_2: complex,
               lambda_3: complex) -> complex:
    """

    :param lambda_1:
    :param lambda_2:
    :param lambda_3:
    :return:
    """
    numerator = (np.abs(lambda_1) ** 2 * (lambda_2 - lambda_3)
                 +
                 np.abs(lambda_2) ** 2 * (lambda_3 - lambda_1)
                 +
                 np.abs(lambda_3) ** 2 * (lambda_1 - lambda_2))
    denominator = 2 * np.imag(lambda_1 * np.conj(lambda_2)
                              +
                              lambda_2 * np.conj(lambda_3)
                              +
                              lambda_3 * np.conj(lambda_1))
    return 1j * numerator / denominator


@nb.njit(fastmath=True)
def radius_3points(mu: complex,
                   lambda_: complex) -> float:
    """

    :param mu:
    :param lambda_:
    :return:
    """
    return abs(mu - lambda_).real


@nb.njit(fastmath=True)
def compute_circle_two_points(lambda_1: complex,
                              lambda_2: complex) -> [complex, float]:
    """

    :param lambda_1:
    :param lambda_2:
    :return:
    """
    center = mu_2points(lambda_1, lambda_2)
    radius = radius_2points(lambda_1, lambda_2)
    return center, radius


@nb.njit(fastmath=True)
def compute_circle_three_points(lambda_1: complex,
                                lambda_2: complex,
                                lambda_3: complex) -> [complex, float]:
    """

    :param lambda_1:
    :param lambda_2:
    :param lambda_3:
    :return:
    """
    x1, y1 = lambda_1.real, lambda_1.imag
    x2, y2 = lambda_2.real, lambda_2.imag
    x3, y3 = lambda_3.real, lambda_3.imag

    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

    if np.abs(d) < 1e-12:
        return 1.0+0.0j, 0.0  # Points are colinear or too close

    center = mu_3points(lambda_1, lambda_2, lambda_3)
    radius = radius_3points(center, lambda_1)
    return center, radius


@nb.jit(fastmath=True)
def circle_contains_points(center: complex,
                           radius: float,
                           points: npt.NDArray[np.complex128],
                           epsilon: float = 1e-8) -> bool:
    """

    :param center:
    :param radius:
    :param points:
    :param epsilon:
    :return:
    """
    return np.prod(np.abs(points - center) <= radius + epsilon) > 0


@nb.njit(fastmath=True)
def circle_contains_origin(center: complex,
                           radius: float,
                           epsilon: float = 1e-8) -> bool:
    """

    :param center:
    :param radius:
    :param epsilon:
    :return:
    """
    return abs(center) <= radius + epsilon


@nb.njit(fastmath=True)
def circle_angle_from_origin(center: complex,
                             radius: float) -> float:
    """

    :param center:
    :param radius:
    :return:
    """
    d = abs(center)
    if d <= radius:
        return np.pi  # Origin is inside the circle
    ratio = radius / d
    ratio = max(-1.0, min(1.0, ratio))  # Clamp to [-1, 1] to avoid math domain error
    theta = 2 * math.asin(ratio)
    return theta


def find_params(points: npt.NDArray[np.complex128]) -> [complex, float]:
    """

    :param points:
    :return:
    """
    # Step 1:
    for point1, point2 in np.array(np.meshgrid(points, points)).T.reshape(-1, 2):
        center, radius = compute_circle_two_points(point1, point2)
        if circle_contains_points(center, radius, points):
            print(f"Circle center: {center}, radius: {radius}")
            return center, radius,

    # Step 2:
    best_theta = None
    best_circle = None
    for point1, point2, point3 in np.array(np.meshgrid(points, points, points)).T.reshape(-1, 3):
        center, radius = compute_circle_three_points(point1, point2, point3)
        if circle_contains_points(center, radius, points) and \
                not circle_contains_origin(center, radius):
            theta = circle_angle_from_origin(center, radius)
            if best_theta is None or theta < best_theta:
                best_theta = theta
                best_circle = (center, radius)

    if best_circle:
        center, radius = best_circle
        print(f"Circle center: {center}, radius: {radius}")
        return center, radius
    else:
        print("No circle found.")
        return 1.0 + 0.0j, 0.0


def __test():
    np.random.seed(1234)
    points_complex = np.random.normal(5, 0.5, 100) + 1j * np.random.normal(5, 0.5, 100)
    mu, radius = find_params(points_complex)
    print(mu, radius)


if __name__ == "__main__":
    __test()
