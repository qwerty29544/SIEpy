import numpy as np
import matplotlib.pyplot as plt


def cross(o, a, b):
    """
    Возвращает векторное произведение OA и OB
    Положительное значение означает поворот против часовой стрелки
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def sequential_chain(points):
    """
    Функция вычисляет минимальную выпуклую оболочку для заданного множества точек на плоскости
    методом последовательной цепи.
    Возвращает список точек в порядке обхода оболочки.
    """

    # Сортируем точки лексикографически (по x, затем по y)
    points = sorted(set(points))
    if len(points) == 1:
        return points

    # Строим нижнюю оболочку
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Строим верхнюю оболочку
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Соединяем верхнюю и нижнюю оболочки
    # Последняя точка каждого списка совпадает с первой точкой другого списка, поэтому исключаем их
    convex = lower[:-1] + upper[:-1]
    return convex


def plot_hull(points, hull, **kwargs):
    points_color = "black" if kwargs.get("points_color") is None else kwargs.get("points_color")
    points_alpha = 0.7 if kwargs.get("points_alpha") is None else points_alpha = kwargs.get("points_alpha")
    hull_color = "black" if kwargs.get("hull_color") is None else kwargs.get("hull_color")
    hull_marker = "x" if kwargs.get("hull_marker") is None else kwargs.get("hull_marker")
    hull_label = "Convex hull" if kwargs.get("hull_label") is None else kwargs.get("hull_label")
    points_label = "Set of points" if kwargs.get("points_label") is None else kwargs.get("points_label")
    plt.scatter(points[:, 0],
                points[:, 1],
                color=points_color,
                label=points_label,
                alpha=points_alpha)
    plt.plot(hull[:, 0],
             hull[:, 1],
             color=hull_color,
             linestyle="--",
             label=hull_label)
    plt.scatter(hull[:, 0],
                hull[:, 1],
                color=hull_color,
                marker=hull_marker)
    return 0


def points_to_list(points_np):
    return [(x[0], x[1]) for x in points_np]


class ConvexHull:
    def __init__(self):
        self.points = None
        self.hull2d = None
        self.hull_locked = False

    def fit(self, list_of_points):
        self.points = [(x[0], x[1]) for x in list_of_points]
        self.hull2d = sequential_chain(self.points)

    def to_numpy(self):
        self.points = np.array(self.points)
        self.hull2d = np.array(self.hull2d)

    def to_list(self):
        self.points = [(x[0], x[1]) for x in self.points]
        self.hull2d = [(x[0], x[1]) for x in self.hull2d]

    def hull_locking(self):
        if not self.hull_locked:
            self.to_list()
            self.hull2d.append(self.hull2d[0])
            self.hull_locked = True

    def plot_hull(self, jitter=False, **kwargs):
        # Params
        points_color = "black" if kwargs.get("points_color") is None else kwargs.get("points_color")
        points_alpha = 0.7 if kwargs.get("points_alpha") is None else points_alpha = kwargs.get("points_alpha")
        hull_color = "black" if kwargs.get("hull_color") is None else kwargs.get("hull_color")
        hull_marker = "x" if kwargs.get("hull_marker") is None else kwargs.get("hull_marker")
        plt_figsize = (6, 6) if kwargs.get("plt_figsize") is None else kwargs.get("plt_figsize")
        plt_dpi = 150 if kwargs.get("plt_dpi") is None else kwargs.get("plt_dpi")
        hull_label = "Convex hull" if kwargs.get("hull_label") is None else kwargs.get("hull_label")
        points_label = "Set of points" if kwargs.get("points_label") is None else kwargs.get("points_label")
        # Check if hull locked of not
        if not self.hull_locked:
            self.hull_locking()

        # Force to numpy
        self.to_numpy()

        # Plot somehow
        plt.figure(figsize=plt_figsize,
                   dpi=plt_dpi,
                   frameon=False)
        if jitter:
            max_points = min((500, len(self.points)))
            order = np.random.permutation(max_points)
            plt.scatter(self.points[order, 0],
                        self.points[order, 1],
                        color=points_color,
                        label=points_label,
                        alpha=points_alpha)
        else:
            plt.scatter(self.points[:, 0],
                        self.points[:, 1],
                        color=points_color,
                        label=points_label,
                        alpha=points_alpha)
        plt.plot(self.hull2d[:, 0],
                 self.hull2d[:, 1],
                 color=hull_color,
                 linestyle="--",
                 label=hull_label)
        plt.scatter(self.hull2d[:, 0],
                    self.hull2d[:, 1],
                    color=hull_color,
                    marker=hull_marker)
        plt.legend()
        plt.box(False)
        plt.grid(linewidth=0.5)
        plt.show()

