import numpy as np
import matplotlib.pyplot as plt
import os


def plot_cube_scatter(grid, points, save_to=None, show=True):
    if save_to is None:
        save_to = "output\\"

    fig = plt.figure(figsize=(9, 9), dpi=256)
    ax = fig.add_subplot(projection='3d')

    ax.scatter(grid[:, 0], grid[:, 1], grid[:, 2], c=points)
    ax.set_xlabel(fr"$X_1$ axis", fontsize=16)
    ax.set_ylabel(fr"$X_2$ axis", fontsize=16)
    ax.set_zlabel(fr"$X_3$ axis", fontsize=16)

    plt.savefig(save_to)
    if show:
        plt.show()

def plot_plane_vector(grid, vectors, save_to=None, show=True):
    if save_to is None:
        save_to = "output\\"


    return None