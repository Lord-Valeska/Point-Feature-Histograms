#!/usr/bin/env python
import utils
import numpy
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from pfh import PFH, SPFH, FPFH, get_correspondence, get_transform, get_error

if __name__ == '__main__':
    # user
    user = input("Enter the user (valeska / chenzj): ")
    print("Welcome, " + user + "!")

    # Load point clouds
    print("Loading point clouds ......")
    source_mesh = o3d.io.read_triangle_mesh("data/bun_zipper.ply")
    print("Loaded")

    source_pcd = source_mesh.sample_points_uniformly(number_of_points=5000)

    source_matrix = np.matrix(source_pcd.points)
    source_pc = utils.convert_matrix_to_pc(source_matrix.T)
    utils.view_pc([source_pc], None, ['b'], ['o'])
    if user == 'chenzj':
        plt.axis([-0.15, 0.15, -0.15, 0.15, -0.15, 0.15])
    else:
        plt.axis([-0.15, 0.15, -0.15, 0.15])
    plt.show()