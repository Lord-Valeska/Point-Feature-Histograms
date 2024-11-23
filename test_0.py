#!/usr/bin/env python
import utils
import numpy
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from pfh import PFH, SPFH, FPFH, get_correspondence, get_transform, get_error

if __name__ == '__main__':
    # Load point clouds
    print("Loading point clouds ......")
    pcd_source = o3d.io.read_point_cloud("data/bun000_Structured.pcd")
    print("Loaded")

    source_matrix = np.matrix(pcd_source.points)
    print(source_matrix.shape)
    source_pc = utils.convert_matrix_to_pc(source_matrix.T)
    utils.view_pc([source_pc], None, ['b'], ['o'])
    plt.axis([-0.15, 1000000, -0.15, 1000000, -0.1, 1000])
    plt.show()