#!/usr/bin/env python
import utils
import numpy
import matplotlib.pyplot as plt
import numpy as np

from pfh import PFH

if __name__ == '__main__':
    # Load point clouds
    print("Loading point clouds ......")
    pc_source = utils.load_pc('data/cloud_icp_source.csv')
    pc_target = utils.load_pc('data/cloud_icp_target0.csv')
    print("Loaded")

    # Convert point clouds into matrix
    P = utils.convert_pc_to_matrix(pc_source)
    Q = utils.convert_pc_to_matrix(pc_target)

    # Test for kNNs
    k = 16
    r = 0.03
    pfh_source = PFH(P, r, k)
    neighbors = np.matrix(pfh_source.get_kNNs(P[:, 0].squeeze()).T)
    pc_neighbors = utils.convert_matrix_to_pc(neighbors)
    
    fig = utils.view_pc([pc_source, pc_neighbors], None, ['b', 'r'], ['^', 'o'])

    # Test for getting normals
    normals = pfh_source.get_normals()
    for i in range(len(normals)):
        utils.draw_vector(fig, normals[i].squeeze(), P[:, i], color='y')

    plt.axis([-0.15, 0.15, -0.15, 0.15, -0.1, 0.1])
    plt.show()
    
