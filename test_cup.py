#!/usr/bin/env python
import utils
import numpy
import matplotlib.pyplot as plt
import numpy as np

from pfh import PFH, SPFH, FPFH, get_correspondence, get_transform, get_error

if __name__ == '__main__':
    # Load point clouds
    print("Loading point clouds ......")
    pc_source = utils.load_pc('data/cloud_icp_source.csv')
    pc_target = utils.load_pc('data/cloud_icp_target3.csv')
    print("Loaded")

    pc_target = pc_target[150:]
    # pc_target_1 = pc_target[300:]
    # pc_target_2 = pc_target[:150]
    # pc_target = np.concatenate((pc_target_1, pc_target_2))

    # Convert point clouds into matrix
    P = utils.convert_pc_to_matrix(pc_source)
    Q = utils.convert_pc_to_matrix(pc_target)

    # Make a ndarray version of the point clouds
    source = np.asarray(P).T
    target = np.asarray(Q).T

    # Test for kNNs
    k = 8
    r = 0.03
    pfh_source = PFH(source, r, k, 2, 4)
    neighbor_indices = pfh_source.get_kNNs(216)
    neighbors = source[neighbor_indices]
    pc_neighbors = utils.convert_matrix_to_pc(np.matrix(neighbors.T))

    fig = utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['^', 'o'])

    # Test for getting normals
    # normals = pfh_source.get_normals()
    # for i in range(len(normals)):
    #     utils.draw_vector(fig, normals[i].squeeze(), P[:, i], color='y')

    plt.axis([-0.15, 0.15, -0.15, 0.15, -0.1, 0.1])
    plt.show()

    # Test for calculating features for a point
    # features = pfh_source.get_features(216)
    # print(features)

    # Test for getting a histogram for a point
    # histogram = pfh_source.get_feature_histogram(216)
    # bins = np.arange(1, len(histogram) + 1)
    # bar_width = 0.5
    # plt.figure()
    # plt.bar(bins, histogram, width=bar_width, color='red', label='$P_1$', edgecolor='black')
    # plt.xlabel('Bins')
    # plt.ylabel('Ratio of points in one bin (%)')
    # plt.title('Persistent Feature Points Histograms for 3D Geometric Primitives')
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.xticks(bins)
    # plt.show()

    # Test for getting correspondence
    threshold=0.001
    k = 8
    r = 0.03
    pfh_source = FPFH(source, r, k, 3, 3)
    pfh_target = FPFH(target, r, k, 3, 3)

    for i in range(3):
        C = get_correspondence(pfh_source, pfh_target)
        R, t = get_transform(C)
        aligned = pfh_source.transform(R, t)
        error = get_error(C, R, t)
        print(error)
        if error < threshold:
            break
    
    
    pc_aligned = utils.convert_matrix_to_pc(np.matrix(aligned.T))
    utils.view_pc([pc_aligned, pc_target], None, ['b', 'r'], ['o', '^'])
    plt.axis([-0.15, 0.15, -0.15, 0.15, -0.1, 0.1])
    plt.show()

    

    
    
