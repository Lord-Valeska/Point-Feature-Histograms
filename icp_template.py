#!/usr/bin/env python
import utils
import numpy
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
import numpy as np
###YOUR IMPORTS HERE###

def getTransform(C):
    # Step 0
    Cp = C[:, 0, :] # (495, 3)
    Cq = C[:, 1, :] # (495, 3)
    # Step 1
    p_bar = np.mean(Cp, axis=0)
    q_bar = np.mean(Cq, axis=0)
    X = Cp - p_bar
    Y = Cq - q_bar
    # Step 2
    S = X.T @ Y
    U, sigma, V_T = np.linalg.svd(S)
    # Step 3
    M = np.eye(3)
    M[2, 2] = np.linalg.det(V_T.T @ U.T)
    R = V_T.T @ M @ U.T
    t = q_bar.reshape(3, 1) - R @ p_bar.reshape(3, 1)
    return R, t

def getError(C, R, t):
    Cp = C[:, 0, :] # (495, 3)
    Cq = C[:, 1, :] # (495, 3)
    errors = (((R @ Cp.T).T + t.reshape(1, 3)) - Cq) ** 2
    return np.sum(errors)

def main(threshold=0.016, target_name='./data/cloud_icp_target0.csv'):
    #Import the cloud
    pc_source = utils.load_pc('./data/cloud_icp_source.csv')

    ###YOUR CODE HERE###
    pc_target = utils.load_pc(target_name) # Change this to load in a different target

    # Number of pointclouds
    num_source = len(pc_source)
    num_target = len(pc_target)
    P = utils.convert_pc_to_matrix(pc_source)
    Q = utils.convert_pc_to_matrix(pc_target)
    print(f"Source: {num_source}, Shape: {P.shape}") # 495, (3, 495)
    print(f"Target: {num_target}, Shape: {Q.shape}") # 495, (3, 495)
    error_list = []

    while True:
        C = []
        for i in range(num_source):
            p = P[:, i]
            distances = np.linalg.norm(Q - p, axis=0)
            q = Q[:, np.argmin(distances)]
            C.append([p, q])
        C = np.asarray(C).squeeze() # (495, 2, 3)
        R, t = getTransform(C)
        error = getError(C, R, t)
        error_list.append(error)
        # print(error)
        if error < threshold:
            break
        P = R @ P + t
    
    pc_aligned = utils.convert_matrix_to_pc(P)
    # print(pc_source)
    plt.plot(error_list)
    plt.title("Error vs Iteration")
    utils.view_pc([pc_aligned, pc_target], None, ['b', 'r'], ['o', '^'])
    plt.axis([-0.15, 0.15, -0.15, 0.15, -0.15, 0.15])
    ###YOUR CODE HERE###

    plt.show()
    #raw_input("Press enter to end:")


if __name__ == '__main__':
    # main(threshold=0.016, target_name='./data/cloud_icp_target0.csv')
    # main(threshold=0.00001, target_name='./data/cloud_icp_target1.csv')
    # main(threshold=0.00808555, target_name='./data/cloud_icp_target2.csv')
    main(threshold=0.04186485, target_name='./data/cloud_icp_target3.csv')

