#!/usr/bin/env python
import utils
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import time

from pfh import PFH, SPFH, FPFH, get_pfh_correspondence, get_transform, get_error, get_correspondence, \
    get_chamfer_error

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix for rotation about the given axis by theta radians.
    """
    axis = axis / np.linalg.norm(axis)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    u_x, u_y, u_z = axis
    R = np.array([
        [cos_theta + u_x**2*(1 - cos_theta),       u_x*u_y*(1 - cos_theta) - u_z*sin_theta, u_x*u_z*(1 - cos_theta) + u_y*sin_theta],
        [u_y*u_x*(1 - cos_theta) + u_z*sin_theta,  cos_theta + u_y**2*(1 - cos_theta),      u_y*u_z*(1 - cos_theta) - u_x*sin_theta],
        [u_z*u_x*(1 - cos_theta) - u_y*sin_theta,  u_z*u_y*(1 - cos_theta) + u_x*sin_theta, cos_theta + u_z**2*(1 - cos_theta)]
    ])
    return R

if __name__ == '__main__':
    # user
    user = input("Enter the user (valeska / chenzj): ")
    print("Welcome, " + user + "!")

    # Load the bunny mesh and sample points to create the source point cloud
    print("Loading bunny mesh and sampling points...")
    source_mesh = o3d.io.read_triangle_mesh("data/bun_zipper.ply")
    source_pcd = source_mesh.sample_points_uniformly(number_of_points=1000)
    source_points = np.asarray(source_pcd.points).T  # shape (3, N)

    # Apply a known rotation and translation to create the target point cloud
    print("Creating target point cloud with transformation...")
    angle = np.pi / 3  # 30 degrees rotation
    axis = np.array([0, 1, 0])  # Rotation around the z-axis
    R_true = rotation_matrix(axis, angle)
    t_true = np.array([[0.05], [0.02], [-0.02]])  # Translation vector
    target_points = R_true @ source_points + t_true

    # Add Gaussian noise to the target point cloud
    noise_level = 0.001
    noise = np.random.normal(0, noise_level, target_points.shape)
    target_points = target_points + noise

    # Convert the point clouds to the required format
    source_pc = utils.convert_matrix_to_pc(np.matrix(source_points))
    target_pc = utils.convert_matrix_to_pc(np.matrix(target_points))
    
    # Test for missing points
    # target_pc = target_pc[:700]
    # source_pc = source_pc[300:]

    # Visualize the source and target point clouds
    utils.view_pc([source_pc, target_pc], None, ['b', 'r'], ['o', '^'])
    if user == 'chenzj':
        plt.axis([-0.15, 0.15, -0.15, 0.15, -0.15, 0.15])
    else:
        plt.axis([-0.15, 0.15, -0.15, 0.15])
    plt.show()

    # Prepare the point clouds for ICP
    target_points = np.asarray(utils.convert_pc_to_matrix(target_pc))
    source_points = np.asarray(utils.convert_pc_to_matrix(source_pc))
    P_source = source_points.T  # Shape (N, 3)
    P_target = target_points.T  # Shape (N, 3)

    # Set parameters for ICP
    threshold = 1e-5
    k = 8
    r = 0.03

    # Initialize FPFH features for source and target
    pfh_source = FPFH(P_source, r, k, 2, 3, 0)
    pfh_target = FPFH(P_target, r, k, 2, 3, 0)
    errors = []

    # Initial transform
    current = time.time()
    C = get_pfh_correspondence(pfh_source, pfh_target)
    R, t = get_transform(C)
    aligned = pfh_source.transform(R, t)
    end = time.time()
    print(f"Iteration time: {end - current}")
    error = get_error(C, R, t)  # Compute alignment error
    errors.append(error)
    pc_aligned = utils.convert_matrix_to_pc(np.matrix(aligned.T))
    utils.view_pc([pc_aligned, target_pc], None, ['b', 'r'], ['o', '^'])
    if user == 'chenzj':
        plt.axis([-0.15, 0.15, -0.15, 0.15, -0.15, 0.15])
    else:
        plt.axis([-0.15, 0.15, -0.15, 0.15])
    plt.show()
    
    # Perform ICP iterations
    for i in range(50):
        current = time.time()
        C = get_correspondence(pfh_source, pfh_target)  # Find correspondences
        R_est, t_est = get_transform(C)  # Estimate transformation
        aligned = pfh_source.transform(R_est, t_est)  # Apply transformation
        end = time.time()
        print(f"Iteration {i+1}, time: {end - current:.4f}s")
        error = get_error(C, R, t)  # Compute alignment error
        errors.append(error)
        print(f"Alignment error: {error:.6f}")
        if len(errors) > 1:
            relative_change = abs(errors[-1] - errors[-2]) / errors[-2]
            if relative_change < threshold or error < threshold:
                print(f"Converged at iteration {i} with relative change {relative_change:.6f}")
                break

    # Convert the aligned point cloud to the required format
    pc_aligned = utils.convert_matrix_to_pc(np.matrix(aligned.T))

    # Visualize the aligned source point cloud and the target point cloud
    utils.view_pc([pc_aligned, target_pc], None, ['b', 'r'], ['o', '^'])
    if user == 'chenzj':
        plt.axis([-0.15, 0.15, -0.15, 0.15, -0.15, 0.15])
    else:
        plt.axis([-0.15, 0.15, -0.15, 0.15])
        
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(errors)), errors, marker='o', linestyle='-', color='blue')
    plt.title("Error vs. Iteration", fontsize=16)
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Error", fontsize=14)
    plt.grid(alpha=0.3)
    
    plt.show()

    # Compare the estimated transformation with the true transformation
    print("True rotation matrix (R_true):")
    print(R_true)
    print("\nEstimated rotation matrix (R_est):")
    print(R_est)
    print("\nTrue translation vector (t_true):")
    print(t_true.flatten())
    print("\nEstimated translation vector (t_est):")
    print(t_est.flatten())
