#!/usr/bin/env python
import utils
from utils import rotation_matrix
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import time

from pfh import PFH, SPFH, FPFH, get_pfh_correspondence, get_transform, get_error, get_correspondence, \
    get_chamfer_error



if __name__ == '__main__':


  # ----- Initialization ----- #
  # Load the bunny mesh and sample points to create the source point cloud
  print("Loading terrain test case...")

  # Load the PCD file
  pcd = o3d.io.read_point_cloud("data/test_terrain_1.pcd")
  print(pcd.scale)

  # Apply voxel downsampling
  voxel_size = 10  # Adjust this value to control the downsampling level
  downsampled_pcd = pcd.voxel_down_sample(voxel_size)

  # Convert to NumPy array
  points = np.asarray(downsampled_pcd.points)
  print("Total number of points: ", points.shape[0])

  # # Split data into X, Y, Z components
  # x = points[:, 0]  # First column
  # y = points[:, 1]  # Second column
  # z = points[:, 2]  # Third column

  # # Create a 3D scatter plot
  # fig = plt.figure()
  # ax = fig.add_subplot(111, projection='3d')
  # ax.scatter(x, y, z, c='black', marker='o', s=1)

  # # Add labels
  # ax.set_xlabel('X')
  # ax.set_ylabel('Y')
  # ax.set_zlabel('Z')
  # plt.title('3D Scatter Plot')

  # # Show the plot
  # plt.show()

  # Define splitting criteria (e.g., based on X-coordinate)
  x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
  split_boundary = (x_max + x_min) / 2  # Midpoint for splitting
  overlap_margin = 400  # Adjust for overlap size

  # Define conditions for each part
  part1_condition = points[:, 0] <= (split_boundary + overlap_margin)
  part2_condition = points[:, 0] >= (split_boundary - overlap_margin)

  # Extract points for each part
  part1_points = points[part1_condition].T
  part2_points = points[part2_condition].T

  # visualize the separated parts
  part1_points_vis = utils.convert_matrix_to_pc(np.matrix(part1_points))
  part2_points_vis = utils.convert_matrix_to_pc(np.matrix(part2_points))
  utils.view_pc([part1_points_vis, part2_points_vis], None, ['b', 'r'], ['o', '^'])
  plt.show()

  # Calculate centroids for both source and target point clouds
  source_centroid = np.mean(part1_points, axis=1, keepdims=True)  # Shape (3, 1)
  target_centroid = np.mean(part2_points, axis=1, keepdims=True)  # Shape (3, 1)

  # Zero-center the point clouds
  part1_points = part1_points - source_centroid
  part2_points = part2_points - target_centroid

  # Apply a known rotation and translation to create the source point cloud
  print("Creating target point cloud with transformation...")
  angle = np.pi / 6  # 30 degrees rotation
  axis = np.array([0, 1, 0])  # Rotation around the z-axis
  R_true = rotation_matrix(axis, angle)
  t_true = np.array([[600], [100], [10]]) * np.ones(part1_points.shape)  # Translation vector
  part1_points = R_true @ part1_points + t_true
  target_points = part2_points # (3, N)
  source_points = part1_points

  # # Add Gaussian noise to the target point cloud
  # noise_level = 0.001
  # noise = np.random.normal(0, noise_level, target_points.shape)
  # target_points = target_points + noise

  # Convert the point clouds to the required format
  source_pc = utils.convert_matrix_to_pc(np.matrix(source_points))
  target_pc = utils.convert_matrix_to_pc(np.matrix(target_points))

  # Visualize the zero-centered source and target point clouds
  # utils.view_pc([source_pc, target_pc], None, ['b', 'r'], ['o', '^'])
  # plt.show()

  # --- tested --- #


  # ----- PFH ----- #
  target_points = np.asarray(utils.convert_pc_to_matrix(target_pc))
  source_points = np.asarray(utils.convert_pc_to_matrix(source_pc))
  P_source = source_points.T  # Shape (N, 3)
  P_target = target_points.T  # Shape (N, 3)

  # Set parameters for PFH
  threshold = 1e-5 # no
  k = 8            # changeable # TODO
  r = 20           # changeable # TODO

  # Initialize FPFH features for source and target
  pfh_source = FPFH(P_source, r, k, 2, 3, 0) # 2(bin), 0(percentile) changeable # TODO
  pfh_target = FPFH(P_target, r, k, 2, 3, 0) # 2(bin), 0(percentile) changeable # TODO
  # pfh_source = PFH(P_source, r, k, 2, 3, 0) # 2(bin), 0(percentile) changeable # TODO
  # pfh_target = PFH(P_target, r, k, 2, 3, 0) # 2(bin), 0(percentile) changeable # TODO
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
