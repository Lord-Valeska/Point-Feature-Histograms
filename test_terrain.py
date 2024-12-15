#!/usr/bin/env python
import utils
from utils import rotation_matrix
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import time

from pfh import PFH, SPFH, FPFH, get_pfh_correspondence, get_transform, get_error, get_correspondence, \
    get_chamfer_error

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def test_terrain(div, k, percentile):
  # div = int(input("div: "))
  # k = int(input("k: "))
  # percentile = int(input("percentile: "))

  # ----- Initialization ----- #
  # Load the bunny mesh and sample points to create the source point cloud
  print(f"Loading terrain test case with div = {div}, k = {k}, percentile = {percentile}")

  # Load the PCD file
  pcd = o3d.io.read_point_cloud("data/test_terrain_1.pcd")

  # Apply voxel downsampling
  voxel_size = 20  # Adjust this value to control the downsampling level
  downsampled_pcd = pcd.voxel_down_sample(voxel_size)

  # Convert to NumPy array
  points = np.asarray(downsampled_pcd.points)
  points *= 0.0001 # Scale
  point_centroid = np.mean(points, axis=0, keepdims=True)
  points = points - point_centroid
  # print("Total number of points: ", points.shape[0])

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
  overlap_margin = (x_max - x_min) * 0.5  # Adjust for overlap size

  # Define conditions for each part
  part1_condition = (
    (points[:, 0] >= (split_boundary - (x_max - x_min) * 0.3)) &
    (points[:, 0] <= (split_boundary + (x_max - x_min) * 0.3))
  )
  part2_condition = points[:, 0] <= (split_boundary + overlap_margin)
  
  source_idx = np.where(np.array(part1_condition))[0]
  # print(source_idx)

  # Extract points for each part
  part1_points = points[part1_condition].T
  part2_points = points[part2_condition].T

  # visualize the separated parts
  part1_points_vis = utils.convert_matrix_to_pc(np.matrix(part1_points))
  part2_points_vis = utils.convert_matrix_to_pc(np.matrix(part2_points))
  utils.view_pc([part1_points_vis, part2_points_vis], "", None, ['b', 'r'], ['o', '^'])
  # plt.show()
  plt.title("Preprocessing the terrain point cloud")
  plt.draw()  # Update the figure
  plt.pause(3)  # Pause for 5 seconds
  plt.close() 

  # Apply a known rotation and translation to create the source point cloud
  # print("Creating target point cloud with transformation...")
  angle = np.pi / 3  # 60 degrees rotation
  axis = np.array([0, 0, 1])  # Rotation around the z-axis
  R_true = rotation_matrix(axis, angle)
  t_true = np.array([[-0.05], [0.02], [0.003]]).reshape(3, 1)
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
  utils.view_pc([source_pc, target_pc], "", None, ['b', 'r'], ['o', '^'])
  plt.title("Starting pose of source and target. Calculating...")
  plt.draw()  # Update the figure
  plt.pause(0.1)  # Pause for 5 seconds
  plt.close() 

  # ----- PFH ----- #
  target_points = np.asarray(utils.convert_pc_to_matrix(target_pc))
  source_points = np.asarray(utils.convert_pc_to_matrix(source_pc))
  P_source = source_points.T  # Shape (N, 3)
  P_target = target_points.T  # Shape (N, 3)

  # Set parameters for PFH
  threshold = 1e-10 # no
  # k = 8            # changeable # TODO
  r = 0.5
  # div = 2          # changeable # TODO

  # Initialize FPFH features for source and target
  pfh_source = FPFH(P_source, r, k, div, 3, percentile) # 2(bin), 0(percentile) changeable # TODO
  pfh_target = FPFH(P_target, r, k, div, 3, percentile) # 2(bin), 0(percentile) changeable # TODO
  # pfh_source = PFH(P_source, r, k, div, 3, percentile) # 2(bin), 0(percentile) changeable # TODO
  # pfh_target = PFH(P_target, r, k, div, 3, percentile) # 2(bin), 0(percentile) changeable # TODO
  errors = []

  # Initial transform
  start = time.time()
  current = start
  C = get_pfh_correspondence(pfh_source, pfh_target)
  R, t = get_transform(C)
  aligned = pfh_source.transform(R, t)
  end = time.time()
  print(f"Iteration time: {(end - start):.4f}")
  error = get_error(C, R, t)  # Compute alignment error
  errors.append(error)
  pc_aligned = utils.convert_matrix_to_pc(np.matrix(aligned.T))
  total_error = np.mean((aligned-target_points.T[source_idx,:])**2)
  utils.view_pc([pc_aligned, target_pc], f"Total error = {total_error}", None, ['b', 'r'], ['o', '^'])
  plt.title("Initial transform by FPFH")
  # plt.show()
  plt.draw()  # Update the figure
  plt.pause(3)  # Pause for 5 seconds
  plt.close() 

  # Perform ICP iterations
  for i in range(5):
    current = time.time()
    C = get_correspondence(pfh_source, pfh_target)  # Find correspondences
    R_est, t_est = get_transform(C)  # Estimate transformation
    aligned = pfh_source.transform(R_est, t_est)  # Apply transformation
    # end = time.time()
    # print(f"Iteration {i+1}, time: {end - current:.4f}s")
    error = get_error(C, R, t)  # Compute alignment error
    errors.append(error)
    # Convert the aligned point cloud to the required format
    pc_aligned = utils.convert_matrix_to_pc(np.matrix(aligned.T))
    # Total error
    total_error = np.mean((aligned-target_points.T[source_idx,:])**2)
    # Visualize the aligned source point cloud and the target point cloud
    utils.view_pc([pc_aligned, target_pc], f"Total error = {total_error}", None, ['b', 'r'], ['o', '^'])
    plt.title(f"The {i+1}-th iteration of ICP")
    # Add text to the graph
    # plt.text(5, 0.5, 1, f"Total error = {total_error}", fontsize=12, color='blue')  # Place text at (5, 0.5)
    plt.draw()  # Update the figure
    plt.pause(0.5)  # Pause for 5 seconds
    plt.close()  # Close the current figure to prepare for the next
    # print(f"Alignment error: {error:.6f}")
    if len(errors) > 1:
      relative_change = abs(errors[-1] - errors[-2]) / errors[-2]
      if relative_change < threshold or error < threshold:
        # print(f"Converged at iteration {i} with relative change {relative_change:.6f}")
        # print(f"Final total error: {error:.8f}")
        break

  # Convert the aligned point cloud to the required format
  pc_aligned = utils.convert_matrix_to_pc(np.matrix(aligned.T))
  end = time.time()
  total_runtime = end - start
  print(f"Total runtime is {total_runtime:.4f}")

  # Total error
  total_error = np.mean((aligned-target_points.T[source_idx,:])**2)
  print(f"Final total error: {total_error}")

  utils.view_pc([pc_aligned, target_pc], f"Total error = {total_error}", None, ['b', 'r'], ['o', '^'])
  plt.title("Final result of alignment using FPFH-ICP algorithm")
  # plt.figure(figsize=(8, 6))
  # plt.plot(range(len(errors)), errors, marker='o', linestyle='-', color='blue')
  # plt.title("Error vs. Iteration", fontsize=16)
  # plt.xlabel("Iteration", fontsize=14)
  # plt.ylabel("Error", fontsize=14)
  # plt.grid(alpha=0.3)
  # plt.show()

  plt.draw()  # Update the figure
  plt.pause(5)  # Pause for 5 seconds
  plt.close()  # Close the current figure to prepare for the next

  return [total_runtime, total_error]


  # Update function for animation
  def update(frame):
      line.set_data(x, y)  # Update X and Y data
      line.set_3d_properties(z_data[frame])  # Update Z data
      return line,

if __name__ == '__main__':
  div_list = [2, 3, 5]
  k_list = [8, 9, 10, 12]
  percentile_list = [0, 25, 50, 75, 100]
  with open('result_6.py', 'w') as file:
    file.write(f"# result of runtime and error in terms of div and k of FPFH\n")
    file.write(f"import numpy as np\n\n")

    file.write(f"div_list = {div_list}\n")
    file.write(f"k_list = {k_list}\n")
    file.write(f"percentile_list = {percentile_list}\n")
    file.write(f"performance_res = []\n")

    for i in range(len(div_list)):
      result = []
      for j in range(len(k_list)):
        div = div_list[i]
        k = k_list[j]
        percentile = percentile_list[0]
        [runtime, error] = test_terrain(div, k, percentile)
        result.append([runtime, error])
      file.write(f"performance_res.append({result})\n")

# if __name__ == '__main__':
#   [_, _] = test_terrain(4, 10, 0)
#   [_, _] = test_terrain(3, 8, 0)