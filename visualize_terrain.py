import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go



# import result_1
# div_list = result_1.div_list
# k_list = result_1.k_list
# percentile_list = result_1.percentile_list
# performance_res = np.array(result_1.performance_res)
# # (len(div), len(percentile_list), [runtime, error]), (n, 5, 2)

# runtime_list = []
# error_list = []
# for i in range(len(div_list)):
#   runtime_list.append(performance_res[i,:,0])
#   error_list.append(performance_res[i,:,1])


# x = np.array(div_list)
# y = np.array(percentile_list)
# z1 = np.array(runtime_list)
# z2 = np.array(error_list)

# # Create a meshgrid for plotting
# X, Y = np.meshgrid(x, y)

# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the first surface
# ax.plot_surface(X, Y, z1.T, color = 'blue', alpha=0.7, label='Z1')


# # Add labels and legend
# ax.set_xlabel('div')
# ax.set_ylabel('percentile')
# ax.set_zlabel('runtime')

# # Show the plot
# plt.title('Runtime influenced by div and percentile')


# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # Plot the second surface
# ax.plot_surface(X, Y, z2.T, color = 'red', alpha=0.7, label='Z2')


# # Add labels and legend
# ax.set_xlabel('div')
# ax.set_ylabel('percentile')
# ax.set_zlabel('error')
# ax.set_zscale('log')
# # ax.set_zlim(10^(-6), 10^(-2))
# ax.set_zlim(1 * 10**-7, 1 * 10**-2.8)

# # Show the plot
# plt.title('Error influenced by div and percentile')
# plt.show()




# import result_1
# div_list = result_1.div_list
# k_list = result_1.k_list
# percentile_list = result_1.percentile_list
# performance_res = np.array(result_1.performance_res)
# # (len(div), len(percentile_list), [runtime, error]), (n, 5, 2)

# runtime_list = []
# error_list = []
# for i in range(len(div_list)):
#   runtime_list.append(performance_res[i,:,0])
#   error_list.append(performance_res[i,:,1])


# x = np.array(k_list)
# y = np.array(percentile_list)
# z1 = np.array(runtime_list)
# z2 = np.array(error_list)

# # Create a meshgrid for plotting
# X, Y = np.meshgrid(x, y)

# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the first surface
# ax.plot_surface(X, Y, z1.T, color = 'blue', alpha=0.7, label='Z1')


# # Add labels and legend
# ax.set_xlabel('k')
# ax.set_ylabel('percentile')
# ax.set_zlabel('runtime')

# # Show the plot
# plt.title('Runtime influenced by k and percentile')


# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # Plot the second surface
# ax.plot_surface(X, Y, z2.T, color = 'red', alpha=0.7, label='Z2')


# # Add labels and legend
# ax.set_xlabel('k')
# ax.set_ylabel('percentile')
# ax.set_zlabel('error')
# ax.set_zscale('log')
# # ax.set_zlim(10^(-6), 10^(-2))
# ax.set_zlim(1 * 10**-7, 1 * 10**-2.8)

# # Show the plot
# plt.title('Error influenced by k and percentile')
# plt.show()





# # (div,k) = (3,3)
# import result_3
# div_list = result_3.div_list
# k_list = result_3.k_list
# percentile_list = result_3.percentile_list
# performance_res = np.array(result_3.performance_res)
# # (len(div_list), len(k_list), [runtime, error]), (3, 3, 2)

# runtime_list = []
# error_list = []
# for i in range(len(div_list)):
#   runtime_list.append(performance_res[i,:,0])
#   error_list.append(performance_res[i,:,1])


# x = np.array(div_list)
# y = np.array(k_list)
# z1 = np.array(runtime_list)
# z2 = np.array(error_list)

# # Create a meshgrid for plotting
# X, Y = np.meshgrid(x, y)

# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the first surface
# ax.plot_surface(X, Y, z1.T, color = 'blue', alpha=0.7, label='Z1')


# # Add labels and legend
# ax.set_xlabel('div')
# ax.set_ylabel('k')
# ax.set_zlabel('runtime')

# # Show the plot
# plt.title('Runtime influenced by div and k')
# # plt.show()


# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # Plot the second surface
# ax.plot_surface(X, Y, z2.T, color = 'red', alpha=0.7, label='Z2')


# # Add labels and legend
# ax.set_xlabel('div')
# ax.set_ylabel('k')
# ax.set_zlabel('error')
# ax.set_zscale('log')
# # ax.set_zlim(10^(-6), 10^(-2))
# ax.set_zlim(1 * 10**-7, 5 * 10**-5)

# # Show the plot
# plt.title('Error influenced by div and k')
# plt.show()





# # smooth result of performance vs div and k
# import result_4
# div_list = result_4.div_list
# k_list = result_4.k_list
# percentile_list = result_4.percentile_list
# performance_res = np.array(result_4.performance_res)
# # (len(div_list), len(k_list), [runtime, error]), (3, 3, 2)

# runtime_list = []
# error_list = []
# for i in range(len(div_list)):
#   runtime_list.append(performance_res[i,:,0])
#   error_list.append(performance_res[i,:,1])


# x = np.array(div_list)
# y = np.array(k_list)
# z1 = np.array(runtime_list)
# z2 = np.array(error_list)

# # Create a meshgrid for plotting
# X, Y = np.meshgrid(x, y)

# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the first surface
# ax.plot_surface(X, Y, z1.T, color = 'blue', alpha=0.7, label='Z1')


# # Add labels and legend
# ax.set_xlabel('div')
# ax.set_ylabel('k')
# ax.set_zlabel('runtime')

# # Show the plot
# plt.title('Runtime influenced by div and k')
# # plt.show()


# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # Plot the second surface
# ax.plot_surface(X, Y, z2.T, color = 'red', alpha=0.7, label='Z2')


# # Add labels and legend
# ax.set_xlabel('div')
# ax.set_ylabel('k')
# ax.set_zlabel('error')
# ax.set_zscale('log')
# # ax.set_zlim(1*10**-50, 1*10**-6)
# # ax.set_zlim(1 * 10**-7, 5 * 10**-5)

# # Show the plot
# plt.title('Error influenced by div and k')
# plt.show()







# # smooth result of performance of PFH vs div and k
# import result_5
# div_list = result_5.div_list
# k_list = result_5.k_list
# percentile_list = result_5.percentile_list
# performance_res = np.array(result_5.performance_res)
# # (len(div_list), len(k_list), [runtime, error]), (3, 3, 2)

# runtime_list = []
# error_list = []
# for i in range(len(div_list)):
#   runtime_list.append(performance_res[i,:,0])
#   error_list.append(performance_res[i,:,1])


# x = np.array(div_list)
# y = np.array(k_list)
# z1 = np.array(runtime_list)
# z2 = np.array(error_list)

# # Create a meshgrid for plotting
# X, Y = np.meshgrid(x, y)

# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the first surface
# ax.plot_surface(X, Y, z1.T, color = 'blue', alpha=0.7, label='Z1')


# # Add labels and legend
# ax.set_xlabel('div')
# ax.set_ylabel('k')
# ax.set_zlabel('runtime')

# # Show the plot
# plt.title('Runtime influenced by div and k')
# # plt.show()


# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # Plot the second surface
# ax.plot_surface(X, Y, z2.T, color = 'red', alpha=0.7, label='Z2')


# # Add labels and legend
# ax.set_xlabel('div')
# ax.set_ylabel('k')
# ax.set_zlabel('error')
# ax.set_zscale('log')
# # ax.set_zlim(1*10**-50, 1*10**-6)
# # ax.set_zlim(1 * 10**-7, 5 * 10**-5)

# # Show the plot
# plt.title('Error influenced by div and k')
# plt.show()








# # smooth result of performance of PFH vs div and k
# import result_6
# div_list = result_6.div_list
# k_list = result_6.k_list
# percentile_list = result_6.percentile_list
# performance_res = np.array(result_6.performance_res)
# # (len(div_list), len(k_list), [runtime, error]), (3, 3, 2)

# runtime_list = []
# error_list = []
# for i in range(len(div_list)):
#   runtime_list.append(performance_res[i,:,0])
#   error_list.append(performance_res[i,:,1])


# x = np.array(div_list)
# y = np.array(k_list)
# z1 = np.array(runtime_list)
# z2 = np.array(error_list)

# # Create a meshgrid for plotting
# X, Y = np.meshgrid(x, y)

# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the first surface
# ax.plot_surface(X, Y, z1.T, color = 'blue', alpha=0.7, label='Z1')


# # Add labels and legend
# ax.set_xlabel('div')
# ax.set_ylabel('k')
# ax.set_zlabel('runtime')

# # Show the plot
# plt.title('Runtime influenced by div and k')
# # plt.show()


# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # Plot the second surface
# ax.plot_surface(X, Y, z2.T, color = 'red', alpha=0.7, label='Z2')


# # Add labels and legend
# ax.set_xlabel('div')
# ax.set_ylabel('k')
# ax.set_zlabel('error')
# ax.set_zscale('log')
# # ax.set_zlim(1*10**-50, 1*10**-6)
# # ax.set_zlim(1 * 10**-7, 5 * 10**-5)

# # Show the plot
# plt.title('Error influenced by div and k')
# plt.show()







# smooth result of performance vs div and k
import result_4, result_6
div_list = result_4.div_list
k_list = result_4.k_list
percentile_list = result_4.percentile_list
performance_res_fpfh = np.array(result_4.performance_res)
performance_res_pfh = np.array(result_6.performance_res)
# (len(div_list), len(k_list), [runtime, error]), (3, 3, 2)


runtime_list_fpfh = []
error_list_fpfh = []
for i in range(len(div_list)):
  runtime_list_fpfh.append(performance_res_fpfh[i,:,0])
  error_list_fpfh.append(performance_res_fpfh[i,:,1])

runtime_list_pfh = []
error_list_pfh = []
for i in range(len(div_list)):
  runtime_list_pfh.append(performance_res_pfh[i,:,0])
  error_list_pfh.append(performance_res_pfh[i,:,1])


x = np.array(div_list)
y = np.array(k_list)
z1 = np.array(runtime_list_fpfh)
z2 = np.array(error_list_fpfh)
z3 = np.array(runtime_list_pfh)
z4 = np.array(error_list_pfh)


# Create a meshgrid for plotting
X, Y = np.meshgrid(x, y)
Z1 = z1.T
Z2 = z2.T
Z3 = z3.T
Z4 = z4.T

print(X.shape)
print(Y.shape)
print(Z1.shape)
print(Z2.shape)
print(Z3.shape)
print(Z4.shape)

# Create a 3D plot
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')

# Plot the first Z-plane with a fixed color
ax1.plot_surface(X, Y, Z1, color='blue', alpha=0.8, label='FPFH')

# Plot the second Z-plane with a fixed color
ax1.plot_surface(X, Y, Z3, color='red', alpha=0.6, label='PFH')

# Add labels
ax1.set_xlabel('div')
ax1.set_ylabel('k')
ax1.set_zlabel('runtime')
ax1.legend()

# Add a title
plt.title('Runtime influenced by div and k')

# Show the plot
# plt.show()



# Create a 3D plot
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')

# Plot the first Z-plane with a fixed color
ax2.plot_surface(X, Y, Z2, color='blue', alpha=0.8, label='FPFH')

# Plot the second Z-plane with a fixed color
ax2.plot_surface(X, Y, Z4, color='red', alpha=0.6, label='PFH')

# Add labels
ax2.set_xlabel('div')
ax2.set_ylabel('k')
ax2.set_zlabel('error')
# ax2.set_zscale('log')
ax2.legend()

# Add a title
plt.title('Eerror influenced by div and k')

plt.show()
