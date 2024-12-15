import numpy as np

# Highlight a specific point
point_x = np.pi
point_y = np.sin(point_x)

# Create the plot
plt.plot(x, y, label='sin(x)')
plt.scatter([point_x], [point_y], color='red')  # Mark the point
plt.annotate('Maximum Point', xy=(point_x, point_y), xytext=(point_x + 1, point_y - 0.5),
             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=12, color='green')

# Customize the plot
plt.title('Graph with Annotation')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

# Show the plot
plt.show()
