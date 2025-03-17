import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x, y):
    return (1/20) * x**2 + y**2

# Gradient of the function
def grad_f(x, y):
    df_dx = (1/10) * x
    df_dy = 2 * y
    return np.array([df_dx, df_dy])

# Stochastic Gradient Descent (SGD) parameters
learning_rate = 0.9
num_iterations = 100

# Initial point
x, y = -5.0, 5.0

# Store the path
path = [(x, y)]

# Perform SGD
for _ in range(num_iterations):
    grad = grad_f(x, y)
    x -= learning_rate * grad[0]
    y -= learning_rate * grad[1]
    path.append((x, y))

# Convert path to numpy array for plotting
path = np.array(path)

# Plot the function and the path taken by SGD
X, Y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
Z = f(X, Y)
plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=30, cmap='viridis')
plt.plot(path[:, 0], path[:, 1], 'r.-', markersize=10)  # Increase markersize
plt.xlim(-6, 6)  # Set x-axis limits to zoom in
plt.ylim(-6, 6)  # Set y-axis limits to zoom in
plt.title('SGD Path on f(x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('images/sgd_path.png')  # Save the plot to a file

