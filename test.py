from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X0, X1, Z, cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
ax.scatter(points[:, 0], points[:, 1], [f(p[0], p[1]) for p in points], color='red', s=50, marker='o')
ax.set_title('3D Surface Plot of f(x0, x1) = x0^2 + x1^2')
ax.set_xlabel('x0')
ax.set_ylabel('x1')
ax.set_zlabel('f(x0, x1)')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='f(x0, x1) value')

# Save the 3D plot to the images directory
plt.savefig('images/gradient_descent_3d.png', dpi=300)
