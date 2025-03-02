import numpy as np
import matplotlib.pyplot as plt

# Define matrices
A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([[7, 8, 9], [10, 11, 12]])

# Compute matrix multiplication
C = np.dot(A, B)

# Plot matrices
fig, ax = plt.subplots(1, 3, figsize=(12, 4))

# Plot matrix A
ax[0].imshow(A, cmap='Blues')
ax[0].set_title('Matrix A')
ax[0].set_xticks(np.arange(A.shape[1]))
ax[0].set_yticks(np.arange(A.shape[0]))
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        ax[0].text(j, i, A[i, j], ha='center', va='center', color='black')

# Plot matrix B
ax[1].imshow(B, cmap='Blues')
ax[1].set_title('Matrix B')
ax[1].set_xticks(np.arange(B.shape[1]))
ax[1].set_yticks(np.arange(B.shape[0]))
for i in range(B.shape[0]):
    for j in range(B.shape[1]):
        ax[1].text(j, i, B[i, j], ha='center', va='center', color='black')

# Plot matrix C
ax[2].imshow(C, cmap='Blues')
ax[2].set_title('Matrix C = A * B')
ax[2].set_xticks(np.arange(C.shape[1]))
ax[2].set_yticks(np.arange(C.shape[0]))
for i in range(C.shape[0]):
    for j in range(C.shape[1]):
        ax[2].text(j, i, C[i, j], ha='center', va='center', color='black')

plt.tight_layout()
plt.savefig('images/matrix_multiplication.png')
plt.show()
