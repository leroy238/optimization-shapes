import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

p = 3
N = 100_000

a = None
def init_dataset():
    global a
    path = os.path.join(os.getcwd(), "dataset.pkl")
    with open(path, "rb") as f:
        a = pickle.load(f)
    #end with
#end init_dataset

# Load pickled error data
with open("error_sgd.pkl", "rb") as f:
    error_sgd = pickle.load(f)

with open("error_adam.pkl", "rb") as f:
    error_adam = pickle.load(f)

# Plotting the errors
plt.figure(figsize=(10, 6))
plt.plot(error_sgd, label="SGD", color="blue", linewidth=2)
plt.plot(error_adam, label="Adam", color="green", linewidth=2)

# Chart details
plt.title("Training Error over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig("errors.png")
plt.close()

def f(x, c, p):
    x_p = x ** p
    return 1/(2 * N) * np.sum((c * np.prod(x_p) - 1)**2)
#end for

# Load pickled data
with open("xs_sgd.pkl", "rb") as f_sgd:
    xs_sgd = pickle.load(f_sgd)

with open("xs_adam.pkl", "rb") as f_adam:
    xs_adam = pickle.load(f_adam)

#print(xs_sgd)
#print(xs_adam)

# Create meshgrid for contours
x_vals = np.linspace(0, 3.5, 100)
y_vals = np.linspace(0, 3.5, 100)
X, Y = np.meshgrid(x_vals, y_vals)
grid_points = np.stack([X, Y], axis=-1)  # Shape (100, 100, 2)

# Define 'c' parameter for f(x, c, 3)
init_dataset()
c = a
Z = np.array([[f(grid_points[i,j], c, 3) for j in range(100)] for i in range(100)])
print(Z)

# Start plotting
plt.figure(figsize=(8, 6))

# Plot contours
plt.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)

# Plot paths from SGD
colors = np.linspace(0.1, 1.0, len(xs_sgd))
xs_sgd = np.stack(xs_sgd, axis = 0)
xs_adam = np.stack(xs_adam, axis = 0)
#for i in range(xs_sgd.shape[0]):
#    plt.scatter(xs_sgd[i:i+1, 0], xs_sgd[i:i+1, 1], label='SGD', color=plt.get_cmap("Reds")(colors[i]), alpha=0.5)

# Plot paths from Adam
for i in range(xs_adam.shape[0]):
    plt.scatter(xs_adam[i:i+1, 0], xs_adam[i:i+1, 1], label='Adam', color=plt.get_cmap("Blues")(colors[i]), alpha=0.5)

# Optional: Add legend only once per optimizer
#plt.plot([], [], color='red', label='SGD')
#plt.plot([], [], color='blue', label='Adam')
#plt.legend()

plt.title("Optimizer Paths and Function Contours")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)

# Save to file
plt.savefig("xs_adam_flat.png")
plt.close()
