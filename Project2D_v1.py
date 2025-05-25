import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve
from mpl_toolkits.mplot3d import Axes3D


def discretizare_patrat(n, x, y):
    patrate = []
    for i in range(n):
        for j in range(n):
            x0, x1 = x[i], x[i + 1]
            y0, y1 = y[j], y[j + 1]
            patrate.append(((x0, y0), (x1, y1)))
    return patrate

def discretizez_patrate(n, x, y):
    X, Y = np.meshgrid(x, y)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    points = np.vstack((X_flat, Y_flat))
    return points

def discretizare_ecuatii(puncte, x, y):
    n = len(x) - 1
    total_puncte = (n + 1)**2
    A = np.zeros((total_puncte, total_puncte))
    h = x[1] - x[0]
    
    for i in range(n + 1):
        for j in range(n + 1):
            index = i * (n + 1) + j
            if i == 0 or i == n or j == 0 or j == n:
                A[index, index] = 1
            else:
                A[index, index] = 4/h**2
                A[index, index - 1] = -1/h**2
                A[index, index + 1] = -1/h**2
                A[index, index - (n + 1)] = -1/h**2
                A[index, index + (n + 1)] = -1/h**2
    return A

def plot_discretizare_patrat(squares, points):
    fig, ax = plt.subplots()
    for (x0, y0), (x1, y1) in squares:
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
    ax.plot(points[0], points[1], 'ro', markersize=4, label="Noduri")
    ax.set_xlim(points[0].min() - 0.1, points[0].max() + 0.1)
    ax.set_ylim(points[1].min() - 0.1, points[1].max() + 0.1)
    ax.set_aspect('equal')
    plt.title("Discretizarea pătratului și nodurile")
    plt.legend()
    plt.grid(True)
    plt.show()

f = lambda x, y: x*2 + y*2 + 1
g = lambda x, y: x**2 + np.log(y + 1e-10)

n = 3
a = 2
b = 5

x_vals = np.linspace(a, b, n + 1)
y_vals = np.linspace(a, b, n + 1)

# Discretizare și vizualizare
patrate = discretizare_patrat(n, x_vals, y_vals)
puncte = discretizez_patrate(n, x_vals, y_vals)
plot_discretizare_patrat(patrate, puncte)

A = discretizare_ecuatii(puncte, x_vals, y_vals)

total_puncte = (n + 1)**2
B = np.zeros(total_puncte)
for i in range(n + 1):
    for j in range(n + 1):
        index = i * (n + 1) + j
        if i == 0 or i == n or j == 0 or j == n:
            B[index] = g(x_vals[i], y_vals[j])
        else:
            B[index] = f(x_vals[i], y_vals[j])

lu, piv = lu_factor(A)
U_sys = lu_solve((lu, piv), B)

print("Matricea sistemului A:")
print(A)

print("\nVectorul termenilor liberi B:")
print(B)

print("\nSoluția sistemului U_sys:")
print(U_sys)


x_grid = puncte[0, :].reshape((n + 1, n + 1))
y_grid = puncte[1, :].reshape((n + 1, n + 1))

# Reshape solution to grid
z_grid = U_sys.reshape((n + 1, n + 1))

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x_grid, y_grid, z_grid, cmap='coolwarm', edgecolor='none')
fig.colorbar(surf)

ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.title("Soluția numerică pe domeniu 2D")

plt.show()