import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

def discretizez_patrate(x, y):
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

def vizualizare_rezultate(x_vals, y_vals, U_sys):
    X, Y = np.meshgrid(x_vals, y_vals)
    x_fine = np.linspace(a, b, 50)
    y_fine = np.linspace(a, b, 50)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
    U_interp = griddata((X.flatten(), Y.flatten()), U_sys, (X_fine, Y_fine), method='cubic')


    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_fine, Y_fine, U_interp, cmap='coolwarm', edgecolor='none')

    plt.title("Interpolated 3D Surface")
    plt.xlabel("X")
    plt.ylabel("Y")
    ax.set_zlabel("Interpolated Values")
    plt.show()

f = lambda x, y: x*2 + y*2 + 1
g = lambda x, y: x**2 + np.log(y + 1e-10)

n = 3
a = 2
b = 5

x_vals = np.linspace(a, b, n + 1)
y_vals = np.linspace(a, b, n + 1)

puncte = discretizez_patrate( x_vals, y_vals)

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

vizualizare_rezultate(x_vals, y_vals, U_sys)
