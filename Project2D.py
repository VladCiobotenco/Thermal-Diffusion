import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


def discretizare_domeniu(n, x, y):
    X, Y = np.meshgrid(x, y)  # Creează o rețea (meshgrid) de puncte în planul (x, y)
    # Aplatizează matricile în vectori simpli (1D)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    points = np.vstack((X_flat, Y_flat)) # Combină vectorii într-o matrice 2xN, unde fiecare coloană este un punct (x_i, y_i)
    return points

def discretizare_ecuatii(n, x, y):
    total_puncte = (n + 1)**2  # Numărul total de noduri din rețea
    A = np.zeros((total_puncte, total_puncte)) # Inițializează matricea de sistem (coeficienți) cu zerouri
    
    h_x = x[1] - x[0]  # Pasul pe axa x
    h_y = y[1] - y[0]  # Pasul pe axa y
      # Parcurge toate nodurile interioare și de pe frontieră
    for i in range(n + 1):
        for j in range(n + 1):
            index = i * (n + 1) + j #Calculăm indexul punctului curent în vectorul 1D
            if i == 0 or j == 0 or (i == n and j == n): # Aici setam caldura pentru Frotierele 3 si 4
                A[index, index] = 1
            else:
                if i == n and j > 0 and j < n: # Setam fluxul de caldura pentru Frontiera 1
                    A[index,index] = 3/(2*h_x) # Coeficientul pentru punctul u_{i,j}
                    A[index,index-1] = -2/h_x # Coeficientul pentru punctul central u_{i,j-1}
                    A[index,index-2] = 1/(2*h_x) # Coeficientul pentru punctul central u_{i,j-2}
                else:
                    if i > 0 and i < n and j == n: # Setam fluxul de caldura pentru Frontiera 2
                        A[index,index] = 3/(2*h_y) # Coeficientul pentru punctul u_{i,j}
                        A[index,index - (n + 1)] = -2/h_y # Coeficientul pentru punctul central u_{i-1,j}
                        A[index,index - 2*(n + 1)] = 1/(2*h_y) # Coeficientul pentru punctul central u_{i-2,j}
                    else: # Setam Caldura pentru punctele din interior 
                        A[index, index] = 2*(1+x[i])/h_x**2 + 4*np.log(y[j])/h_y**2 # Coeficientul pentru punctul central u_{i,j}
                        A[index, index - 1] = 1/2*h_x - (1+x[i])/h_x**2 # Coeficientul pentru punctul central u_{i,j-1}
                        A[index, index + 1] = 1/2*h_x - (1+x[i])/h_x**2 # Coeficientul pentru punctul central u_{i,j+1}
                        A[index, index - (n + 1)] = 1/(y[j]*h_y) - 2*np.log(y[j])/h_y**2 # Coeficientul pentru punctul central u_{i-1,j}
                        A[index, index + (n + 1)] = 1/(y[j]*h_y) - 2*np.log(y[j])/h_y**2 # Coeficientul pentru punctul central u_{i+1,j}
                
    return A # Returnează matricea sistemului de ecuații
    



def vizualizare_rezultate(x_vals, y_vals, U_sys):
    X, Y = np.meshgrid(x_vals, y_vals)
    #U_grid = U_sys.reshape((n + 1, n + 1))  # Reshaping solution into grid format

    # Define finer grid for interpolation
    x_fine = np.linspace(a, b, 50)
    y_fine = np.linspace(a, b, 50)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

    # Interpolate using Scipy's griddata
    U_interp = griddata((X.flatten(), Y.flatten()), U_sys, (X_fine, Y_fine), method='cubic')

    # 3D Plot of interpolated results
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_fine, Y_fine, U_interp, cmap='coolwarm', edgecolor='none')

    plt.title("Interpolated 3D Surface")
    plt.xlabel("X")
    plt.ylabel("Y")
    ax.set_zlabel("Interpolated Values")
    plt.show()

f = lambda x, y: - 4*x - 4 - 2*(1-np.log(y))/y**2
g_D = lambda x, y: x**2 + np.log(y)
g_N_Fr1 = lambda x: 2*x**2 + 2*x 
g_N_Fr2 = lambda y: 2*np.log(y)/y 
n = 3
a = 2
b = 6
c = 2
d = 4
x_vals = np.linspace(a, b, n + 1)
y_vals = np.linspace(c, d, n + 1)

# Discretizare și vizualizare

puncte = discretizare_domeniu(n, x_vals, y_vals)

A = discretizare_ecuatii(n, x_vals, y_vals)

total_puncte = (n + 1)**2
B = np.zeros(total_puncte)
for i in range(n + 1):
    for j in range(n + 1):
        index = i * (n + 1) + j #Calculăm indexul punctului curent în vectorul 1D
        if i == 0 or j == 0 or (i == n and j == n): 
            B[index] = g_D(x_vals[i], y_vals[j]) # Valoarea punctului u_{i,j}
        else:
            if i == n and j > 0 and j < n: 
                B[index] = g_N_Fr1(x_vals[i]) # Valoarea punctului u_{i,j}
            else:
                if i > 0 and i < n and j == n:
                    B[index] = g_N_Fr2(y_vals[j]) # Valoarea punctului u_{i,j}
                else:
                    B[index] = f(x_vals[i], y_vals[j]) # Valoarea punctului u_{i,j}

lu, piv = lu_factor(A)
U_sys = lu_solve((lu, piv), B)

print("Matricea sistemului A:")
print(A)

print("\nVectorul termenilor liberi B:")
print(B)

print("\nSoluția sistemului U_sys:")
print(U_sys)

vizualizare_rezultate(x_vals, y_vals, U_sys)


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RectBivariateSpline

# ======================== FUNCȚII AUXILIARE =====================================

def spline_patratic(deriv_init, X, Y, x):
    
    n = len(X) - 1
    if len(Y) != n + 1:
        raise ValueError("Listele X și Y trebuie să aibă aceeași lungime")
    if n < 1:
        raise ValueError("Sunt necesare cel puțin 2 puncte pentru interpolare")
    if any(X[i] >= X[i+1] for i in range(n)):
        raise ValueError("Nodurile X trebuie să fie strict crescătoare")

    h = [X[i+1] - X[i] for i in range(n)]
    a = Y.copy()
    b = [0.0] * n
    c = [0.0] * n

    b[0] = deriv_init

    for i in range(1, n):
        b[i] = 2 * (a[i] - a[i-1]) / h[i-1] - b[i-1]

    for i in range(n):
        c[i] = (a[i+1] - a[i] - b[i] * h[i]) / (h[i] ** 2)

    for i in range(n):
        if X[i] <= x <= X[i+1]:
            dx = x - X[i]
            return a[i] + b[i] * dx + c[i] * dx * dx

    return None

def discretizare_domeniu(n, x, y):
    X, Y = np.meshgrid(x, y)
    return np.vstack((X.flatten(), Y.flatten())), X, Y

def matrice_sistem(n, x, y):
    total = (n + 1)**2
    A = np.zeros((total, total))
    h_x = x[1] - x[0]
    h_y = y[1] - y[0]

    for j in range(n + 1):
        for i in range(n + 1):
            idx = i + j * (n + 1)
            if i == 0 or j == 0 or (i == n and j == n):
                A[idx, idx] = 1
            elif i == n and 0 < j < n:
                A[idx, idx] = 3 / (2 * h_x)
                A[idx, idx - 1] = -2 / h_x
                A[idx, idx - 2] = 1 / (2 * h_x)
            elif 0 < i < n and j == n:
                A[idx, idx] = 3 / (2 * h_y)
                A[idx, idx - (n + 1)] = -2 / h_y
                A[idx, idx - 2 * (n + 1)] = 1 / (2 * h_y)
            else:
                A[idx, idx] = 2 * (1 + x[i]) / h_x**2 + 4 * np.log(y[j]) / h_y**2
                A[idx, idx - 1] = 1 / (2 * h_x) - (1 + x[i]) / h_x**2
                A[idx, idx + 1] = 1 / (2 * h_x) - (1 + x[i]) / h_x**2
                A[idx, idx - (n + 1)] = 1 / (y[j] * h_y) - 2 * np.log(y[j]) / h_y**2
                A[idx, idx + (n + 1)] = 1 / (y[j] * h_y) - 2 * np.log(y[j]) / h_y**2
    return A

def spline_2D_interp(x_vals, y_vals, U_vals_flat):
    n = len(x_vals)
    m = len(y_vals)
    U_grid = U_vals_flat.reshape((m, n))
    spline = RectBivariateSpline(y_vals, x_vals, U_grid, kx=2, ky=2)
    x_fine = np.linspace(x_vals[0], x_vals[-1], 100)
    y_fine = np.linspace(y_vals[0], y_vals[-1], 100)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
    U_fine = spline(y_fine, x_fine)
    return X_fine, Y_fine, U_fine

def vizualizare_rezultate(x_vals, y_vals, U_sys, U_teoretic, max_erori):
    X_fine, Y_fine, U_num_interp = spline_2D_interp(x_vals, y_vals, U_sys)
    _, _, U_exact_interp = spline_2D_interp(x_vals, y_vals, U_teoretic)
    eroare_rel = np.abs(U_num_interp - U_exact_interp) / np.abs(U_exact_interp)

    max_erori.append(np.nanmax(eroare_rel))

    z_max = max(np.nanmax(U_num_interp), np.nanmax(U_exact_interp))
    z_min = min(np.nanmin(U_num_interp), np.nanmin(U_exact_interp))

    fig = plt.figure(figsize=(18, 5))

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X_fine, Y_fine, U_num_interp, cmap='viridis')
    ax1.set_title('Soluția Numerică')
    ax1.set_zlim(z_min, z_max)

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X_fine, Y_fine, U_exact_interp, cmap='plasma')
    ax2.set_title('Soluția Teoretică')
    ax2.set_zlim(z_min, z_max)

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X_fine, Y_fine, eroare_rel, cmap='coolwarm')
    ax3.set_title('Eroare Relativă |U_num - U_exact| / |U_exact|')
    ax3.set_zlim(z_min, z_max)

    plt.tight_layout()
    plt.show()

def factorizareQR(A):
    m = A.shape[0]
    n = A.shape[1]
    Q = np.copy(A).astype(float)
    R = np.zeros((n, n))
    for k in range(n):
        for i in range(k):
            R[i, k] = Q[:, i].T @ Q[:, k]
            Q[:, k] -= R[i, k] * Q[:, i]
        R[k, k] = np.linalg.norm(Q[:, k])
        Q[:, k] /= R[k, k]
    return Q, R

def substitutie_descendenta(U, b):
    n = len(b)
    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = (b[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x

def rezolva_QR(A, b):
    Q, R = factorizareQR(A)
    b_modificat = Q.T @ b
    return substitutie_descendenta(R, b_modificat)

def interpolare_lagrange(x, xi, yi):
    n = len(xi)
    L = np.zeros_like(x, dtype=float)
    for i in range(n):
        li = np.ones_like(x, dtype=float)
        for j in range(n):
            if i != j:
                li *= (x - xi[j]) / (xi[i] - xi[j])
        L += yi[i] * li
    return L

# ======================== EXECUTARE PRINCIPALĂ ==================================

f = lambda x, y: -4 * x - 4 - 2 * (1 - np.log(y)) / y**2
g_D = lambda x, y: x**2 + np.log(y)
g_N_Fr1 = lambda x: 2 * x**2 + 2 * x
g_N_Fr2 = lambda y: 2 * np.log(y) / y

max_erori = []
n_vals = (5, 10, 20, 30, 40)

for n in n_vals:
    x_vals = np.linspace(2, 6, n + 1)
    y_vals = np.linspace(2, 4, n + 1)
    puncte, X, Y = discretizare_domeniu(n, x_vals, y_vals)
    A = matrice_sistem(n, x_vals, y_vals)

    total = (n + 1)**2
    B = np.zeros(total)
    U_teoretic = np.zeros(total)

    for j in range(n + 1):
        for i in range(n + 1):
            idx = i + j * (n + 1)
            x = x_vals[i]
            y = y_vals[j]
            U_teoretic[idx] = g_D(x, y)
            if i == 0 or j == 0 or (i == n and j == n):
                B[idx] = g_D(x, y)
            elif i == n and 0 < j < n:
                B[idx] = g_N_Fr1(x)
            elif 0 < i < n and j == n:
                B[idx] = g_N_Fr2(y)
            else:
                B[idx] = f(x, y)

    U_sys = rezolva_QR(A, B)
    vizualizare_rezultate(x_vals, y_vals, U_sys, U_teoretic, max_erori)

# Analiză erori
n_array = np.array(n_vals)
erori_array = np.array(max_erori)
masca_valida = np.isfinite(erori_array) & (erori_array > 0)
n_array = n_array[masca_valida]
erori_array = erori_array[masca_valida]

n_vals_fine = np.linspace(min(n_array), max(n_array), 200)
erori_interp = interpolare_lagrange(n_vals_fine, n_array, erori_array)

plt.figure()
plt.semilogy(n_array, erori_array, 'o', label='Eroare relativă maximă (puncte)')
plt.semilogy(n_vals_fine, erori_interp, '-', label='Interpolare Lagrange')
plt.xlabel('n')
plt.ylabel('Eroare relativă maximă (log scale)')
plt.title('Eroare relativă maximă în funcție de n')
plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.legend()
plt.tight_layout()
plt.show()



# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RectBivariateSpline

# ======================== DEFINIRE FUNCȚII ======================================

def discretizare_domeniu(n, x, y):
    X, Y = np.meshgrid(x, y)
    return np.vstack((X.flatten(), Y.flatten())), X, Y

def matrice_sistem(n, x, y):
    total = (n + 1)**2
    A = np.zeros((total, total))
    h_x = x[1] - x[0]
    h_y = y[1] - y[0]

    for j in range(n + 1):
        for i in range(n + 1):
            idx = i + j * (n + 1)
            if i == 0 or j == 0 or (i == n and j == n):
                A[idx, idx] = 1
            elif i == n and 0 < j < n:
                A[idx, idx] = 3 / (2 * h_x)
                A[idx, idx - 1] = -2 / h_x
                A[idx, idx - 2] = 1 / (2 * h_x)
            elif 0 < i < n and j == n:
                A[idx, idx] = 3 / (2 * h_y)
                A[idx, idx - (n + 1)] = -2 / h_y
                A[idx, idx - 2 * (n + 1)] = 1 / (2 * h_y)
            else:
                A[idx, idx] = 2 * (1 + x[i]) / h_x**2 + 4 * np.log(y[j]) / h_y**2
                A[idx, idx - 1] = 1 / (2 * h_x) - (1 + x[i]) / h_x**2
                A[idx, idx + 1] = 1 / (2 * h_x) - (1 + x[i]) / h_x**2
                A[idx, idx - (n + 1)] = 1 / (y[j] * h_y) - 2 * np.log(y[j]) / h_y**2
                A[idx, idx + (n + 1)] = 1 / (y[j] * h_y) - 2 * np.log(y[j]) / h_y**2
    return A

def spline_2D_interp(x_vals, y_vals, U_vals_flat):
    n = len(x_vals)
    m = len(y_vals)
    U_grid = U_vals_flat.reshape((m, n))
    spline = RectBivariateSpline(y_vals, x_vals, U_grid, kx=2, ky=2)
    x_fine = np.linspace(x_vals[0], x_vals[-1], 100)
    y_fine = np.linspace(y_vals[0], y_vals[-1], 100)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
    U_fine = spline(y_fine, x_fine)
    return X_fine, Y_fine, U_fine

def vizualizare_rezultate(x_vals, y_vals, U_sys, U_teoretic, max_erori):
    X_fine, Y_fine, U_num_interp = spline_2D_interp(x_vals, y_vals, U_sys)
    _, _, U_exact_interp = spline_2D_interp(x_vals, y_vals, U_teoretic)
    eroare_rel = np.abs(U_num_interp - U_exact_interp) / np.abs(U_exact_interp)

    max_erori.append(np.nanmax(eroare_rel))

    z_max = max(np.nanmax(U_num_interp), np.nanmax(U_exact_interp))
    z_min = min(np.nanmin(U_num_interp), np.nanmin(U_exact_interp))

    fig = plt.figure(figsize=(18, 5))

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X_fine, Y_fine, U_num_interp, cmap='viridis')
    ax1.set_title('Soluția Numerică')
    ax1.set_zlim(z_min, z_max)

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X_fine, Y_fine, U_exact_interp, cmap='plasma')
    ax2.set_title('Soluția Teoretică')
    ax2.set_zlim(z_min, z_max)

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X_fine, Y_fine, eroare_rel, cmap='coolwarm')
    ax3.set_title('Eroare Relativă |U_num - U_exact| / |U_exact|')
    ax3.set_zlim(z_min, z_max)

    plt.tight_layout()
    plt.show()

def factorizareQR(A):
    m = A.shape[0]
    n = A.shape[1]
    Q = np.copy(A).astype(float)
    R = np.zeros((n, n))
    for k in range(n):
        for i in range(k):
            R[i, k] = Q[:, i].T @ Q[:, k]
            Q[:, k] -= R[i, k] * Q[:, i]
        R[k, k] = np.linalg.norm(Q[:, k])
        Q[:, k] /= R[k, k]
    return Q, R

def substitutie_descendenta(U, b):
    n = len(b)
    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = (b[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x

def rezolva_QR(A, b):
    Q, R = factorizareQR(A)
    b_modificat = Q.T @ b
    return substitutie_descendenta(R, b_modificat)

# ======================== DEFINIRE PARAMETRI ȘI EXECUTARE =======================

f = lambda x, y: -4 * x - 4 - 2 * (1 - np.log(y)) / y**2
g_D = lambda x, y: x**2 + np.log(y)
g_N_Fr1 = lambda x: 2 * x**2 + 2 * x
g_N_Fr2 = lambda y: 2 * np.log(y) / y

max_erori = []
n_vals = (5, 10, 20, 30, 40)

for n in n_vals:
    x_vals = np.linspace(2, 6, n + 1)
    y_vals = np.linspace(2, 4, n + 1)
    puncte, X, Y = discretizare_domeniu(n, x_vals, y_vals)
    A = matrice_sistem(n, x_vals, y_vals)

    total = (n + 1)**2
    B = np.zeros(total)
    U_teoretic = np.zeros(total)

    for j in range(n + 1):
        for i in range(n + 1):
            idx = i + j * (n + 1)
            x = x_vals[i]
            y = y_vals[j]
            U_teoretic[idx] = g_D(x, y)
            if i == 0 or j == 0 or (i == n and j == n):
                B[idx] = g_D(x, y)
            elif i == n and 0 < j < n:
                B[idx] = g_N_Fr1(x)
            elif 0 < i < n and j == n:
                B[idx] = g_N_Fr2(y)
            else:
                B[idx] = f(x, y)

    U_sys = rezolva_QR(A, B)
    vizualizare_rezultate(x_vals, y_vals, U_sys, U_teoretic, max_erori)

# Grafic logaritmic al erorilor maxime cu interpolare Lagrange

def interpolare_lagrange(x, xi, yi):
    n = len(xi)
    L = np.zeros_like(x, dtype=float)
    for i in range(n):
        li = np.ones_like(x, dtype=float)
        for j in range(n):
            if i != j:
                li *= (x - xi[j]) / (xi[i] - xi[j])
        L += yi[i] * li
    return L

n_array = np.array(n_vals)
erori_array = np.array(max_erori)
masca_valida = np.isfinite(erori_array) & (erori_array > 0)
n_array = n_array[masca_valida]
erori_array = erori_array[masca_valida]

n_vals_fine = np.linspace(min(n_array), max(n_array), 200)
erori_interp = interpolare_lagrange(n_vals_fine, n_array, erori_array)

plt.figure()
plt.semilogy(n_array, erori_array, 'o', label='Eroare relativă maximă (puncte)')
plt.semilogy(n_vals_fine, erori_interp, '-', label='Interpolare Lagrange')
plt.xlabel('n')
plt.ylabel('Eroare relativă maximă (log scale)')
plt.title('Eroare relativă maximă în funcție de n')
plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.legend()
plt.tight_layout()
plt.show()
