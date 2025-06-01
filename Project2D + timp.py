import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RectBivariateSpline
import scipy as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


# ======================== FUNCȚII AUXILIARE =====================================
# Functia f + conditii Neumann margini + solutie exacta pt verificarea erorii
f = lambda x, y: -4 * x - 2 * (1 - np.log(y)) / y**2
g_N_Fr1 = lambda x: 2 * x**2 + 2 * x
g_N_Fr2 = lambda y: 2 * np.log(y) / y
g_D = lambda x, y, t: x**2 + np.log(y) + 2*t
 
# Discretizeaza domeniul [2,6]x[2,6] in n+1 puncte pe fiecare axa
def discretizare_domeniu(n):
    X = np.linspace(2, 6, n+1)
    Y = np.linspace(2, 6, n+1)
    return X, Y
    
# Construieste matricea sistemului si vectorul de termen liber
def matrice_sistem(n, x, y, t):
    total = (n + 1)**2
    B = np.zeros(total)
    U_teoretic = np.zeros(total)
    A = np.zeros((total, total))
    h_x = x[1] - x[0]
    h_y = y[1] - y[0]

    for j in range(n + 1):
        for i in range(n + 1):
            idx = i + j * (n + 1)
            U_teoretic[idx] = g_D(x[i], y[j], t)
              # Conditii de Dirichlet
            if i == 0 or j == 0 or (i == n and j == n):
                A[idx, idx] = 1
                B[idx] = g_D(x[i], y[j], t)
                 # Conditie de Neumann pe marginea dreapta
            elif i == n and 0 < j < n:
                A[idx, idx] = 3 / (2 * h_x)
                A[idx, idx - 1] = -2 / h_x
                A[idx, idx - 2] = 1 / (2 * h_x)
                B[idx] = g_N_Fr1(x[i])
                  # Conditie de Neumann pe marginea de sus
            elif 0 < i < n and j == n:
                A[idx, idx] = 3 / (2 * h_y)
                A[idx, idx - (n + 1)] = -2 / h_y
                A[idx, idx - 2 * (n + 1)] = 1 / (2 * h_y)
                B[idx] = g_N_Fr2(y[j])
                # Nucleu interior - discretizare centrala
            else:
                A[idx, idx] = 2 * (1 + x[i]) / h_x**2 + 4 * np.log(y[j]) / h_y**2
                A[idx, idx - 1] = 1 / (2 * h_x) - (1 + x[i]) / h_x**2
                A[idx, idx + 1] = 1 / (2 * h_x) - (1 + x[i]) / h_x**2
                A[idx, idx - (n + 1)] = 1 / (y[j] * h_y) - 2 * np.log(y[j]) / h_y**2
                A[idx, idx + (n + 1)] = 1 / (y[j] * h_y) - 2 * np.log(y[j]) / h_y**2
                B[idx] = f(x[i], y[j])
    return A, B, U_teoretic
# Interpolarea spline bidimensionala
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
# Vizualizeaza solutia numerica, teoretica si eroarea dintre ele
def vizualizare_rezultate(x_vals, y_vals, U_sys, U_teoretic, max_erori):
    X_fine, Y_fine, U_num_interp = spline_2D_interp(x_vals, y_vals, U_sys)
    _, _, U_exact_interp = spline_2D_interp(x_vals, y_vals, U_teoretic)
    eroare_rel = np.abs(U_num_interp - U_exact_interp)

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
    ax3.set_title('Eroare absolută |U_num - U_exact|')
    ax3.set_zlim(z_min, z_max)

    plt.tight_layout()
    plt.show()
#interpolarea polinomiala de tip Lagrange (pt eroare)
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
#rezolvarea sistemului liniar
def rezolva_sparse(A,b):
    A_sparse=csr_matrix(A)
    x=spsolve(A_sparse,b)
    return x

# ======================== EXECUTARE PRINCIPALĂ ==================================

max_erori = []
n_vals = (5, 10, 20, 30, 40)
for n in n_vals:
    print("N = " + str(n))
    X, Y = discretizare_domeniu(n)

    total = (n + 1)**2
    B = np.zeros(total)
    U_teoretic = np.zeros(total)
    A, B, U_teoretic = matrice_sistem(n, X, Y, 0)
    
    U_sys = rezolva_sparse(A, B)
    U_prec = U_sys.copy()
    timp = 10 
    T = np.linspace(0, timp, timp+1)
    h_t = T[1] - T[0]

    B_init = B.copy()
    vizualizare_rezultate(X, Y, U_sys, U_teoretic, max_erori)
    frames = []
    for t in T:
        B = B_init + U_prec / h_t

        # Suprascrie B cu valorile corecte pentru condițiile de frontieră
        for j in range(n+1):
            for i in range(n+1):
                idx = i + j * (n + 1)
                U_teoretic[idx] = g_D(X[i], Y[j], t)
                if i == 0 or j == 0 or (i == n and j == n):
                    B[idx] = g_D(X[i], Y[j], t)
                elif i == n and 0 < j < n:
                    B[idx] = g_N_Fr1(X[i])
                elif 0 < i < n and j == n:
                    B[idx] = g_N_Fr2(Y[j])

        if t == T[-1]:  # salvează eroarea doar la ultimul pas de timp
            vizualizare_rezultate(X, Y, U_sys, U_teoretic, [])
        else:
            vizualizare_rezultate(X, Y, U_sys, U_teoretic, [])
        U_sys = rezolva_sparse(A, B)
        U_prec = U_sys.copy()  # pregătește pentru următoarea iterație
        frames.append(U_sys.reshape((n+1, n+1)))
        
# Analiză erori
n_array = np.array(n_vals)
erori_array = np.array(max_erori)
masca_valida = np.isfinite(erori_array) & (erori_array > 0)
n_array = n_array[masca_valida]
erori_array = erori_array[masca_valida]

n_vals_fine = np.linspace(min(n_array), max(n_array), 200)
erori_interp = interpolare_lagrange(n_vals_fine, n_array, erori_array)

#grafic
plt.plot(n_array, erori_array, 'o', label='log10(Eroare absolută maximă)')
plt.plot(n_vals_fine, erori_interp, '-', label='Interpolare Lagrange (log10)')
plt.xlabel('n')
plt.ylabel('log10(Eroare absolută maximă)')
plt.title('Eroare absolută maximă în funcție de n (formă logaritmică)')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.minorticks_on()
plt.legend()
plt.tight_layout()
plt.show()
