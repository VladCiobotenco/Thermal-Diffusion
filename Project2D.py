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
