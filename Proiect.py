import numpy as np
import matplotlib.pyplot as plt

def factorizareLU(A):
    n=np.shape(A)[0]
    L=np.eye(n)
    U=np.zeros((n,n))
    A_copy=np.copy(A)

    for k in range(n):
        U[k,k]=A_copy[k,k]
        for i in range(k+1,n):
            L[i,k]=A_copy[i,k]/A_copy[k,k]
            U[k,i]=A_copy[k,i]
        for i in range(k+1,n):
            for j in range(k+1,n):
                A_copy[i,j]=A_copy[i,j]-1/A_copy[k,k]*A_copy[i,k]*A_copy[k,j]
    return L,U

def Subs_Asc(L, b):
    n=np.shape(L)[0]
    for i in range(0,n):
        if L[i,i]==0:
            return 0
    x = np.zeros((n, 1))
    for i in range(0,n):
        x[i]=(b[i]-L[i,:i]@x[:i])/L[i,i]
    return x

def Subs_Desc(U, b):
    n=np.shape(U)[0]
    for i in range(0,n):
        if U[i,i]==0:
            return 0
    x = np.zeros((n, 1))
    for i in range(n-1,-1,-1):
        x[i]=(b[i]-U[i,i+1:]@x[i+1:])/U[i,i]
    return x

def polinomLagrange(X,k,n,x):
    L=1
    for i in range(n+1):
        if i!=k:
            L=L* (x-X[i])/(X[k]-X[i])
    return L

def interpolareLagrange(X,Y,x):
    y=0
    for i in range(len(X)):
        y+=Y[i]*polinomLagrange(X,i,len(X)-1,x)
    return y

def grafic_interpolare(X,Y):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
    ax1.scatter(X,Y, c='green',label='Puncte de interpolare')

    x_grafic=np.linspace(X[0],X[-1],100)
    y_grafic = []
    y_grafic_eroare = [] 

    for x in x_grafic:
            y_grafic.append(interpolareLagrange(X, Y, x))

    ax1.plot(x_grafic, y_grafic, 'r', linewidth=3)
    
    ax1.axvline(0, c='black')
    ax1.axhline(0, c='black')
    ax1.grid(color='grey', linestyle='--', linewidth=0.5)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax2.axvline(0, c='black')
    ax2.axhline(0, c='black')
    ax2.grid(color='grey', linestyle='--', linewidth=0.5)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    plt.tight_layout()
    plt.show()

n=5
a,b=0,10
timp=5

f = lambda x,t: (x*x+1)*t

X=np.linspace(a,b,n+1)          #discretizarea lungimii
T=np.linspace(0,timp,100)          #discretizarea timpului
A=np.zeros((n+1,n+1))
B=np.zeros((n+1,1))

for t in T:
    for i in range(1,n):
        B[i,0]=f(X[i],t)
    B[0,0]=0
    B[n,0]=17*t


    A[0,0]=1
    A[n,n]=1
    for i in range(1,n):
        A[i,i-1]=-1/(X[i+1]-X[i])
        A[i,i]=2/(X[i+1]-X[i])
        A[i,i+1]=-1/(X[i+1]-X[i])

    l,u=factorizareLU(A)
    y=Subs_Asc(l,B)
    U=Subs_Desc(u,y)
    grafic_interpolare(X,B)
