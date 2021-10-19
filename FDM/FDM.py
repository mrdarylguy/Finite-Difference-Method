import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import scipy.sparse as sp
import scipy.sparse.linalg as la


#Initialize the 1000x1000 matrix 
N = 100
u0_t, u0_b = 5, -5
u0_l, u0_r = 3, -1
dx = 1./(N+1) 

A1_d = (sp.eye(N, k=-1) + 
        sp.eye(N, k=1) - 
        4 * sp.eye(N)) / dx**2

A = (sp.kron(sp.eye(N), A1_d) + 
    sp.eye(N**2, k=-N) + 
    sp.eye(N**2, k=N)) / dx**2


#Column of eigenvalues
b = np.zeros((N,N))

b[0, :] += u0_b #bottom
b[-1, :] += u0_t #top
b[:, 0] += u0_l #left
b[:, -1] += u0_r #right
b = -b.reshape(N**2) / dx**2

#Solve the eigenproblem
v = la.spsolve(A, b)
u = v.reshape(N, N)

U = np.vstack([np.ones((1, N+2)) * u0_b,
               np.hstack([np.ones((N, 1)) * u0_l,
               u,
               np.ones((N, 1)) * u0_r]), 
               np.ones((1, N+2)) * u0_t])

x = np.linspace(0, 1, N+2)
X, Y = np.meshgrid(x, x)

fig = plt.figure(figsize=(12, 5.5))
cmap = mpl.cm.get_cmap('cool_r')

ax = fig.add_subplot(1, 2, 1)
c = ax.pcolor(X, Y, U, vmin=-5, vmax=5, cmap=cmap, shading='auto')
ax.set_xlabel(r'$x_1$', fontsize=18)
ax.set_ylabel(r'$x_2$', fontsize=18)

ax = fig.add_subplot(1, 2, 2, projection='3d')
p = ax.plot_surface(X, Y, U, 
                    vmin=-5, vmax=5, 
                    rstride=3, cstride=3,
                    linewidth=0,
                    cmap=cmap
                    )

ax.set_xlabel(r'$x_1$', fontsize=18)
ax.set_ylabel(r'$x_2$', fontsize=18)
cb = plt.colorbar(p, ax=ax, shrink=0.75)
cb.set_label(r'$u(x_1, x_2)$', fontsize=18)

plt.show()