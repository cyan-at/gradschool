from scipy.optimize import linprog
from scipy import sparse
import numpy as np
from scipy.stats import truncnorm

from scipy.stats import multivariate_normal

N = nSample = 100

state_min = 0
state_max = 6
T_t = 5

sigma_0 = 0.1
sigma_T = 0.08

x_grid = np.transpose(np.linspace(state_min, state_max, nSample))
y_grid = np.transpose(np.linspace(state_min, state_max, nSample))
[X,Y] = np.meshgrid(x_grid,x_grid)
C = (X - Y)**2

# cvector = C.flatten('F')
cvector = C.reshape(nSample**2,1)

A = np.concatenate(
    (
        np.kron(
            np.ones((1,nSample)),
            sparse.eye(nSample).toarray()
        ),
        np.kron(
            sparse.eye(nSample).toarray(),
            np.ones((1,nSample))
        )
    ), axis=0)
# 2*nSample

mu_0 = 5.0
mu_T = 5.0
rv0 = multivariate_normal([mu_0, mu_0, mu_0], sigma_0 * np.eye(3))
rvT = multivariate_normal([mu_T, mu_T, mu_T], sigma_T * np.eye(3))

def pdf1d_0(x,y,z):
    return rv0.pdf(np.hstack((x, y, z)))

def pdf1d_T(x,y,z):
    return rvT.pdf(np.hstack((x, y, z)))

################################

# 6 define data and net

x_T = np.transpose(np.linspace(state_min, state_max, N))
y_T = np.transpose(np.linspace(state_min, state_max, N))
z_T = np.transpose(np.linspace(state_min, state_max, N))

x_T=x_T.reshape(len(x_T),1)
y_T=y_T.reshape(len(y_T),1)
z_T=z_T.reshape(len(z_T),1)

terminal_time=np.hstack((x_T,y_T,z_T,T_t*np.ones((len(x_T), 1))))

rho_T=pdf1d_T(x_T,y_T,z_T).reshape(len(x_T),1)

rho_0=pdf1d_0(x_T,y_T,z_T).reshape(len(x_T),1)

bvector = np.concatenate((rho_0, rho_T), axis=0)
res = linprog(cvector, A_eq=A, b_eq=bvector, options={"disp": True})
print(res.fun)