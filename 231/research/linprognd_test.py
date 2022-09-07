from scipy.optimize import linprog
from scipy import sparse
import numpy as np
from scipy.stats import truncnorm

from scipy.stats import multivariate_normal

N = nSample = 100

x_grid = np.transpose(np.linspace(0., 6., nSample))
y_grid = np.transpose(np.linspace(0., 6., nSample))
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
mu_T = 1.0
rv0 = multivariate_normal([mu_0, mu_0, mu_0], sigma_0 * np.eye(3))
rvT = multivariate_normal([mu_T, mu_T, mu_T], sigma_T * np.eye(3))

def pdf1d_T(x):
    mu = 5.
    sigma = .1
    a, b = (0. - mu) / sigma, (6. - mu) / sigma
    rho_T=truncnorm.pdf(x, a, b, loc = mu, scale = sigma)
    return rho_T

def pdf1d_0(x):
    sigma = .08
    mu=0.5 
    # the smaller this is, the smaller sigma must be to solve
    # i.e.
    # mu=0.6, sigma=0.1
    # mu=0.5, sigma=0.08
    a, b = (0. - mu) / sigma, (6. - mu) / sigma
    rho_0=truncnorm.pdf(x, a, b, loc = mu, scale = sigma)
    return rho_0

x_T = np.transpose(np.linspace(0., 6., N))
rho_0=pdf1d_0(x_T).reshape(len(x_T),1)
rho_T=pdf1d_T(x_T).reshape(len(x_T),1)

bvector = np.concatenate((rho_0, rho_T), axis=0)
res = linprog(cvector, A_eq=A, b_eq=bvector, options={"disp": True})
print(res.fun)