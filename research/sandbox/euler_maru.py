#!/usr/bin/env python3

import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from common import *
 
def lorenz(t, state, sigma, beta, rho, verbose):
    x, y, z = state
     
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    if verbose:
        print(t)
     
    return [dx, dy, dz]

sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0

N = 2000
 
p = (sigma, beta, rho)  # Parameters of the system
 
y0 = [1.0, 1.0, 1.0]  # Initial state of the system

t_span = (0.0, 20.0)

t = np.arange(t_span[0], t_span[-1], (t_span[-1] - t_span[0])/N)
print(t)

 
fig = plt.figure()

ts, result = euler_maru(
    y0,
    t_span,
    lorenz,
    (t_span[-1] - t_span[0])/(N),
    lambda delta_t: 0.0, # np.random.normal(loc=0.0, scale=np.sqrt(delta_t)),
    lambda y, t: 0.0, # 0.06,
    (sigma, beta, rho, True))

print(len(ts))

ax = fig.add_subplot(1, 3, 1, projection='3d')
ax.plot(result[:, 0],
        result[:, 1],
        result[:, 2])
ax.set_title("euler_maru")

'''
https://danielmuellerkomorowska.com/2021/02/16/differential-equations-with-scipy-odeint-or-solve_ivp/
'''
result_odeint = odeint(
    lorenz, y0, t,
    (sigma, beta, rho, False),
    tfirst=True)
ax = fig.add_subplot(1, 3, 2, projection='3d')
ax.plot(result_odeint[:, 0],
        result_odeint[:, 1],
        result_odeint[:, 2])
ax.set_title("odeint")

result_solve_ivp = solve_ivp(
    lorenz,
    t_span,
    y0,
    args=(sigma, beta, rho, False),
    # t_eval=np.linspace(0, 40.0, 10),
    method='LSODA',
    # rtol=1.0,
    # atol=1.0,
    # min_step=1.0,
    # max_step=1.0
)
ax = fig.add_subplot(1, 3, 3, projection='3d')
ax.plot(result_solve_ivp.y[0, :],
        result_solve_ivp.y[1, :],
        result_solve_ivp.y[2, :])
ax.set_title("solve_ivp")

plt.show()
