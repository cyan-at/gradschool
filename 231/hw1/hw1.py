#!/usr/bin/env python3

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def f(x,t):
    a = 0.5
    b = 0.5

    x1 = x[0]
    x2 = x[1]

    x1p1 = a * x2 / (1 + x1**2)
    x2p1 = b * x1 / (1 + x2**2)

    return [x1p1, x2p1]

fig = plt.figure()
ax = fig.add_subplot(111)

bound = 20.0
n = 101
y1 = np.linspace(-bound, bound, n)
y2 = np.linspace(-bound, bound, n)
Y1, Y2 = np.meshgrid(y1, y2)
u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)
NI, NJ = Y1.shape
for i in range(NI):
    for j in range(NJ):
        x1 = Y1[i, j]
        x2 = Y2[i, j]
        xkp1 = f([x1, x2], 0)
        u[i,j] = xkp1[0]
        v[i,j] = xkp1[1]

        # diff = np.abs(u[i, j] - x1) + np.abs(v[i, j] - x2)
        # print(x1, x2, diff)
Q = ax.quiver(Y1, Y2, u, v, color='r')

bound2 = 20
n2 = bound2 + 1
t = np.linspace(0,500)
for x1 in np.linspace(-bound2, bound2, n2):
    for x2 in np.linspace(-bound2, bound2, n2):

# for x1 in np.linspace(-bound2, bound2, n2):
#         x2 = -x1

        path = odeint(f, [x1, x2], t)
        plt.plot(path[:,0], path[:,1], 'b--', linewidth=0.5)

        plt.plot([path[0,0]], [path[0,1]], 'go') # start
        plt.plot([path[-1,0]], [path[-1,1]], 'rs') # end

fig.suptitle('Question 2: a, b = 0.5')
ax.set_title('arrows = fields, lines = paths, green = starts, red = ends')
fig.tight_layout()
ax.set_aspect('equal')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')

z = u + v
ax2.plot_surface(u, v, z,cmap='viridis', edgecolor='none')

plt.show()