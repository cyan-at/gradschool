#!/usr/bin/env python3

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from sympy import *
from sympy.solvers import solve
from sympy import real_root

def gen_2dcircle_pts(r, n):
    pts = []
    for rad in np.linspace(0, 2*np.pi, n):
        y = r * np.sin(rad)
        x = r * np.cos(rad)
        pts.append([x, y])
    return pts

def f(x,t):
    x1 = x[0]
    x2 = x[1]

    x1p1 = x2
    x2p1 = -x1 - x2 + x1**3

    return [x1p1, x2p1]

def gen_pts(v, n):
    pts = []

    x2 = np.sqrt(2 * v)
    x2_r = np.linspace(-x2, x2, n) + [x2]

    x1 = Symbol('x1', real=True)
    f = x1**2 / 2 - x1**4 / 4

    for x2 in x2_r:
        y = x2**2 / 2 - v

        # f + y = x2**2 / 2 + x1**2 / 2 - x1**4 / 4 - v = 0
        ans = solve(f + y, x1)
        # print(ans)

        for a in ans:
            if a < 1 and a > -1: # in D
                pts.append([a, x2])

                # print(x2**2 / 2 + a**2 / 2 - a**4 / 4)
    return pts

fig = plt.figure()
ax = fig.add_subplot(111)

bound = 2.0
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

bound2 = 0.26
n2 = 10

t = np.linspace(0,100,200)

# r = 0.577
# pts = gen_2dcircle_pts(r, 100)

v = 0.213
pts = gen_pts(v, 100)

for pt in pts:
    x1 = pt[0]
    x2 = pt[1]

    path = odeint(f, [x1, x2], t)

    plt.plot(path[:,0], path[:,1], 'b--', linewidth=0.5)
    plt.plot([path[0,0]], [path[0,1]], 'go') # start
    plt.plot([path[-1,0]], [path[-1,1]], 'rs') # end

fig.suptitle('state-space v = %.3f' % (v))
ax.set_title('arrows = fields, lines = paths, green = starts, red = ends')
fig.tight_layout()
ax.set_aspect('equal')

lim = 2
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)

# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111, projection='3d')

# z = u + v
# ax2.plot_surface(u, v, z,cmap='viridis', edgecolor='none')

plt.show()