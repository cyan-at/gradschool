#!/usr/bin/env python

import argparse
import matplotlib
import numpy as np
try:
    matplotlib.use("TkAgg")
except:
    print("no tkagg")
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--state_bound_min', type=float, default=-0.1, help='')
    parser.add_argument('--state_bound_max', type=float, default=0.6, help='')

    parser.add_argument('--dat',
        type=str, required=True)

    args, _ = parser.parse_known_args()

    buffer = 0.1

    fig = plt.figure(figsize=(8,8))
    ax = fig.gca()
    ax.set_xlim(args.state_bound_min - buffer, args.state_bound_max + buffer)
    ax.set_ylim(args.state_bound_min - buffer, args.state_bound_max + buffer)

    d = 2
    N = 22
    M = N**d
    linspaces = []
    for i in range(d):
        linspaces.append(np.transpose(
            np.linspace(args.state_bound_min, args.state_bound_max, N))
        )
    meshes = np.meshgrid(*linspaces)

    f = np.loadtxt(args.dat).reshape((N, N))

    cfset = ax.contourf(meshes[0], meshes[1], f, cmap='coolwarm')

    ax.imshow(np.rot90(f), cmap='coolwarm', extent=[
        args.state_bound_min - buffer, args.state_bound_max + buffer,
        args.state_bound_min - buffer, args.state_bound_max + buffer
    ])
    cset = ax.contour(meshes[0], meshes[1], f, colors='k')

    ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.title(args.dat)

    plt.show()