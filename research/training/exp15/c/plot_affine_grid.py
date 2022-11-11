#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt
try:
    matplotlib.use("TkAgg")
except:
    print("no tkagg")
from matplotlib import cm

import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--f',
        type=str,
        required=True)

    parser.add_argument('--v_scale',
        type=str,
        default="1.0,100.0")
    parser.add_argument('--bias',
        type=str,
        default=" -5.0,5.0")
    parser.add_argument('--search_grid',
        type=int,
        default=15)

    args = parser.parse_args()

    v_scales = [float(x) for x in args.v_scale.split(",")]
    biases = [float(x) for x in args.bias.split(",")]
    if len(v_scales) > 1:
        v_scales = np.linspace(
            v_scales[0], v_scales[-1], args.search_grid)
    if len(biases) > 1:
        biases = np.linspace(
            biases[0], biases[-1], args.search_grid)

    x = np.load(args.f, allow_pickle=True).item()
    mus = np.array([y[0] for y in x.values()])

    # parameter_state = 

    # state, space, dynamics, trajectories, transforms

    l1_norms = np.sum(np.abs(mus), axis=1) # /l1 norm
    tokens = np.array([[float(s) for s in t.split('_')] for t in x.keys()])

    x = tokens[:, 0].reshape(15, 15)
    y = tokens[:, 1].reshape(15, 15)
    l = l1_norms.reshape(15, 15)

    fig = plt.figure()
    ax3 = fig.add_subplot(1, 1, 1, projection='3d')

    sc3=ax3.scatter(
        x,
        y,
        l,
        c=l,
        s=10,
        cmap=cm.jet,
        alpha=1.0)
    plt.colorbar(sc3, shrink=0.25)


    ax3.set_xlabel('v scale')
    ax3.set_ylabel('bias')
    ax3.set_zlabel('l1 norm')
    ax3.set_title('affine v transform vs. mu l1 norm (goal was 0)')

    plt.show()

    # x.values
    # x.values()
    # np.array(x.values())
    # x.values()
    # [y[0] for y in x.values()]
    # np.array([y[0] for y in x.values()])
    # np.sum(mus)
    # np.sum(mus, axis=0)
    # np.sum(mus, axis=1)
    # np.abs(np.sum(mus, axis=1))
    # np.min(np.abs(np.sum(mus, axis=1)))
    # np.argmin(np.abs(np.sum(mus, axis=1)))
    # mus[23]
    # np.sum(np.abs(mus), axis=1)
    # np.min(np.sum(np.abs(mus), axis=1))
    # np.argmin(np.sum(np.abs(mus), axis=1))
    # mus[9]
    # np.array([y[1] for y in x.values()])
    # mus
    # mu.shape
    # mus.shape
    # np.abs(mus)
    # history
