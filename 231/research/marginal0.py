#!/usr/bin/python3

import argparse

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import os, pickle, time, sys
from matplotlib import cm
import matplotlib.pyplot as plt

import scipy.integrate as integrate

from distribution0 import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--times',
        type=str,
        default="0,1.0,2.0,3.0,4.0,5.0,10.0",
        required=False)

    parser.add_argument('--mu_0',
        type=float,
        default=2.0,
        required=False)

    parser.add_argument('--sampling',
        type=str,
        default="15,15,15,15,15,15,100,200",
        required=False)

    args = parser.parse_args()

    # system
    alpha2 = 0.5

    # distribution
    mu_0 = np.array([args.mu_0]*3)
    cov_0 = np.eye(3)

    # sampling
    ts = [float(x) for x in args.times.split(",")]

    sampling = [int(x) for x in args.sampling.split(",")]
    window0 = sampling[0]
    window1 = sampling[1]
    window2 = sampling[2]
    window3 = sampling[3]
    window4 = sampling[4]
    window5 = sampling[5]
    windows = sampling[:6]
    N = sampling[6]
    distribution_samples = sampling[7]

    #############################################################################

    initial_sample, te_to_data, X1, X2, X3 = init_data(
        mu_0, cov_0,
        windows, distribution_samples, N, ts,
        alpha2)

    x1 = X1[0, :, 0]
    x2 = X2[:, 0, 0]
    x3 = X3[0, 0, :]

    #############################################################################

    for t_e in ts:
        '''
        te_to_data[t_e] = {
            "probs" : probs,
            "all_time_data" : all_time_data
        }

        all_time_data = np.empty(
            (
                initial_sample.shape[0],
                initial_sample.shape[1],
                len(t_samples))
            )
        # x/y slice is all samples at that time, 1 x/y slice per z time initial_sample
        '''
        probs_reshape = te_to_data[t_e]["probs"].reshape(N, N, N)

        te_to_data[t_e]["x1_marginal"] = np.array([
            np.trapz(
                np.array([
                    np.trapz(probs_reshape[j, i, :], x=x3) # x3 slices for one x2 => R
                    for i in range(len(x2))]) # x3 slices across all x2 => Rn
                , x=x2) # x2 slice for one x1 => R
        for j in range(len(x1))])

        te_to_data[t_e]["x2_marginal"] = np.array([
            np.trapz(
                np.array([
                    np.trapz(probs_reshape[i, j, :], x=x3) # x3 slices for one x1 => R
                    for i in range(len(x1))]) # x3 slices across all x1 => Rn
                , x=x1) # x1 slice for one x2 => R
        for j in range(len(x2))])

        te_to_data[t_e]["x3_marginal"] = np.array([
            np.trapz(
                np.array([
                    np.trapz(probs_reshape[i, :, j], x=x2) # x2 slices for one x1 => R
                    for i in range(len(x1))]) # x2 slices across all x1 => Rn
                , x=x1) # x1 slice for one x3 => R
        for j in range(len(x3))])

fig = plt.figure(1)
ax1 = plt.subplot(121, frameon=False)
ax1.set_aspect('equal')
ax1.grid()

ax2 = plt.subplot(122, frameon=False)
ax2.set_aspect('equal')
ax2.grid()

# ax3 = plt.subplot(133, frameon=False)

fig2 = plt.figure(2)
ax3 = plt.subplot(111, frameon=False)
ax3.set_aspect('equal')
ax3.grid()

colors="rgbymkc"

for i, t_e in enumerate(ts):
    ax1.plot(x1,
        te_to_data[t_e]["x1_marginal"],
        colors[i % len(colors)],
        linewidth=1,
        label=t_e)
    ax1.legend(loc='lower right')

    ax2.plot(x2,
        te_to_data[t_e]["x2_marginal"],
        colors[i % len(colors)],
        linewidth=1,
        label=t_e)
    ax2.legend(loc='lower right')

    ax3.plot(x3,
        te_to_data[t_e]["x3_marginal"],
        colors[i % len(colors)],
        linewidth=1,
        label=t_e)
    ax3.legend(loc='lower right')

plt.show()
