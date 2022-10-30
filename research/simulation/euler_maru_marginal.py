#!/usr/bin/env python3

'''
#!/usr/bin/python3

USAGE:

./marginal0.py

./marginal0.py --plot 1
'''

import argparse

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import os, pickle, time, sys
from matplotlib import cm
import matplotlib.pyplot as plt

import scipy.integrate as integrate

from common import *

from RSB_traj import plot_params

import matplotlib
matplotlib.use("TkAgg")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--times',
        type=str,
        default="0,1.0,2.0,3.0,4.0,5.0",
        required=False)

    parser.add_argument('--mu_0',
        type=float,
        default=2.0,
        required=False)

    parser.add_argument('--sampling',
        type=str,
        default="15,15,15,15,15,15,100,200",
        required=False)

    parser.add_argument('--system',
        type=str,
        default="1,1,2", # 3,2,1
        required=False)

    parser.add_argument('--N',
        type=int,
        default=2000,
        required=False)

    parser.add_argument('--M',
        type=int,
        default=10,
        required=False)

    parser.add_argument('--control_data',
        type=str,
        default="",
        required=False)

    parser.add_argument('--v_scale',
        type=float,
        default=1.0)

    args = parser.parse_args()

    j1, j2, j3 = [float(x) for x in args.system.split(",")]

    d = 3

    # T_t = 20.0

    t_span = (T_0, T_t)
    N = args.N
    dt = (t_span[-1] - t_span[0])/(N)
    ts = np.arange(t_span[0], t_span[1] + dt, dt)
    print(len(ts))

    a = 0.05

    control_data = None
    if len(args.control_data) > 0:
        control_data = np.load(
            args.control_data,
            allow_pickle=True).item()

        # import ipdb; ipdb.set_trace()

    initial_sample = np.random.multivariate_normal(
        np.array([mu_0]*d), np.eye(d)*0.1, args.M) # 100 x 3

    ##############################

    with_control = np.empty(
        (
            initial_sample.shape[0],
            initial_sample.shape[1],
            len(ts),
        ))

    for i in range(initial_sample.shape[0]):
        # x[i] is sample [i]
        # y[i] is state dim [i]
        # z[i] is time [i]
        print(i)
        _, tmp = euler_maru(
            initial_sample[i, :],
            t_span,
            dynamics,
            (t_span[-1] - t_span[0])/(N),
            lambda delta_t: 0.0, # np.random.normal(loc=0.0, scale=np.sqrt(delta_t)),
            lambda y, t: 0.0, # 0.06,
            (j1, j2, j3, control_data, args.v_scale))
        with_control[i, :, :] = tmp.T

    without_control = np.empty(
        (
            initial_sample.shape[0],
            initial_sample.shape[1],
            len(ts),
        ))

    for i in range(initial_sample.shape[0]):
        # x[i] is sample [i]
        # y[i] is state dim [i]
        # z[i] is time [i]
        print(i)
        _, tmp = euler_maru(
            initial_sample[i, :],
            t_span,
            dynamics,
            (t_span[-1] - t_span[0])/(N),
            lambda delta_t: 0.0, # np.random.normal(loc=0.0, scale=np.sqrt(delta_t)),
            lambda y, t: 0.0, # 0.06,
            (j1, j2, j3, None, args.v_scale))
        without_control[i, :, :] = tmp.T

    ##############################

    fig = plt.figure()

    params = plot_params()
    plt.rcParams.update(params)

    fig = plt.figure(1,
        figsize=params["figure.figsize"]) 
    # figsize accepts only inches.
    fig.subplots_adjust(
        left=0.04,
        right=0.98,
        top=0.93,
        bottom=0.15,
        hspace=0.05,
        wspace=0.02)
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    axs = [ax1, ax2, ax3]

    ##############################

    h = 0.5

    for i in range(initial_sample.shape[0]):

        for j in range(3):
            axs[j].plot(
                without_control[i, j, :],
                ts,
                [0.0]*len(ts),
                lw=.3,
                c='b')

            ########################################

            axs[j].plot(
                with_control[i, j, :],
                ts,
                [0.0]*len(ts),
                lw=.3,
                c='g')

            ########################################
            ########################################

            axs[j].plot(
                [with_control[i, j, -1]]*2,
                [ts[-1]]*2,
                [0.0, h],
                lw=1,
                c='g')

            axs[j].scatter(
                with_control[i, j, -1],
                ts[-1],
                h,
                c='g',
                s=50,
            )

            axs[j].plot(
                [without_control[i, j, -1]]*2,
                [ts[-1]]*2,
                [0.0, h],
                lw=1,
                c='b')

            axs[j].scatter(
                without_control[i, j, -1],
                ts[-1],
                h,
                c='b',
                s=50,
            )

            ########################################
            ########################################

            axs[j].plot(
                [with_control[i, j, 0]]*2,
                [ts[0]]*2,
                [0.0, h],
                lw=1,
                c='g')

            axs[j].scatter(
                with_control[i, j, 0],
                ts[0],
                h,
                c='g',
                s=50,
            )

            axs[j].plot(
                [without_control[i, j, 0]]*2,
                [ts[0]]*2,
                [0.0, h],
                lw=1,
                c='b')

            axs[j].scatter(
                without_control[i, j, 0],
                ts[0],
                h,
                c='b',
                s=50,
            )

    ##############################

    ax1.set_aspect('equal', 'box')
    ax2.set_aspect('equal', 'box')
    ax3.set_aspect('equal', 'box')

    ax1.set_zlim(-0.1, 2*h)
    ax2.set_zlim(-0.1, 2*h)
    ax3.set_zlim(-0.1, 2*h)

    ##############################

    s = args.control_data
    if s is None:
        s = "None"

    plt.suptitle("euler_maru g: with, b: without, T_0 %.3f, T_t %.3f, v_scale=%.2f\ncontrol_data=%s" % (
        T_0,
        T_t,
        args.v_scale,
        s,
    ))

    plt.show()
