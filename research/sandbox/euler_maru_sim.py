#!/usr/bin/env python3

import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

from common import *

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
        default=500,
        required=False)

    parser.add_argument('--control_data',
        type=str,
        default="",
        required=False)

    args = parser.parse_args()

    j1, j2, j3 = [float(x) for x in args.system.split(",")]

    d = 3

    t_span = (T_0, T_t)
    N = args.N

    a = 0.05

    control_data = None
    if len(args.control_data) > 0:
        control_data = np.load(
            args.control_data,
            allow_pickle=True).item()

        # import ipdb; ipdb.set_trace()

    initial_sample = np.random.multivariate_normal(
        np.array([mu_0]*d), np.eye(d)*0.1, 10) # 100 x 3

    ##############################

    with_control = np.empty(
        (
            initial_sample.shape[0],
            initial_sample.shape[1],
            N+1
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
            (j1, j2, j3, control_data))
        with_control[i, :, :] = tmp.T

    without_control = np.empty(
        (
            initial_sample.shape[0],
            initial_sample.shape[1],
            N+1
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
            (j1, j2, j3, None))
        without_control[i, :, :] = tmp.T

    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ##############################

    ax.scatter(
        with_control[:, 0, 0],
        with_control[:, 1, 0],
        with_control[:, 2, 0],
        c='r',
        marker='.')

    ##############################

    for i in range(initial_sample.shape[0]):
        # x[i] is sample [i]
        # y[i] is state dim [i]
        # z[i] is time [i]
        ax.plot(with_control[i, 0, :],
                with_control[i, 1, :],
                with_control[i, 2, :],
                alpha=a,
                c='g')

    ax.scatter(
        with_control[:, 0, -1],
        with_control[:, 1, -1],
        with_control[:, 2, -1],
        c='g',
        marker='.')

    ##############################

    for i in range(initial_sample.shape[0]):
        # x[i] is sample [i]
        # y[i] is state dim [i]
        # z[i] is time [i]
        ax.plot(without_control[i, 0, :],
                without_control[i, 1, :],
                without_control[i, 2, :],
                alpha=a,
                c='b')

    ax.scatter(
        without_control[:, 0, -1],
        without_control[:, 1, -1],
        without_control[:, 2, -1],
        c='b',
        marker='.')

    ##############################

    ax.set_title("euler_maru g: with, b: without, T_0 %.3f, T_t %.3f" % (T_0, T_t))

    ax.set_aspect('equal', 'box')

    plt.show()
