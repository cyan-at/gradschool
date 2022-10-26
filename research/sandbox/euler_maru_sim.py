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

    control_data = None
    if len(args.control_data) > 0:
        control_data = np.load(
            args.control_data,
            allow_pickle=True).item()

    initial_sample = np.random.multivariate_normal(
        np.array([mu_0]*d), np.eye(d)*0.1, 100) # 100 x 3

    all_time_data = np.empty(
        (
            initial_sample.shape[0],
            initial_sample.shape[1],
            N+1
        ))

    for i in range(initial_sample.shape[0]):
        # x[i] is sample [i]
        # y[i] is state dim [i]
        # z[i] is time [i]
        _, tmp = euler_maru(
            initial_sample[i, :],
            t_span,
            dynamics,
            (t_span[-1] - t_span[0])/(N),
            lambda delta_t: 0.0, # np.random.normal(loc=0.0, scale=np.sqrt(delta_t)),
            lambda y, t: 0.0, # 0.06,
            (j1, j2, j3, {}))
        all_time_data[i, :, :] = tmp.T

    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1, projection='3d')

    for i in range(initial_sample.shape[0]):
        # x[i] is sample [i]
        # y[i] is state dim [i]
        # z[i] is time [i]
        ax.plot(all_time_data[i, 0, :],
                all_time_data[i, 1, :],
                all_time_data[i, 2, :],
                alpha=0.1)

    ax.scatter(
        all_time_data[:, 0, 0],
        all_time_data[:, 1, 0],
        all_time_data[:, 2, 0],
        c='r',
        marker='.')

    ax.scatter(
        all_time_data[:, 0, -1],
        all_time_data[:, 1, -1],
        all_time_data[:, 2, -1],
        c='b',
        marker='.')

    ax.set_title("euler_maru")

    ax.set_aspect('equal', 'box')

    plt.show()
