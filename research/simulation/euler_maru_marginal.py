#!/usr/bin/env python3

'''
USAGE:

./euler_maru_marginal.py --control_data ../training/exp8/model-100000_500_0_1_linear_30_all_control_data.npy --M 4 --v_scale 12000.0 --bias 0.0 --M 20
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

from concurrent.futures import ThreadPoolExecutor

def hash_func(v_scale, bias):
    return "%.3f_%.3f" % (v_scale, bias)

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
        type=str,
        default="1.0")
    parser.add_argument('--bias',
        type=str,
        default="0.0")

    parser.add_argument('--d',
        type=int,
        default=3)

    parser.add_argument('--workers',
        type=int,
        default=4)

    parser.add_argument('--search_grid',
        type=int,
        default=10)

    parser.add_argument('--headless',
        action='store_true') 

    args = parser.parse_args()

    v_scales = [float(x) for x in args.v_scale.split(",")]
    biases = [float(x) for x in args.bias.split(",")]

    if len(v_scales) > 1:
        v_scales = np.linspace(
            v_scales[0], v_scales[-1], args.search_grid)
    if len(biases) > 1:
        biases = np.linspace(
            biases[0], biases[-1], args.search_grid)

    j1, j2, j3 = [float(x) for x in args.system.split(",")]

    d = args.d

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
        np.array([mu_0]*d), np.eye(d)*sigma_0, args.M) # 100 x 3

    ##############################

    all_results = {}

    def task(i, target, control_data, affine):
        # print("starting {}".format(i))
        _, tmp = euler_maru(
            initial_sample[i, :],
            t_span,
            dynamics,
            (t_span[-1] - t_span[0])/(N),
            lambda delta_t: 0.0, # np.random.normal(loc=0.0, scale=np.sqrt(delta_t)),
            lambda y, t: 0.0, # 0.06,
            (
                j1, j2, j3,
                control_data,
                affine
            ))
        target[i, :, :] = tmp.T
        return i

    for vs in v_scales:
        for b in biases:
            if len(args.control_data) > 0:
                with_control_affine = lambda v: v * vs + b
                with_control = np.empty(
                    (
                        initial_sample.shape[0],
                        initial_sample.shape[1],
                        len(ts),
                    ))
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    results = executor.map(
                        task,
                        list(range(initial_sample.shape[0])),
                        [with_control]*initial_sample.shape[0],
                        [control_data]*initial_sample.shape[0],
                        [with_control_affine]*initial_sample.shape[0]
                    )
                    if len(v_scales) == 1:
                        for result in results:
                            print("done with {}".format(result))

                ##############################

                mus = np.zeros(args.d)
                variances = np.zeros(args.d)
                for j in range(args.d):
                    tmp = with_control[:, j, -1]
                    mus[j] = np.mean(tmp)
                    variances[j] = np.var(tmp)
                mu_s = "{}".format(mus)
                var_s = "{}".format(variances)
                print("vs %.3f, b %.3f" % (vs, b))
                print("mu_s", mu_s)
                print("var_s", var_s)

                all_results[hash_func(vs, b)] = [mus, variances]

                del with_control

            # for i in range(initial_sample.shape[0]):
            #     # x[i] is sample [i]
            #     # y[i] is state dim [i]
            #     # z[i] is time [i]
            #     print(i)
            #     _, tmp = euler_maru(
            #         initial_sample[i, :],
            #         t_span,
            #         dynamics,
            #         (t_span[-1] - t_span[0])/(N),
            #         lambda delta_t: 0.0, # np.random.normal(loc=0.0, scale=np.sqrt(delta_t)),
            #         lambda y, t: 0.0, # 0.06,
            #         (
            #             j1, j2, j3,
            #             control_data,
            #             lambda v: v * args.v_scale + args.bias
            #         ))
            #     with_control[i, :, :] = tmp.T

            '''
            without_control = np.empty(
                (
                    initial_sample.shape[0],
                    initial_sample.shape[1],
                    len(ts),
                ))
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                results = executor.map(
                    task,
                    list(range(initial_sample.shape[0])),
                    [without_control]*initial_sample.shape[0],
                    [None]*initial_sample.shape[0],
                    [None]*initial_sample.shape[0]
                )
                if len(v_scales) == 1:
                    for result in results:
                        print("done with {}".format(result))
            '''

            # for i in range(initial_sample.shape[0]):
            #     # x[i] is sample [i]
            #     # y[i] is state dim [i]
            #     # z[i] is time [i]
            #     print(i)
            #     _, tmp = euler_maru(
            #         initial_sample[i, :],
            #         t_span,
            #         dynamics,
            #         (t_span[-1] - t_span[0])/(N),
            #         lambda delta_t: 0.0, # np.random.normal(loc=0.0, scale=np.sqrt(delta_t)),
            #         lambda y, t: 0.0, # 0.06,
            #         (
            #             j1, j2, j3,
            #             None,
            #             None
            #         ))
            #     without_control[i, :, :] = tmp.T

    if len(args.control_data) > 0:
        np.save(
            "%s_all_results" % (args.control_data),
            all_results)
    else:
        print("no control data, no all_results")

    ##############################

    if args.headless:
        print("headless, no plot")
        sys.exit(0)

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
        for j in range(args.d):
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

            ########################################
            ########################################

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

    plt.suptitle("euler_maru g: with, b: without, T_0 %.3f, T_t %.3f, v_scale=%.2f, bias=%.2f, M=%d\ncontrol_data=%s\ncontrolled mu=%s, var=%s" % (
        T_0,
        T_t,
        v_scales[-1],
        biases[-1],
        args.M,
        s,
        mu_s,
        var_s,
    ))

    plt.show()
