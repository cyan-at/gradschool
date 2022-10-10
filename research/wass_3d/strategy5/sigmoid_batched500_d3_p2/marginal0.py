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

from distribution0 import *

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
        default=3.0,
        required=False)

    parser.add_argument('--sampling',
        type=str,
        default="15,15,15,15,15,15,100,100",
        required=False)

    parser.add_argument('--system',
        type=str,
        default="1,1,2",
        required=False)

    parser.add_argument('--ignore_symmetry',
        type=int,
        default=0,
        required=False)

    parser.add_argument('--control_prefix',
        type=str,
        default="run_tensorflow_-2.500_2.500__5.000__2000__0.001__1_1_2__0.000__2.000_1.000__0.000_1.000__12000_1000__20000",
        required=False)

    parser.add_argument('--plot',
        type=int,
        default=1,
        required=False)

    args = parser.parse_args()

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

    j1, j2, j3 = [float(x) for x in args.system.split(",")]

    #############################################################################

    control_data = None
    if len(args.control_prefix) > 0:
        vinterp_N = 10
        vinterp_T = 20
        state_min = -2.5
        state_max = 2.5
        T_t = 5.0

        t0_v_mat_fname = '%s/%s/%s_post_predict_t0_%d_%d_v.mat' % (
            os.path.abspath("./"), args.control_prefix, args.control_prefix, vinterp_N, vinterp_T)

        mid_v_mat_fname = '%s/%s/%s_post_predict_mid_%d_%d_v.mat' % (
            os.path.abspath("./"), args.control_prefix, args.control_prefix, vinterp_N, vinterp_T)

        t5_v_mat_fname = '%s/%s/%s_post_predict_t5_%d_%d_v.mat' % (
            os.path.abspath("./"), args.control_prefix, args.control_prefix, vinterp_N, vinterp_T)

        x_1_ = np.linspace(state_min, state_max, vinterp_N)
        x_2_ = np.linspace(state_min, state_max, vinterp_N)
        x_3_ = np.linspace(state_min, state_max, vinterp_N)
        t_ = np.linspace(0, T_t, vinterp_T)
        # t5_v_mat_fname = '%s/%s/%s_post_predict_x1_x2_x3_t.mat' % (
        #     os.path.abspath("./"), args.control_prefix, args.control_prefix)

        if os.path.exists(t0_v_mat_fname) and os.path.exists(mid_v_mat_fname) and os.path.exists(t5_v_mat_fname):
            control_data = {
                "x_1_" : x_1_,
                "x_2_" : x_2_,
                "x_3_" : x_3_,
                "t_" : t_,
            }

            mat_contents = scipy.io.loadmat(t0_v_mat_fname)
            for k in ["t0_V1", "t0_V2", "t0_V3"]:
                control_data[k] = mat_contents[k]
            del mat_contents

            mat_contents = scipy.io.loadmat(mid_v_mat_fname)
            for k in ["mid_V1", "mid_V2", "mid_V3"]:
                control_data[k] = mat_contents[k]
            del mat_contents

            mat_contents = scipy.io.loadmat(t5_v_mat_fname)
            for k in ["t5_V1", "t5_V2", "t5_V3"]:
                control_data[k] = mat_contents[k]
            del mat_contents
        else:
            print("missing one of the control v files")

    #############################################################################

    initial_sample, te_to_data, X1, X2, X3 = init_data(
        mu_0, cov_0,
        windows, distribution_samples, N, ts,
        j1, j2, j3,
        control_data,
        args.ignore_symmetry)

    x1 = X1[0, :, 0]
    x2 = X2[:, 0, 0]
    x3 = X3[0, 0, :]

    #############################################################################

    cloud_t_e = np.empty((distribution_samples, 3, len(ts)))
    # n x 3 slices of end points at time t_e
    # one layer per t_e
    # xi state points at time t_e is
    # cloud_t_e[:, i, t_e]

    for t_e_i, t_e in enumerate(ts):
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

        # import ipdb; ipdb.set_trace();

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

        # normalize all the pdfs so area under curve ~= 1.0
        x1_pdf_area = np.trapz(te_to_data[t_e]["x1_marginal"], x=x1)
        x2_pdf_area = np.trapz(te_to_data[t_e]["x2_marginal"], x=x2)
        x3_pdf_area = np.trapz(te_to_data[t_e]["x3_marginal"], x=x3)
        print("prior to normalization: %.2f, %.2f, %.2f" % (
            x1_pdf_area,
            x2_pdf_area,
            x3_pdf_area))

        te_to_data[t_e]["x1_marginal"] /= x1_pdf_area
        te_to_data[t_e]["x2_marginal"] /= x2_pdf_area
        te_to_data[t_e]["x3_marginal"] /= x3_pdf_area

        x1_pdf_area = np.trapz(te_to_data[t_e]["x1_marginal"], x=x1)
        x2_pdf_area = np.trapz(te_to_data[t_e]["x2_marginal"], x=x2)
        x3_pdf_area = np.trapz(te_to_data[t_e]["x3_marginal"], x=x3)
        print("after to normalization: %.2f, %.2f, %.2f" % (
            x1_pdf_area,
            x2_pdf_area,
            x3_pdf_area))

        # swap x1 and x2, reflect assignment swap
        # to undo the swapping for plotting
        tmp = te_to_data[t_e]["x2_marginal"]
        te_to_data[t_e]["x2_marginal"] = te_to_data[t_e]["x1_marginal"]
        te_to_data[t_e]["x1_marginal"] = tmp

        if te_to_data[t_e]["all_time_data"] is not None:
            cloud_t_e[:, :3, t_e_i] = te_to_data[t_e]["all_time_data"][:, :3, -1]
        else:
            cloud_t_e[:, :3, t_e_i] = initial_sample[:, :3]

    #############################################################################

    if args.plot == 0:
        fig = plt.figure(1)
        ax1 = plt.subplot(121, frameon=False)
        # ax1.set_aspect('equal')
        ax1.grid()

        ax2 = plt.subplot(122, frameon=False)
        # ax2.set_aspect('equal')
        ax2.grid()

        # ax3 = plt.subplot(133, frameon=False)

        fig2 = plt.figure(2)
        ax3 = plt.subplot(111, frameon=False)
        # ax3.set_aspect('equal')
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
    elif args.plot == 1:
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

        # fig2 = plt.figure(2,
        #     figsize=params["figure.figsize"]) 
        # # figsize accepts only inches.
        # fig2.subplots_adjust(
        #     left=0.04,
        #     right=0.98,
        #     top=0.93,
        #     bottom=0.15,
        #     hspace=0.05,
        #     wspace=0.02)

        ax3 = fig.add_subplot(1, 3, 3, projection='3d')

        for i, t_e in enumerate(ts):
            c1 = 'k'
            c2 = 'gray'
            alpha = 0.3
            if (i == 0):
                c1 = 'g'
                c2 = 'green'
                alpha = 0.15
            elif (i == len(ts) - 1):
                c1 = 'r'
                c2 = 'red'
                alpha = 0.15

            ax1.plot(
                x1,
                t_e*np.ones((len(x1),1)),
                te_to_data[t_e]["x1_marginal"],
                color=c1,
                lw=1)
            ax1.add_collection3d(
                ax1.fill_between(
                    x1,
                    te_to_data[t_e]["x1_marginal"],
                    color=c2,
                    alpha=alpha),
                zs=t_e, zdir='y')

            ax1.set_title('x1')

            ax2.plot(
                x2,
                t_e*np.ones((len(x2),1)),
                te_to_data[t_e]["x2_marginal"],
                color=c1,
                lw=1)
            ax2.add_collection3d(
                ax2.fill_between(
                    x2,
                    te_to_data[t_e]["x2_marginal"],
                    color=c2,
                    alpha=alpha),
                zs=t_e, zdir='y')

            ax2.set_title('x2')

            ax3.plot(
                x3,
                t_e*np.ones((len(x3),1)),
                te_to_data[t_e]["x3_marginal"],
                color=c1,
                lw=1)
            ax3.add_collection3d(
                ax3.fill_between(
                    x3,
                    te_to_data[t_e]["x3_marginal"],
                    color=c2,
                    alpha=alpha),
                zs=t_e, zdir='y')

            ax3.set_title('x3')

            fig.suptitle('j1=%.3f, j2=%.3f, j3=%.3f' % (j1, j2, j3))

        for i in range(distribution_samples):
            ax1.plot(
                cloud_t_e[i, 0, :],
                ts,
                [0.0]*len(ts),
                lw=.3)

            ax2.plot(
                cloud_t_e[i, 1, :],
                ts,
                [0.0]*len(ts),
                lw=.3)

            ax3.plot(
                cloud_t_e[i, 2, :],
                ts,
                [0.0]*len(ts),
                lw=.3)

    plt.show()
