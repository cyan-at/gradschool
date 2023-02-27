#!/usr/bin/env python
# coding: utf-8

'''
USAGE:
./plot_bc.py --testdat ./wass_2d_model-99875-99875.dat --modelpt ./wass_2d_model-99875.pt
'''

# 0 define backend
import sys, os, time

# %env DDE_BACKEND=tensorflow.compat.v1
# %env XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/home/cyan3/miniforge/envs/tf

os.environ['DDE_BACKEND'] = "pytorch" # v2
os.environ['XLA_FLAGS'] = "--xla_gpu_cuda_data_dir=/usr/local/home/cyan3/miniforge/envs/tf"

# https://stackoverflow.com/questions/68614547/tensorflow-libdevice-not-found-why-is-it-not-found-in-the-searched-path
# this directory has /nvvm/libdevice/libdevice.10.bc

print(os.environ['DDE_BACKEND'])

import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.current_device())
torch.cuda.set_device(0)

# https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
try:
    torch.jit.enable_onednn_fusion(True)
except:
    print("no onednn")

cuda0 = torch.device('cuda:0')
cpu = torch.device('cpu')

import deepxde as dde
import numpy as np
from numpy import linalg as LA
import math

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm

class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)
    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)
        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)
        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)
def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''
    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)
setattr(Axes3D, 'arrow3D', _arrow3D)
def show_assignments_3d(state_a, state_b, P): 
    np_p = P.cpu().detach().numpy()
    norm_P = np_p/np_p.max()
    for i in range(state_a.shape[0]):
        for j in range(state_b.shape[0]):
            ax.arrow3D(
                state_a[i, 0], state_a[i, 1], z1,
                state_b[j, 0] - state_a[i, 0],
                state_b[j, 1] - state_a[i, 1],
                z2 - z1,
                alpha=norm_P[i,j].item(),
                mutation_scale=20,
                arrowstyle="-|>",
                linestyle='dashed')

from os.path import dirname, join as pjoin

from scipy import stats
import scipy.io
from scipy.stats import truncnorm, norm
from scipy.optimize import linprog
from scipy import sparse
from scipy.stats import multivariate_normal
from scipy.interpolate import griddata as gd

if dde.backend.backend_name == "pytorch":
    exp = dde.backend.torch.exp
else:
    from deepxde.backend import tf

    exp = tf.exp
    
import cvxpy as cp
import numpy as np

import argparse

from common import *

call_dir = os.getcwd()
sys.path.insert(0,call_dir)
print("expect train.py in %s" % (call_dir))
from train import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--modelpt',
        type=str, default='')
    parser.add_argument('--testdat',
        type=str, required=True)
    parser.add_argument('--plot',
        type=int, default=0)

    parser.add_argument('--diff_on_cpu',
        type=int, default=1)
    parser.add_argument('--fullstate',
        type=int, default=1)
    parser.add_argument('--interp_mode',
        type=str, default="nearest")
    parser.add_argument('--grid_n',
        type=int, default=30)
    parser.add_argument('--control_strategy',
        type=int,
        default=0)

    parser.add_argument('--optimizer',
        type=str, default="adam", help='')

    parser.add_argument('--mu_0',
        type=str, default="", help='')
    parser.add_argument('--mu_T',
        type=str, default="", help='')
    parser.add_argument('--T_t',
        type=str, default='', help='')
    parser.add_argument('--a', type=str, default="-1,1,2", help='')
    parser.add_argument('--N', type=int, default=15, help='')
    parser.add_argument('--state_bound_min', type=float, default=-5, help='')
    parser.add_argument('--state_bound_max', type=float, default=5, help='')

    parser.add_argument('--train_distribution', type=str, default="Hammersley", help='')
    parser.add_argument('--timemode', type=int, default=0, help='')
    # timemode  0 = linspace, 1 = even time samples
    parser.add_argument('--ni', type=int, default=-1, help='')
    parser.add_argument('--bif', type=int, default=1000, help='')
    parser.add_argument('--loss_func', type=str, default="wass3", help='')
    parser.add_argument('--pde_key', type=str, default="", help='')
    parser.add_argument('--batchsize',
        type=str,
        default="")
    parser.add_argument('--batchsize2',
        type=str,
        default="")

    parser.add_argument('--do_integration',
        type=int,
        default=1)

    parser.add_argument('--integrate_N',
        type=int,
        default=2000,
        required=False)
    parser.add_argument('--M',
        type=int,
        default=100,
        required=False)
    parser.add_argument('--workers',
        type=int,
        default=4)
    parser.add_argument('--noise',
        action='store_true')
    parser.add_argument('--v_scale',
        type=str,
        default="1.0")
    parser.add_argument('--bias',
        type=str,
        default="0.0")

    parser.add_argument('--headless',
        type=int,
        default=0)

    parser.add_argument('--plot_bc',
        type=int,
        default=0)

    args = parser.parse_args()

    if len(args.mu_T) > 0:
        mu_T = float(args.mu_T)
    if len(args.mu_0) > 0:
        mu_0 = float(args.mu_0)

    print("mu_0", mu_0)

    if len(args.T_t) > 0:
        T_t = float(args.T_t)
    print("T_t", T_t)

    ################################################

    N = args.N

    test = np.loadtxt(args.testdat)

    ################################################

    print("test.shape", test.shape)
    d = test.shape[1] - 1 - 2 # one for time, 2 for pinn output
    print("found %d dimension state space" % (d))
    M = N**d
    batchsize = M

    if len(args.batchsize) > 0:
        batchsize = int(args.batchsize)

    Ns = tuple([N]*d)

    ################################################

    model = None
    meshes = None
    if os.path.exists(args.modelpt):
        print("loading model")

        ni = initial_samples
        if args.ni >= 0:
            ni = args.ni

        model, meshes = get_model(
            d,
            N,
            batchsize,
            0,
            "tanh",
            mu_0,
            mu_T,
            T_t,
            args,
            [float(x) for x in args.a.split(',')],
            args.optimizer,
            train_distribution=args.train_distribution,
            timemode=args.timemode,
            ni=ni
            )
        model.restore(args.modelpt)
    else:
        print("no model")
        sys.exit(0)

    ########################################

    inputs = np.float32(test[:, :d+1])

    print("test.shape", test.shape)

    inputs = np.vstack((
        model.data.bc_points(),
        model.data.train_x_all
    ))

    print("test.shape", test.shape)

    # import ipdb; ipdb.set_trace()

    # test, rho0, rhoT, T_t, control_data,\
    #     t0s, tTs, tts, grids = make_control_data(
    #     model, inputs, N, d, meshes, args)

    test, T_t,\
    rho0, rhoT,\
    bc_grids, domain_grids, grid_n_meshes,\
    control_data = make_control_data(
        model, inputs, N, d, meshes, args)

    fname = '%s_%d_%d_%s_%d_%d_all_control_data.npy' % (
            args.modelpt.replace(".pt", ""),
            batchsize,
            args.diff_on_cpu,
            args.interp_mode,
            args.grid_n,
            T_t,
        )
    np.save(
        fname, 
        control_data
    )
    print("saved control_data to %s" % (fname))

    ########################################################

    # training specific volume / inverse of density: total state space / total sample points in that space
    v = (args.state_bound_max - args.state_bound_min) ** d
    v *= T_t
    m = args.bif
    print("specific volume", v / m)
    print("density", m / v)

    ########################################################

    fig = plt.figure()

    ax_count = 0
    if args.plot_bc > 0:
        ax_count += 2
    if args.do_integration > 0:
        ax_count += d

    ########################################################

    ax1 = fig.add_subplot(1, ax_count, 1, projection='3d')
    ax2 = fig.add_subplot(1, ax_count, 2, projection='3d')
    ax3 = fig.add_subplot(1, ax_count, 3, projection='3d')
    axs = [ax1, ax2, ax3]

    ax_i = 0

    z1 = T_0
    z2 = T_t
    p = 0.01

    ########################################################

    if d == 2 and args.plot_bc > 0:
        axs[ax_i].contourf(
            meshes[0],
            meshes[1],
            rho0.reshape(*Ns),
            50, zdir='z',
            cmap=cm.jet,
            offset=z1,
            alpha=0.4
        )

        axs[ax_i].contourf(
            meshes[0],
            meshes[1],
            rhoT.reshape(*Ns),
            50, zdir='z',
            cmap=cm.jet,
            offset=z2,
            alpha=0.4,
        )

        axs[ax_i].set_xlim(
            args.state_bound_min, args.state_bound_max)
        axs[ax_i].set_zlim(T_0 - 0.1, T_t + 0.1)

        axs[ax_i].set_xlabel('x')
        axs[ax_i].set_ylabel('y')
        axs[ax_i].set_zlabel('t')
        axs[ax_i].set_title('rho_opt')

        ax_i += 1

        ########################################################

        # import ipdb; ipdb.set_trace()

        for d_i in range(d):
            axs[ax_i].contourf(
                meshes[0],
                meshes[1],
                control_data['t0'][str(d_i)].reshape(N, N),
                50, zdir='z',
                cmap=cm.jet,
                offset=z1,
                alpha=0.4
            )

            axs[ax_i].contourf(
                meshes[0],
                meshes[1],
                control_data['tT'][str(d_i)].reshape(N, N),
                50, zdir='z',
                cmap=cm.jet,
                offset=z2,
                alpha=0.4,
            )

            # import ipdb; ipdb.set_trace()

            # tt control is to grid_n**d
            sc2=axs[ax_i].scatter(
                *grid_n_meshes,
                c=control_data['tt'][str(d_i)],
                s=np.abs(control_data['tt'][str(d_i)]*p),
                cmap=cm.jet,
                alpha=1.0)
            plt.colorbar(sc2, shrink=0.25)

            axs[ax_i].set_xlim(args.state_bound_min, args.state_bound_max)
            axs[ax_i].set_zlim(T_0 - 0.1, T_t + 0.1)

            axs[ax_i].set_xlabel('x')
            axs[ax_i].set_ylabel('y')
            axs[ax_i].set_zlabel('t')
            axs[ax_i].set_title('u' + str(d_i))

            ax_i += 1

    ########################################################

    if d == 3 and args.plot_bc > 0:
        # import ipdb; ipdb.set_trace()

        sc1=axs[ax_i].scatter(
            *meshes,
            c=rho0,
            s=np.abs(rho0*p),
            cmap=cm.jet,
            alpha=1.0)
        plt.colorbar(sc1, shrink=0.25)
        # axs[ax_i].set_title(
        #     'rho0:\nmu=%.3f\nsigma=%.3f\nsum=%.3f\nmin=%.3f\nmax=%.3f' % (
        #     mu_0,
        #     sigma_0,
        #     np.sum(t0[:, -1]),
        #     np.min(t0[:, -1]),
        #     np.max(t0[:, -1])
        # ))
        axs[ax_i].set_xlabel('x')
        axs[ax_i].set_ylabel('y')
        axs[ax_i].set_zlabel('z')

        ax_i += 1

        sc2=axs[ax_i].scatter(
            *meshes,
            c=rhoT,
            s=np.abs(rhoT*p),
            cmap=cm.jet,
            alpha=1.0)
        plt.colorbar(sc2, shrink=0.25)
        # axs[ax_i].set_title(
        #     'rhoT:\nmu=%.3f\nsigma=%.3f\nsum=%.3f\nmin=%.3f\nmax=%.3f' % (
        #     mu_T,
        #     sigma_T,
        #     np.sum(tT[:, -1]),
        #     np.min(tT[:, -1]),
        #     np.max(tT[:, -1])
        # ))
        axs[ax_i].set_xlabel('x')
        axs[ax_i].set_ylabel('y')
        axs[ax_i].set_zlabel('z')

        ax_i += 1

        # import ipdb; ipdb.set_trace()

        # tmp = gd(
        #   (t0[:, 0], t0[:, 1], t0[:, 2]),
        #   dphi_dinput_t0_dx,
        #   (grid_x1, grid_x2, grid_x3),
        #   method=args.interp_mode,
        #   fill_value=0.0)

        # sc3=axs[ax_i].scatter(
        #     grid_x1,
        #     grid_x2,
        #     grid_x3,
        #     c=tmp,
        #     s=np.abs(tmp*p),
        #     cmap=cm.jet,
        #     alpha=1.0)
        # plt.colorbar(sc3, shrink=0.25)
        # axs[ax_i].set_title('DPHI_DINPUT_tt_0, [0] range=%.3f, %.3f' % (np.min(tmp), np.max(tmp)))
        # axs[ax_i].set_xlabel('x')
        # axs[ax_i].set_ylabel('y')
        # axs[ax_i].set_zlabel('z')

    ########################################################

    title_str = args.modelpt
    title_str += "\n"
    # title_str += "N=15, d=%d, T_t=%.3f, batch=full, %s, mu_0=%.3f, mu_T=%.3f" % (
    #     d,
    #     T_t,
    #     "tanh",
    #     mu_0,
    #     mu_T,)

    # title_str += "\n"
    # title_str += "rho0_sum=%.3f, rhoT_sum=%.3f" % (
    #         np.sum(rho0),
    #         np.sum(rhoT),
    #     )

    # title_str += "\n"
    # title_str += "dphi_dinput_t0_dx={%.3f, %.3f}, dphi_dinput_t0_dy={%.3f, %.3f}" % (
    #         np.min(dphi_dinput_t0_dx),
    #         np.max(dphi_dinput_t0_dx),
    #         np.min(dphi_dinput_t0_dy),
    #         np.max(dphi_dinput_t0_dy)
    #     )

    # title_str += "\n"
    # title_str += "dphi_dinput_tT_dx={%.3f, %.3f}, dphi_dinput_tT_dy={%.3f, %.3f}" % (
    #         np.min(dphi_dinput_tT_dx),
    #         np.max(dphi_dinput_tT_dx),
    #         np.min(dphi_dinput_tT_dy),
    #         np.max(dphi_dinput_tT_dy)
    #     )

    plt.suptitle(title_str)

    manager = plt.get_current_fig_manager()

    c = Counter()
    fig.canvas.mpl_connect('key_press_event', lambda e: c.on_press_saveplot(e,
            '%s_Tt=%.3f_rho_opt_bc_batch=%d_%d_%s_%d' % (
                args.modelpt.replace(".pt", ""),
                T_t,
                batchsize,
                args.diff_on_cpu,
                args.interp_mode,
                args.grid_n
            )
        )
    )

    if args.do_integration > 0:
        print("T_t", T_t)

        ts, initial_sample, with_control, without_control,\
            all_results, mus, variances = do_integration(control_data, d, T_0, T_t, mu_0, sigma_0, args)

        if ax_i == d-1:
            axs.append(fig.add_subplot(1, ax_count, 4, projection='3d'))
            axs.append(fig.add_subplot(1, ax_count, 5, projection='3d'))

        h1 = 0.2
        h2 = 0.3
        b = -0.05

        for i in range(initial_sample.shape[0]):
            for j in range(d):
                axs[ax_i + j].plot(
                    without_control[i, j, :],
                    ts,
                    [0.0]*len(ts),
                    lw=.3,
                    c='b',
                    alpha=0.5)

                ########################################

                axs[ax_i + j].plot(
                    with_control[i, j, :],
                    ts,
                    [0.0]*len(ts),
                    lw=.3,
                    c='g',
                    alpha=0.5)

                ########################################
                ########################################

                axs[ax_i + j].plot(
                    [with_control[i, j, 0]]*2,
                    [ts[0]]*2,
                    [0.0, h1],
                    lw=1,
                    c='g')

                axs[ax_i + j].scatter(
                    with_control[i, j, 0],
                    ts[0],
                    h1,
                    c='g',
                    s=50,
                )

                axs[ax_i + j].plot(
                    [with_control[i, j, -1]]*2,
                    [ts[-1]]*2,
                    [0.0, h1],
                    lw=1,
                    c='g')

                axs[ax_i + j].scatter(
                    with_control[i, j, -1],
                    ts[-1],
                    h1,
                    c='g',
                    s=50,
                )

                ########################################
                ########################################

                axs[ax_i + j].plot(
                    [without_control[i, j, 0]]*2,
                    [ts[0]]*2,
                    [0.0, h2],
                    lw=1,
                    c='b')

                axs[ax_i + j].scatter(
                    without_control[i, j, 0],
                    ts[0],
                    h2,
                    c='b',
                    s=50,
                )

                axs[ax_i + j].plot(
                    [without_control[i, j, -1]]*2,
                    [ts[-1]]*2,
                    [0.0, h2],
                    lw=1,
                    c='b')

                axs[ax_i + j].scatter(
                    without_control[i, j, -1],
                    ts[-1],
                    h2,
                    c='b',
                    s=50,
                )

        ##############################

        for j in range(d):
            axs[ax_i + j].set_aspect('equal', 'box')
            axs[ax_i + j].set_zlim(b, 2*np.max([h1, h2]))
            axs[ax_i + j].set_title(
                'mu %.2f, var %.2f' % (mus[j], variances[j]))

        ##############################

    for ax in axs:
        ax.dist = 9 # default is 10

    if args.headless > 0:
        print("headless")
    else:
        plt.show()
