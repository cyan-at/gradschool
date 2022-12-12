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
        type=str, default="linear")
    parser.add_argument('--grid_n',
        type=int, default=30)
    parser.add_argument('--optimizer',
        type=str, default="adam", help='')

    parser.add_argument('--mu_0',
        type=str, default="", help='')
    parser.add_argument('--mu_T',
        type=str, default="", help='')
    parser.add_argument('--system',
        type=str,
        default="1,1,2", # 3,2,1
        required=False)
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

    args = parser.parse_args()

    test = np.loadtxt(args.testdat)

    d = test.shape[1] - 1 - 2 # one for time, 2 for pinn output
    print("found %d dimension state space" % (d))

    if os.path.exists(args.modelpt):
        print("loading model")

        N = 15;
        M = N**d
        batchsize = M

        # instead of using test output, use model and 
        # generate / predict a new output
        inputs = test[:, :d+1]

        T_t = inputs[batchsize, -1]
        print("found T_t", T_t)

        if len(args.mu_T) > 0:
            mu_T = float(args.mu_T)
        if len(args.mu_0) > 0:
            mu_0 = float(args.mu_0)

        inputs = np.float32(inputs)

        activation = "tanh"

        model, meshes = get_model(
            d,
            N,
            0,
            activation,
            mu_0,
            mu_T,
            T_t,
            args.optimizer,
            )
        model.restore(args.modelpt)

        # output = model.predict(inputs)

        inputs_tensor = torch.from_numpy(
            inputs).requires_grad_(True)

        if args.diff_on_cpu == 0:
            print("moving input to cuda")
            inputs_tensor = inputs_tensor.type(torch.FloatTensor).to(cuda0).requires_grad_(True)
        else:
            print("keeping input on cpu")

        # move the MODEL to the cpu
        # to compute the gradient there, not on CUDA
        # because input to cuda makes it non-leaf
        # so it does not catch backward()'d backprop'd gradient
        if args.diff_on_cpu > 0:
            model.net = model.net.cpu()
        else:
            print("keeping model on cuda")

        # import ipdb; ipdb.set_trace()

        output_tensor = model.net(inputs_tensor)

        # only possible if tensors on cpu
        # maybe moving to cuda makes input non-leaf
        if args.diff_on_cpu > 0:
            output_tensor[:, 0].backward(torch.ones_like(output_tensor[:, 0]))
            dphi_dinput = inputs_tensor.grad
        else:
            # OR do grad like so
            dphi_dinput = torch.autograd.grad(outputs=output_tensor[:, 0], inputs=inputs_tensor, grad_outputs=torch.ones_like(output_tensor[:, 0]))[0]

        if args.diff_on_cpu > 0:
            dphi_dinput = dphi_dinput.numpy()
        else:
            print("moving dphi_dinput off cuda")
            dphi_dinput = dphi_dinput.cpu().numpy()

        # import ipdb; ipdb.set_trace()

        dphi_dinput_fname = args.modelpt.replace(".pt", "_dphi_dinput_%d_%d_%.3f.txt" % (
            args.diff_on_cpu,
            batchsize,
            T_t,
        ))
        np.savetxt(
            dphi_dinput_fname,
            dphi_dinput)

        if args.diff_on_cpu > 0:
            output = output_tensor.detach().numpy()
        else:
            print("moving output off cuda")
            output = output_tensor.detach().cpu().numpy()

        test = np.hstack((inputs, output))
    else:
        print("no model, using test dat alone")

    ########################################################

    t0 = test[:batchsize, :]
    tT = test[batchsize:2*batchsize, :]
    tt = test[2*batchsize:, :]

    ########################################################

    rho0 = t0[:, -1]
    rhoT = tT[:, -1]

    x_1_ = np.linspace(state_min, state_max, args.grid_n)
    x_2_ = np.linspace(state_min, state_max, args.grid_n)
    x_3_ = np.linspace(state_min, state_max, args.grid_n)
    t_ = np.linspace(T_0, T_t, args.grid_n*2)

    ########################################################

    fig = plt.figure()

    ########################################################

    ax1 = fig.add_subplot(2, 3, 1, projection='3d')

    z1 = T_0
    z2 = T_t

    ax1.contourf(
        meshes[0],
        meshes[1],
        rho0.reshape(N, N),
        50, zdir='z',
        cmap=cm.jet,
        offset=z1,
        alpha=0.4
    )

    ax1.contourf(
        meshes[0],
        meshes[1],
        rhoT.reshape(N, N),
        50, zdir='z',
        cmap=cm.jet,
        offset=z2,
        alpha=0.4,
    )

    ax1.set_xlim(state_min, state_max)
    ax1.set_zlim(T_0 - 0.1, T_t + 0.1)

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('t')
    ax1.set_title('rho_opt')

    ########################################################

    dphi_dinput_t0 = dphi_dinput[:batchsize, :]
    dphi_dinput_tT = dphi_dinput[batchsize:2*batchsize, :]
    dphi_dinput_tt = dphi_dinput[2*batchsize:, :]
    print(
        np.max(dphi_dinput_t0),
        np.max(dphi_dinput_tT),
        np.max(dphi_dinput_tt)
    )

    ########################################################

    meshes[0].reshape(-1)
    meshes[1].reshape(-1)
    grid0 = np.array((
        meshes[0].reshape(-1),
        meshes[1].reshape(-1),
    )).T

    dphi_dinput_t0_dx = dphi_dinput_t0[:, 0]
    dphi_dinput_t0_dy = dphi_dinput_t0[:, 1]

    t0={
        '0': dphi_dinput_t0_dx.reshape(-1),
        '1': dphi_dinput_t0_dy.reshape(-1),
        'grid' : grid0,
    }

    dphi_dinput_tT_dx = dphi_dinput_tT[:, 0]
    dphi_dinput_tT_dy = dphi_dinput_tT[:, 1]

    tT={
        '0': dphi_dinput_tT_dx.reshape(-1),
        '1': dphi_dinput_tT_dy.reshape(-1),
        'grid' : grid0,
    }

    x_1_ = np.linspace(state_min, state_max, args.grid_n)
    x_2_ = np.linspace(state_min, state_max, args.grid_n)
    t_ = np.linspace(T_0, T_t, args.grid_n * 2)
    grid_x1, grid_x2, grid_t = np.meshgrid(
        x_1_,
        x_2_,
        t_, copy=False) # each is NxNxN

    grid1 = np.array((
        grid_x1.reshape(-1),
        grid_x2.reshape(-1),
        grid_t.reshape(-1),
    )).T

    # import ipdb; ipdb.set_trace()
    DPHI_DINPUT_tt_0 = gd(
      (tt[:, 0], tt[:, 1], tt[:, 2]),
      dphi_dinput_tt[:, 0],
      (grid_x1, grid_x2, grid_t),
      method=args.interp_mode)

    DPHI_DINPUT_tt_1 = gd(
      (tt[:, 0], tt[:, 1], tt[:, 2]),
      dphi_dinput_tt[:, 1],
      (grid_x1, grid_x2, grid_t),
      method=args.interp_mode)

    # import ipdb; ipdb.set_trace()

    print("# DPHI_DINPUT_tt_0 nans:", np.count_nonzero(np.isnan(DPHI_DINPUT_tt_0)), DPHI_DINPUT_tt_0.size)
    print("# DPHI_DINPUT_tt_1 nans:", np.count_nonzero(np.isnan(DPHI_DINPUT_tt_1)), DPHI_DINPUT_tt_1.size)

    DPHI_DINPUT_tt_0 = np.nan_to_num(DPHI_DINPUT_tt_0)
    DPHI_DINPUT_tt_1 = np.nan_to_num(DPHI_DINPUT_tt_1)

    tt={
        '0': DPHI_DINPUT_tt_0.reshape(-1),
        '1': DPHI_DINPUT_tt_1.reshape(-1),
        'grid' : grid1,
    }

    ########################################################

    ax2 = fig.add_subplot(2, 3, 2, projection='3d')

    z1 = T_0
    z2 = T_t

    p = 1.

    ax2.contourf(
        meshes[0],
        meshes[1],
        dphi_dinput_t0_dx.reshape(N, N),
        50, zdir='z',
        cmap=cm.jet,
        offset=z1,
        alpha=0.4
    )

    ax2.contourf(
        meshes[0],
        meshes[1],
        dphi_dinput_tT_dx.reshape(N, N),
        50, zdir='z',
        cmap=cm.jet,
        offset=z2,
        alpha=0.4,
    )

    sc2=ax2.scatter(
        grid_x1,
        grid_x2,
        grid_t,
        c=DPHI_DINPUT_tt_0,
        s=np.abs(DPHI_DINPUT_tt_0*p),
        cmap=cm.jet,
        alpha=1.0)
    plt.colorbar(sc2, shrink=0.25)

    ax2.set_xlim(state_min, state_max)
    ax2.set_zlim(T_0 - 0.1, T_t + 0.1)

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('t')
    ax2.set_title('dphi_dx')

    ########################################################

    ax3 = fig.add_subplot(2, 3, 3, projection='3d')

    z1 = T_0
    z2 = T_t

    ax3.contourf(
        meshes[0],
        meshes[1],
        dphi_dinput_t0_dy.reshape(N, N),
        50, zdir='z',
        cmap=cm.jet,
        offset=z1,
        alpha=0.4
    )

    ax3.contourf(
        meshes[0],
        meshes[1],
        dphi_dinput_tT_dy.reshape(N, N),
        50, zdir='z',
        cmap=cm.jet,
        offset=z2,
        alpha=0.4,
    )

    sc3=ax3.scatter(
        grid_x1,
        grid_x2,
        grid_t,
        c=DPHI_DINPUT_tt_1,
        s=np.abs(DPHI_DINPUT_tt_1*p),
        cmap=cm.jet,
        alpha=1.0)
    plt.colorbar(sc3, shrink=0.25)

    ax3.set_xlim(state_min, state_max)
    ax3.set_zlim(T_0 - 0.1, T_t + 0.1)

    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('t')
    ax3.set_title('dphi_dy')

    ########################################################

    title_str = args.modelpt
    title_str += "\n"
    title_str += "N=15, d=%d, T_t=%.3f, batch=full, %s, mu_0=%.3f, mu_T=%.3f" % (
        d,
        T_t,
        activation,
        mu_0,
        mu_T,)

    title_str += "\n"
    title_str += "rho0_sum=%.3f, rhoT_sum=%.3f" % (
            np.sum(rho0),
            np.sum(rhoT),
        )

    title_str += "\n"
    title_str += "dphi_dinput_t0_dx={%.3f, %.3f}, dphi_dinput_t0_dy={%.3f, %.3f}" % (
            np.min(dphi_dinput_t0_dx),
            np.max(dphi_dinput_t0_dx),
            np.min(dphi_dinput_t0_dy),
            np.max(dphi_dinput_t0_dy)
        )

    title_str += "\n"
    title_str += "dphi_dinput_tT_dx={%.3f, %.3f}, dphi_dinput_tT_dy={%.3f, %.3f}" % (
            np.min(dphi_dinput_tT_dx),
            np.max(dphi_dinput_tT_dx),
            np.min(dphi_dinput_tT_dy),
            np.max(dphi_dinput_tT_dy)
        )

    plt.suptitle(title_str)

    # Option 1
    # QT backend
    manager = plt.get_current_fig_manager()

    # try:
    #     manager.window.showMaximized()
    # except:
    #     pass

    # # Option 2
    # # TkAgg backend
    # try:
    #     manager.resize(*manager.window.maxsize())
    # except:
    #     pass

    # # Option 3
    # # WX backend
    # try:
    #     manager.frame.Maximize(True)
    # except:
    #     pass
    c = Counter()
    fig.canvas.mpl_connect('key_press_event', lambda e: c.on_press_saveplot(e,
            '%s_Tt=%.3f_rho_opt_bc_batch=%d_%d_%s_%d.png' % (
                args.modelpt.replace(".pt", ""),
                T_t,
                batchsize,
                args.diff_on_cpu,
                args.interp_mode,
                args.grid_n
            )
        )
    )

    gen_control_data = input("generate control data? ")
    print(gen_control_data)

    if gen_control_data != "1":
        sys.exit(0)

    fname = '%s_%d_%d_%s_%d_%d_all_control_data.npy' % (
            args.modelpt.replace(".pt", ""),
            batchsize,
            args.diff_on_cpu,
            args.interp_mode,
            args.grid_n,
            T_t,
        )
    control_data = {
            't0' : t0,
            'tT' : tT,
            'tt' : tt
        }
    np.save(
        fname, 
        control_data
    )
    print("saved control_data to %s" % (fname))

    n = input("run marginal? ")
    print(n)

    if n != "1":
        sys.exit(0)

    ##############################

    t_span = (T_0, T_t)
    dt = (t_span[-1] - t_span[0])/(args.integrate_N)
    ts = np.arange(t_span[0], t_span[1] + dt, dt)

    initial_sample = np.random.multivariate_normal(
        np.array([mu_0]*d), np.eye(d)*sigma_0, args.M) # 100 x 3

    v_scales = [float(x) for x in args.v_scale.split(",")]
    biases = [float(x) for x in args.bias.split(",")]

    ##############################

    all_results = {}

    t_span = (T_0, T_t)

    integrator = Integrator(initial_sample, t_span, args, dynamics)

    without_control = np.empty(
        (
            initial_sample.shape[0],
            initial_sample.shape[1],
            len(ts),
        ))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = executor.map(
            integrator.task,
            list(range(initial_sample.shape[0])),
            [without_control]*initial_sample.shape[0],
            [None]*initial_sample.shape[0],
            [None]*initial_sample.shape[0]
        )
        if len(v_scales) == 1:
            for result in results:
                print("done with {}".format(result))

    mus = np.zeros(d)
    variances = np.zeros(d)

    for vs in v_scales:
        for b in biases:
            with_control_affine = lambda v: v * vs + b
            with_control = np.empty(
                (
                    initial_sample.shape[0],
                    initial_sample.shape[1],
                    len(ts),
                ))
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                results = executor.map(
                    integrator.task,
                    list(range(initial_sample.shape[0])),
                    [with_control]*initial_sample.shape[0],
                    [control_data]*initial_sample.shape[0],
                    [with_control_affine]*initial_sample.shape[0]
                )
                if len(v_scales) == 1:
                    for result in results:
                        print("done with {}".format(result))

            ##############################

            for j in range(d):
                tmp = with_control[:, j, -1]
                mus[j] = np.mean(tmp)
                variances[j] = np.var(tmp)
            mu_s = "{}".format(mus)
            var_s = "{}".format(variances)
            print("vs %.3f, b %.3f" % (vs, b))
            print("mu_s", mu_s)
            print("var_s", var_s)

            all_results[hash_func(vs, b)] = [mus, variances]

            if len(v_scales) > 1:
                del with_control

    ax1 = fig.add_subplot(2, 3, 4, projection='3d')
    ax2 = fig.add_subplot(2, 3, 5, projection='3d')
    axs = [ax1, ax2]

    h = 0.5
    b = -0.05

    for i in range(initial_sample.shape[0]):
        for j in range(d):
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

    for j, ax in enumerate(axs):
        ax.set_aspect('equal', 'box')
        ax.set_zlim(b, 2*h)
        ax.set_title('mu %.2f, var %.2f' % (mus[j], variances[j]))

    plt.show()
