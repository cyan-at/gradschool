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

# ###patch start###
# from mpl_toolkits.mplot3d.axis3d import Axis
# if not hasattr(Axis, "_get_coord_info_old"):
#     def _get_coord_info_new(self, renderer):
#         mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
#         mins += deltas / 4
#         maxs -= deltas / 4
#         return mins, maxs, centers, deltas, tc, highs
#     Axis._get_coord_info_old = Axis._get_coord_info  
#     Axis._get_coord_info = _get_coord_info_new
# ###patch end###

def make_control_data(model, inputs, N, d, meshes, args):
    M = N**d
    batchsize = M

    if len(args.batchsize) > 0:
        batchsize = int(args.batchsize)
    print("batchsize", batchsize, "inputs.shape", inputs.shape)

    T_t = inputs[batchsize, -1]
    print("found T_t", T_t)

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

    output_tensor = model.net(inputs_tensor)

    ################################################

    if args.diff_on_cpu > 0:
        output = output_tensor.detach().numpy()
    else:
        print("moving output off cuda")
        output = output_tensor.detach().cpu().numpy()

    test = np.hstack((inputs, output))

    t0 = test[:batchsize, :]
    tT = test[batchsize:2*batchsize, :]
    tt = test[2*batchsize:, :]

    ################################################

    t0_u = test[:batchsize, inputs.shape[1] + 3 - 1:inputs.shape[1] + 5 - 1]
    tT_u = test[batchsize:2*batchsize, inputs.shape[1] + 3 - 1:inputs.shape[1] + 5 - 1]
    tt_u = test[2*batchsize:, inputs.shape[1] + 3 - 1:inputs.shape[1] + 5 - 1]

    ################################################

    rho0 = t0[:, inputs.shape[1] + 2 - 1]
    rhoT = tT[:, inputs.shape[1] + 2 - 1]

    ################################################

    x_1_ = np.linspace(args.state_bound_min, args.state_bound_max, args.grid_n)
    x_2_ = np.linspace(args.state_bound_min, args.state_bound_max, args.grid_n)
    x_3_ = np.linspace(args.state_bound_min, args.state_bound_max, args.grid_n)
    t_ = np.linspace(T_0, T_t, args.grid_n*2)

    ########################################################

    if d == 2:
        grid0 = np.array((
            meshes[0].reshape(-1),
            meshes[1].reshape(-1),
        )).T
    elif d == 3:
        grid0 = np.array((
            meshes[0].reshape(-1),
            meshes[1].reshape(-1),
            meshes[2].reshape(-1),
        )).T

    ########################################################

    t0_u1 = t0_u[:, 0]
    t0_u2 = t0_u[:, 1]

    tT_u1 = tT_u[:, 0]
    tT_u2 = tT_u[:, 1]

    tt_u1 = tt_u[:, 0]
    tt_u2 = tt_u[:, 1]

    t0_u3 = None
    tT_u3 = None
    if d == 3:
        t0_u3 = dphi_dinput_t0[:, 2]
        tT_u3 = dphi_dinput_tT[:, 2]
        tt_u3 = dphi_dinput_tt[:, 2]

    if len(args.batchsize) == 0:
        t0={
            '0': t0_u1.reshape(-1),
            '1': t0_u2.reshape(-1),
            'grid' : grid0,
        }

        tT={
            '0': tT_u1.reshape(-1),
            '1': tT_u2.reshape(-1),
            'grid' : grid0,
        }

        if d == 3:
            t0['2'] = t0_u3.reshape(-1)
            tT['2'] = tT_u3.reshape(-1)
    else:
        print("interpolating t0 and tt also since batchsize is not enough")

        grid_x1, grid_x2, grid_x3 = np.meshgrid(
            x_1_,
            x_2_,
            x_3_, copy=False) # each is NxNxN

        t0_u1 = gd(
          (t0[:, 0], t0[:, 1], t0[:, 2]),
          t0_u1,
          (grid_x1, grid_x2, grid_x3),
          method=args.interp_mode)

        t0_u2 = gd(
          (t0[:, 0], t0[:, 1], t0[:, 2]),
          t0_u2,
          (grid_x1, grid_x2, grid_x3),
          method=args.interp_mode)

        if t0_u3 is not None:
            t0_u3 = gd(
              (t0[:, 0], t0[:, 1], t0[:, 2]),
              t0_u3,
              (grid_x1, grid_x2, grid_x3),
              method=args.interp_mode)

        t0={
            '0': t0_u1.reshape(-1),
            '1': t0_u2.reshape(-1),
            'grid' : grid0,
        }

        ##########################

        tT_u1 = gd(
          (tT[:, 0], tT[:, 1], tT[:, 2]),
          tT_u1,
          (grid_x1, grid_x2, grid_x3),
          method=args.interp_mode)

        tT_u2 = gd(
          (tT[:, 0], tT[:, 1], tT[:, 2]),
          tT_u2,
          (grid_x1, grid_x2, grid_x3),
          method=args.interp_mode)

        if tT_u3 is not None:
            tT_u3 = gd(
              (tT[:, 0], tT[:, 1], tT[:, 2]),
              tT_u3,
              (grid_x1, grid_x2, grid_x3),
              method=args.interp_mode)

        tT={
            '0': tT_u1.reshape(-1),
            '1': tT_u2.reshape(-1),
            'grid' : grid0,
        }

        ##########################

        if d == 3:
            t0['2'] = t0_u3.reshape(-1)
            tT['2'] = tT_u3.reshape(-1)

    ###########################

    grid_x3 = None
    if d == 2:
        grid_x1, grid_x2, grid_t = np.meshgrid(
            x_1_,
            x_2_,
            t_, copy=False) # each is NxNxN

        grid1 = np.array((
            grid_x1.reshape(-1),
            grid_x2.reshape(-1),
            grid_t.reshape(-1),
        )).T
    elif d == 3:
        grid_x1, grid_x2, grid_x3, grid_t = np.meshgrid(
            x_1_,
            x_2_,
            x_3_,
            t_, copy=False) # each is NxNxN

        grid1 = np.array((
            grid_x1.reshape(-1),
            grid_x2.reshape(-1),
            grid_x3.reshape(-1),
            grid_t.reshape(-1),
        )).T

    ###########################

    tt_U3 = None
    if d == 2:
        # import ipdb; ipdb.set_trace()
        tt_U1 = gd(
          (tt[:, 0], tt[:, 1], tt[:, 2]),
          tt_u1,
          (grid_x1, grid_x2, grid_t),
          method=args.interp_mode)

        tt_U2 = gd(
          (tt[:, 0], tt[:, 1], tt[:, 2]),
          tt_u2,
          (grid_x1, grid_x2, grid_t),
          method=args.interp_mode)

        # import ipdb; ipdb.set_trace()

        print("# tt_U1 nans:", np.count_nonzero(np.isnan(tt_U1)), tt_U1.size)
        print("# tt_U2 nans:", np.count_nonzero(np.isnan(tt_U2)), tt_U2.size)

        tt_U1 = np.nan_to_num(tt_U1)
        tt_U2 = np.nan_to_num(tt_U2)

        tt={
            '0': tt_U1.reshape(-1),
            '1': tt_U2.reshape(-1),
            'grid' : grid1,
        }
    elif d == 3:
        tt_U1 = gd(
          (tt[:, 0], tt[:, 1], tt[:, 2], tt[:, 3]),
          tt_u1,
          (grid_x1, grid_x2, grid_x3, grid_t),
          method=args.interp_mode)

        tt_U2 = gd(
          (tt[:, 0], tt[:, 1], tt[:, 2], tt[:, 3]),
          tt_u2,
          (grid_x1, grid_x2, grid_x3, grid_t),
          method=args.interp_mode)

        tt_U3 = gd(
          (tt[:, 0], tt[:, 1], tt[:, 2], tt[:, 3]),
          tt_u3,
          (grid_x1, grid_x2, grid_x3, grid_t),
          method=args.interp_mode)

        print("# tt_U1 nans:", np.count_nonzero(np.isnan(tt_U1)), tt_U1.size)
        print("# tt_U2 nans:", np.count_nonzero(np.isnan(tt_U2)), tt_U2.size)
        print("# tt_U2 nans:", np.count_nonzero(np.isnan(tt_U2)), tt_U2.size)

        tt_U1 = np.nan_to_num(tt_U1)
        tt_U2 = np.nan_to_num(tt_U2)
        tt_U3 = np.nan_to_num(tt_U3)

        tt={
            '0': tt_U1.reshape(-1),
            '1': tt_U2.reshape(-1),
            '2': tt_U2.reshape(-1),
            'grid' : grid1,
        }

    tt['grid_tree'] = KDTree(grid1, leaf_size=2)

    ########################################################

    control_data = {
            't0' : t0,
            'tT' : tT,
            'tt' : tt,
            'time_slices' : None,
        }

    return test, rho0, rhoT, T_t, control_data,\
        [t0_u1, t0_u2, t0_u3],\
        [tT_u1, tT_u2, tT_u3],\
        [tt_U1, tt_U2, tt_U3],\
        [grid_x1, grid_x2, grid_x3, grid_t]

def do_integration(control_data, d, T_0, T_t, mu_0, sigma_0, args, sde):
    dt = (T_t - T_0)/(args.integrate_N)
    ts = np.arange(T_0, T_t + dt, dt)

    initial_sample = np.random.multivariate_normal(
        np.array([mu_0]*d), np.eye(d)*sigma_0, args.M) # 100 x 3

    v_scales = [float(x) for x in args.v_scale.split(",")]
    biases = [float(x) for x in args.bias.split(",")]

    ##############################

    all_results = {}

    mus = np.zeros(d)
    variances = np.zeros(d)

    pde_key = d
    if len(args.pde_key) > 0:
        pde_key = int(args.pde_key)
    print("pde_key", pde_key)

    # integrator = Integrator(
    #     initial_sample,
    #     (T_0, T_t),
    #     args,
    #     dynamics_map[pde_key])

    without_control = np.empty(
        (
            initial_sample.shape[0],
            initial_sample.shape[1],
            len(ts),
        ))

    return ts, initial_sample, None, without_control,\
        None, mus, variances

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
        type=float, default=5.0, help='')
    parser.add_argument('--a', type=str, default="-1,1,2", help='')
    parser.add_argument('--N', type=int, default=22, help='')
    parser.add_argument('--state_bound_min', type=float, default=0.1, help='')
    parser.add_argument('--state_bound_max', type=float, default=0.5, help='')

    parser.add_argument('--train_distribution', type=str, default="Hammersley", help='')
    parser.add_argument('--timemode', type=int, default=0, help='')
    # timemode  0 = linspace, 1 = even time samples
    parser.add_argument('--bif', type=int, default=1000, help='')
    parser.add_argument('--ni', type=int, default=-1, help='')
    parser.add_argument('--loss_func', type=str, default="wass3", help='')
    parser.add_argument('--pde_key', type=str, default="", help='')
    parser.add_argument('--batchsize',
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

    parser.add_argument('--sdept',
        type=str,
        default="../../sde/4fold_3_2_layer_model.pt",
        help='')

    args = parser.parse_args()

    if len(args.mu_T) > 0:
        mu_T = float(args.mu_T)
    if len(args.mu_0) > 0:
        mu_0 = float(args.mu_0)

    print("mu_0", mu_0)

    ################################################

    N = args.N

    test = np.loadtxt(args.testdat)

    ################################################

    print("test.shape", test.shape)
    d = test.shape[1] - 1 - 4 # one for time, 4 for pinn output
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

        sde = SDE()
        # state path to model information file
        # load model parameters
        sde.load_state_dict(torch.load(args.sdept))
        if torch.cuda.is_available():
            print("Using GPU.")
            sde = sde.to(cuda0)
        # set model to evaluation mode
        sde.eval()

        bcc = np.array([0.41235, 0.37605])
        fcc = np.array([0.012857, 0.60008])
        sc = np.array([0.41142, 0.69550])

        target = bcc

        model, meshes = get_model(
            d,
            N,
            batchsize,
            0,
            "tanh",
            [0.3525, 0.3503], # y0
            target, # bcc crystal
            T_t,
            args,
            sde.network_f,
            sde.network_g,
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

    # import ipdb; ipdb.set_trace()

    test, rho0, rhoT, T_t, control_data,\
        t0s, tTs, tts, grids = make_control_data(
        model, inputs, N, d, meshes, args)

    dphi_dinput_t0_dx, dphi_dinput_t0_dy, dphi_dinput_t0_dz = t0s
    dphi_dinput_tT_dx, dphi_dinput_tT_dy, dphi_dinput_tT_dz = tTs
    DPHI_DINPUT_tt_0, DPHI_DINPUT_tt_1, DPHI_DINPUT_tt_2 = tts
    grid_x1, grid_x2, grid_x3, grid_t = grids

    ########################################################

    fig = plt.figure()

    ax_count = 3
    if args.do_integration > 0:
        ax_count += 2

    ########################################################

    ax1 = fig.add_subplot(1, ax_count, 1, projection='3d')
    ax2 = fig.add_subplot(1, ax_count, 2, projection='3d')
    ax3 = fig.add_subplot(1, ax_count, 3, projection='3d')
    axs = [ax1, ax2, ax3]

    for ax in axs:
        ax.dist = 9 # default is 10

    ax_i = 0

    z1 = T_0
    z2 = T_t
    p = 0.01

    if d == 2:
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

        axs[ax_i].set_xlim(args.state_bound_min, args.state_bound_max)
        axs[ax_i].set_zlim(T_0 - 0.1, T_t + 0.1)

        axs[ax_i].set_xlabel('x')
        axs[ax_i].set_ylabel('y')
        axs[ax_i].set_zlabel('t')
        axs[ax_i].set_title('rho_opt')

        ax_i += 1

        ########################################################

        axs[ax_i].contourf(
            meshes[0],
            meshes[1],
            dphi_dinput_t0_dx.reshape(N, N),
            50, zdir='z',
            cmap=cm.jet,
            offset=z1,
            alpha=0.4
        )

        axs[ax_i].contourf(
            meshes[0],
            meshes[1],
            dphi_dinput_tT_dx.reshape(N, N),
            50, zdir='z',
            cmap=cm.jet,
            offset=z2,
            alpha=0.4,
        )

        sc2=axs[ax_i].scatter(
            grid_x1,
            grid_x2,
            grid_t,
            c=DPHI_DINPUT_tt_0,
            s=np.abs(DPHI_DINPUT_tt_0*p),
            cmap=cm.jet,
            alpha=1.0)
        plt.colorbar(sc2, shrink=0.25)

        axs[ax_i].set_xlim(args.state_bound_min, args.state_bound_max)
        axs[ax_i].set_zlim(T_0 - 0.1, T_t + 0.1)

        axs[ax_i].set_xlabel('x')
        axs[ax_i].set_ylabel('y')
        axs[ax_i].set_zlabel('t')
        axs[ax_i].set_title('u1')

        ax_i += 1

        ########################################################

        axs[ax_i].contourf(
            meshes[0],
            meshes[1],
            dphi_dinput_t0_dy.reshape(N, N),
            50, zdir='z',
            cmap=cm.jet,
            offset=z1,
            alpha=0.4
        )

        axs[ax_i].contourf(
            meshes[0],
            meshes[1],
            dphi_dinput_tT_dy.reshape(N, N),
            50, zdir='z',
            cmap=cm.jet,
            offset=z2,
            alpha=0.4,
        )

        sc3=axs[ax_i].scatter(
            grid_x1,
            grid_x2,
            grid_t,
            c=DPHI_DINPUT_tt_1,
            s=np.abs(DPHI_DINPUT_tt_1*p),
            cmap=cm.jet,
            alpha=1.0)
        plt.colorbar(sc3, shrink=0.25)

        axs[ax_i].set_xlim(args.state_bound_min, args.state_bound_max)
        axs[ax_i].set_zlim(T_0 - 0.1, T_t + 0.1)

        axs[ax_i].set_xlabel('x')
        axs[ax_i].set_ylabel('y')
        axs[ax_i].set_zlabel('t')
        axs[ax_i].set_title('u2')

        ax_i += 1


    ########################################################

    title_str = args.modelpt
    title_str += "\n"
    title_str += "N=15, d=%d, T_t=%.3f, batch=full, %s, mu_0=%.3f, mu_T=%.3f" % (
        d,
        T_t,
        "tanh",
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
            all_results, mus, variances = do_integration(control_data, d, T_0, T_t, mu_0, sigma_0, args, sde)

    if args.headless > 0:
        print("headless")
    else:
        plt.show()
