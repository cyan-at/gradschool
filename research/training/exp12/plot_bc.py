#!/usr/bin/env python
# coding: utf-8

'''
USAGE:
./plot_bc.py --testdat ./wass_3d_model-60970-60970.dat --modelpt ./wass_3d_model-60970.pt --interp_mode nearest --grid_n 15
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

    args = parser.parse_args()

    test = np.loadtxt(args.testdat)

    if os.path.exists(args.modelpt):
        print("loading model")

        d = 3; N = 15;
        M = N**d
        batchsize = 500

        # instead of using test output, use model and 
        # generate / predict a new output
        inputs = test[:, :d+1]

        T_t = inputs[batchsize, -1]
        print("found T_t", T_t)

        inputs = np.float32(inputs)

        if args.fullstate > 0:
            # BE CAREFUL ABOUT TIME VALUES HERE
            # TIME = FUNCTION(DATA), not FUNCTION(COMMON)
            linspaces = []
            for i in range(d):
                linspaces.append(np.transpose(np.linspace(state_min, state_max, N)))
            meshes = np.meshgrid(*linspaces)
            rest = inputs[2*batchsize:, :]
            mesh_vectors = []
            for i in range(d):
                mesh_vectors.append(meshes[i].reshape(M,1))
            state = np.hstack(tuple(mesh_vectors))
            time_0=np.hstack((
                state,
                T_0*np.ones((len(mesh_vectors[0]), 1))
            ))
            time_t=np.hstack((
                state,
                T_t*np.ones((len(mesh_vectors[0]), 1))
            ))
            inputs = np.vstack((time_0, time_t, rest))
            inputs = np.float32(inputs)
            batchsize = N**d

        '''
        inputs_tensor = torch.from_numpy(
            inputs).requires_grad_(True)
        inputs_tensor = inputs_tensor.requires_grad_(True)

        if args.diff_on_cpu == 0:
            print("moving input to cuda")
            inputs_tensor = inputs_tensor.type(torch.FloatTensor).to(cuda0).requires_grad_(True)
        else:
            print("keeping input on cpu")
        '''

        model, meshes = get_model(
            d,
            N,
            batchsize,
            0,
            "tanh")
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

        output_tensor = model.net(inputs_tensor)

        # import ipdb; ipdb.set_trace()

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

        dphi_dinput_fname = args.modelpt.replace(".pt", "_dphi_dinput_%d_%d.txt" % (
            args.diff_on_cpu,
            batchsize
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

    # import ipdb; ipdb.set_trace()

    ########################################################

    rho0 = t0[:batchsize, -1]
    rhoT = tT[:, -1]

    x_1_ = np.linspace(state_min, state_max, args.grid_n)
    x_2_ = np.linspace(state_min, state_max, args.grid_n)
    x_3_ = np.linspace(state_min, state_max, args.grid_n)
    t_ = np.linspace(T_0, T_t, args.grid_n*2)

    grid_x1, grid_x2, grid_x3 = np.meshgrid(
        x_1_,
        x_2_,
        x_3_, copy=False) # each is NxNxN

    RHO_0 = gd(
      (t0[:, 0], t0[:, 1], t0[:, 2]),
      t0[:, -1],
      (grid_x1, grid_x2, grid_x3),
      method=args.interp_mode,
      fill_value=0.0)

    # import ipdb; ipdb.set_trace()

    RHO_T = gd(
      (tT[:, 0], tT[:, 1], tT[:, 2]),
      tT[:, -1],
      (grid_x1, grid_x2, grid_x3),
      method=args.interp_mode,
      fill_value=0.0)

    print("RHO_0 sum", np.sum(t0[:, -1]))
    print("RHO_T sum", np.sum(tT[:, -1]))

    ########################################################

    fig = plt.figure()

    ########################################################

    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')

    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')

    ########################################################


    ########################################################

    dphi_dinput_t0 = dphi_dinput[:batchsize, :]
    dphi_dinput_tT = dphi_dinput[batchsize:2*batchsize, :]
    dphi_dinput_tt = dphi_dinput[2*batchsize:, :]
    print(
        np.max(dphi_dinput_t0),
        np.max(dphi_dinput_tT),
        np.max(dphi_dinput_tt)
    )

    dphi_dinput_t0_l2_norm = np.linalg.norm(
        dphi_dinput_t0[:, 0:3], ord=2, axis=1)

    DPHI_DINPUT_T0_L2_NORM = gd(
      (t0[:, 0], t0[:, 1], t0[:, 2]),
      dphi_dinput_t0_l2_norm,
      (grid_x1, grid_x2, grid_x3),
      method=args.interp_mode,
      fill_value=0.0)

    p = 1

    ########################################################

    sc1=ax1.scatter(
        grid_x1,
        grid_x2,
        grid_x3,
        c=RHO_0,
        s=np.abs(RHO_0*p),
        cmap=cm.jet,
        alpha=1.0)
    plt.colorbar(sc1, shrink=0.25)
    ax1.set_title('rho0: mu=%.3f, sigma=%.3f, sum=%.3f, min=%.3f, max=%.3f' % (
        mu_0,
        sigma_0,
        np.sum(t0[:, -1]),
        np.min(t0[:, -1]),
        np.max(t0[:, -1])
    ))
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    sc2=ax2.scatter(
        grid_x1,
        grid_x2,
        grid_x3,
        c=RHO_T,
        s=np.abs(RHO_T*p),
        cmap=cm.jet,
        alpha=1.0)
    plt.colorbar(sc2, shrink=0.25)
    ax2.set_title('rhoT: mu=%.3f, sigma=%.3f, sum=%.3f, min=%.3f, max=%.3f' % (
        mu_T,
        sigma_T,
        np.sum(tT[:, -1]),
        np.min(tT[:, -1]),
        np.max(tT[:, -1])
    ))
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')

    sc3=ax3.scatter(
        grid_x1,
        grid_x2,
        grid_x3,
        c=DPHI_DINPUT_T0_L2_NORM,
        s=np.abs(DPHI_DINPUT_T0_L2_NORM*p),
        cmap=cm.jet,
        alpha=1.0)
    plt.colorbar(sc3, shrink=0.25)
    ax3.set_title('dphi_dinput_t0 l2_norm, [0] range=%.3f, %.3f' % (np.min(dphi_dinput[:, 0]), np.max(dphi_dinput[:, 0])))
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    '''
    '''

    ########################################################

    DPHI_DINPUT_T0_X = gd(
      (t0[:, 0], t0[:, 1], t0[:, 2]),
      dphi_dinput_t0[:, 0],
      (grid_x1, grid_x2, grid_x3),
      method=args.interp_mode,
      fill_value=0.0)

    DPHI_DINPUT_T0_Y = gd(
      (t0[:, 0], t0[:, 1], t0[:, 2]),
      dphi_dinput_t0[:, 1],
      (grid_x1, grid_x2, grid_x3),
      method=args.interp_mode,
      fill_value=0.0)

    DPHI_DINPUT_T0_Z = gd(
      (t0[:, 0], t0[:, 1], t0[:, 2]),
      dphi_dinput_t0[:, 2],
      (grid_x1, grid_x2, grid_x3),
      method=args.interp_mode,
      fill_value=0.0)

    ########################################################

    '''
    '''
    sc4=ax4.scatter(
        grid_x1,
        grid_x2,
        grid_x3,
        c=DPHI_DINPUT_T0_X,
        s=np.abs(DPHI_DINPUT_T0_X*p),
        cmap=cm.jet,
        alpha=1.0)
    plt.colorbar(sc4, shrink=0.25)
    ax4.set_title('scaled %.3f dpsi_dx\nmin=%.3f, max=%.3f' % (
        p,
        np.min(dphi_dinput[:, 0]*p),
        np.max(dphi_dinput[:, 0]*p)
    ))
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('z')

    sc5=ax5.scatter(
        grid_x1,
        grid_x2,
        grid_x3,
        c=DPHI_DINPUT_T0_Y,
        s=np.abs(DPHI_DINPUT_T0_Y*p),
        cmap=cm.jet,
        alpha=1.0)
    plt.colorbar(sc5, shrink=0.25)
    ax5.set_title('scaled %.3f dpsi_dy\nmin=%.3f, max=%.3f' % (
        p,
        np.min(dphi_dinput[:, 1]*p),
        np.max(dphi_dinput[:, 1]*p)
    ))
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_zlabel('z')

    sc6=ax6.scatter(
        grid_x1,
        grid_x2,
        grid_x3,
        c=DPHI_DINPUT_T0_Z,
        s=np.abs(DPHI_DINPUT_T0_Z*p),
        cmap=cm.jet,
        alpha=1.0)
    plt.colorbar(sc6, shrink=0.25)
    ax6.set_title('scaled %.3f dpsi_dz\nmin=%.3f, max=%.3f' % (
        p,
        np.min(dphi_dinput[:, 2]*p),
        np.max(dphi_dinput[:, 2]*p)
    ))
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_zlabel('z')

    ########################################################

    # u1_t0 = u1[:batchsize]

    # U1 = gd(
    #   (t0[:, 0], t0[:, 1], t0[:, 2]),
    #   u1_t0,
    #   (grid_x1, grid_x2, grid_x3),
    #   method=args.interp_mode,
    #   fill_value=0.0)

    # ax3 = fig.add_subplot(1, 4, 4, projection='3d')
    # sc3=ax3.scatter(
    #     grid_x1,
    #     grid_x2,
    #     grid_x3,
    #     c=U1,
    #     s=np.abs(U1*1000),
    #     cmap=cm.jet,
    #     alpha=1.0)
    # plt.colorbar(sc3, shrink=0.25)
    # ax3.set_title('u1_t0 min=%.3f, max=%.3f' % (np.min(u1), np.max(u1)))
    # ax3.set_xlabel('x')
    # ax3.set_ylabel('y')
    # ax3.set_zlabel('z')

    ########################################################

    title_str = args.modelpt
    title_str += "\n"
    title_str += "N=15, d=3, batch=%d + %s, %s + not clamped weights" % (
        batchsize,
        args.interp_mode,
        "tanh",
    )
    title_str += "\n"
    # title_str += "99875     [5.95e-08, 5.44e-04, 2.50e-02, 2.51e-02]    [5.95e-08, 5.44e-04, 2.50e-02, 2.51e-02]    []"

    plt.suptitle(title_str)

    c = Counter()
    fig.canvas.mpl_connect('key_press_event', lambda e: c.on_press_saveplot(e,
            '%s_rho_opt_bc_batch=%d_%d_%d_%s_%d.png'  %(
                args.modelpt.replace(".pt", ""),
                batchsize,
                args.fullstate,
                args.diff_on_cpu,
                args.interp_mode,
                args.grid_n
            )
        )
    )

    plt.show()

    ########################################################

    gen_control_data = input("generate control data? ")
    print(gen_control_data)

    if gen_control_data != "1":
        sys.exit(0)

    grid = np.array((grid_x1.reshape(-1), grid_x2.reshape(-1), grid_x3.reshape(-1))).T

    DPHI_DINPUT_t0_0 = gd(
      (t0[:, 0], t0[:, 1], t0[:, 2]), dphi_dinput_t0[:, 0],
      (grid_x1, grid_x2, grid_x3),
      method=args.interp_mode,
      fill_value=0.0)
    DPHI_DINPUT_t0_1 = gd(
      (t0[:, 0], t0[:, 1], t0[:, 2]), dphi_dinput_t0[:, 1],
      (grid_x1, grid_x2, grid_x3),
      method=args.interp_mode,
      fill_value=0.0)
    DPHI_DINPUT_t0_2 = gd(
      (t0[:, 0], t0[:, 1], t0[:, 2]), dphi_dinput_t0[:, 2],
      (grid_x1, grid_x2, grid_x3),
      method=args.interp_mode,
      fill_value=0.0)
    t0={
        '0': DPHI_DINPUT_t0_0.reshape(-1),
        '1': DPHI_DINPUT_t0_1.reshape(-1),
        '2': DPHI_DINPUT_t0_2.reshape(-1),
        'grid' : grid,
    }

    DPHI_DINPUT_tT_0 = gd(
      (tT[:, 0], tT[:, 1], tT[:, 2]), dphi_dinput_tT[:, 0],
      (grid_x1, grid_x2, grid_x3),
      method=args.interp_mode,
      fill_value=0.0)
    DPHI_DINPUT_tT_1 = gd(
      (tT[:, 0], tT[:, 1], tT[:, 2]), dphi_dinput_tT[:, 1],
      (grid_x1, grid_x2, grid_x3),
      method=args.interp_mode,
      fill_value=0.0)
    DPHI_DINPUT_tT_2 = gd(
      (tT[:, 0], tT[:, 1], tT[:, 2]), dphi_dinput_tT[:, 2],
      (grid_x1, grid_x2, grid_x3),
      method=args.interp_mode,
      fill_value=0.0)
    tT={
        '0': DPHI_DINPUT_tT_0.reshape(-1),
        '1': DPHI_DINPUT_tT_1.reshape(-1),
        '2': DPHI_DINPUT_tT_2.reshape(-1),
        'grid' : grid,
    }

    grid_x1, grid_x2, grid_x3, grid_t = np.meshgrid(
        x_1_,
        x_2_,
        x_3_,
        t_, copy=False) # each is NxNxN

    grid = np.array((
        grid_x1.reshape(-1),
        grid_x2.reshape(-1),
        grid_x3.reshape(-1),
        grid_t.reshape(-1),
    )).T

    # import ipdb; ipdb.set_trace()

    DPHI_DINPUT_tt_0 = gd(
      (tt[:, 0], tt[:, 1], tt[:, 2], tt[:, 3]),
      dphi_dinput_tt[:, 0],
      (grid_x1, grid_x2, grid_x3, grid_t),
      method=args.interp_mode,
      fill_value=0.0)
    DPHI_DINPUT_tt_1 = gd(
      (tt[:, 0], tt[:, 1], tt[:, 2], tt[:, 3]),
      dphi_dinput_tt[:, 1],
      (grid_x1, grid_x2, grid_x3, grid_t),
      method=args.interp_mode,
      fill_value=0.0)
    DPHI_DINPUT_tt_2 = gd(
      (tt[:, 0], tt[:, 1], tt[:, 2], tt[:, 3]),
      dphi_dinput_tt[:, 2],
      (grid_x1, grid_x2, grid_x3, grid_t),
      method=args.interp_mode,
      fill_value=0.0)
    tt={
        '0': DPHI_DINPUT_tt_0.reshape(-1),
        '1': DPHI_DINPUT_tt_1.reshape(-1),
        '2': DPHI_DINPUT_tt_2.reshape(-1),
        'grid' : grid,
    }

    fname = '%s_%d_%d_%d_%s_%d_all_control_data.npy' % (
            args.modelpt.replace(".pt", ""),
            batchsize,
            args.fullstate,
            args.diff_on_cpu,
            args.interp_mode,
            args.grid_n
        )
    np.save(
        fname,
        {
            't0' : t0,
            'tT' : tT,
            'tt' : tt
        })
    print("saved control_data to %s" % (fname))
