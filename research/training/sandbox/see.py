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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dat',
        type=str, required=True)

    parser.add_argument('--indices',
        type=str, default="")

    args = parser.parse_args()

    if not os.path.exists(args.dat):
        print("bad dat")
        sys.exit(0)

    x = None
    try:
        x = np.load(args.dat, allow_pickle=True)
    except Exception as e:
        print(str(e))

    if x is None:
        try:
            x = np.loadtxt(args.dat)
            print("loaded as txt")
        except Exception as e:
            print("unable to load", str(e))
            sys.exit(1)
    else:
        print("loaded numpy data")

    try:
        x = x.item()
    except Exception as e:
        print(e)

    ########################################################

    all_data_to_plot = []

    if type(x) == dict:
        while type(x) == dict:
            t = [type(y) for y in x.keys()]
            print(t)

            s = [str(y) for y in x.keys()]

            s_to_t = {}
            for s_i, q in enumerate(s):
                s_to_t[q] = t[s_i]

            k = input('found dict, enter keys (%s): ' % (",".join(s)))
            k_tokens = k.strip().split(",")

            pending_level = x[s_to_t[k_tokens[0]](k_tokens[0])]

            for k in k_tokens:
                if type(pending_level) != dict:
                    print('k', k, type(pending_level))
                    all_data_to_plot.append(x[s_to_t[k](k)])

            x = pending_level
    else:
        all_data_to_plot.append(x)

    ########################################################

    colors = 'rgbymck'
    fig = plt.figure()

    for x_i, x in enumerate(all_data_to_plot):
        # import ipdb; ipdb.set_trace()
        c = colors[x_i % len(colors)]

        print("x.shape", x.shape)

        x = np.float32(x)

        if len(x.shape) == 1:
            x = x[:, np.newaxis]

        ########################################################

        if len(args.indices) > 0:
            indices = [int(x) for x in args.indices.split(",")]
        else:
            indices = list(range(x.shape[1]))

        print(indices)

        if len(indices) > 3:
            print("bad indices")
            sys.exit(0)

        ########################################################

        if len(indices) == 3:
            ax1 = fig.add_subplot(1, 1, 1, projection='3d')

            sc1=ax1.scatter(
                x[:, 0],
                x[:, 1],
                x[:, 2],
                c=0.5*np.ones(x.shape[0]),
                s=1.0*np.ones(x.shape[0]),
                cmap=cm.jet,
                alpha=1.0)
            plt.colorbar(sc1, shrink=0.25)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('z')
        elif len(indices) == 2:
            plt.plot(x[:, indices[0]], x[:, indices[1]],
                c=c,
                alpha=1/len(all_data_to_plot))
        elif len(indices) == 1:
            plt.plot(x[:, indices[0]],
                c=c,
                alpha=1/len(all_data_to_plot))

    ########################################################

    plt.suptitle(args.dat)

    c = Counter()
    fig.canvas.mpl_connect('key_press_event', lambda e: c.on_press_saveplot(e,
            '%s.png'  %(
                args.dat.replace(".npy", ""),
            )
        )
    )

    ########################################################

    plt.grid(True)
    plt.tight_layout()
    plt.show()
