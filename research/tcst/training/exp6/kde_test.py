#!/usr/bin/env python
# coding: utf-8

'''
USAGE:
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

'''
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
if dde.backend.backend_name == "pytorch":
    exp = dde.backend.torch.exp
else:
    from deepxde.backend import tf

    exp = tf.exp
from common import *

call_dir = os.getcwd()
sys.path.insert(0,call_dir)
print("expect train.py in %s" % (call_dir))
from train import *
'''

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

import cvxpy as cp
import numpy as np

import argparse


from scipy.signal import hilbert, butter, filtfilt
import scipy.stats
from sklearn.neighbors import KernelDensity

plt.rcParams['text.usetex'] = True

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import os, pickle, time, sys
from matplotlib import cm

import argparse
import matplotlib
import numpy as np
try:
    matplotlib.use("TkAgg")
except:
    print("no tkagg")
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture

print('hello')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--state_bound_min', type=float, default=-0.1, help='')
    parser.add_argument('--state_bound_max', type=float, default=0.6, help='')
    parser.add_argument('--sigma_0', type=float, default=0.001, help='')

    # parser.add_argument('--dat',
    #     type=str, required=True)

    args, _ = parser.parse_known_args()

    # nx2
    # dat = np.loadtxt(args.dat)
    mu_0 = [0.2, 0.2]
    sigma_0 = args.sigma_0
    d = 2
    dat = multivariate_normal(mu_0, sigma_0 * np.eye(d))

    N = 22
    M = N**d
    linspaces = []
    for i in range(d):
        linspaces.append(np.transpose(
            np.linspace(args.state_bound_min, args.state_bound_max, N))
        )

    meshes = np.meshgrid(*linspaces)
    mesh_vectors = []
    for i in range(d):
        mesh_vectors.append(meshes[i].reshape(M,1))
    state = np.hstack(tuple(mesh_vectors))

    rho0 = np.float32(dat.pdf(state))

    initial_sample = np.random.multivariate_normal(
        mu_0, np.eye(d)*sigma_0, M) # 100 x 3

    tmp = population_to_pdf(initial_sample, state)

    l = np.linspace(args.state_bound_min, args.state_bound_max, N)
    tmp /= square2d_pdfnormalize(tmp, l)
    rho0 /= square2d_pdfnormalize(rho0, l)

    # import ipdb; ipdb.set_trace()

    buffer = 0.1

    fig = plt.figure(figsize=(8,8))

    # ax = fig.gca()
    ax = fig.add_subplot(1, 2, 1)
    ax.set_xlim(args.state_bound_min - buffer, args.state_bound_max + buffer)
    ax.set_ylim(args.state_bound_min - buffer, args.state_bound_max + buffer)

    # f = np.loadtxt(args.dat).reshape((N, N))

    cfset = ax.contourf(meshes[0], meshes[1],
        rho0.reshape((N,N)),
        cmap='coolwarm', alpha=0.75)

    ax = fig.add_subplot(1, 2, 2)
    ax.set_xlim(args.state_bound_min - buffer, args.state_bound_max + buffer)
    ax.set_ylim(args.state_bound_min - buffer, args.state_bound_max + buffer)

    ax.scatter(
        initial_sample[:, 0],
        initial_sample[:, 1],
        alpha=1.0)

    # ax.imshow(np.rot90(f), cmap='coolwarm', extent=[
    #     args.state_bound_min - buffer, args.state_bound_max + buffer,
    #     args.state_bound_min - buffer, args.state_bound_max + buffer
    # ])
    cset = ax.contourf(meshes[0], meshes[1],
        tmp.reshape((N,N)),
        cmap='coolwarm', alpha=0.75)

    print(np.sum(np.abs(rho0.reshape((N, N)) - tmp.reshape((N, N)))))

    # ax.clabel(cset, inline=1, fontsize=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.title(args.sigma_0)

    plt.show()