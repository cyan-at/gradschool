#!/usr/bin/env python
# coding: utf-8

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
torch.set_printoptions(precision=3)
torch.set_printoptions(sci_mode=False)
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
device = cuda0

import deepxde as dde

import numpy as np
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

from numpy import linalg as LA
import math
import matplotlib.pyplot as plt
import pylab
from os.path import dirname, join as pjoin
from scipy import stats
import scipy.io
from scipy.stats import truncnorm, norm
from scipy.optimize import linprog
from scipy import sparse
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
if dde.backend.backend_name == "pytorch":
    exp = dde.backend.torch.exp
else:
    from deepxde.backend import tf
    exp = tf.exp
import cvxpy as cp
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.linalg import sqrtm

from cvxpylayers.torch import CvxpyLayer

######################################

import torch
from torch.autograd import Function
import numpy as np
import scipy.linalg

class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """
    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input
sqrtm = MatrixSquareRoot.apply

from common import *

import argparse
parser = argparse.ArgumentParser(description="")
parser.add_argument('--N', type=int, default=50, help='')
parser.add_argument('--js', type=str, default="1,1,2", help='')
parser.add_argument('--q', type=float, default=0.0, help='')
parser.add_argument('--debug', type=int, default=False, help='')
parser.add_argument('--modelpt',
    type=str, default='')
parser.add_argument('--testdat',
    type=str, required=True)
args = parser.parse_args()

N = args.N
j1, j2, j3 = [float(x) for x in args.js.split(",")] # axis-symmetric case
q_statepenalty_gain = args.q # 0.5
print("N: ", N)
print("js: ", j1, j2, j3)
print("q: ", q_statepenalty_gain)

###############################################

# a 3d grid, not the sparse diagonal elements as above
x1 = np.transpose(np.linspace(state_min, state_max, N)).astype(f)
x2 = np.transpose(np.linspace(state_min, state_max, N)).astype(f)
x3 = np.transpose(np.linspace(state_min, state_max, N)).astype(f)
[X,Y,Z] = np.meshgrid(x1,x2,x3)
x_T=X.reshape(N**3,1)
y_T=Y.reshape(N**3,1)
z_T=Z.reshape(N**3,1)

###############################################

test = np.loadtxt(args.testdat)

###############################################

t0 = test[:N**3, :]

rho0 = t0[:, RHO_OPT_IDX]
rho0 = np.where(rho0 < 0, 0, rho0)

mu, cov_matrix, pmf_cube_normed, x1m, x2m, x3m = get_pmf_stats(rho0, x_T, y_T, z_T, x1, x2, x3)
print("mu\n", mu)
print("cov_matrix\n", cov_matrix)

rho0_name = 'rho0_%.3f_%.3f__%.3f_%.3f__%d.dat' % (
    mu_0, sigma_0,
    state_min, state_max,
    N)
trunc_rho0_pdf = get_multivariate_truncated_norm(x_T, y_T, z_T, mu_0, sigma_0, state_min, state_max, N, f, rho0_name)
rho0_mu, rho0_cov_matrix, _, true_x1m, true_x2m, true_x3m = get_pmf_stats(
    trunc_rho0_pdf, x_T, y_T, z_T, x1, x2, x3)
print("rho0_mu\n", rho0_mu)
print("rho0_cov_matrix\n", rho0_cov_matrix)

###############################################

tt = test[N**3:2*N**3, :]

rhoT = tt[:, RHO_OPT_IDX]
rhoT = np.where(rhoT < 0, 0, rhoT)

muT, cov_matrixT, pmf_cube_normedT, x1mT, x2mT, x3mT = get_pmf_stats(rhoT, x_T, y_T, z_T, x1, x2, x3)
print("muT\n", muT)
print("cov_matrixT\n", cov_matrixT)

rhoT_name = 'rhoT_%.3f_%.3f__%.3f_%.3f__%d.dat' % (
    mu_T, sigma_T,
    state_min, state_max,
    N)
trunc_rhoT_pdf = get_multivariate_truncated_norm(x_T, y_T, z_T, mu_T, sigma_T, state_min, state_max, N, f, rhoT_name)
rhoT_mu, rhoT_cov_matrix, _, true_x1mT, true_x2mT, true_x3mT = get_pmf_stats(
    trunc_rhoT_pdf, x_T, y_T, z_T, x1, x2, x3)
print("rhoT_mu\n", rhoT_mu)
print("rhoT_cov_matrix\n", rhoT_cov_matrix)

###############################################

colors = 'rgbymck'

fig = plt.figure(1, figsize=(20, 10))
ax1 = plt.subplot(131, frameon=False)
# ax1.set_aspect('equal')
ax1.grid()

ax2 = plt.subplot(132, frameon=False)
# ax2.set_aspect('equal')
ax2.grid()

ax3 = plt.subplot(133, frameon=False)
# fig2 = plt.figure(2)
# ax3 = plt.subplot(111, frameon=False)
# ax3.set_aspect('equal')
ax3.grid()

###############################################

ax1.plot(x1, x1m,
  color=colors[0 % len(colors)],
  lw=1,
  label='rho0 pred x1')
ax1.plot(x1, true_x1m,
  color=colors[1 % len(colors)],
  lw=1,
  label='rho0 true x1')

ax2.plot(x2, x2m,
  color=colors[2 % len(colors)],
  lw=1,
  label='rho0 pred x2')
ax2.plot(x2, true_x2m,
  color=colors[3 % len(colors)],
  lw=1,
  label='rho0 true x2')

ax3.plot(x3, x3m,
  color=colors[4 % len(colors)],
  lw=1,
  label='rho0 pred x3')
ax3.plot(x3, true_x3m,
  color=colors[5 % len(colors)],
  lw=1,
  label='rho0 true x3')

###############################################

ax1.plot(x1, x1mT,
  color=colors[0 % len(colors)],
  lw=1,
  label='rhoT pred x1')
ax1.plot(x1, true_x1mT,
  color=colors[1 % len(colors)],
  lw=1,
  label='rhoT true x1')

ax2.plot(x2, x2mT,
  color=colors[2 % len(colors)],
  lw=1,
  label='rhoT pred x2')
ax2.plot(x2, true_x2mT,
  color=colors[3 % len(colors)],
  lw=1,
  label='rhoT true x2')

ax3.plot(x3, x3mT,
  color=colors[4 % len(colors)],
  lw=1,
  label='rhoT pred x3')
ax3.plot(x3, true_x3mT,
  color=colors[5 % len(colors)],
  lw=1,
  label='rhoT true x3')

###############################################

plt.suptitle('loss (%s)=%s\nPMF: mu0=%.3f, sigma0=%.3f\nmuT=%.3f, sigmaT=%.3f\nN=%d' % (
    "MSE, MSE, wass, wass",
    "[5.02e-04, 1.98e-02, 1.78e-02, 1.16e-02]",
    mu_0, sigma_0,
    mu_T, sigma_T,
    N))

plt.tight_layout()

# ax.grid()
ax1.legend(loc="lower left")
ax2.legend(loc="lower left")
ax3.legend(loc="lower left")

# ax.set_title('training error/residual plots')
# ax.set_yscale('log')
# ax.set_xscale('log')

plot_fname = "%s/%s.png" % (
    os.path.abspath("./"),
    args.testdat.replace(".dat", "")
)
plt.savefig(plot_fname, dpi=500)
print("saved plot")

plt.show()
