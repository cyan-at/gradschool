#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 0 define backend
import sys, os, time

# %env DDE_BACKEND=tensorflow.compat.v1
# %env XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/home/cyan3/miniforge/envs/tf

os.environ['DDE_BACKEND'] = "pytorch" # v2
os.environ['XLA_FLAGS'] = "--xla_gpu_cuda_data_dir=/usr/local/home/cyan3/miniforge/envs/tf"

# https://stackoverflow.com/questions/68614547/tensorflow-libdevice-not-found-why-is-it-not-found-in-the-searched-path
# this directory has /nvvm/libdevice/libdevice.10.bc

print(os.environ['DDE_BACKEND'])


# In[12]:


# import tensorflow as tf
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in gpu_devices:
#     print(device)

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
import pylab

from os.path import dirname, join as pjoin

from scipy import stats
import scipy.io
from scipy.stats import truncnorm, norm
from scipy.optimize import linprog
from scipy import sparse
from scipy.stats import multivariate_normal

if dde.backend.backend_name == "pytorch":
    exp = dde.backend.torch.exp
else:
    from deepxde.backend import tf

    exp = tf.exp
    
import cvxpy as cp
import numpy as np
import tensorflow as tf
from scipy.linalg import solve_discrete_are
from scipy.linalg import sqrtm

# from cvxpylayers.tensorflow.cvxpylayer import CvxpyLayer
from cvxpylayers.torch import CvxpyLayer


# In[34]:

N = nSample = 50

# must be floats
state_min = 0.0
state_max = 6.0

mu_0 = 5.0
sigma_0 = 1.0

mu_T = 3.0
sigma_T = 1.0

j1, j2, j3 =1,1,2 # axis-symmetric case
q_statepenalty_gain = 0 # 0.5

T_0=0. #initial time
T_t=200. #Terminal time

x_grid = np.transpose(np.linspace(state_min, state_max, nSample))
y_grid = np.transpose(np.linspace(state_min, state_max, nSample))
[X,Y] = np.meshgrid(x_grid,x_grid)
C = (X - Y)**2

# cvector = C.flatten('F')
cvector = C.reshape(nSample**2,1)

A = np.concatenate(
    (
        np.kron(
            np.ones((1,nSample)),
            sparse.eye(nSample).toarray()
        ),
        np.kron(
            sparse.eye(nSample).toarray(),
            np.ones((1,nSample))
        )
    ), axis=0)
# 2*nSample

# nb_name = __file__
# id_prefix = nb_name.replace(".ipynb", "").replace("-", "_")

id_prefix = "empty"

def pdf1d(x, mu, sigma):
    a, b = (state_min - mu) / sigma, (state_max - mu) / sigma
    rho_x=truncnorm.pdf(x, a, b, loc = mu, scale = sigma)

    # do NOT use gaussian norm, because it is only area=1
    # from -inf, inf, will not be for finite state/grid
    # rho_x = norm.pdf(x, mu, sigma)
    return rho_x

X_IDX = 0
T_IDX = 1
EQ_IDX = 3


test_ti = np.loadtxt('./test.dat')
test_ti = test_ti[0:N, :] # first BC test data
ind = np.lexsort((test_ti[:,X_IDX],test_ti[:,T_IDX]))
test_ti = test_ti[ind]

s1 = np.trapz(test_ti[:,EQ_IDX], axis=0, x=test_ti[:,X_IDX])
s2 = np.sum(test_ti[:,EQ_IDX])

fig = plt.figure(1)
ax1 = plt.subplot(111, frameon=False)
# ax1.set_aspect('equal')
ax1.grid()
ax1.set_title('trapz=%.3f, sum=%.3f' % (s1, s2))

ax1.plot(
    test_ti[:, 0],
    test_ti[:, EQ_IDX],
    linewidth=1,
    label='rho_0')
ax1.legend(loc='lower right')

test_rho0=pdf1d(test_ti[:, 0], 2.0, 3.0).reshape(test_ti.shape[0],1)
test_rho0 = test_rho0 / np.sum(np.abs(test_rho0))

ax1.plot(
    test_ti[:, 0],
    test_rho0,
    c='r',
    linewidth=1,
    label='rho_0')

plt.show()

