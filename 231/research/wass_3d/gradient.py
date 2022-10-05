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

from common import *

import argparse
parser = argparse.ArgumentParser(description="")
parser.add_argument('--N', type=int, default=50, help='')
parser.add_argument('--js', type=str, default="1,1,2", help='')
parser.add_argument('--q', type=float, default=0.0, help='')
parser.add_argument('--debug', type=int, default=0, help='')
parser.add_argument('--save', type=int, default=1, help='')
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

model = dde.Model(data, net)
