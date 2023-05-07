#!/usr/bin/env python
# coding: utf-8

'''
run ./train.py --sdept ../../trained_sde_model/4fold_3_2_layer_model.pt
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
torch.set_printoptions(precision=3)
torch.set_printoptions(sci_mode=False)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)
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
from scipy.linalg import sqrtm as sqrtm2

######################################

import torch
from torch.autograd import Function
import numpy as np
import scipy.linalg

sys.path.insert(0, "..")

sde_path = '.'
sys.path.insert(0,sde_path)

import torchsde

from trained_sde_model import *
from common import *
from layers import *

import argparse
import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--N', type=int, default=22, help='')
    parser.add_argument('--state_bound_min', type=float, default=0.1, help='')
    parser.add_argument('--state_bound_max', type=float, default=0.5, help='')
    parser.add_argument('--bound_u', type=int, default=0, help='')

    parser.add_argument('--mu_0', type=str, default="0.2,0.2", help='')
    # parser.add_argument('--mu_T', type=str, default="", help='')
    parser.add_argument('--sigma',
        type=float,
        default=0.01)
    parser.add_argument('--crystal',
        type=str,
        required=True)

    parser.add_argument('--T_0', type=str, default="0", help='')
    parser.add_argument('--T_t', type=str, default="200", help='')

    parser.add_argument('--optimizer', type=str, default="adam", help='')
    parser.add_argument('--train_distribution', type=str, default="Hammersley", help='')
    parser.add_argument('--timemode', type=int, default=0, help='')
    # timemode  0 = linspace, 1 = even time samples
    parser.add_argument('--ni', type=int, default=0, help='')
    parser.add_argument('--bif', type=int, default=1000, help='')
    parser.add_argument('--loss_func', type=str, default="wass3", help='')

    parser.add_argument('--ck_path', type=str, default=".", help='')
    parser.add_argument('--model_name', type=str, default="", help='')
    parser.add_argument('--debug', type=int, default=False, help='')
    parser.add_argument('--epochs', type=int, default=-1, help='')
    parser.add_argument('--restore', type=str, default="", help='')
    parser.add_argument('--batchsize',
        type=str,
        default="")
    parser.add_argument('--batchsize2',
        type=str,
        default="")
    parser.add_argument('--batch2_period',
        type=int,
        default=5)

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

    parser.add_argument('--control_data',
        type=str,
        required=True)

    parser.add_argument('--population',
        type=str,
        default="")

    args = parser.parse_args()

    N = args.N
    print("N: ", N)

    if len(args.T_0) > 0:
        T_0 = float(args.T_0)
    print("T_0", T_0)

    if len(args.T_t) > 0:
        T_t = float(args.T_t)
    print("T_t", T_t)

    d = 2

    mu_0 = [float(x) for x in args.mu_0.strip().split(",")]

    ###########################################

    sde = SDE_3()
    # state path to model information file
    # load model parameters
    t = np.float32(np.array([1.0089, 1.8874])) 
    sde.start_val = torch.from_numpy(
        t).to(device)
    t = np.float32(np.array([[ 0.0036, -0.0007]]))
    sde.r = torch.from_numpy(
        t
    ).requires_grad_(False).to(device)

    files = glob.glob(
        sde_path + "/27hyper*.pt", 
        recursive = False)
    assert(len(files) == 1)
    print("using model: ", files[0])
    sde.load_state_dict(torch.load(files[0]))

    if torch.cuda.is_available():
        print("Using GPU.")
        sde = sde.to(cuda0)
    # set model to evaluation mode
    sde.eval()

    ###########################################

    control_data = np.load(args.control_data, allow_pickle=True).item()

    sde2 = SDE2(control_data)
    sde2.load_state_dict(torch.load(files[0]))
    if torch.cuda.is_available():
        print("Using GPU.")
        sde2 = sde2.to(cuda0)
    # set model to evaluation mode
    sde2.eval()

    ts, initial_sample, with_control, _,\
        all_results, mus, variances = do_integration2(
            control_data, d, T_0, T_t, mu_0, args.sigma,
            args, sde, sde2)

    with_control_fname, _ = Util.get_next_valid_name_increment(
        './',
        args.control_data.replace(".npy", "").split("_control_data")[0] + '_with_control',
        0,
        '',
        'npy')

    with_control_plot_fname, _ = Util.get_next_valid_name_increment(
        './',
        args.control_data.replace(".npy", "").split("_control_data")[0] + '_with_control',
        0,
        '',
        'png')

    ##############################################

    np.save(with_control_fname, with_control)
    print("WITH_CONTROL:", with_control_fname)

    ##############################################

    final_population = with_control[:, :, -1]

    linspaces = []
    for i in range(d):
        linspaces.append(np.transpose(
            np.linspace(args.state_bound_min, args.state_bound_max, N))
        )

    meshes = np.meshgrid(*linspaces)
    mesh_vectors = []
    for i in range(d):
        mesh_vectors.append(meshes[i].reshape(N**d,1))
    state = np.hstack(tuple(mesh_vectors))

    tmp = population_to_pdf(final_population, state)
    l = np.linspace(args.state_bound_min, args.state_bound_max, N)
    tmp /= square2d_pdfnormalize(tmp, l)

    # import ipdb; ipdb.set_trace()

    ##############################################

    buffer = 0.1

    fig = plt.figure(figsize=(8,8))

    # ax = fig.gca()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(args.state_bound_min - buffer, args.state_bound_max + buffer)
    ax.set_ylim(args.state_bound_min - buffer, args.state_bound_max + buffer)

    ax.scatter(
        final_population[:, 0],
        final_population[:, 1],
        alpha=1.0)

    # f = np.loadtxt(args.dat).reshape((N, N))

    cfset = ax.contourf(meshes[0], meshes[1],
        tmp.reshape((N,N)),
        cmap='coolwarm', alpha=0.75)

    # ax.clabel(cset, inline=1, fontsize=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # plt.show()
    '''
    '''

    plt.savefig(with_control_plot_fname, dpi=300)