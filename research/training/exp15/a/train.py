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

sys.path.insert(0,'..')
from common import *

from layers import *

import argparse

######################################

def get_model(
    d,
    N,
    model_type,
    activations, # sigmoid, tanh
    mu_0,
    mu_T,
    T_t,
    optimizer="adam",
    init="Glorot normal",
    ):
    M = N**d

    linspaces = []
    for i in range(d):
        linspaces.append(np.transpose(np.linspace(state_min, state_max, N)))

    linspace_tensors = []
    for i in range(d):
        t = torch.from_numpy(
            linspaces[i]).requires_grad_(False)
        t = t.to(device)
        linspace_tensors.append(t)

    meshes = np.meshgrid(*linspaces)
    mesh_vectors = []
    for i in range(d):
        mesh_vectors.append(meshes[i].reshape(M,1))
    state = np.hstack(tuple(mesh_vectors))

    # state_tensor = torch.tensor(
    #     state,
    #     dtype=torch.float,
    #     requires_grad=True,
    #     device=device)

    ######################################

    rv0 = multivariate_normal([mu_0]*d, sigma_0 * np.eye(d))
    rvT = multivariate_normal([mu_T]*d, sigma_T * np.eye(d))

    rho0=rv0.pdf(state)
    rho0 = np.float32(rho0)

    rhoT= rvT.pdf(state)
    rhoT = np.float32(rhoT)

    rho0_tensor = torch.from_numpy(
        rho0,
    )
    rho0_tensor = rho0_tensor.to(device).requires_grad_(False)

    rhoT_tensor = torch.from_numpy(
        rhoT
    )
    rhoT_tensor = rhoT_tensor.to(device).requires_grad_(False)

    ######################################

    time_0=np.hstack((
        state,
        T_0*np.ones((len(mesh_vectors[0]), 1))
    ))
    rho_0_BC = dde.icbc.PointSetBC(
        time_0,
        rho0[..., np.newaxis],
        component=1)

    ######################################

    time_t=np.hstack((
        state,
        T_t*np.ones((len(mesh_vectors[0]), 1))
    ))
    rho_T_BC = dde.icbc.PointSetBC(
        time_t,
        rhoT[..., np.newaxis],
        component=1)

    ######################################

    sinkhorn = SinkhornDistance(eps=0.1, max_iter=200)

    # C = sinkhorn._cost_matrix(state_tensor, state_tensor)
    C = cdist(state, state, 'sqeuclidean')
    C = torch.from_numpy(
        C)
    C = C.to(device).requires_grad_(False)

    ######################################

    geom=dde.geometry.geometry_3d.Cuboid(
        [state_min]*d,
        [state_max]*d)
    timedomain = dde.geometry.TimeDomain(0., T_t)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    data = WASSPDE(
        geomtime,
        euler_pdes[d],
        [rho_0_BC,rho_T_BC],
        num_domain=samples_between_initial_and_final,
        num_initial=initial_and_final_samples,
        train_distribution="uniform")

    # d+1 inputs: <state> + t
    # 5 outputs: 2 eq
    net = dde.nn.FNN(
        [d+1] + [70] *3  + [2],
        # "sigmoid",
        activations,
        init
        # "zeros",
    )
    model = model_types[model_type](data, net)

    ######################################

    rho0_WASS = lambda y_true, y_pred: WASS_3(y_true, y_pred, sinkhorn, rho0_tensor, C)
    rho0_WASS.__name__ = "rho0_WASS"
    rhoT_WASS = lambda y_true, y_pred: WASS_3(y_true, y_pred, sinkhorn, rhoT_tensor, C)
    rhoT_WASS.__name__ = "rhoT_WASS"
    losses=[
        "MSE","MSE",
        rho0_WASS,
        rhoT_WASS,
    ]
    # loss functions are based on PDE + BC: eq outputs, BCs

    model.compile("adam", lr=1e-3,loss=losses)

    return model, meshes

######################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--N', type=int, default=15, help='')
    parser.add_argument('--js', type=str, default="1,1,2", help='')
    parser.add_argument('--q', type=float, default=0.5, help='')
    parser.add_argument('--ck_path', type=str, default=".", help='')
    parser.add_argument('--model_name', type=str, default="", help='')
    parser.add_argument('--debug', type=int, default=False, help='')
    parser.add_argument('--epochs', type=int, default=-1, help='')
    parser.add_argument('--optimizer', type=str, default="adam", help='')
    parser.add_argument('--restore', type=str, default="", help='')
    parser.add_argument('--mu_0', type=str, default="", help='')
    parser.add_argument('--mu_T', type=str, default="", help='')
    parser.add_argument('--T_t', type=str, default="", help='')
    args = parser.parse_args()

    N = args.N
    j1, j2, j3 = [float(x) for x in args.js.split(",")] # axis-symmetric case
    q_statepenalty_gain = args.q # 0.5
    print("N: ", N)
    print("js: ", j1, j2, j3)
    print("q: ", q_statepenalty_gain)

    if len(args.mu_0) > 0:
        mu_0 = float(args.mu_0)
    print("mu_0", mu_0)

    if len(args.mu_T) > 0:
        mu_T = float(args.mu_T)
    print("mu_T", mu_T)

    if len(args.T_t) > 0:
        T_t = float(args.T_t)
    print("T_t", T_t)

    d = 2

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    model, _ = get_model(
        d,
        N,
        0,
        "tanh",
        mu_0,
        mu_T,
        T_t,
        args.optimizer,
        )
    if len(args.restore) > 0:
        model.restore(args.restore)

    de = 1

    model_name = "model"
    if len(args.model_name) > 0:
        model_name = args.model_name

    ck_path = "%s/%s" % (args.ck_path, model_name)

    earlystop_cb = EarlyStoppingFixed(
        ck_path,
        baseline=1e-3,
        patience=0)
    modelcheckpt_cb = ModelCheckpoint2(
        ck_path,
        verbose=True,
        save_better_only=True,
        period=1)

    if args.epochs > 0:
        num_epochs = args.epochs

    losshistory, train_state = model.train(
        iterations=num_epochs,
        display_every=de,
        callbacks=[earlystop_cb, modelcheckpt_cb],
        model_save_path=ck_path)

    ######################################

    dde.saveplot(losshistory, train_state, issave=True, isplot=False)
    model_path = model.save(ck_path)
    print(model_path)
