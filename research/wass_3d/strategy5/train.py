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

# Before running your code, run this shell command to tell torch that there are no GPUs:
# https://stackoverflow.com/questions/53266350/how-to-tell-pytorch-to-not-use-the-gpu
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
torch.set_printoptions(precision=3)
torch.set_printoptions(sci_mode=False)
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))
# print(torch.version.cuda)
# print(torch.cuda.current_device())
# torch.cuda.set_device(0)

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

def get_model(d, N):
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
    rhoT_tensor = rhoT_tensor.to(cpu).requires_grad_(False)

    ######################################

    # rho0_name = 'rho0_%.3f_%.3f__%.3f_%.3f__%d.dat' % (
    #     mu_0, sigma_0,
    #     state_min, state_max,
    #     N)
    # trunc_rho0_pdf = get_multivariate_truncated_pdf(
    #     mesh_vectors[0],
    #     mesh_vectors[1],
    #     mesh_vectors[2],
    #     mu_0, sigma_0, state_min, state_max, N, f, rho0_name)
    time_0=np.hstack((
        state,
        T_0*np.ones((len(mesh_vectors[0]), 1))
    ))
    rho_0_BC = dde.icbc.PointSetBC(
        time_0,
        rho0[..., np.newaxis],
        batch_size=args.batchsize,
        component=1,
        shuffle=True)

    ######################################

    # rhoT_name = 'rhoT_%.3f_%.3f__%.3f_%.3f__%d.dat' % (
    #     mu_T, sigma_T,
    #     state_min, state_max,
    #     N)
    # trunc_rhoT_pdf = get_multivariate_truncated_pdf(
    #     mesh_vectors[0],
    #     mesh_vectors[1],
    #     mesh_vectors[2],
    #     mu_T, sigma_T, state_min, state_max, N, f, rhoT_name)
    time_t=np.hstack((
        state,
        T_t*np.ones((len(mesh_vectors[0]), 1))
    ))
    rho_T_BC = dde.icbc.PointSetBC(
        time_t,
        rhoT[..., np.newaxis],
        batch_size=args.batchsize,
        component=1,
        shuffle=True)

    ######################################

    sinkhorn0 = SinkhornDistance(eps=0.1, max_iter=200)
    sinkhornT = SinkhornDistance(eps=0.1, max_iter=200) #.cpu()

    # C = sinkhorn._cost_matrix(state_tensor, state_tensor)
    C = cdist(state, state, 'sqeuclidean')
    C_device = torch.from_numpy(
        C)
    C_device = C_device.to(device).requires_grad_(False)

    C_cpu = torch.from_numpy(
        C)
    C_cpu = C_cpu.to(cpu).requires_grad_(False)

    ######################################

    # import ipdb; ipdb.set_trace()

    def rho0_WASS_batch_cuda0(y_true, y_pred):
        n = torch.numel(y_pred)

        p1 = (y_pred<0).sum()  # negative terms

        # print(y_pred.shape)
        # print(y_true.shape)

        rho0_temp_tensor = torch.from_numpy(
            rho0[y_true],
        )
        rho0_temp_tensor = rho0_temp_tensor.to(device).requires_grad_(False)

        C_temp = cdist(state[y_true, :], state[y_true, :], 'sqeuclidean')
        C_temp_device = torch.from_numpy(
            C_temp)
        C_temp_device = C_temp_device.to(device).requires_grad_(False)
    
        y_pred_max = -torch.max(y_pred)

        y_pred = torch.where(y_pred < 0, 0, y_pred)

        s = torch.sum(y_pred)
        p2 = torch.abs(s - 1)

        if s < 1e-3:
            # mostly negative, then do not compute
            # wass distance
            # to avoid getting 'stuck' in 0 gradient
            
            # this is BETTER than checking if max is negative
            # because pinn could output a very small
            # positive value, and overcome that case
            return -y_pred_max

        # if s > 1e-2:
        #     y_pred /= s # into pmf

        dist, _, _ = sinkhorn0(
            C_temp_device,
            y_pred.reshape(-1),
            rho0_temp_tensor)
        # print("Sinkhorn distance: {:.3f}".format(dist.item()))

        return dist + p2 / n + p1 / n

    def rhoT_WASS_batch_cuda0(y_true, y_pred):
        n = torch.numel(y_pred)

        p1 = (y_pred<0).sum()  # negative terms

        # print(y_pred.shape)
        # print(y_true.shape)

        rhoT_temp_tensor = torch.from_numpy(
            rhoT[y_true],
        )
        rhoT_temp_tensor = rhoT_temp_tensor.to(device).requires_grad_(False)

        C_temp = cdist(state[y_true, :], state[y_true, :], 'sqeuclidean')
        C_temp_device = torch.from_numpy(
            C_temp)
        C_temp_device = C_temp_device.to(device).requires_grad_(False)

        y_pred_max = -torch.max(y_pred)

        y_pred = torch.where(y_pred < 0, 0, y_pred)

        s = torch.sum(y_pred)
        p2 = torch.abs(s - 1)

        if s < 1e-3:
            # mostly negative, then do not compute
            # wass distance
            # to avoid getting 'stuck' in 0 gradient
            
            # this is BETTER than checking if max is negative
            # because pinn could output a very small
            # positive value, and overcome that case
            return -y_pred_max

        # if s > 1e-3:
        #     y_pred /= s # into pmf

        # import ipdb; ipdb.set_trace()

        dist, _, _ = sinkhornT(
            C_temp_device,
            y_pred.reshape(-1),
            rhoT_temp_tensor)
        # print("Sinkhorn distance: {:.3f}".format(dist.item()))

        return dist + p2 / n + p1 / n

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
        # num_boundary=50,
        # train_distribution="pseudo",
        # num_initial=10,
    )

    # resampler = dde.callbacks.PDEResidualResampler(period=1)

    # 4 inputs: x,y,z,t
    # 5 outputs: 2 eq + 3 control vars
    net = dde.nn.FNN(
        [d+1] + [70] *3  + [2],
        # "sigmoid",
        "tanh",

        "Glorot normal"
        # "zeros",
    )

    model = dde.Model(data, net)
    # model = NonNeg_LastLayer_Model(data, net)

    ######################################

    loss_func=[
        "MSE","MSE",
        rho0_WASS_batch_cuda0,
        rhoT_WASS_batch_cuda0
    ]
    # loss functions are based on PDE + BC: eq outputs, BCs

    model.compile("adam", lr=1e-3,loss=loss_func)

    return model, meshes

######################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--N', type=int, default=15, help='')
    parser.add_argument('--js', type=str, default="1,1,2", help='')
    parser.add_argument('--q', type=float, default=0.0, help='')
    parser.add_argument('--debug', type=int, default=False, help='')
    parser.add_argument('--batchsize', type=int, default=500, help='')
    args = parser.parse_args()

    N = args.N
    j1, j2, j3 = [float(x) for x in args.js.split(",")] # axis-symmetric case
    q_statepenalty_gain = args.q # 0.5
    print("N: ", N)
    print("js: ", j1, j2, j3)
    print("q: ", q_statepenalty_gain)

    d = 3

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    model, _ = get_model(d, N)

    de = 1

    modelcheckpt_cb = ModelCheckpoint2(
        ck_path,
        verbose=True,
        save_better_only=True,
        period=1,
        extra_comment="{}:{}:{}".format(d, N, "nonneg-sigmoid"))

    losshistory, train_state = model.train(
        iterations=num_epochs,
        display_every=de,
        callbacks=[
            # resampler,
            earlystop_cb,
            modelcheckpt_cb
        ])

    dde.saveplot(losshistory, train_state, issave=True, isplot=False)
    model_path = model.save(ck_path)
    print(model_path)
