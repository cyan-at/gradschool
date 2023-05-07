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

######################################

def get_model(
    d,
    N,
    batchsize,
    model_type,
    activations, # sigmoid, tanh
    mu_0,
    sigma_0,
    mu_T,
    sigma_T,
    T_0,
    T_t,
    args,
    network_f,
    network_g,
    optimizer="adam",
    init="Glorot normal",
    train_distribution="Hammersley",
    timemode=0,
    ni=0,
    epsilon=1e-3
    ):
    print("mu_0", mu_0, sigma_0)
    print("mu_T", mu_T, sigma_T)
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

    linspace_tensors = []
    for i in range(d):
        t = torch.from_numpy(
            linspaces[i]).requires_grad_(False)
        t = t.to(device)
        linspace_tensors.append(t)

    # state_tensor = torch.tensor(
    #     state,
    #     dtype=torch.float,
    #     requires_grad=True,
    #     device=device)

    ######################################

    if len(args.population) > 0:
        print("population specified, overriding mu_0 ", args.population)

        population = np.load(args.population, allow_pickle=True)
        final_population = population[:, :, -1]

        rho0 = population_to_pdf(final_population, state)
        l = np.linspace(args.state_bound_min, args.state_bound_max, N)
        rho0 /= square2d_pdfnormalize(rho0, l)

        # import ipdb; ipdb.set_trace()
    else:
        print("rho0 is multivariate_normal")
        rv0 = multivariate_normal(mu_0, sigma_0 * np.eye(d))
        rho0 = np.float32(rv0.pdf(state))

    rho0_tensor = torch.from_numpy(
        rho0,
    )
    rho0_tensor = rho0_tensor.to(device).requires_grad_(False)

    ######################################

    rvT = multivariate_normal(mu_T, sigma_T * np.eye(d))
    rhoT= rvT.pdf(state)
    rhoT = np.float32(rhoT)
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
        component=1,
        # batch_size=batchsize,
        # shuffle=True
        )

    ######################################

    time_t=np.hstack((
        state,
        T_t*np.ones((len(mesh_vectors[0]), 1))
    ))
    rho_T_BC = dde.icbc.PointSetBC(
        time_t,
        rhoT[..., np.newaxis],
        component=1,
        # batch_size=batchsize,
        # shuffle=True
        )

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

    geom=dde.geometry.geometry_3d.Cuboid(
        [args.state_bound_min]*d,
        [args.state_bound_max]*d)
    timedomain = dde.geometry.TimeDomain(T_0, T_t)

    if timemode == 0:
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    elif timemode == 1:
        geomtime = CustomGeometryXTime2(geom, timedomain, M)

    bif = samples_between_initial_and_final
    if args.bif > 0:
        bif = args.bif

    batchsize2 = None
    if len(args.batchsize2) > 0:
        batchsize2 = int(args.batchsize2)

    data = WASSPDE(
        geomtime,
        lambda x, y: tcst_pdes[0](
            x,  y, network_f, network_g, args),
        [rho_0_BC,rho_T_BC],
        num_domain=bif,
        num_initial=ni, # initial_samples,
        train_distribution=train_distribution,
        # domain_batch_size=batchsize2
    )

    # d+1 inputs: <state> + t
    # 5 outputs: 2 eq
    net = dde.nn.FNN(
        [d+1] + [70] *4  + [4],
        # "sigmoid",
        activations,
        init
        # "zeros",
    )
    model = model_types[model_type](data, net)

    ######################################

    dx = linspaces[0][1] - linspaces[0][0]
    print("dx", dx)

    name_tmp = "WASS"
    if "batch" in args.loss_func:
        name_tmp = "WASS_batch"
    print("name_tmp", name_tmp)

    # rho0_WASS = lambda y_true, y_pred: loss_func_dict[args.loss_func](y_true, y_pred, device, sinkhorn0, rho0, state)
    rho0_WASS = lambda y_true, y_pred: loss_func_dict[args.loss_func](
        y_true, y_pred, device, sinkhorn0, rho0_tensor, C_device, N, dx)
    rho0_WASS.__name__ = name_tmp
    # rhoT_WASS = lambda y_true, y_pred: loss_func_dict[args.loss_func](y_true, y_pred, device, sinkhornT, rhoT, state)
    rhoT_WASS = lambda y_true, y_pred: loss_func_dict[args.loss_func](
        y_true, y_pred, device, sinkhorn0, rhoT_tensor, C_device, N, dx)
    rhoT_WASS.__name__ = name_tmp
    losses=[
        "MSE","MSE", "MSE", "MSE",
        rho0_WASS,
        rhoT_WASS,
    ]
    # loss functions are based on PDE + BC: eq outputs, BCs

    model.compile("adam", lr=1e-3,loss=losses)

    # import ipdb; ipdb.set_trace()

    return model, meshes, C_device, rho0_tensor, rhoT_tensor

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

    parser.add_argument('--population',
        type=str,
        default="")

    '''
    '''

    args = parser.parse_args()

    N = args.N
    print("N: ", N)

    if len(args.T_0) > 0:
        T_0 = float(args.T_0)
    print("T_0", T_0)

    if len(args.T_t) > 0:
        T_t = float(args.T_t)
    print("T_t", T_t)

    if args.bound_u > 0:
        print("bounding u")

    d = 2

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    ni = initial_samples
    if args.ni >= 0:
        ni = args.ni

    batchsize = N**d
    if len(args.batchsize) > 0:
        batchsize = int(args.batchsize)

    mu_0 = [float(x) for x in args.mu_0.strip().split(",")]

    target = tcst_map[args.crystal]

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

    model, meshes, _, _, _ = get_model(
        d,
        N,
        batchsize,
        0,
        "tanh",
        mu_0,
        args.sigma,
        target,
        args.sigma,
        T_0,
        T_t,
        args,
        sde.network_f,
        sde.network_g,
        args.optimizer,
        train_distribution=args.train_distribution,
        timemode=args.timemode,
        ni=ni
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

    def save_minibatch():
        fname, _ = Util.get_next_valid_name_increment(
            './', model_name + '_minibatch', 0, '', 'npy')
        np.save(
            fname,
            model.train_state.X_test)
        print("saving minibatch to %s" % (fname))

        _, _, _, _, control_data,\
            _, _, _, _ = make_control_data(
            model, model.train_state.X_test, N, d, meshes, args)
        fname, _ = Util.get_next_valid_name_increment(
            './', model_name + '_minibatch_control_data', 0, '', 'npy')
        np.save(
            fname, 
            control_data
        )
        print("saved control_data to %s" % (fname))

    resampler_cb = PDEPointResampler2(
        pde_points=True,
        bc_points=False,
        period=args.batch2_period,
        cbs=[save_minibatch])

    if args.epochs > 0:
        num_epochs = args.epochs

    start = time.time()
    losshistory, train_state = model.train(
        iterations=num_epochs,
        display_every=de,
        callbacks=[
            earlystop_cb,
            modelcheckpt_cb,
            # resampler_cb
        ],
        model_save_path=ck_path)
    end = time.time()

    print("training dt", end - start)

    ######################################

    dde.saveplot(losshistory, train_state, issave=True, isplot=False)
    model_path = model.save(ck_path)
    print(model_path)

    ######################################

    test, T_t, rho0, rhoT, bc_grids, domain_grids, grid_n_meshes, control_data, tt_u = make_control_data(
        model, model.train_state.X_test, N, d, meshes, args)

    # fname = '%s_%d_%d_%s_%d_%d_all_control_data.npy' % (
    #         model_path.replace(".pt", ""),
    #         N**d,
    #         args.diff_on_cpu,
    #         args.interp_mode,
    #         args.grid_n,
    #         T_t,
    #     )
    fname, _ = Util.get_next_valid_name_increment(
        './',
        model_path.replace(".pt", "") + '_control_data',
        0,
        '',
        'npy')

    np.save(
        fname, 
        control_data
    )
    print("CONTROL_DATA: %s" % (fname))

    ######################################

    '''
    sde2 = SDE2(control_data)
    sde2.load_state_dict(torch.load(files[0]))
    if torch.cuda.is_available():
        print("Using GPU.")
        sde2 = sde2.to(cuda0)
    # set model to evaluation mode
    sde2.eval()

    ts, initial_sample, with_control, without_control,\
        all_results, mus, variances = do_integration2(
            control_data, d, T_0, T_t, mu_0, args.sigma,
            args, sde, sde2)
    '''