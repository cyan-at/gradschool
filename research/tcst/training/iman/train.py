#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

'''
run ./train.py --sdept ../../trained_sde_model/4fold_3_2_layer_model.pt
'''

# 0 define backend
import sys, os, time
import argparse
import glob

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
from layers import *

sde_path = './sde/T_t200_2D/'
sys.path.insert(0,sde_path)
from trained_sde_model import *

from common import *


# In[2]:


def tcst1(x, y, network_f, network_g, args):
    psi, rho, u1, u2 = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]

    # x = c10, c12, t

    # psi eq (4a), rho eq (4b), u1 eq (6), u2 eq (6)
    dpsi_c10 = dde.grad.jacobian(psi, x, j=0)
    dpsi_c12 = dde.grad.jacobian(psi, x, j=1)
    dpsi_t = dde.grad.jacobian(psi, x, j=2)

    hpsi_c10 = dde.grad.hessian(psi, x, i=0, j=0)
    hpsi_c12 = dde.grad.hessian(psi, x, i=1, j=1)

    drho_t = dde.grad.jacobian(rho, x, j=2)

    drho_c10 = dde.grad.hessian(rho, x, i=0, j=0)
    drho_c12 = dde.grad.hessian(rho, x, i=1, j=1)

    # d1
    leaf_x = x[:, 0:2].detach()
    leaf_u1_u2 = y[:, 2:4].detach()
    leaf_t = x[:, 2].detach().unsqueeze(1)

    ###########################################

    leaf_vec = torch.cat(
        (
            x[:, 0:2], # leaf_x,
            # i think this makes sense since we
            # take jacobian of it w.r.t x for divergence
            y[:, 2:4],
            x[:, 2].unsqueeze(1),
        ),
        dim=1)
    leaf_vec = leaf_vec.requires_grad_(True)

    d1 = network_f.forward(leaf_vec)
    d2 = network_g.forward(leaf_vec)**2 / 2 # elementwise
    # divergence terms
    d_rhod1_c10 = dde.grad.jacobian(rho*d1[:, 0], x, j=0)
    d_rhod1_c12 = dde.grad.jacobian(rho*d1[:, 1], x, j=1)

    ###########################################

    # divergence = trace of jacobian
    # divergence is a scalar

    u_term = torch.mul(dpsi_c10.squeeze(), d1[:, 0])\
    + torch.mul(dpsi_c12.squeeze(), d1[:, 1])\
    + torch.mul(d2[:, 0], hpsi_c10.squeeze())\
    + torch.mul(d2[:, 1], hpsi_c12.squeeze()).unsqueeze(dim=0)

    # import ipdb; ipdb.set_trace()

    d_uterm_du1_du2 = torch.autograd.grad(
        outputs=u_term,
        inputs=leaf_vec,
        grad_outputs=torch.ones_like(u_term),
        retain_graph=True)[0]

    l_u1 = u1 - d_uterm_du1_du2[:, 2]
    l_u2 = u2 - d_uterm_du1_du2[:, 3]
    if args.bound_u > 0:
        # print("bounding u")
        l_u1_bound = -torch.sum(u1[u1 < -0.005]) +\
            torch.sum(u1[u1 > 0.005]) 
        l_u2_bound = -torch.sum(u2[u2 < -0.005]) +\
            torch.sum(u2[u2 > 0.005])

        l_u1 += args.bound_u * l_u1_bound
        l_u2 += args.bound_u * l_u2_bound

    return [
        -dpsi_t + 0.5 * (u1**2 + u2**2)\
        - (dpsi_c10 * d1[:, 0] + dpsi_c12 * d1[:, 1])\
        - (d2[:, 0] * hpsi_c10 + d2[:, 1] * hpsi_c12),

        -drho_t - (d_rhod1_c10 + d_rhod1_c12)\
        + (d2[:, 0] * drho_c10 + d2[:, 1] * drho_c12),

        l_u1,
        l_u2
    ]

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
    M = N**d

    linspaces = []
    for i in range(d):
        linspaces.append(np.transpose(
            np.linspace(args.state_bound_min, args.state_bound_max, N))
        )

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

    ######################################

    rv0 = multivariate_normal(mu_0, sigma_0 * np.eye(d))
    rvT = multivariate_normal(mu_T, sigma_T * np.eye(d))

    rho0=rv0.pdf(state)
    rho0 = np.float32(rho0)

    rhoT= rvT.pdf(state)
    rhoT = np.float32(rhoT)

    ######################################

    time_0=np.hstack((
        state,
        T_0*np.ones((len(mesh_vectors[0]), 1))
    ))
    
    if batchsize is not None:
        rho_0_BC = dde.icbc.PointSetBC(
            time_0,
            rho0[..., np.newaxis],
            component=1,
            batch_size=batchsize,
            shuffle=True
        )
    else:
        rho_0_BC = dde.icbc.PointSetBC(
            time_0,
            rho0[..., np.newaxis],
            component=1,
        )

    ######################################

    time_t=np.hstack((
        state,
        T_t*np.ones((len(mesh_vectors[0]), 1))
    ))
    
    if batchsize is not None:
        rho_T_BC = dde.icbc.PointSetBC(
            time_t,
            rhoT[..., np.newaxis],
            component=1,
            batch_size=batchsize,
            shuffle=True
        )
    else:
        rho_T_BC = dde.icbc.PointSetBC(
            time_t,
            rhoT[..., np.newaxis],
            component=1,
        )

    ######################################

    geom=dde.geometry.geometry_3d.Cuboid(
        [args.state_bound_min]*d,
        [args.state_bound_max]*d)
    timedomain = dde.geometry.TimeDomain(0., T_t)

    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bif = samples_between_initial_and_final
    if args.bif > 0:
        bif = args.bif

    batchsize2 = None
    if len(args.batchsize2) > 0:
        batchsize2 = int(args.batchsize2)

    # dde.data.TimePDE
    data = WASSPDE(
        geomtime,
        lambda x, y: tcst1(
            x,  y, network_f, network_g, args),
        [rho_0_BC,rho_T_BC],
        num_domain=bif,
        num_initial=ni, # initial_samples,
        train_distribution=train_distribution,
        domain_batch_size=batchsize2
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

    losses=[
        "MSE","MSE", "MSE", "MSE",
        "MSE",
        "MSE",
    ]
    # loss functions are based on PDE + BC: eq outputs, BCs

    model.compile("adam", lr=1e-3,loss=losses)

    # import ipdb; ipdb.set_trace()

    return model, meshes

if __name__ == '__main__':

    # In[3]:


    sde = SDE()
    # state path to model information file
    # load model parameters

    files = glob.glob(
        sde_path + "/*.pt", 
        recursive = False)
    assert(len(files) == 1)
    print("using model: ", files[0])
    sde.load_state_dict(torch.load(files[0]))

    if torch.cuda.is_available():
        print("Using GPU.")
        sde = sde.to(cuda0)
    # set model to evaluation mode
    sde.eval()


    # In[4]:


    d = 2
    N = 15
    batchsize = None

    mu_0 = [0.35, 0.35]

    sigma = 0.1
    T_t = 200.0
    bcc = np.array([0.41235, 0.37605])

    class Container(object):
        state_bound_min = 0.1
        state_bound_max = 0.6
        bound_u = 0
        
        bif = 100000
        batchsize2 = "5000"
        batch2_period = 5000
    args = Container()

    num_epochs = 15000
    de = 1000


    # In[5]:


    model, meshes = get_model(
        d,
        N,
        batchsize,
        0,
        "tanh",

        mu_0,
        sigma,

        bcc,
        sigma,

        T_t,
        args,
        sde.network_f,
        sde.network_g,
    )

    print(model)


    # In[6]:


    resampler_cb = PDEPointResampler2(
        pde_points=True,
        bc_points=False,
        period=args.batch2_period)
    ck_path = "./tt200_2d_mse"

    start = time.time()
    losshistory, train_state = model.train(
        iterations=num_epochs,
        display_every=de,
        callbacks=[resampler_cb],
        model_save_path=ck_path)
    end = time.time()

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    model_path = model.save(ck_path)
    print(model_path)

    # 
