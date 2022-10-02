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

import argparse
parser = argparse.ArgumentParser(description="")
parser.add_argument('--N', type=int, default=20, help='')
parser.add_argument('--js', type=str, default="1,1,2", help='')
parser.add_argument('--q', type=float, default=0.0, help='')
parser.add_argument('--debug', type=int, default=False, help='')
args = parser.parse_args()

if args.debug:
    torch.autograd.set_detect_anomaly(True)

N = args.N
j1, j2, j3 = [float(x) for x in args.js.split(",")] # axis-symmetric case
q_statepenalty_gain = args.q # 0.5
print("N: ", N)
print("js: ", j1, j2, j3)
print("q: ", q_statepenalty_gain)

######################################

d = 3
M = N**d

linspaces = []
for i in range(d):
    linspaces.append(np.transpose(np.linspace(state_min, state_max, N)))

meshes = np.meshgrid(*linspaces)
mesh_vectors = []
for i in range(d):
    mesh_vectors.append(meshes[i].reshape(M,1))
state = np.hstack(tuple(mesh_vectors))

######################################

def euler_pde(x, y):
    """Euler system.
    dy1_t = g(x)-1/2||Dy1_x||^2-<Dy1_x,f>-epsilon*Dy1_xx
    dy2_t = -D.(y2*(f)+Dy1_x)+epsilon*Dy2_xx
    All collocation-based residuals are defined here
    """
    y1, y2 = y[:, 0:1], y[:, 1:2]

    dy1_x = dde.grad.jacobian(y1, x, j=0)
    dy1_y = dde.grad.jacobian(y1, x, j=1)
    dy1_z = dde.grad.jacobian(y1, x, j=2)
    dy1_t = dde.grad.jacobian(y1, x, j=3)
    dy1_xx = dde.grad.hessian(y1, x, i=0, j=0)
    dy1_yy = dde.grad.hessian(y1, x, i=1, j=1)
    dy1_zz = dde.grad.hessian(y1, x, i=2, j=2)

    dy2_x = dde.grad.jacobian(y2, x, j=0)
    dy2_y = dde.grad.jacobian(y2, x, j=1)
    dy2_z = dde.grad.jacobian(y2, x, j=2)
    dy2_t = dde.grad.jacobian(y2, x, j=3)

    dy2_xx = dde.grad.hessian(y2, x, i=0, j=0)
    dy2_yy = dde.grad.hessian(y2, x, i=1, j=1)
    dy2_zz = dde.grad.hessian(y2, x, i=2, j=2)

    """Compute Jacobian matrix J: J[i][j] = dy_i / dx_j, where i = 0, ..., dim_y - 1 and
    j = 0, ..., dim_x - 1.
    Use this function to compute first-order derivatives instead of ``tf.gradients()``
    or ``torch.autograd.grad()``, because

    - It is lazy evaluation, i.e., it only computes J[i][j] when needed.
    - It will remember the gradients that have already been computed to avoid duplicate
      computation.
    """

    """Compute Hessian matrix H: H[i][j] = d^2y / dx_i dx_j, where i,j=0,...,dim_x-1.

    Use this function to compute second-order derivatives instead of ``tf.gradients()``
    or ``torch.autograd.grad()``, because

    - It is lazy evaluation, i.e., it only computes H[i][j] when needed.
    - It will remember the gradients that have already been computed to avoid duplicate
      computation."""

    f1=x[:, 1:2]*x[:, 2:3]*(j2-j3)/j1
    f2=x[:, 0:1]*x[:, 2:3]*(j3-j1)/j2
    f3=x[:, 0:1]*x[:, 1:2]*(j1-j2)/j3
    
    # d_f1dy1_y2_x=tf.gradients((f1+dy1_x)*y2, x)[0][:, 0:1]
    # d_f2dy1_y2_y=tf.gradients((f2+dy1_y)*y2, x)[0][:, 1:2]
    # d_f3dy1_y2_z=tf.gradients((f3+dy1_z)*y2, x)[0][:, 2:3]
    d_f1dy1_y2_x = dde.grad.jacobian((f1+dy1_x)*y2, x, j=0)
    d_f2dy1_y2_y = dde.grad.jacobian((f2+dy1_y)*y2, x, j=1)
    d_f3dy1_y2_z = dde.grad.jacobian((f3+dy1_z)*y2, x, j=2)

    # stay close to origin while searching, penalizes large state distance solutions
    q = q_statepenalty_gain*(
        x[:, 0:1] * x[:, 0:1]\
        + x[:, 1:2] * x[:, 1:2]\
        + x[:, 2:3] * x[:, 2:3])
    # also try
    # q = 0 # minimum effort control

    psi = -dy1_t + q - .5*(dy1_x*dy1_x+dy1_y*dy1_y+dy1_z*dy1_z) - (dy1_x*f1 + dy1_y*f2 + dy1_z*f3) - epsilon*(dy1_xx+dy1_yy+dy1_zz)

    dpsi_x = dde.grad.jacobian(psi, x, j=0)
    dpsi_y = dde.grad.jacobian(psi, x, j=1)
    dpsi_z = dde.grad.jacobian(psi, x, j=2)

    # TODO: verify this expression
    return [
        psi,
        -dy2_t-(d_f1dy1_y2_x+d_f2dy1_y2_y+d_f3dy1_y2_z)+epsilon*(dy2_xx+dy2_yy+dy2_zz),
        #U1 - dpsi_x,
        #U2 - dpsi_y,
        #U3 - dpsi_z,
    ]

######################################

rho0_name = 'rho0_%.3f_%.3f__%.3f_%.3f__%d.dat' % (
    mu_0, sigma_0,
    state_min, state_max,
    N)
trunc_rho0_pdf = get_multivariate_truncated_pdf(
    mesh_vectors[0],
    mesh_vectors[1],
    mesh_vectors[2],
    mu_0, sigma_0, state_min, state_max, N, f, rho0_name)
time_0=np.hstack((
    mesh_vectors[0],
    mesh_vectors[1],
    mesh_vectors[2],
    T_0*np.ones((len(mesh_vectors[0]), 1))
))
rho_0_BC = dde.icbc.PointSetBC(
    time_0,
    trunc_rho0_pdf, component=1)

######################################

rhoT_name = 'rhoT_%.3f_%.3f__%.3f_%.3f__%d.dat' % (
    mu_T, sigma_T,
    state_min, state_max,
    N)
trunc_rhoT_pdf = get_multivariate_truncated_pdf(
    mesh_vectors[0],
    mesh_vectors[1],
    mesh_vectors[2],
    mu_T, sigma_T, state_min, state_max, N, f, rhoT_name)
time_t=np.hstack((
    mesh_vectors[0],
    mesh_vectors[1],
    mesh_vectors[2],
    T_t*np.ones((len(mesh_vectors[0]), 1))
))
rho_T_BC = dde.icbc.PointSetBC(time_t, trunc_rhoT_pdf, component=1)

######################################

rho0_tensor = torch.from_numpy(
    trunc_rho0_pdf
).requires_grad_(False).type(torch.FloatTensor).view(-1)
rho0_tensor = rho0_tensor.to(device)

rhoT_tensor = torch.from_numpy(
    trunc_rhoT_pdf
).requires_grad_(False).type(torch.FloatTensor).view(-1)
rhoT_tensor = rhoT_tensor.to(device)

######################################

C = cdist(state, state, 'sqeuclidean')
cvector = C.reshape((M)**2)

reg = 10e-1 # gamma, 10e-2, 5e-2
C_tensor = torch.from_numpy(
    -C / reg - 1
).requires_grad_(False).type(torch.FloatTensor)
C_tensor = C_tensor.to(device)
c_tensor = torch.from_numpy(
    cvector
).requires_grad_(False).type(torch.FloatTensor)
c_tensor = c_tensor.to(device)
M = torch.exp(C_tensor).type(torch.FloatTensor)
M = M.to(device)

u_vec0 = torch.ones(rho0_tensor.shape[0], dtype=torch.float32).requires_grad_(True)
u_vec0 = u_vec0.to(device)
v_vec0 = torch.ones(rho0_tensor.shape[0], dtype=torch.float32).requires_grad_(True)
v_vec0 = v_vec0.to(device)

u_vecT = torch.ones(rho0_tensor.shape[0], dtype=torch.float32).requires_grad_(True)
u_vecT = u_vecT.to(device)
v_vecT = torch.ones(rho0_tensor.shape[0], dtype=torch.float32).requires_grad_(True)
v_vecT = v_vecT.to(device)

p_opt0 = torch.zeros_like(M).requires_grad_(True)
p_opt0 = p_opt0.to(device)

p_optT = torch.zeros_like(M).requires_grad_(True)
p_optT = p_optT.to(device)

######################################

# import ipdb; ipdb.set_trace()

def rho0_WASS_cuda0(y_true, y_pred):
    return sinkhorn_torch(M,
        c_tensor,
        rho0_tensor,
        y_pred.view(-1),
        u_vec0,
        v_vec0,
        p_opt0,
        device,
        delta=1e-1,
        lam=1e-6)

def rhoT_WASS_cuda0(y_true, y_pred):
    return sinkhorn_torch(M,
        c_tensor,
        rhoT_tensor,
        y_pred.view(-1),
        u_vecT,
        v_vecT,
        p_optT,
        device,
        delta=1e-1,
        lam=1e-6)

######################################

geom=dde.geometry.geometry_3d.Cuboid(
    [state_min, state_min, state_min],
    [state_max, state_max, state_max])
timedomain = dde.geometry.TimeDomain(0., T_t)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

data = dde.data.TimePDE(
    geomtime,
    euler_pde,
    [rho_0_BC,rho_T_BC],
    num_domain=samples_between_initial_and_final,
    num_initial=initial_and_final_samples)

# 4 inputs: x,y,z,t
# 5 outputs: 2 eq + 3 control vars
net = dde.nn.FNN(
    [4] + [70] *3  + [2],
    # "sigmoid",
    # "zeros",
    "tanh",
    "Glorot normal"
)
model = dde.Model(data, net)

######################################

ck_path = "%s/%s_model" % (os.path.abspath("./"), id_prefix)
class EarlyStoppingFixed(dde.callbacks.EarlyStopping):
    def on_epoch_end(self):
        current = self.get_monitor_value()
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            # must meet baseline first
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = self.model.train_state.epoch
                self.model.stop_training = True
        else:
            self.wait = 0
                
    def on_train_end(self):
        if self.stopped_epoch > 0:
            print("Epoch {}: early stopping".format(self.stopped_epoch))
        
        self.model.save(ck_path, verbose=True)

    def get_monitor_value(self):
        if self.monitor == "train loss" or self.monitor == "loss_train":
            data = self.model.train_state.loss_train
        elif self.monitor == "test loss" or self.monitor == "loss_test":
            data = self.model.train_state.loss_test
        else:
            raise ValueError("The specified monitor function is incorrect.", self.monitor)

        result = max(data)
        if min(data) < 1e-50:
            print("likely a numerical error")
            # numerical error
            return 1.0

        return result
earlystop_cb = EarlyStoppingFixed(baseline=1e-3, patience=0)

class ModelCheckpoint2(dde.callbacks.ModelCheckpoint):
    def on_epoch_end(self):
        current = self.get_monitor_value()
        if self.monitor_op(current, self.best) and current < 1e-1:
            save_path = self.model.save(self.filepath, verbose=0)
            print(
                "Epoch {}: {} improved from {:.2e} to {:.2e}, saving model to {} ...\n".format(
                    self.model.train_state.epoch,
                    self.monitor,
                    self.best,
                    current,
                    save_path,
                ))

            test_path = save_path.replace(".pt", "-%d.dat" % (
                self.model.train_state.epoch))
            test = np.hstack((
                self.model.train_state.X_test,
                self.model.train_state.y_pred_test))
            np.savetxt(test_path, test, header="x, y_pred")
            print("saved test data to ", test_path)

            self.best = current

    def get_monitor_value(self):
        if self.monitor == "train loss" or self.monitor == "loss_train":
            data = self.model.train_state.loss_train
        elif self.monitor == "test loss" or self.monitor == "loss_test":
            data = self.model.train_state.loss_test
        else:
            raise ValueError("The specified monitor function is incorrect.", self.monitor)

        result = max(data)
        if min(data) < 1e-50:
            print("likely a numerical error")
            # numerical error
            return 1.0

        return result
modelcheckpt_cb = ModelCheckpoint2(
    ck_path, verbose=True, save_better_only=True, period=1)

######################################

loss_func=[
    "MSE","MSE",
    rho0_WASS_cuda0,
    rhoT_WASS_cuda0
]
# loss functions are based on PDE + BC: 2 eq outputs, 2 BCs

model.compile("adam", lr=1e-3,loss=loss_func)
de = 1
losshistory, train_state = model.train(
    iterations=num_epochs,
    display_every=de,
    callbacks=[earlystop_cb, modelcheckpt_cb])

######################################

dde.saveplot(losshistory, train_state, issave=True, isplot=False)
model_path = model.save(ck_path)
print(model_path)

######################################
