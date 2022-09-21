#!/usr/bin/env python
# coding: utf-8

# 0 define backend
import sys, os, time
import numpy as np
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt
import pylab
from os.path import dirname, join as pjoin
import os, glob
import numpy as np

os.environ['XLA_FLAGS'] = "--xla_gpu_cuda_data_dir=/usr/local/home/cyan3/miniforge/envs/tf"
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
device = cuda0

from scipy import stats
import scipy.io
from scipy.stats import truncnorm, norm
from scipy.optimize import linprog
from scipy import sparse
from scipy.stats import multivariate_normal
from scipy.linalg import solve_discrete_are
from scipy.linalg import sqrtm

os.environ['DDE_BACKEND'] = "pytorch" # v2
# https://stackoverflow.com/questions/68614547/tensorflow-libdevice-not-found-why-is-it-not-found-in-the-searched-path
# this directory has /nvvm/libdevice/libdevice.10.bc
print(os.environ['DDE_BACKEND'])
import deepxde as dde
if dde.backend.backend_name == "pytorch":
    exp = dde.backend.torch.exp
else:
    from deepxde.backend import tf
    exp = tf.exp

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from common import *

############################################################

x_T = np.transpose(np.linspace(state_min, state_max, N))
y_T = np.transpose(np.linspace(state_min, state_max, N))
z_T = np.transpose(np.linspace(state_min, state_max, N))
x_T=x_T.reshape(len(x_T),1)
y_T=y_T.reshape(len(y_T),1)
z_T=z_T.reshape(len(z_T),1)

xT_tensor = torch.from_numpy(
    x_T
).requires_grad_(False)
xT_tensor = xT_tensor.to(device)

############################################################

rho_0=pdf1d(x_T, mu_0, sigma_0).reshape(len(x_T),1)
rho_0 = np.where(rho_0 < 0, 0, rho_0)
rho_0 /= np.trapz(rho_0, x=x_T, axis=0)[0] # pdf
rho_0 = rho_0 / np.sum(np.abs(rho_0)) # pmf
time_0=np.hstack((x_T,T_0*np.ones((len(x_T), 1))))
rho_0_BC = dde.icbc.PointSetBC(time_0, rho_0, component=1)
rho_0_tensor = torch.from_numpy(
    rho_0
).requires_grad_(False)
rho_0_tensor = rho_0_tensor.to(device)

rho_T=pdf1d(x_T, mu_T, sigma_T).reshape(len(x_T),1)
rho_T = np.where(rho_T < 0, 0, rho_T)
rho_T /= np.trapz(rho_T, x=x_T, axis=0)[0] # pdf
rho_T = rho_T / np.sum(np.abs(rho_T)) # pmf
time_t=np.hstack((x_T,T_t*np.ones((len(x_T), 1))))
rho_T_BC = dde.icbc.PointSetBC(time_t, rho_T, component=1)
rho_T_tensor = torch.from_numpy(
    rho_T
).requires_grad_(False)
rho_T_tensor = rho_T_tensor.to(device)

############################################################

S3 = dde.Variable(1.0)
a, b, c, d, f= 10., 2.1, 0.75, .0045, 0.0005
K, T=1.38066*10**-23, 293.
def pde(x, y):
    """Self assembly system.
    dy1_t = 1/2*(y3^2)-dy1_x*D1-dy1_xx*D2
    dy2_t = -dD1y2_x +dD2y2_xx
    y3=dy1_x*dD1_y3+dy1_xx*dD2_y3
    All collocation-based residuals are defined here
    Including a penalty function for negative solutions
    """
    y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:]
    dy1_t = dde.grad.jacobian(y1, x, j=1)
    dy1_x = dde.grad.jacobian(y1, x, j=0)
    dy1_xx = dde.grad.hessian(y1, x, j=0)

    D2=d*torch.exp(-(x[:, 0:1]-b-c*y3)*(x[:, 0:1]-b-c*y3))+f
    F=a*K*T*(x[:, 0:1]-b-c*y3)*(x[:, 0:1]-b-c*y3)
#     dD2_x=dde.grad.jacobian(D2, x, j=0)
#     dF_x=dde.grad.jacobian(F, x, j=0)
#     D1=dD2_x-dF_x*(D2/(K*T))
    D1=-2*(x[:, 0:1]-b-c*y3)*((d*torch.exp(-(x[:, 0:1]-b-c*y3)*(x[:, 0:1]-b-c*y3)))+a*D2)
    dy2_t = dde.grad.jacobian(y2, x, j=1)
    dD1y2_x=dde.grad.jacobian(D1*y2, x, j=0)
    dD2y2_xx = dde.grad.hessian(D2*y2, x,  j=0)
    dD1_y3=dde.grad.jacobian(D1, y3)
    dD2_y3=dde.grad.jacobian(D2, y3)
    tt=100
    return [
        dy1_t-.5*(S3*y3*S3*y3)+D1*dy1_x+D2*dy1_xx,
        dy2_t+dD1y2_x-dD2y2_xx,
        S3*y3-dy1_x*dD1_y3-dy1_xx*dD2_y3,
#         neg_loss,
#         neg_loss_y2,
    ]

############################################################

[X,Y] = np.meshgrid(x_T, y_T)
C = (X - Y)**2 # 1D squared euclidean distance
cvector = C.reshape(N**2,1)
cvector_tensor = torch.from_numpy(
    cvector.reshape(-1)
).requires_grad_(False)
cvector_tensor = cvector_tensor.to(device)

A = np.concatenate(
    (
        np.kron(
            np.ones((1,N)),
            sparse.eye(N).toarray()
        ),
        np.kron(
            sparse.eye(N).toarray(),
            np.ones((1,N))
        )
    ), axis=0)
# 2*N x N**2

# Define and solve the CVXPY problem.
x = cp.Variable(
    cvector.shape[0],
    nonneg=True
)
pred = cp.Parameter((A.shape[0],))
problem = cp.Problem(
    cp.Minimize(cvector.T @ x),
    [
        A @ x == pred,
    ],
)
assert problem.is_dpp()
cvxpylayer = CvxpyLayer(
    problem,
    parameters=[pred],
    variables=[x])

############################################################

def rho0_WASS_cuda0(y_true, y_pred):
    # avoid moving to speed up
    # y_pred = y_pred.to(cuda0)
    # y_pred.retain_grad()
    # total = fail_cost

    # # normalize to pdf
    y_pred = torch.where(y_pred < 0, 0, y_pred)
    y_pred /= torch.trapz(y_pred, x=xT_tensor, dim=0)[0]

    # normalize to PMF
    y_pred = y_pred / torch.sum(torch.abs(y_pred))

    param = torch.cat((rho_0_tensor, y_pred), 0)
    param = torch.reshape(param, (2*N,))
    # print(type(param))
    # try:
    x_sol, = cvxpylayer(param, solver_args={
        'max_iters': 500000,
        # 'eps' : 1e-5,
        'solve_method' : 'ECOS'
    }) # or ECOS, ECOS is faster
    wass_dist = torch.matmul(cvector_tensor, x_sol)
    wass_dist = torch.sqrt(wass_dist)

    # ECOS might return nan
    # SCS is slower, and you need 'luck'?
    wass_dist = torch.nan_to_num(wass_dist, 1e3)

    return wass_dist
    # total = wass_dist
    # except:
    #     pass

    # return total

def rhoT_WASS_cuda0(y_true, y_pred):
    # avoid moving to speed up
    # y_pred = y_pred.to(cuda0)
    # y_pred.retain_grad()
    # total = fail_cost

    # # normalize to pdf
    y_pred = torch.where(y_pred < 0, 0, y_pred)
    y_pred /= torch.trapz(y_pred, x=xT_tensor, dim=0)[0]

    # normalize to PMF
    y_pred = y_pred / torch.sum(torch.abs(y_pred))

    param = torch.cat((rho_T_tensor, y_pred), 0)
    param = torch.reshape(param, (2*N,))
    # print(type(param))
    # try:
    x_sol, = cvxpylayer(param, solver_args={
        'max_iters': 10000,
        # 'eps' : 1e-5,
        'solve_method' : 'ECOS'
    }) # or ECOS, ECOS is faster
    wass_dist = torch.matmul(cvector_tensor, x_sol)
    wass_dist = torch.sqrt(wass_dist)

    # ECOS might return nan
    # SCS is slower, and you need 'luck'?
    wass_dist = torch.nan_to_num(wass_dist, 1e3)

    return wass_dist
    # total = wass_dist
    # except:
    #     pass

    # return total

############################################################

geom = dde.geometry.Interval(state_min, state_max)
timedomain = dde.geometry.TimeDomain(0., T_t)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)
data = dde.data.TimePDE(
    geomtime,
    pde,
    [rho_0_BC,rho_T_BC],
    num_domain=5000,
    num_initial=500)
# 2 inputs: x + t
# 3 outputs: 3 eqs
net = dde.nn.FNN([2] + [70] *3  + [3], "tanh", "Glorot normal")
model = dde.Model(data, net)

############################################################

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

loss_func=[
    "MSE","MSE","MSE",
    rho0_WASS_cuda0,
    rhoT_WASS_cuda0]
# loss functions are based on PDE + BC: 3 eq outputs, 2 BCs

############################################################

model.compile("adam", lr=1e-3,loss=loss_func)
losshistory, train_state = model.train(
    iterations=10000,
    display_every=de,
    callbacks=[earlystop_cb, modelcheckpt_cb])

############################################################

dde.saveplot(losshistory, train_state, issave=True, isplot=False)
model_path = model.save(ck_path)
print(model_path)

############################################################

loss_loaded = np.genfromtxt('./loss.dat')
# [0] epoch
# [1] y1, psi, hjb
# [2] y2, rho, plank pde
# [3] rho0, initial
# [4] rhoT, terminal
epoch = loss_loaded[:, 0]
y1_psi_hjb = loss_loaded[:, 1]
y2_rho_plankpde = loss_loaded[:, 2]
loss3 = loss_loaded[:, 3]
rho0_initial = loss_loaded[:, 4]
rhoT_terminal = loss_loaded[:, 5]

fig, ax = plt.subplots()

line1, = ax.plot(epoch, y1_psi_hjb, color='orange', lw=1, label='eq1')
line2, = ax.plot(epoch, y2_rho_plankpde, color='blue', lw=1, label='eq2')
line2, = ax.plot(epoch, loss3, color='green', lw=1, label='eq3')
line3, = ax.plot(epoch, rho0_initial, color='red', lw=1, label='p0 boundary condition')
line4, = ax.plot(epoch, rhoT_terminal, color='purple', lw=1, label='pT boundary condition')

ax.grid()
ax.legend(loc="lower left")
ax.set_title('training error/residual plots: %d epochs' % (len(epoch)*de))
ax.set_yscale('log')
ax.set_xscale('log')
plot_fname = "%s/%s_loss.png" % (
    os.path.abspath("./"),
    id_prefix)
plt.savefig(plot_fname, dpi=300)
print("saved plot")

############################################################

test = np.loadtxt('./test.dat')

############################################################

fig = plt.figure(1)
ax1 = plt.subplot(111, frameon=False)
ax1.grid()

############################################################

s1, s2 = plot_rho_bc('rho_0', test[0:N, :], mu_0, sigma_0, ax1)

test_tt = test[N:2*N, :]
s3, s4 = plot_rho_bc('rho_T', test_tt, mu_T, sigma_T, ax1)

############################################################

ax1.plot(
    test_tt[:, X_IDX],
    test_tt[:, Y3_IDX],
    linewidth=1,
    c='m',
    label='y3')

############################################################

ax1.legend(loc='lower right')
ax1.set_title(
    'rho0: trapz=%.3f, sum=%.3f, rhoT: trapz=%.3f, sum=%.3f' % (s1, s2, s3, s4))

plot_fname = "%s/pinn_vs_rho_%s.png" % (os.path.abspath("./"), id_prefix)
plt.savefig(plot_fname, dpi=300)
# plt.show()

