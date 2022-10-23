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
args = parser.parse_args()

if args.debug:
    torch.autograd.set_detect_anomaly(True)

N = args.N
j1, j2, j3 = [float(x) for x in args.js.split(",")] # axis-symmetric case
q_statepenalty_gain = args.q # 0.5
print("N: ", N)
print("js: ", j1, j2, j3)
print("q: ", q_statepenalty_gain)

k1 = 1.0
k2 = 5.0
k3 = -1e-2

######################################

# a 3d grid, not the sparse diagonal elements as above
x1 = np.transpose(np.linspace(state_min, state_max, N)).astype(f)
x2 = np.transpose(np.linspace(state_min, state_max, N)).astype(f)
x3 = np.transpose(np.linspace(state_min, state_max, N)).astype(f)
[X,Y,Z] = np.meshgrid(x1,x2,x3)
x_T=X.reshape(N**3,1)
y_T=Y.reshape(N**3,1)
z_T=Z.reshape(N**3,1)

######################################

# U1 = dde.Variable(1.0)
# U2 = dde.Variable(1.0)
# U3 = dde.Variable(1.0)
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
trunc_rho0_pdf = get_multivariate_truncated_pdf(x_T, y_T, z_T, mu_0, sigma_0, state_min, state_max, N, f, rho0_name)

time_0=np.hstack((
    x_T,
    y_T,
    z_T,
    T_0*np.ones((len(x_T), 1))
))

rho_0_BC = dde.icbc.PointSetBC(time_0, trunc_rho0_pdf, component=1)

######################################

rhoT_name = 'rhoT_%.3f_%.3f__%.3f_%.3f__%d.dat' % (
    mu_T, sigma_T,
    state_min, state_max,
    N)
trunc_rhoT_pdf = get_multivariate_truncated_pdf(x_T, y_T, z_T, mu_T, sigma_T, state_min, state_max, N, f, rhoT_name)

time_t=np.hstack((
    x_T,
    y_T,
    z_T,
    T_t*np.ones((len(x_T), 1))
))

rho_T_BC = dde.icbc.PointSetBC(time_t, trunc_rhoT_pdf, component=1)

######################################

x1_tensor = torch.from_numpy(
    x1).requires_grad_(False).to(device)
x2_tensor = torch.from_numpy(
    x2).requires_grad_(False).to(device)
x3_tensor = torch.from_numpy(
    x3).requires_grad_(False).to(device)

x_T_tensor = torch.from_numpy(
    x_T).requires_grad_(False).to(device)
y_T_tensor = torch.from_numpy(
    y_T).requires_grad_(False).to(device)
z_T_tensor = torch.from_numpy(
    z_T).requires_grad_(False).to(device)

trunc_rho0_tensor = torch.from_numpy(
    trunc_rho0_pdf
).requires_grad_(False)
trunc_rho0_tensor = trunc_rho0_tensor.to(device)

rho0_m, rho0_sig = get_pmf_stats_torch(
    trunc_rho0_tensor,
    x_T_tensor, y_T_tensor, z_T_tensor,
    x1_tensor, x2_tensor, x3_tensor, dt)
rho0_sig_diagonal = torch.diagonal(rho0_sig)

trunc_rhoT_tensor = torch.from_numpy(
    trunc_rhoT_pdf
).requires_grad_(False)
trunc_rhoT_tensor = trunc_rhoT_tensor.to(device)

rhoT_m, rhoT_sig = get_pmf_stats_torch(
    trunc_rhoT_tensor,
    x_T_tensor, y_T_tensor, z_T_tensor,
    x1_tensor, x2_tensor, x3_tensor, dt)
rhoT_sig_diagonal = torch.diagonal(rhoT_sig)

r = 1e-5
e = torch.ones((3,3)) * r
# regularizer to prevent nan in matrix sqrt
# nan in gradient

rho0_w1 = sqrtm(rho0_sig)
print("rho0_w1\n", rho0_w1)

rhoT_w1 = sqrtm(rhoT_sig)
print("rhoT_w1\n", rhoT_w1)

def rho0_WASS_cuda0(y_true, y_pred):
    p1 = (y_pred< k3).sum() # negative terms

    # if p1 > 0:
    #     import ipdb; ipdb.set_trace()

    # pdf_support = get_pdf_support_torch(y_pred, [x1_tensor, x2_tensor, x3_tensor], 0)
    pmf_support = torch.sum(y_pred)

    # if pmf_support < 0:
    #     import ipdb; ipdb.set_trace()

    p2 = torch.abs(pmf_support - 1)

    # print("p1, p2", p1, p2)
    # if p1 > 0 or p2 > k2:
    #     # print("pmf_support", pmf_support)
    #     return k1 * (p1 + p2)

    # print("non-negative, pmf")

    '''
    # normalize to pdf
    y_pred = torch.where(y_pred < 0, 0, y_pred)

    pdf_support = get_pdf_support_torch(y_pred, [x1_tensor, x2_tensor, x3_tensor], 0)
    if pdf_support > 1e-3:
        y_pred /= pdf_support

    # normalize to PMF
    pmf_support = torch.sum(y_pred)
    if pmf_support > 1e-3:
        y_pred = y_pred / pmf_support
    '''

    ym, ysig = get_pmf_stats_torch(
        y_pred,
        x_T_tensor, y_T_tensor, z_T_tensor,
        x1_tensor, x2_tensor, x3_tensor, dt)

    w1 = torch.norm(ym - rho0_m, p=2)
    w2 = torch.norm(
        torch.diagonal(ysig) - rho0_sig_diagonal)
    w3 = torch.norm(torch.diagonal(ysig, 1), p=2)
    w4 = torch.norm(torch.diagonal(ysig, 2), p=2)

    if torch.isnan(w1) or torch.isnan(w2) or torch.isnan(w3) or torch.isnan(w4):
        # print("bad pmf")
        w1 = w2 = w3 = w4 = 0.0

    return p1 + p2 + w1 + w2 + w3 + w4

    '''
    ysig = torch.nan_to_num(ysig) + e

    # if torch.max(ysig) < 1e-3:
    #     print("degenerate\n", ysig)

    #     import ipdb; ipdb.set_trace();

    #     tmp = torch.mean(torch.square(y_true - y_pred))
    #     return torch.nan_to_num(tmp)

    c = rho0_w1 * ysig * rho0_w1 + e
    # torch.sqrt element-wise sqrt of c introduces
    # nans into the gradient, and then
    # y_pred becomes nan and all losses become nan
    # and learning stops
    a = sqrtm(c)
    # a = torch.nan_to_num(a) + e

    b = torch.trace(rho0_sig + ysig - 2*a)
    b = torch.nan_to_num(b) + r

    # print("c\n", c)
    # print("a max\n", a)
    # print("ysig max\n", ysig)
    # print("b max\n", b)
    # print("ym\n", ym)

    w = torch.norm(rho0_m - ym, p=2) + b

    return p1 + p2 + w
    '''

def rhoT_WASS_cuda0(y_true, y_pred):
    p1 = (y_pred< k3).sum() # negative terms

    # pdf_support = get_pdf_support_torch(y_pred, [x1_tensor, x2_tensor, x3_tensor], 0)
    pmf_support = torch.sum(y_pred)
    p2 = torch.abs(pmf_support - 1)

    # if p1 > 0 or p2 > k2:
    #     return k1 * (p1 + p2)

    # if p1 > 1e-3 or p2 > 1e-3:
    #     return k1 * (p1 + p2)

    # print("non-negative, pmf")

    '''
    # normalize to pdf
    y_pred = torch.where(y_pred < 0, 0, y_pred)

    pdf_support = get_pdf_support_torch(y_pred, [x1_tensor, x2_tensor, x3_tensor], 0)
    if pdf_support > 1e-3:
        y_pred /= pdf_support

    # normalize to PMF
    pmf_support = torch.sum(y_pred)
    if pmf_support > 1e-3:
        y_pred = y_pred / pmf_support
    '''

    ym, ysig = get_pmf_stats_torch(
        y_pred,
        x_T_tensor, y_T_tensor, z_T_tensor,
        x1_tensor, x2_tensor, x3_tensor, dt)

    w1 = torch.norm(ym - rhoT_m, p=2)
    w2 = torch.norm(
        torch.diagonal(ysig) - rhoT_sig_diagonal)
    w3 = torch.norm(torch.diagonal(ysig, 1), p=2)
    w4 = torch.norm(torch.diagonal(ysig, 2), p=2)

    if torch.isnan(w1) or torch.isnan(w2) or torch.isnan(w3) or torch.isnan(w4):
        w1 = w2 = w3 = w4 = 0.0

    return p1 + p2 + w1 + w2 + w3 + w4

    '''
    # if torch.max(ysig) < 1e-3:
    #     print("degenerate\n", ysig)

    #     import ipdb; ipdb.set_trace();

    #     tmp = torch.mean(torch.square(y_true - y_pred))
    #     return torch.nan_to_num(tmp)

    c = rhoT_w1 * ysig * rhoT_w1 + e
    # torch.sqrt element-wise sqrt of c introduces
    # nans into the gradient, and then
    # y_pred becomes nan and all losses become nan
    # and learning stops
    a = sqrtm(c)
    # a = torch.nan_to_num(a) + e

    b = torch.trace(rhoT_sig + ysig - 2*a)
    b = torch.nan_to_num(b) + r

    # print("c\n", c)
    # print("a max\n", a)
    # print("ysig max\n", ysig)
    # print("b max\n", b)
    # print("ym\n", ym)

    w = torch.norm(rhoT_m - ym, p=2) + b

    return p1 + p2 + w
    '''

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
    "sigmoid",
    # "zeros",
    # "tanh",
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
