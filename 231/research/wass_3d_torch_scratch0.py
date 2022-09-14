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



print(time.time())


# In[13]:


nb_name = "none"
# get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.kernel.execute(\'nb_name = "\' + IPython.notebook.notebook_name + \'"\')\n')


# In[28]:


N = nSample = 50

# must be floats
state_min = 0.0
state_max = 6.0

mu_0 = 2.0
sigma_0 = 1.5

mu_T = 4.0
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

id_prefix = "3d"

print(id_prefix)
print(time.time())


# In[ ]:





# In[29]:


def pdf1d(x, mu, sigma):
    a, b = (state_min - mu) / sigma, (state_max - mu) / sigma
    rho_x=truncnorm.pdf(x, a, b, loc = mu, scale = sigma)

    # do NOT use gaussian norm, because it is only area=1
    # from -inf, inf, will not be for finite state/grid
    # rho_x = norm.pdf(x, mu, sigma)
    return rho_x

rv0 = multivariate_normal([mu_0, mu_0, mu_0], sigma_0 * np.eye(3))
rvT = multivariate_normal([mu_T, mu_T, mu_T], sigma_T * np.eye(3))
def pdf3d(x,y,z,rv):
    return rv.pdf(np.hstack((x, y, z)))

def boundary(_, on_initial):
    return on_initial

print(time.time())


# In[30]:


x_T = np.transpose(np.linspace(state_min, state_max, N))
y_T = np.transpose(np.linspace(state_min, state_max, N))
z_T = np.transpose(np.linspace(state_min, state_max, N))
x_T=x_T.reshape(len(x_T),1)
y_T=y_T.reshape(len(y_T),1)
z_T=z_T.reshape(len(z_T),1)
print(time.time())


# In[31]:

time_0=np.hstack((x_T,T_0*np.ones((len(x_T), 1))))

# rho_0=pdf1d(x_T, mu_0, sigma_0).reshape(len(x_T),1)
rho_0=pdf3d(x_T,y_T,z_T, rv0).reshape(len(x_T),1)

rho_0 = np.where(rho_0 < 0, 0, rho_0)
rho_0 /= np.trapz(rho_0, x=x_T, axis=0)[0] # pdf
rho_0 = rho_0 / np.sum(np.abs(rho_0)) # pmf
print(np.sum(rho_0))

rho_0_BC = dde.icbc.PointSetBC(time_0, rho_0, component=1)


time_t=np.hstack((x_T,T_t*np.ones((len(x_T), 1))))

# rho_T=pdf1d(x_T, mu_T, sigma_T).reshape(len(x_T),1)
rho_T=pdf3d(x_T,y_T,z_T, rvT).reshape(len(x_T),1)

rho_T = np.where(rho_T < 0, 0, rho_T)
rho_T /= np.trapz(rho_T, x=x_T, axis=0)[0] # pdf
rho_T = rho_T / np.sum(np.abs(rho_T)) # pmf
print(np.sum(rho_T))

rho_T_BC = dde.icbc.PointSetBC(time_t, rho_T, component=1)

print(time.time())


# In[9]:


# # 1d linprog example
# res = linprog(
#     cvector,
#     A_eq=A,
#     b_eq=np.concatenate((rho_0, rho_T), axis=0),
#     bounds=[(0, np.inf)],
#     options={"disp": True}
# )
# print(res.fun)

# print(time.time())


# In[ ]:





# In[17]:


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

def euler_pde(x, y):
    """Euler system.
    dy1_t = g(x)-1/2||Dy1_x||^2-<Dy1_x,f>-epsilon*Dy1_xx
    dy2_t = -D.(y2*(f)+Dy1_x)+epsilon*Dy2_xx
    All collocation-based residuals are defined here
    """
    y1, y2 = y[:, 0:1], y[:, 1:]

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
    
    # TODO: verify this expression
    return [
        -dy1_t+q-.5*(dy1_x*dy1_x+dy1_y*dy1_y+dy1_z*dy1_z)-dy1_x*f1-dy1_y*f2-dy1_z*f3-epsilon*(dy1_xx+dy1_yy+dy1_zz),
        -dy2_t-(d_f1dy1_y2_x+d_f2dy1_y2_y+d_f3dy1_y2_z)+epsilon*(dy2_xx+dy2_yy+dy2_zz),
    ]


print(time.time())


# In[22]:


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

xT_tensor = torch.from_numpy(
    x_T
).requires_grad_(False)

rho_0_tensor = torch.from_numpy(
    rho_0
).requires_grad_(False)


rho_T_tensor = torch.from_numpy(
    rho_T
).requires_grad_(False)

cvector_tensor = torch.from_numpy(
    cvector.reshape(-1)
).requires_grad_(False)

'''
rho_0_tensor = rho_0_tensor.to(cpu)
cvector_tensor = cvector_tensor.to(cpu)
print(type(rho_0_tensor))
def rho0_WASS_cpu(y_true, y_pred):
#     y_pred = y_pred.to(cpu)
#     y_pred = y_pred.cpu()
    
    y_pred = torch.where(y_pred < 0, 0, y_pred)
    y_pred = y_pred / torch.sum(torch.abs(y_pred))

    param = torch.cat((rho_0_tensor, y_pred), 0)
    param = torch.reshape(param, (2*N,))
    print(type(param))
    x_sol, = cvxpylayer(param)
    # TODO(handle infeasible)
    wass_dist = torch.matmul(cvector_tensor, x_sol)
    return wass_dist
'''

xT_tensor = xT_tensor.to(cuda0)
rho_0_tensor = rho_0_tensor.to(cuda0)
rho_T_tensor = rho_T_tensor.to(cuda0)
cvector_tensor = cvector_tensor.to(cuda0)

# first train on mse, then train on wass?
class LossSeq(object):
    def __init__(self):
        self.mode = 0
        self.print_mode = 0

    def rho0_WASS_cuda1(self, y_true, y_pred):
        total = torch.mean(torch.square(y_true - y_pred))
        return total

# fail_cost = torch.Tensor(1e3, dtype=torch.float32)

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

print(time.time())


# In[ ]:





# In[23]:


geom = dde.geometry.Interval(state_min, state_max)
timedomain = dde.geometry.TimeDomain(0., T_t)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

print(time.time())


# In[24]:





# In[25]:


data = dde.data.TimePDE(
    geomtime,
    pde,
    [rho_0_BC,rho_T_BC],
    num_domain=5000,
    num_initial=500)

# 2 inputs: x + t
# 3 outputs: 3 eqs
net = dde.nn.FNN([2] + [70] *3  + [3], "tanh", "Glorot normal")
# net.apply_output_transform(modify_output)
# net.apply_output_transform(modify_output)
model = dde.Model(data, net)

print(time.time())


# In[26]:


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
        if self.monitor == "loss_train":
            # result = sum(self.model.train_state.loss_train)
            result = max(self.model.train_state.loss_train)
        elif self.monitor == "loss_test":
            # result = sum(self.model.train_state.loss_test)
            result = max(self.model.train_state.loss_test)
        else:
            raise ValueError("The specified monitor function is incorrect.")

        return result

earlystop_cb = EarlyStoppingFixed(baseline=1e-3, patience=0)

class ModelCheckpoint2(dde.callbacks.ModelCheckpoint):
    def on_epoch_end(self):
        current = self.get_monitor_value()
        if self.monitor_op(current, self.best) and current < 1e-3:
            save_path = self.model.save(self.filepath, verbose=0)
            print(
                "Epoch {}: {} improved from {:.2e} to {:.2e}, saving model to {} ...\n".format(
                    self.model.train_state.epoch,
                    self.monitor,
                    self.best,
                    current,
                    save_path,
                ))
            self.best = current

modelcheckpt_cb = ModelCheckpoint2(
    ck_path, verbose=True, save_better_only=True, period=1)

print(time.time())


# In[27]:


loss_seq = LossSeq()
loss_func=["MSE","MSE","MSE", rho0_WASS_cuda0, rhoT_WASS_cuda0]
# loss functions are based on PDE + BC: 3 eq outputs, 2 BCs

model.compile("adam", lr=1e-3,loss=loss_func)
de = 1
losshistory, train_state = model.train(
    iterations=50000,
    display_every=de,
    callbacks=[earlystop_cb, modelcheckpt_cb])

# import ipdb; ipdb.set_trace();
# In[30]:

dde.saveplot(losshistory, train_state, issave=True, isplot=False)
model_path = model.save(ck_path)
print(model_path)


# In[31]:


# params = {'backend': 'ps',
#           'xtick.labelsize': 12,
#           'ytick.labelsize': 12,
#           'legend.handlelength': 1,
#           'legend.borderaxespad': 0,
#           'font.family': 'serif',
#           'font.serif': ['Computer Modern Roman'],
#           'ps.usedistiller': 'xpdf',
#           'text.usetex': True,
#           # include here any neede package for latex
#           'text.latex.preamble': [r'\usepackage{amsmath}'],
#           }
# plt.rcParams.update(params)
# plt.style.use('seaborn-white')


# In[32]:


# 16 plot loss

loss_loaded = np.genfromtxt('./loss.dat')

print("loss_loaded", loss_loaded)

# import ipdb; ipdb.set_trace();

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

plot_fname = "%s/loss.png" % (os.path.abspath("./"))
plt.savefig(plot_fname, dpi=300)
print("saved plot")

# plt.show()


# In[34]:


X_IDX = 0
T_IDX = 1
RHO0_IDX = 3
RHOT_IDX = 4

test_ti = np.loadtxt('./test.dat')
test_ti = test_ti[0:N, :] # first BC test data
ind = np.lexsort((test_ti[:,X_IDX],test_ti[:,T_IDX]))
test_ti = test_ti[ind]

# post process
test_ti[:,RHO0_IDX] = np.where(test_ti[:,RHO0_IDX] < 0, 0, test_ti[:,RHO0_IDX])
s1 = np.trapz(test_ti[:,RHO0_IDX], axis=0, x=test_ti[:,X_IDX])
test_ti[:, RHO0_IDX] /= s1 # to pdf
s2 = np.sum(test_ti[:,RHO0_IDX])
test_ti[:, RHO0_IDX] /= s2 # to pmf

s1 = np.trapz(test_ti[:,RHO0_IDX], axis=0, x=test_ti[:,X_IDX])
s2 = np.sum(test_ti[:,RHO0_IDX])

test_ti[:,RHOT_IDX] = np.where(test_ti[:,RHOT_IDX] < 0, 0, test_ti[:,RHOT_IDX])
s1 = np.trapz(test_ti[:,RHOT_IDX], axis=0, x=test_ti[:,X_IDX])
test_ti[:, RHOT_IDX] /= s1 # to pdf
s2 = np.sum(test_ti[:,RHOT_IDX])
test_ti[:, RHOT_IDX] /= s2 # to pmf

s3 = np.trapz(test_ti[:,RHOT_IDX], axis=0, x=test_ti[:,X_IDX])
s4 = np.sum(test_ti[:,RHOT_IDX])

fig = plt.figure(1)
ax1 = plt.subplot(111, frameon=False)
# ax1.set_aspect('equal')
ax1.grid()
ax1.set_title(
    'rho0: trapz=%.3f, sum=%.3f, rhoT: trapz=%.3f, sum=%.3f' % (s1, s2, s3, s4))

ax1.plot(
    test_ti[:, X_IDX],
    test_ti[:, RHO0_IDX],
    linewidth=1,
    label='test rho_0')

test_rho0=pdf1d(test_ti[:, X_IDX], mu_0, sigma_0).reshape(test_ti.shape[0],1)
test_rho0 /= np.trapz(test_rho0, axis=0, x=test_ti[:,X_IDX])
test_rho0 = test_rho0 / np.sum(np.abs(test_rho0))
ax1.plot(
    test_ti[:, X_IDX],
    test_rho0,
    c='r',
    linewidth=1,
    label='rho_0')

ax1.plot(
    test_ti[:, X_IDX],
    test_ti[:, RHOT_IDX],
    c='g',
    linewidth=1,
    label='test rho_T')

test_rhoT=pdf1d(test_ti[:, X_IDX], mu_T, sigma_T).reshape(test_ti.shape[0],1)
test_rhoT /= np.trapz(test_rhoT, axis=0, x=test_ti[:,X_IDX])
test_rhoT = test_rhoT / np.sum(np.abs(test_rhoT))
ax1.plot(
    test_ti[:, X_IDX],
    test_rhoT,
    c='c',
    linewidth=1,
    label='rho_T')

ax1.legend(loc='lower right')

plot_fname = "%s/pinn_vs_rho.png" % (os.path.abspath("./"))
plt.savefig(plot_fname, dpi=300)
# plt.show()

