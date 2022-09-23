
fname = 'inversed_%d_%d_%d.pkl' % (N, int(window), int(t))
inversed0 = None
inversed1 = None
inversed2 = None
if os.path.exists(fname):
    print("%s found" % (fname))
    with open(fname, 'rb') as handle:
        data = pickle.load(handle)
        inversed0 = data["inversed0"]
        inversed1 = data["inversed1"]
        inversed2 = data["inversed2"]
else:
    '''
    X1, X2, X3 = np.meshgrid(x1,x2,x3) # each is NxNxN

    pos = np.empty(X1.shape + (4,)) # NxNxN, for x10, x20, x30, p0

    t = 0
    omega = alpha2 * X3
    gamma = (X2 - X1 * np.tan(omega*t)) / (X1 + X2*np.tan(omega*t))

    pos[:, :, :, 0] = np.sqrt((X1**2 + X2**2) / (1 + gamma))
    pos[:, :, :, 1] = gamma * np.sqrt((X1**2 + X2**2) / (1 + gamma**2))
    pos[:, :, :, 2] = X3

    pos[:, :, :, 3] = 

    # this takes the linspace at that axis
    # and duplicates it across the other axii
    # so instead of iterating, we sample
    # taking more space to take less time

    inversed = inverse_flow(X1, X2, X3) # 500x500x500, 3D matrix

    inversed1 = inverse_flow_t1(X1, X2, X3) # 500x500x500, 3D matrix

    '''

    inversed0 = np.zeros((N, N, N, 4))
    inversed1 = np.zeros((N, N, N, 4))
    inversed2 = np.zeros((N, N, N, 4))

    for i in range(N):
        for j in range(N):
            for k in range(N):
                inversed0[i, j, k, :3] = inverse_flow(x1[i], x2[j], x3[k], 0, alpha2)
                inversed0[i, j, k, 3] = normal_dist_array(inversed0[i, j, k, :3], mu_0 , cov_0)

                inversed1[i, j, k, :3] = inverse_flow(x1[i], x2[j], x3[k], 1, alpha2)
                inversed1[i, j, k, 3] = normal_dist_array(inversed1[i, j, k, :3], mu_0 , cov_0)

                inversed2[i, j, k, :3] = inverse_flow(x1[i], x2[j], x3[k], 5, alpha2)
                inversed2[i, j, k, 3] = normal_dist_array(inversed2[i, j, k, :3], mu_0 , cov_0)
    with open(fname, 'wb') as handle:
        pickle.dump(
            {
                "inversed0" : inversed0,
                "inversed1" : inversed1,
                "inversed2" : inversed2,
            },
            handle, protocol=pickle.HIGHEST_PROTOCOL)

#############################################################################

size = np.ones((N**3)) * 0.2

pos0 = np.empty((N**3, 3))
color0 = np.zeros((N**3, 4))

pos1 = np.empty((N**3, 3))
color1 = np.zeros((N**3, 4))

pos2 = np.empty((N**3, 3))
color2 = np.zeros((N**3, 4))


max_0 = np.max(inversed0[:, :, :, 3])
max_1 = np.max(inversed1[:, :, :, 3])
max_2 = np.max(inversed2[:, :, :, 3])

print("max_0", max_0)
print("max_1", max_1)
print("max_2", max_2)

m = max([max_0, max_1, max_2])

cmap = pg.ColorMap(pos=np.linspace(0.0, 0.5, len(colors)), color=colors)

for i in range(N):
    for j in range(N):
        for k in range(N):
            pos0[i*N*N+j*N+k] = [x1[i], x2[j], x3[k]] # inversed0[i, j, k, :3]
            # color0[i*N*N+j*N+k] = (inversed0[i, j, k, 3] * 5, 0.0, 0.0, 0.5)

            t = cmap.mapToQColor(inversed0[i, j, k, 3])
            color0[i*N*N+j*N+k] = t.getRgbF()

            pos1[i*N*N+j*N+k] = [x1[i], x2[j], x3[k]] # inversed1[i, j, k, :3]            
            # color1[i*N*N+j*N+k] = (0.0, inversed1[i, j, k, 3] * 5 , 0.0, 0.5)

            t = cmap.mapToQColor(inversed1[i, j, k, 3])
            color1[i*N*N+j*N+k] = t.getRgbF()

            pos2[i*N*N+j*N+k] = [x1[i], x2[j], x3[k]] # inversed2[i, j, k, :3]            
            # color2[i*N*N+j*N+k] = (0.0, 0.0, inversed2[i, j, k, 3] * 5, 0.5)

            t = cmap.mapToQColor(inversed2[i, j, k, 3])
            color2[i*N*N+j*N+k] = t.getRgbF()

sp0 = gl.GLScatterPlotItem(pos=pos0, size=size, color=color0, pxMode=False)
w.addItem(sp0)

sp1 = gl.GLScatterPlotItem(pos=pos1, size=size, color=color1, pxMode=False)
sp1.translate(5, 5,0)
w.addItem(sp1)

sp2 = gl.GLScatterPlotItem(pos=pos2, size=size, color=color2, pxMode=False)
sp2.translate(10,10,0)
w.addItem(sp2)


#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

x, y, z = np.mgrid[-10:10:.1, -10:10:.1, -10:10:.1]
pos = np.empty(x.shape + (3,))
pos[:, :, :, 0] = x
pos[:, :, :, 1] = y
pos[:, :, :, 2] = z
v = multivariate_normal(
  [0, 0, 0],
  np.eye(3))


fig,ax = plt.subplots(ncols=1,nrows=1,subplot_kw=dict(projection='3d'))

N=200
stride=1

u = np.linspace(0, 2 * np.pi, N)
v = np.linspace(0, np.pi, N)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, linewidth=0.0, cstride=stride, rstride=stride)
ax.set_title('{0}x{0} data points, stride={1}'.format(N,stride))

plt.show()



## Saddle example with x and y specified
x = np.linspace(-8, 8, 50)
y = np.linspace(-8, 8, 50)
z = 0.1 * ((x.reshape(50,1) ** 2) - (y.reshape(1,50) ** 2))

p2 = gl.GLSurfacePlotItem(x=x, y=y, z=z, shader='normalColor')
p2.translate(-10,-10,0)
w.addItem(p2)

def r_theta_phi(theta, phi, k, l):
    return np.absolute((np.cos((k*l/2)*np.cos(theta)) -np.cos(k*l/2))/np.sin(theta))

p = 2*np.pi
q = 0.5

md = Sphere(100, 100, r_theta_phi, args=(p, q))
colors = np.ones((md.faceCount(), 4), dtype=float)
colors[:,0] = np.linspace(0.1, 0.1, colors.shape[0])
colors[:,1] = np.linspace(0.6, 0.9, colors.shape[0])
colors[:,2] = np.linspace(0.0, 0.0, colors.shape[0])
md.setFaceColors(colors)
m = gl.GLMeshItem(meshdata=md, smooth=False)
w.addItem(m)


## Manually specified colors
z = pg.gaussianFilter(np.random.normal(size=(50,50)), (1,1))
x = np.linspace(-12, 12, 50)
y = np.linspace(-12, 12, 50)
colors = np.ones((50,50,4), dtype=float)
colors[...,0] = np.clip(np.cos(((x.reshape(50,1) ** 2) + (y.reshape(1,50) ** 2)) ** 0.5), 0, 1)
colors[...,1] = colors[...,0]

p3 = gl.GLSurfacePlotItem(z=z, colors=colors.reshape(50*50,4), shader='shaded', smooth=False)
p3.scale(16./49., 16./49., 1.0)
p3.translate(2, -18, 0)
w.addItem(p3)


## Animated example
## compute surface vertex data
cols = 90
rows = 100
x = np.linspace(-8, 8, cols+1).reshape(cols+1,1)
y = np.linspace(-8, 8, rows+1).reshape(1,rows+1)
d = (x**2 + y**2) * 0.1
d2 = d ** 0.5 + 0.1

## precompute height values for all frames
phi = np.arange(0, np.pi*2, np.pi/20.)
z = np.sin(d[np.newaxis,...] + phi.reshape(phi.shape[0], 1, 1)) / d2[np.newaxis,...]


## create a surface plot, tell it to use the 'heightColor' shader
## since this does not require normal vectors to render (thus we 
## can set computeNormals=False to save time when the mesh updates)
p4 = gl.GLSurfacePlotItem(x=x[:,0], y = y[0,:], shader='heightColor', computeNormals=False, smooth=False)
p4.shader()['colorMap'] = np.array([0.2, 2, 0.5, 0.2, 1, 1, 0.2, 0, 2])
p4.translate(10, 10, 0)
w.addItem(p4)

index = 0
def update():
    global p4, z, index
    index -= 1
    p4.setData(z=z[index%z.shape[0]])
    
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(30)


#! /usr/bin/env python3

import numpy as np

# this runs from wsb python
from pyqtgraph.Qt import QtCore, QtGui

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

def inverse_flow(alpha2, state, t):
  '''
  inverse flow map [x1, x2, x3] -> [x10, x20, x30]
  '''
  x1 = state[0]
  x2 = state[1]
  x3 = state[2]

  omega = alpha2 * x3
  gamma = (x2 - x1 * np.tan(omega*t)) / (x1 + x2*np.tan(omega*t))

  x10 = np.sqrt((x1**2 + x2**2) / (1 + gamma))
  x20 = gamma * np.sqrt((x1**2 + x2**2) / (1 + gamma**2))

  return np.array([x10, x20, x30])

#######################################################
# init_data

# p0 is distribution with mean at 0.5
mu_0 = np.array([0.5, 0.5, 0.5])
cov_0 = np.eye(3) * 1.0


# time sample t = 0

x1 = np.linspace(mu_0[0] - window, mu_0[0] + window, 500)
x2 = np.linspace(mu_0[1] - window, mu_0[1] + window, 500)
x3 = np.linspace(mu_0[2] - window, mu_0[2] + window, 500)

X1, X2, X3 = np.meshgrid(x1,x2,x3)


#######################################################


#######################################################

X1, X2, X3 = np.meshgrid(x1,x2,x3)
# this takes the linspace at that axis
# and duplicates it across the other axii

pos = np.empty(X1.shape + (3,))
pos[:, :, :, 0] = X1
pos[:, :, :, 1] = X2
pos[:, :, :, 2] = X3

v = multivariate_normal(
  [mu_x1, mu_x2, mu_x3],
  [
    [variance_x1, 0, 0],
    [0, variance_x2, 0],
    [0, 0, variance_x3],
  ])

pdf = v.pdf(pos) # 3D matrix

no_z = np.sum(pdf, axis=1)

#Make a 3D plot
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_surface(
  X1[:, :, 0],
  X2[:, :, 0],
  no_z,
  cmap='viridis',
  linewidth=0)

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.show()




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


# In[34]:

N = nSample = 50

# must be floats
state_min = 0.0
state_max = 6.0

mu_0 = 5.0
sigma_0 = 1.0

mu_T = 3.0
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

id_prefix = "empty"

def pdf1d(x, mu, sigma):
    a, b = (state_min - mu) / sigma, (state_max - mu) / sigma
    rho_x=truncnorm.pdf(x, a, b, loc = mu, scale = sigma)

    # do NOT use gaussian norm, because it is only area=1
    # from -inf, inf, will not be for finite state/grid
    # rho_x = norm.pdf(x, mu, sigma)
    return rho_x

X_IDX = 0
T_IDX = 1
EQ_IDX = 3

test_ti = np.loadtxt('./test.dat')
test_ti = test_ti[0:N, :] # first BC test data
ind = np.lexsort((test_ti[:,X_IDX],test_ti[:,T_IDX]))
test_ti = test_ti[ind]

# post process
test_ti[:,EQ_IDX] = np.where(test_ti[:,EQ_IDX] < 0, 0, test_ti[:,EQ_IDX])
print("test_ti[:,EQ_IDX]", test_ti[:,EQ_IDX])
s1 = np.trapz(test_ti[:,EQ_IDX], axis=0, x=test_ti[:,X_IDX])
print(s1)

test_ti[:, EQ_IDX] /= s1 # to pdf
test_ti[:, EQ_IDX] = np.nan_to_num(test_ti[:, EQ_IDX], 0.0)

# s2 = np.sum(test_ti[:,EQ_IDX])
# test_ti[:, EQ_IDX] /= s2 # to pmf

s1 = np.trapz(test_ti[:,EQ_IDX], axis=0, x=test_ti[:,X_IDX])
s2 = np.sum(test_ti[:,EQ_IDX])

fig = plt.figure(1)
ax1 = plt.subplot(111, frameon=False)
# ax1.set_aspect('equal')
ax1.grid()
ax1.set_title('trapz=%.3f, sum=%.3f' % (s1, s2))

ax1.plot(
    test_ti[:, 0],
    test_ti[:, EQ_IDX],
    linewidth=1,
    label='test rho_0')

test_rho0=pdf1d(test_ti[:, 0], 2.0, 1.5).reshape(test_ti.shape[0],1)
test_rho0 /= np.trapz(test_rho0, axis=0, x=test_ti[:,X_IDX])
# test_rho0 = test_rho0 / np.sum(np.abs(test_rho0))
ax1.plot(
    test_ti[:, 0],
    test_rho0,
    c='r',
    linewidth=1,
    label='rho_0')

ax1.legend(loc='lower right')

plt.show()





# first train on mse, then train on wass?
class LossSeq(object):
    def __init__(self):
        self.mode = 0
        self.print_mode = 0

    def rho0_WASS_cuda1(self, y_true, y_pred):
        total = torch.mean(torch.square(y_true - y_pred))
        return total



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


rho_0_tensor = torch.from_numpy(
    rho_0,
).requires_grad_(False)
rho_0_cube_tensor = torch.reshape(rho_0_tensor, (N,N,N))
rho_0_cube_tensor = rho_0_cube_tensor.to(device)
rho_0_tensor_support = torch_get_pdf_support(rho_0_cube_tensor,
    [x1_tensor, x2_tensor, x3_tensor],
    0,
    support_buffer0,
    support_buffer1)
print("rho_0_tensor_support", rho_0_tensor_support)

get_marginal(rho_0_cube, [x1, x2, x3], 0)

  get_marginal(
    rho_0_cube, [x2, x1, x3], 1)

  get_marginal(
    rho_0_cube, [x3, x1, x2], 2)



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



# first train on mse, then train on wass?
class LossSeq(object):
    def __init__(self):
        self.mode = 0
        self.print_mode = 0

    def rho0_WASS_cuda1(self, y_true, y_pred):
        total = torch.mean(torch.square(y_true - y_pred))
        return total

# fail_cost = torch.Tensor(1e3, dtype=torch.float32)


#!/usr/bin/env python3

import torch

import numpy as np

def slice(matrix_3d, i, j, mode):
  if mode == 0:
    return matrix_3d[j, i, :]
  elif mode == 1:
    return matrix_3d[i, j, :]
  else:
    return matrix_3d[i, :, j]

def pdf3d(x,y,z,rv):
  return rv.pdf(np.hstack((x, y, z)))

def get_marginal(matrix_3d, xs, mode, normalize=True):
  marginal = np.array([
    np.trapz(
      np.array([
          np.trapz(
            slice(matrix_3d, i, j, mode)
            , x=xs[2]) # x3 slices for one x2 => R
          for i in range(len(xs[1]))
        ]) # x3 slices across all x2 => Rn
      , x=xs[1]) # x2 slice for one x1 => R
    for j in range(len(xs[0]))
  ])
  if normalize:
    marginal /= np.trapz(marginal, x=xs[0])
  return marginal

def get_pdf_support(matrix_3d, xs, mode):
  marginal = np.array([
    np.trapz(
      np.array([
          np.trapz(
            slice(matrix_3d, i, j, mode)
            , x=xs[2]) # x3 slices for one x2 => R
          for i in range(len(xs[1]))
        ]) # x3 slices across all x2 => Rn
      , x=xs[1]) # x2 slice for one x1 => R
    for j in range(len(xs[0]))
  ])
  return np.trapz(marginal, x=xs[0])

def torch_get_pdf_support(
  tensor_3d,
  xtensors,
  mode,
  buffer0,
  buffer1):
  for j in range(len(xtensors[0])):
    # collapse 1 dimension away into buffer0
    for i in range(len(xtensors[1])):
      buffer0[i] = torch.trapz(
        slice(tensor_3d, i, j, mode)
        , x=xtensors[2])
    # collapse 2 dimensions into 1 scalar
    buffer1[j] = torch.trapz(
      buffer0,
      x=xtensors[1])
  return torch.trapz(buffer1, x=xtensors[0])

N = 10

nb_name = "none"

# must be floats
# to recover mu from np.trapz(marginal, x)
# this must be sufficient broad
state_min = -3.5
state_max = 3.5

mu_0 = 2.0
sigma_0 = 1.5

mu_T = 0.0
sigma_T = 1.0

T_0=0. #initial time
T_t=50. #Terminal time

epsilon=.001

samples_between_initial_and_final = 20000 # 10^4 order, 20k = out of memory
initial_and_final_samples = 2000 # some 10^3 order

num_epochs = 100000


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



######################################

rv0 = multivariate_normal([mu_0, mu_0, mu_0], sigma_0 * np.eye(3))
rvT = multivariate_normal([mu_T, mu_T, mu_T], sigma_T * np.eye(3))


'''
xyzs=np.hstack((
    x_T,
    y_T,
    z_T,
))
C = cdist(xyzs, xyzs, 'sqeuclidean')
cvector = C.reshape((N**3)**2,1)
A = np.concatenate(
    (
        np.kron(
            np.ones((1,N**3)),
            sparse.eye(N**3).toarray()
        ),
        np.kron(
            sparse.eye(N**3).toarray(),
            np.ones((1,N**3))
        )
    ), axis=0)
# 2*N**3
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
    param = torch.reshape(param, (2*N**3,))
    # print(type(param))
    try:
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
    except:
        print("cvx failed, returning mse")
        return torch.mean(torch.square(y_true - y_pred))

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
    param = torch.reshape(param, (2*N**3,))
    # print(type(param))
    try:
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
    except:
        print("cvx failed, returning mse")
        return torch.nan_to_num(torch.mean(torch.square(y_true - y_pred)), 1e3)
'''


##########################################

#!/usr/bin/env python
# coding: utf-8

# In[42]:


# 0 define backend
import sys, os, time

# %env DDE_BACKEND=tensorflow.compat.v1
# %env XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/home/cyan3/miniforge/envs/tf

os.environ['DDE_BACKEND'] = "tensorflow.compat.v1"
os.environ['XLA_FLAGS'] = "--xla_gpu_cuda_data_dir=/usr/local/home/cyan3/miniforge/envs/tf"

# https://stackoverflow.com/questions/68614547/tensorflow-libdevice-not-found-why-is-it-not-found-in-the-searched-path
# this directory has /nvvm/libdevice/libdevice.10.bc


# In[2]:


import tensorflow.compat.v1 as tf

import deepxde as dde
import numpy as np
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pylab
from numpy import linalg as LA
import math
import scipy.io
from os.path import dirname, join as pjoin
from scipy.stats import truncnorm, norm
# import tensorflow as tf
from scipy.optimize import linprog
from scipy import sparse
from scipy.stats import multivariate_normal

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    print(device)
    
if dde.backend.backend_name == "pytorch":
    exp = dde.backend.torch.exp
else:
    from deepxde.backend import tf

    exp = tf.exp


# In[ ]:





# In[12]:


N = nSample = 100

# must be floats
state_min = 0.0
state_max = 6.0

mu_0 = 5.0
sigma_0 = 1.0

mu_T = 2.0
sigma_T = 0.5

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

print(time.time())


# In[ ]:





# In[9]:


def boundary(_, on_initial):
    return on_initial

def pdf1d_T(x):
    a, b = (state_min - mu_T) / sigma_T, (state_max - mu_T) / sigma_T
    rho_x=truncnorm.pdf(x, a, b, loc = mu_0, scale = sigma_0)

#     rho_x = norm.pdf(x, mu_T, sigma_T)
    return rho_x

def pdf1d_0(x):
    a, b = (state_min - mu_0) / sigma_0, (state_max - mu_0) / sigma_0
    rho_x=truncnorm.pdf(x, a, b, loc = mu_0, scale = sigma_0)

#     rho_x = norm.pdf(x, mu_0, sigma_0)
    return rho_x

print(time.time())


# In[ ]:


rv0 = multivariate_normal([mu_0, mu_0, mu_0], sigma_0 * np.eye(3))
rvT = multivariate_normal([mu_T, mu_T, mu_T], sigma_T * np.eye(3))

def pdf3d_0(x,y,z):
    return rv0.pdf(np.hstack((x, y, z)))

def pdf3d_T(x,y,z):
    return rvT.pdf(np.hstack((x, y, z)))

print(time.time())


# In[10]:


x_T = np.transpose(np.linspace(state_min, state_max, N))
y_T = np.transpose(np.linspace(state_min, state_max, N))
z_T = np.transpose(np.linspace(state_min, state_max, N))
x_T=x_T.reshape(len(x_T),1)
y_T=y_T.reshape(len(y_T),1)
z_T=z_T.reshape(len(z_T),1)
print(time.time())


# In[13]:


# 3d linprog example, with bounds

'''
rho_0=pdf3d_0(x_T,y_T,z_T).reshape(len(x_T),1)
rho_T=pdf3d_T(x_T,y_T,z_T).reshape(len(x_T),1)

# very important to make it solvable
rho_0 = np.where(rho_0 < 0, 0, rho_0)
rho_0 = rho_0 / np.sum(rho_0)

rho_T = np.where(rho_T < 0, 0, rho_T)
rho_T = rho_T / np.sum(rho_T)

res = linprog(
    cvector,
    A_eq=A,
    b_eq=np.concatenate((rho_0, rho_T), axis=0),
    options={"disp": True},
    bounds=[(0, np.inf)], 
    # x >= 0
    # x < inf
)

if res.fun is not None:
    print(np.sqrt(res.fun))
'''

# In[ ]:





# In[14]:


# 1d linprog example

rho_0_1d=pdf1d_0(x_T).reshape(len(x_T),1)
rho_T_1d=pdf1d_T(x_T).reshape(len(x_T),1)

rho_0_1d = np.where(rho_0_1d < 0, 0, rho_0_1d)
rho_0_1d = rho_0_1d / np.sum(np.abs(rho_0_1d))

# rho_0_1d_trapz = np.trapz(rho_0_1d, axis=0)[0]
# rho_0_1d = rho_0_1d / rho_0_1d_trapz
# rho_0_1d_trapz = np.trapz(rho_0_1d, axis=0)
# print("rho_0_1d_trapz=",rho_0_1d_trapz)

print(rho_0_1d)

rho_T_1d = np.where(rho_T_1d < 0, 0, rho_T_1d)

rho_T_1d = rho_T_1d / np.sum(np.abs(rho_T_1d))

# rho_T_1d_trapz = np.trapz(rho_T_1d, axis=0)[0]
# rho_T_1d = rho_T_1d / rho_T_1d_trapz
# rho_T_1d_trapz = np.trapz(rho_T_1d, axis=0)
# print("rho_T_1d_trapz=",rho_T_1d_trapz)

res = linprog(
    cvector,
    A_eq=A,
    b_eq=np.concatenate((rho_0_1d, rho_T_1d), axis=0),
    options={"disp": True},
    bounds=[(0, np.inf)], 
    # x >= 0
    # x < inf
)

if res.fun is not None:
    print(np.sqrt(res.fun))


# In[ ]:


# def modify_output(X, Y):
#     y1, y2, y3  = Y[:, 0:1], Y[:, 1:2], Y[:, 2:3]
#     u_new = tf.clip_by_value(y3, clip_value_min = .5, clip_value_max = 4)
#     return tf.concat((y1, y2, u_new), axis=1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[15]:


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

    D2=d*tf.math.exp(-(x[:, 0:1]-b-c*y3)*(x[:, 0:1]-b-c*y3))+f
    F=a*K*T*(x[:, 0:1]-b-c*y3)*(x[:, 0:1]-b-c*y3)
#     dD2_x=dde.grad.jacobian(D2, x, j=0)
#     dF_x=dde.grad.jacobian(F, x, j=0)
#     D1=dD2_x-dF_x*(D2/(K*T))
    D1=-2*(x[:, 0:1]-b-c*y3)*((d*tf.math.exp(-(x[:, 0:1]-b-c*y3)*(x[:, 0:1]-b-c*y3)))+a*D2)
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


print(time.time())


# In[31]:


def penalty(y_pred):
    p1 = 10 * (y_pred<0).sum()
    
    t = np.trapz(y_pred, axis=0)[0]
    p2 = 10 * np.abs(t - 1)
    return p1 + p2

def WASS(y_true, y_pred):
    xpdf = y_pred.numpy()
    
#     print(xpdf.shape)

    xpdf = np.where(xpdf < 0, 0, xpdf)
    if np.sum(xpdf) > 1e-8:
        xpdf = xpdf / np.sum(xpdf)

    res = linprog(
        cvector,
        A_eq=A,
        b_eq=np.concatenate((xpdf, rho_0_1d), axis=0),
        options={"disp": False},
        bounds=[(0, np.inf)],
    )
    
    # we are cheating here, we are using tf.reduce_sum
    # so that tf system will like our output 'type'
    # but we are 0'ing the sum and adding our scalar cost
    cand = tf.reduce_sum(y_pred) * 0.0
    
    if res.fun is None:
        return np.inf + cand
    else:
        return np.sqrt(res.fun) + cand + penalty(xpdf)

def WASS2(y_true, y_pred):
    loss = tf.py_function(
        func=WASS,
        inp=[y_true, y_pred],
        Tout=tf.float32
    )
    return loss

print(time.time())


# In[ ]:





# In[ ]:





# In[32]:


geom = dde.geometry.Interval(state_min, state_max)
timedomain = dde.geometry.TimeDomain(0., T_t)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

print(time.time())


# In[33]:


time_0=np.hstack((x_T,T_0*np.ones((len(x_T), 1))))
rho_0=pdf1d_0(x_T).reshape(len(x_T),1)
rho_0_BC = dde.icbc.PointSetBC(time_0, rho_0, component=1)

time_t=np.hstack((x_T,T_t*np.ones((len(x_T), 1))))
rho_T=pdf1d_T(x_T).reshape(len(x_T),1)
rho_T_BC = dde.icbc.PointSetBC(time_t, rho_T, component=1)

print(time.time())


# In[34]:


data = dde.data.TimePDE(
    geomtime,
    pde,
    [rho_0_BC,rho_T_BC],
    num_domain=5000,
    num_initial=500)
net = dde.nn.FNN([2] + [70] *3  + [3], "tanh", "Glorot normal")
# net.apply_output_transform(modify_output)
# net.apply_output_transform(modify_output)
model = dde.Model(data, net)

print(time.time())


# In[35]:


ck_path = "%s/model" % (os.path.abspath("./"))

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

earlystop_cb = EarlyStoppingFixed(baseline=1e-3, patience=0)


# In[37]:


loss_func=["MSE","MSE","MSE", WASS2,"MSE"]
model.compile("adam", lr=1e-3,loss=loss_func)
losshistory, train_state = model.train(
    epochs=200000,
    display_every=1000,
    callbacks=[earlystop_cb])


# In[38]:
dde.saveplot(losshistory, train_state, issave=True, isplot=False)
model_path = model.save(ck_path)
print(model_path)

# In[39]:


params = {'backend': 'ps',
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'legend.handlelength': 1,
          'legend.borderaxespad': 0,
          'font.family': 'serif',
          'font.serif': ['Computer Modern Roman'],
          'ps.usedistiller': 'xpdf',
          'text.usetex': True,
          # include here any neede package for latex
          'text.latex.preamble': [r'\usepackage{amsmath}'],
          }
plt.rcParams.update(params)
plt.style.use('seaborn-white')


# In[40]:


# 16 plot loss

'''
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
ax.set_yscale('log')
ax.set_xscale('log')

line1, = ax.plot(epoch, y1_psi_hjb, color='orange', lw=1, label='HJB PDE')
line2, = ax.plot(epoch, y2_rho_plankpde, color='blue', lw=1, label='Controlled Fokker-Planck PDE')
line2, = ax.plot(epoch, y2_rho_plankpde, color='green', lw=1, label='loss3')
line3, = ax.plot(epoch, rho0_initial, color='red', lw=1, label='p0 boundary condition')
line4, = ax.plot(epoch, rhoT_terminal, color='purple', lw=1, label='pT boundary condition')

ax.grid()
ax.legend(loc="upper right")
ax.set_title('training error/residual plots: mu0=2 -> muT=0')
ax.set_yscale('log')
ax.set_xscale('log')

plot_fname = "%s/loss.png" % (os.path.abspath("./"))
plt.savefig(plot_fname, dpi=300)
print("saved plot")

plt.show()


# In[41]:


# 18 load test data

test = np.genfromtxt('./test.dat')
# test_timesorted = test[test[:, 3].argsort()]
# sort AGAIN by output because a lot of samples @ t=0, t=5
ind = np.lexsort((test[:,2],test[:,1])) # sorts by [3] (t) then by [4] (psi)
test_timesorted = test[ind]
source_t = test_timesorted[:, 1]

print(test_timesorted)


# In[ ]:


# 35 plot rho at t=5, t=0

target_t = 0.0

N = 100

test_timesorted = test[test[:, 1].argsort()]
timesorted = test_timesorted[:, 1]
test_ti = test_timesorted[np.where(np.abs(timesorted - target_t) < 1e-8), :][0] # 2k

ti_rho_opt = test_ti[:, 5]

ti_rho_opt = np.where(ti_rho_opt<0, 0, ti_rho_opt)

ti_x1_x2_x3 = test_ti[:, 0:3]

####################################################################

d = 0.0
x1 = np.linspace(state_min - d, state_max + d, N)
x2 = np.linspace(state_min - d, state_max + d, N)
x3 = np.linspace(state_min - d, state_max + d, N)
X1, X2, X3 = np.meshgrid(x1,x2,x3,copy=False) # each is NxNxN

rho_opt = np.zeros((N,N,N))

closest_1 = [(np.abs(x1 - ti_x1_x2_x3[i, 0])).argmin() for i in range(ti_x1_x2_x3.shape[0])]
closest_2 = [(np.abs(x2 - ti_x1_x2_x3[i, 1])).argmin() for i in range(ti_x1_x2_x3.shape[0])]
closest_3 = [(np.abs(x3 - ti_x1_x2_x3[i, 2])).argmin() for i in range(ti_x1_x2_x3.shape[0])]

# some transposing going on in some reshape
# swapping closest_1/2 works well
rho_opt[closest_1, closest_2, closest_3] = ti_rho_opt

####################################################################

# RHO_OPT = gd(
#   (ti_x1_x2_x3[:, 0], ti_x1_x2_x3[:, 1], ti_x1_x2_x3[:, 2]),
#   ti_rho_opt,
#   (X1, X2, X3),
#   method='linear')

####################################################################

x1_marginal = np.array([
    np.trapz(
        np.array([
            np.trapz(rho_opt[j, i, :], x=x3) # x3 slices for one x2 => R
            for i in range(len(x2))]) # x3 slices across all x2 => Rn
        , x=x2) # x2 slice for one x1 => R
for j in range(len(x1))])

x2_marginal = np.array([
    np.trapz(
        np.array([
            np.trapz(rho_opt[i, j, :], x=x3) # x3 slices for one x1 => R
            for i in range(len(x1))]) # x3 slices across all x1 => Rn
        , x=x1) # x1 slice for one x2 => R
for j in range(len(x2))])

x3_marginal = np.array([
    np.trapz(
        np.array([
            np.trapz(rho_opt[i, :, j], x=x2) # x2 slices for one x1 => R
            for i in range(len(x1))]) # x2 slices across all x1 => Rn
        , x=x1) # x1 slice for one x3 => R
for j in range(len(x3))])

####################################################################

# normalize all the pdfs so area under curve ~= 1.0
x1_pdf_area = np.trapz(x1_marginal, x=x1)
x2_pdf_area = np.trapz(x2_marginal, x=x2)
x3_pdf_area = np.trapz(x3_marginal, x=x3)
print("prior to normalization: %.2f, %.2f, %.2f" % (
    x1_pdf_area,
    x2_pdf_area,
    x3_pdf_area))

x1_marginal /= x1_pdf_area
x2_marginal /= x2_pdf_area
x3_marginal /= x3_pdf_area

print(x1_marginal.shape)

x1_pdf_area = np.trapz(x1_marginal, x=x1)
x2_pdf_area = np.trapz(x2_marginal, x=x2)
x3_pdf_area = np.trapz(x3_marginal, x=x3)
print("after to normalization: %.2f, %.2f, %.2f" % (
    x1_pdf_area,
    x2_pdf_area,
    x3_pdf_area))

fig = plt.figure(1)
ax1 = plt.subplot(131, frameon=False)
# ax1.set_aspect('equal')
ax1.grid()
ax1.set_title('x1 marginal')

ax2 = plt.subplot(132, frameon=False)
# ax2.set_aspect('equal')
ax2.grid()
ax2.set_title('x2 marginal')

# ax3 = plt.subplot(133, frameon=False)

ax3 = plt.subplot(133, frameon=False)
# ax3.set_aspect('equal')
ax3.grid()
ax3.set_title('x3 marginal')

colors="rgbymkc"

i = 0
t_e = 0
ax1.plot(x1,
    x1_marginal,
    colors[i % len(colors)],
    linewidth=1,
    label=t_e)
ax1.legend(loc='lower right')

ax2.plot(x2,
    x2_marginal,
    colors[i % len(colors)],
    linewidth=1,
    label=t_e)
ax2.legend(loc='lower right')

ax3.plot(x3,
    x3_marginal,
    colors[i % len(colors)],
    linewidth=1,
    label=t_e)
ax3.legend(loc='lower right')

fig.suptitle('t=%.2f' % (target_t), fontsize=16)

plt.show()


# In[ ]:




'''

#########################################


