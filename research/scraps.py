
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

import argparse
parser = argparse.ArgumentParser(description="")
parser.add_argument('--N', type=int, default=50, help='')
parser.add_argument('--js', type=str, default="1,1,2", help='')
parser.add_argument('--q', type=float, default=0.0, help='')
args = parser.parse_args()

print(time.time())


# In[13]:


nb_name = "none"
# get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.kernel.execute(\'nb_name = "\' + IPython.notebook.notebook_name + \'"\')\n')


# In[28]:


N = nSample = args.N

# must be floats
state_min = -3.5
state_max = 3.5

mu_0 = 2.0
sigma_0 = 1.5

mu_T = 0.0
sigma_T = 1.0

j1, j2, j3 = [float(x) for x in args.js.split(",")] # axis-symmetric case
q_statepenalty_gain = args.q # 0.5

T_0=0. #initial time
T_t=50. #Terminal time

epsilon=.001

samples_between_initial_and_final = 12000 # 10^4 order, 20k = out of memory
initial_and_final_samples = 1000 # some 10^3 order

num_epochs = 100000

print("N: ", N)
print("js: ", j1, j2, j3)
print("q: ", q_statepenalty_gain)

######################################

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

id_prefix = "wass_3d"

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

time_0=np.hstack((
    x_T,
    y_T,
    z_T,
    T_0*np.ones((len(x_T), 1))
))

# rho_0=pdf1d(x_T, mu_0, sigma_0).reshape(len(x_T),1)
rho_0=pdf3d(x_T,y_T,z_T, rv0).reshape(len(x_T),1)

rho_0 = np.where(rho_0 < 0, 0, rho_0)
rho_0 /= np.trapz(rho_0, x=x_T, axis=0)[0] # pdf
rho_0 = rho_0 / np.sum(np.abs(rho_0)) # pmf
print(np.sum(rho_0))

rho_0_BC = dde.icbc.PointSetBC(time_0, rho_0, component=1)


time_t=np.hstack((
    x_T,
    y_T,
    z_T,
    T_t*np.ones((len(x_T), 1))
))

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

U1 = dde.Variable(1.0)
U2 = dde.Variable(1.0)
U3 = dde.Variable(1.0)

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
        U1 - dpsi_x,
        U2 - dpsi_y,
        U3 - dpsi_z,
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
    param = torch.reshape(param, (2*N,))
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

print(time.time())

geom=dde.geometry.geometry_3d.Cuboid(
    [state_min, state_min, state_min],
    [state_max, state_max, state_max])
timedomain = dde.geometry.TimeDomain(0., T_t)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

print(time.time())

data = dde.data.TimePDE(
    geomtime,
    euler_pde,
    [rho_0_BC,rho_T_BC],
    num_domain=samples_between_initial_and_final,
    num_initial=initial_and_final_samples)

# 4 inputs: x,y,z,t
# 5 outputs: 2 eq + 3 control vars
net = dde.nn.FNN([4] + [70] *3  + [5], "tanh", "Glorot normal")
model = dde.Model(data, net)

print(time.time())

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
        if self.monitor_op(current, self.best) and current < 6e-2:
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

print(time.time())


# In[27]:

loss_func=[
    "MSE","MSE",
    "MSE", "MSE", "MSE",
    rho0_WASS_cuda0, rhoT_WASS_cuda0]
# loss functions are based on PDE + BC: 3 eq outputs, 2 BCs

model.compile("adam", lr=1e-3,loss=loss_func)
de = 1
losshistory, train_state = model.train(
    iterations=num_epochs,
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



    if args.plot == 0:
        ax1 = plt.subplot(111, frameon=False)
        ax1.grid()

        ########################################################

        s1, s2 = plot_rho_bc('rho_0', test[0:N, :], mu_0, sigma_0, ax1)

        test_tt = test[N:2*N, :]
        s3, s4 = plot_rho_bc('rho_T', test_tt, mu_T, sigma_T, ax1)

        ########################################################

        ax1.plot(
            test_tt[:, X_IDX],
            test_tt[:, Y3_IDX],
            linewidth=1,
            c='m',
            label='y3')

        ########################################################

        ax1.legend(loc='lower right')
        ax1.set_title(
        'rho0: trapz=%.3f, sum=%.3f, rhoT: trapz=%.3f, sum=%.3f' % (s1, s2, s3, s4))
    elif args.plot == 1:
        ax1 = fig.add_subplot(projection='3d')

        ts = test[:, T_IDX]
        xs = test[:, X_IDX]
        rho_opt = test[:, RHO_OPT_IDX]

        ax1.scatter(ts, xs, rho_opt, marker='o')

        ax1.set_xlabel('t')
        ax1.set_ylabel('x')
        ax1.set_zlabel('rho_opt')




        x_T = np.transpose(np.linspace(state_min, state_max, N))
        y_T = np.transpose(np.linspace(state_min, state_max, N))
        z_T = np.transpose(np.linspace(state_min, state_max, N))
        x_T=x_T.reshape(len(x_T),1)
        y_T=y_T.reshape(len(y_T),1)
        z_T=z_T.reshape(len(z_T),1)
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
          ]
        geom = dde.geometry.Interval(state_min, state_max)
        timedomain = dde.geometry.TimeDomain(0., T_t)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)
        time_0=np.hstack((x_T,T_0*np.ones((len(x_T), 1))))
        rho_0=pdf1d(x_T, mu_0, sigma_0).reshape(len(x_T),1)
        rho_0 = np.where(rho_0 < 0, 0, rho_0)
        rho_0 = rho_0 / np.sum(np.abs(rho_0))
        rho_0_BC = dde.icbc.PointSetBC(time_0, rho_0, component=1)
        time_t=np.hstack((x_T,T_t*np.ones((len(x_T), 1))))
        rho_T=pdf1d(x_T, mu_T, sigma_T).reshape(len(x_T),1)
        rho_T = np.where(rho_T < 0, 0, rho_T)
        rho_T = rho_T / np.sum(np.abs(rho_T))
        rho_T_BC = dde.icbc.PointSetBC(time_t, rho_T, component=1)
        data = dde.data.TimePDE(
          geomtime,
          pde,
          [rho_0_BC,rho_T_BC],
          num_domain=5000,
          num_initial=500)
        net = dde.nn.FNN([2] + [70] *3  + [3], "tanh", "Glorot normal")



######################################

C = cdist(state, state, 'sqeuclidean')
cvector = C.reshape((M)**2)

reg = 1e-1 # gamma, 10e-2, 5e-2
C_tensor = torch.from_numpy(
    -C / reg - 1
).requires_grad_(False).type(torch.FloatTensor)
C_tensor = C_tensor.to(device)
c_tensor = torch.from_numpy(
    cvector
).requires_grad_(False).type(torch.FloatTensor)
c_tensor = c_tensor.to(device)

M = torch.exp(C_tensor).requires_grad_(False).type(torch.FloatTensor)
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
# p_opt0 = p_opt0.to(device)

p_optT = torch.zeros_like(M).requires_grad_(True)
# p_optT = p_optT.to(device)


    '''
    z1 = T_0
    z2 = T_t

    ax1.contourf(
        meshes[0],
        meshes[1],
        rho0.reshape(N, N),
        50, zdir='z',
        cmap=cm.jet,
        offset=z1,
        alpha=0.4
    )

    ax1.contourf(
        meshes[0],
        meshes[1],
        rhoT.reshape(N, N),
        50, zdir='z',
        cmap=cm.jet,
        offset=z2,
        alpha=0.4,
    )

    ax1.set_xlim(state_min, state_max)
    ax1.set_zlim(T_0 - 0.1, T_t + 0.1)

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('t')
    ax1.set_title('rho_opt')

    ########################################################

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')

    z1 = T_0
    z2 = T_t

    ax2.contourf(
        meshes[0],
        meshes[1],
        dphi_dinput_t0_dx.reshape(N, N),
        50, zdir='z',
        cmap=cm.jet,
        offset=z1,
        alpha=0.4
    )

    ax2.contourf(
        meshes[0],
        meshes[1],
        dphi_dinput_tT_dx.reshape(N, N),
        50, zdir='z',
        cmap=cm.jet,
        offset=z2,
        alpha=0.4,
    )

    sc2=ax2.scatter(
        grid_x1,
        grid_x2,
        grid_t,
        c=PSI,
        s=np.abs(PSI*20000),
        cmap=cm.jet,
        alpha=1.0)
    plt.colorbar(sc2, shrink=0.25)

    ax2.set_xlim(state_min, state_max)
    ax2.set_zlim(T_0 - 0.1, T_t + 0.1)

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('t')
    ax2.set_title('dphi_dx')

    ########################################################

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    z1 = T_0
    z2 = T_t

    ax3.contourf(
        meshes[0],
        meshes[1],
        dphi_dinput_t0_dy.reshape(N, N),
        50, zdir='z',
        cmap=cm.jet,
        offset=z1,
        alpha=0.4
    )

    ax3.contourf(
        meshes[0],
        meshes[1],
        dphi_dinput_tT_dy.reshape(N, N),
        50, zdir='z',
        cmap=cm.jet,
        offset=z2,
        alpha=0.4,
    )

    sc3=ax3.scatter(
        grid_x1,
        grid_x2,
        grid_t,
        c=PSI2,
        s=np.abs(PSI2*20000),
        cmap=cm.jet,
        alpha=1.0)
    plt.colorbar(sc3, shrink=0.25)

    ax3.set_xlim(state_min, state_max)
    ax3.set_zlim(T_0 - 0.1, T_t + 0.1)

    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('t')
    ax3.set_title('dphi_dy')
    '''



    print(tt.shape)



    '''

    ########################################################

    source_x1 = t0[:, 0]
    source_x2 = t0[:, 1]

    dphi_dinput_t0_dx = dphi_dinput_t0[:, 0]
    dphi_dinput_t0_dy = dphi_dinput_t0[:, 1]

    dphi_dinput_tT_dx = dphi_dinput_tT[:, 0]
    dphi_dinput_tT_dy = dphi_dinput_tT[:, 1]

    x_1_ = np.linspace(state_min, state_max, N)
    x_2_ = np.linspace(state_min, state_max, N)
    t_ = np.linspace(T_0, T_t, N)
    grid_x1, grid_x2, grid_t = np.meshgrid(
        x_1_,
        x_2_,
        t_, copy=False) # each is NxNxN

    # import ipdb; ipdb.set_trace()
    PSI = gd(
      (tt[:, 0], tt[:, 1], tt[:, 2]),
      dphi_dinput_tt[:, 0],
      (grid_x1, grid_x2, grid_t),
      method='nearest')

    PSI2 = gd(
      (tt[:, 0], tt[:, 1], tt[:, 2]),
      dphi_dinput_tt[:, 1],
      (grid_x1, grid_x2, grid_t),
      method='nearest')
    '''


        vinterp_N = 50
        vinterp_T = 50

        state_min = -2.5
        state_max = 2.5
        T_t = 5.0

        t0_v_mat_fname = '%s/%s/notebook_post_predict_t0_%d_%d_v.mat' % (
            os.path.abspath("./"), args.control_prefix, vinterp_N, vinterp_T)

        mid_v_mat_fname = '%s/%s/notebook_post_predict_mid_%d_%d_v.mat' % (
            os.path.abspath("./"), args.control_prefix, vinterp_N, vinterp_T)

        t5_v_mat_fname = '%s/%s/notebook_post_predict_t5_%d_%d_v.mat' % (
            os.path.abspath("./"), args.control_prefix, vinterp_N, vinterp_T)

        # t5_v_mat_fname = '%s/%s/%s_post_predict_x1_x2_x3_t.mat' % (
        #     os.path.abspath("./"), args.control_prefix, args.control_prefix)

        if os.path.exists(mid_v_mat_fname):
            control_data = {
                "x_1_" : np.linspace(state_min, state_max, vinterp_N),
                "x_2_" : np.linspace(state_min, state_max, vinterp_N),
                "x_3_" : np.linspace(state_min, state_max, vinterp_N),
                "t_" : np.linspace(0, T_t, vinterp_T),
            }

            # mat_contents = scipy.io.loadmat(t0_v_mat_fname)
            # for k in ["t0_V1", "t0_V2", "t0_V3"]:
            #     control_data[k] = mat_contents[k]
            # del mat_contents

            mat_contents = scipy.io.loadmat(mid_v_mat_fname)
            for k in ["mid_V1", "mid_V2", "mid_V3"]:
                control_data[k] = mat_contents[k]
            # del mat_contents

            # mat_contents = scipy.io.loadmat(t5_v_mat_fname)
            # for k in ["t5_V1", "t5_V2", "t5_V3"]:
            #     control_data[k] = mat_contents[k]
            # del mat_contents
        else:
            print("missing one of the control v files")



    # import ipdb; ipdb.set_trace();

    # if (t < 1e-8):
    #     # print("using t0")
    #     V1 = control_data["t0_V1"]
    #     V2 = control_data["t0_V2"]
    #     V3 = control_data["t0_V3"]
    # elif (np.abs(t-5.0) < 1e-8):
    #     # print("using t5")
    #     V1 = control_data["t5_V1"]
    #     V2 = control_data["t5_V2"]
    #     V3 = control_data["t5_V3"]
    # else:
    #     # print("using mid")
    #     V1 = control_data["mid_V1"]
    #     V2 = control_data["mid_V2"]
    #     V3 = control_data["mid_V3"]

    # if len(V1.shape) == 3:
    #     v1 = V1[closest_1, closest_2, closest_3]
    #     v2 = V2[closest_1, closest_2, closest_3]
    #     v3 = V3[closest_1, closest_2, closest_3]
    # elif len(V1.shape) == 4:
    #     v1 = V1[closest_1, closest_2, closest_3, closest_t]
    #     v2 = V2[closest_1, closest_2, closest_3, closest_t]
    #     v3 = V3[closest_1, closest_2, closest_3, closest_t]
    # else:
    #     print("ERROR")
    #     import ipdb; ipdb.set_trace();


#!/usr/bin/env python3

'''
#!/usr/bin/python3

USAGE: ./distribution0.py

n: next
p: prev

k: hide lines
l: show lines

a: hide blue
b: show blue
'''

import argparse

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import os, pickle, time, sys
from matplotlib import cm

import scipy.integrate as integrate
from scipy.interpolate import griddata as gd

import scipy.io

from common import *

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
np.set_printoptions(linewidth=190)

def normal_dist_array(x , mean , cov_matrix):
    '''
    x: vector
    mean: vector
    sd: matrix
    '''
    prob_density = np.exp(
        np.linalg.multi_dot([x-mean, np.linalg.inv(cov_matrix), x-mean]) * -0.5) \
        / (np.sqrt(2*np.pi*np.linalg.norm(cov_matrix)))
    # this is a SCALAR value

    return prob_density

def inverse_flow(x1, x2, x3, t, alpha2):
    '''
    inverse flow map [x1, x2, x3] -> [x10, x20, x30]
    '''
    if (np.abs(x1) < 1e-8 and np.abs(x2) < 1e-8):

        print("noo")
        raise Exception("singularity")
    elif (t < 1e-8):
        return np.array([x1, x2, x3])

    alpha2 = 1.0

    omega = alpha2 * x3
    gamma = (x2 - x1 * np.tan(omega*t)) / (x1 + x2*np.tan(omega*t))

    x10 = np.sqrt((x1**2 + x2**2) / (1 + gamma))
    x20 = gamma * np.sqrt((x1**2 + x2**2) / (1 + gamma**2))
    x30 = x3

    return np.array([x10, x20, x30])

def composited(x1, x2, x3, t, alpha2, mean, covariance):
    try:
        inversed_state = inverse_flow(x1, x2, x3, t, alpha2)
        return normal_dist_array(inversed_state, mean, covariance)
    except Exception as e:
        print(str(e))
        return 0.0

def Sphere(rows, cols, func, args=None):
    verts = np.empty((rows+1, cols, 3), dtype=float)
    phi = (np.arange(rows+1) * 2*np.pi *(1+2/rows)/ rows).reshape(rows+1, 1)
    th = ((np.arange(cols) * np.pi / cols).reshape(1, cols)) 

    # if args is not None:
    #     r = func(th, phi, *args)
    # else:
    #     r = func(th, phi)

    r = 1.0
    s =  r* np.sin(th)
    verts[...,0] = s * np.cos(phi)
    verts[...,1] = s * np.sin(phi)
    verts[...,2] = r * np.cos(th)

    verts = verts.reshape((rows+1)*cols, 3)[cols-1:-(cols-1)]  ## remove redundant vertexes from top and bottom
    faces = np.empty((rows*cols*2, 3), dtype=np.uint)
    rowtemplate1 = ((np.arange(cols).reshape(cols, 1) + np.array([[0, 1, 0]])) % cols) + np.array([[0, 0, cols]])
    rowtemplate2 = ((np.arange(cols).reshape(cols, 1) + np.array([[0, 1, 1]])) % cols) + np.array([[cols, 0, cols]])
    for row in range(rows):
        start = row * cols * 2 
        faces[start:start+cols] = rowtemplate1 + row * cols
        faces[start+cols:start+(cols*2)] = rowtemplate2 + row * cols
    faces = faces[cols:-cols]  ## cut off zero-area triangles at top and bottom

    ## adjust for redundant vertexes that were removed from top and bottom
    vmin = cols-1
    faces[faces<vmin] = vmin
    faces -= vmin  
    vmax = verts.shape[0]-1
    faces[faces>vmax] = vmax

    return gl.MeshData(vertexes=verts, faces=faces)

X1_index = 0
X2_index = 1
X3_index = 2
def dynamics(state, t, j1, j2, j3, control_data):
    print("t", t)
    print("state", state)

    statedot = np.zeros_like(state)
    # implicit is that all state dimension NOT set
    # have 0 dynamics == do not change in value

    alpha1 = (j2 - j3) / j1
    alpha2 = (j3 - j1) / j2
    alpha3 = (j1 - j2) / j3

    ########################################

    statedot[X1_index] = alpha1 * state[X2_index] * state[X3_index]
    statedot[X2_index] = alpha2 * state[X3_index] * state[X1_index]
    statedot[X3_index] = alpha3 * state[X1_index] * state[X2_index]

    ########################################

    if control_data is None:
        return statedot

    ########################################

    # print(state)

    # print(t)
    if np.abs(t - T_0) < 1e-8:
        t_key = 't0'
    elif np.abs(t - T_t) < 1e-8:
        t_key = 'tT'
    else:
        t_key = 'tt'

    t_control_data = control_data[t_key]

    query = state
    # if t_key == 'tt':
    if t_control_data['grid'].shape[1] == 4:
        query = np.append(query, t)

    # if np.abs(t - T_0) < 1e-8:
    #     print("t_key", t_key)
    #     print("state", query)

    # grid_l2_norms = np.linalg.norm(query - t_control_data['grid'], ord=2, axis=1)
    # closest_grid_idx = grid_l2_norms.argmin()

    closest_grid_idx = np.linalg.norm(query - t_control_data['grid'], ord=1, axis=1).argmin()
    print("query",
        query,
        closest_grid_idx,
        t_control_data['grid'][closest_grid_idx],
        t_control_data['0'][closest_grid_idx],
        t_control_data['1'][closest_grid_idx],
        t_control_data['2'][closest_grid_idx])

    statedot[X1_index] = statedot[X1_index] + t_control_data['0'][closest_grid_idx]
    statedot[X2_index] = statedot[X2_index] + t_control_data['1'][closest_grid_idx]
    statedot[X3_index] = statedot[X3_index] + t_control_data['2'][closest_grid_idx]

    return statedot

colors = [
    (0, 0, 0),
    (45, 5, 61),
    (84, 42, 55),
    (150, 87, 60),
    (208, 171, 141),
    (255, 255, 255)
]
cmap = pg.ColorMap(pos=np.linspace(0.0, 0.25, len(colors)), color=colors)
cmap = cm.get_cmap('gist_heat') # you want a colormap that for 0 is close to clearColor (black)

red = (1, 0, 0, 1)
green = (0, 1, 0, 1)
gray = (0.5, 0.5, 0.5, 0.3)
blue = (0, 0, 1, 1)
yellow = (1, 1, 0, 1)

def pyqtgraph_plot_line(
    view,
    line_points_row_xyz,
    mode = 'lines', # 'line_strip' = all points are one line
    color = red,
    linewidth = 5.0):
    plt = gl.GLLinePlotItem(
        pos = line_points_row_xyz,
        mode = mode,
        color = color,
        width = linewidth
    )
    view.addItem(plt)

def pyqtgraph_plot_gnomon(view, g, length = 0.5, linewidth = 5):
    o = g.dot(np.array([0.0, 0.0, 0.0, 1.0]))
    x = g.dot(np.array([length, 0.0, 0.0, 1.0]))
    y = g.dot(np.array([0.0, length, 0.0, 1.0]))
    z = g.dot(np.array([0.0, 0.0, length, 1.0]))

    # import ipdb; ipdb.set_trace();

    pyqtgraph_plot_line(view, np.vstack([o, x])[:, :-1], color = red, linewidth = linewidth)
    pyqtgraph_plot_line(view, np.vstack([o, y])[:, :-1], color = green, linewidth = linewidth)
    pyqtgraph_plot_line(view, np.vstack([o, z])[:, :-1], color = blue, linewidth = linewidth)

class MyGLViewWidget(gl.GLViewWidget):
    def __init__(self, initial_pdf, data, point_size, distribution_samples, parent=None, devicePixelRatio=None, rotationMethod='euler'):
        super(MyGLViewWidget, self).__init__(parent, devicePixelRatio)

        self.initial_pdf = initial_pdf
        self.addItem(self.initial_pdf)
        self.showing_initial_pdf = True

        self._data = data
        self.distribution_samples = distribution_samples

        self.X1, self.X2, self.X3 = te_to_data["grid"]
        self.N = te_to_data["N"]
        self.pdf_pos = np.vstack([self.X1.reshape(-1), self.X2.reshape(-1), self.X3.reshape(-1)]).T

        self.i = 0

        data = self._data[self._data["keys"][self.i]]

        self.pdf_scale = 5.0

        self.pdf = gl.GLScatterPlotItem(
            pos=self.pdf_pos,
            size=np.ones((self.N**3)) * 0.05,
            color=cmap(data["probs"])*self.pdf_scale,
            pxMode=False)
        self.addItem(self.pdf)

        # import ipdb; ipdb.set_trace();

        self.lines = []
        for i in range(distribution_samples):
            lines = np.zeros((self.distribution_samples, 3))
            if data["all_time_data"] is not None:
                lines = data["all_time_data"][i, :3, :].T

            self.lines.append(gl.GLLinePlotItem(
                pos = lines,
                width = 0.05,
                color = gray))
            self.addItem(self.lines[-1])
        self.showing_lines = True

        if data["all_time_data"] is not None:
            ends = data["all_time_data"][:, :3, -1]
        else:
            ends = np.zeros((self.distribution_samples, 3))

        self.endpoints = gl.GLScatterPlotItem(
            pos=ends,
            size=point_size,
            color=blue,
            pxMode=False)
        self.addItem(self.endpoints)
        self.showing_endpoints = True

        if data["unforced_all_time_data"] is not None:
            unforced_ends = data["unforced_all_time_data"][:, :3, -1]
        else:
            unforced_ends = np.zeros((self.distribution_samples, 3))

        self.unforced_endpoints = gl.GLScatterPlotItem(
            pos=unforced_ends,
            size=point_size,
            color=yellow,
            pxMode=False)
        self.addItem(self.unforced_endpoints)

    def keyPressEvent(self, ev):
        print("keyPressEvent",
            str(ev.text()), str(ev.key()))
        super(MyGLViewWidget, self).keyPressEvent(ev)

        if ev.text() == "n":
            self.i = min(self.i + 1, len(self._data["keys"])-1)

            data = self._data[self._data["keys"][self.i]]

            self.pdf.setData(
                color=cmap(data["probs"])*self.pdf_scale)

            if data["all_time_data"] is not None:
                ends = data["all_time_data"][:, :3, -1]
            else:
                ends = np.zeros((self.distribution_samples, 3))
            self.endpoints.setData(pos=ends)

            if data["unforced_all_time_data"] is not None:
                unforced_ends = data["unforced_all_time_data"][:, :3, -1]
            else:
                unforced_ends = np.zeros((self.distribution_samples, 3))
            self.unforced_endpoints.setData(pos=unforced_ends)

            for i in range(self.distribution_samples):
                lines = np.zeros((self.distribution_samples, 3))
                if data["all_time_data"] is not None:
                    lines = data["all_time_data"][i, :3, :].T

                self.lines[i].setData(
                    pos=lines)

        elif ev.text() == "p":
            self.i = max(self.i - 1, 0)

            data = self._data[self._data["keys"][self.i]]

            self.pdf.setData(
                color=cmap(data["probs"])*self.pdf_scale)

            if data["all_time_data"] is not None:
                ends = data["all_time_data"][:, :3, -1]
            else:
                ends = np.zeros((self.distribution_samples, 3))
            self.endpoints.setData(pos=ends)

            if data["unforced_all_time_data"] is not None:
                unforced_ends = data["unforced_all_time_data"][:, :3, -1]
            else:
                unforced_ends = np.zeros((self.distribution_samples, 3))
            self.unforced_endpoints.setData(pos=unforced_ends)

            for i in range(self.distribution_samples):
                lines = np.zeros((self.distribution_samples, 3))
                if data["all_time_data"] is not None:
                    lines = data["all_time_data"][i, :3, :].T

                self.lines[i].setData(
                    pos=lines)

        elif ev.text() == "k":
            if self.showing_lines:
                for i in range(self.distribution_samples):
                    self.removeItem(self.lines[i])
                self.showing_lines = False

        elif ev.text() == "l":
            if not self.showing_lines:
                for i in range(self.distribution_samples):
                    self.addItem(self.lines[i])
                self.showing_lines = True

        elif ev.text() == "a":
            if self.showing_endpoints:
                self.removeItem(self.endpoints)
                self.showing_endpoints = False

        elif ev.text() == "b":
            if not self.showing_endpoints:
                self.addItem(self.endpoints)
                self.showing_endpoints = True

        elif ev.text() == "c":
            if not self.showing_initial_pdf:
                self.addItem(self.initial_pdf)
                self.showing_initial_pdf = True
            else:
                self.removeItem(self.initial_pdf)
                self.showing_initial_pdf = False

def init_data(
    mu_0, cov_0,
    windows, distribution_samples, N, ts,
    j1, j2, j3,
    control_data,
    ignore_symmetry=False):

    alpha1 = (j2 - j3) / j1
    alpha2 = (j3 - j1) / j2
    alpha3 = (j1 - j2) / j3
    # if j1 = j2 != j3
    # alpha1 = j1 - j3 / j1
    # alpha2 = j3 - j1 / j1 = -alpha1
    # alpha3 = 0

    #############################################################################

    x1 = np.linspace(mu_0[0] - windows[0], mu_0[0] + windows[1], N)
    x2 = np.linspace(mu_0[1] - windows[2], mu_0[1] + windows[3], N)
    x3 = np.linspace(mu_0[2] - windows[4], mu_0[2] + windows[5], N)

    X1, X2, X3 = np.meshgrid(x1,x2,x3,copy=False) # each is NxNxN

    #############################################################################
    # using broadcasting: 0.12s

    # den = np.sqrt(np.linalg.det(2*np.pi*cov_0))
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html
    den = (2*np.pi)**(len(mu_0)/2) * np.linalg.det(cov_0)**(1/2)
    cov_inv = np.linalg.inv(cov_0)

    initial_sample = np.random.multivariate_normal(
        mu_0, cov_0, distribution_samples) # 100 x 3

    #############################################################################

    x1_closest_index = [(np.abs(x1 - initial_sample[i, 0])).argmin() for i in range(initial_sample.shape[0])]
    x2_closest_index = [(np.abs(x2 - initial_sample[i, 1])).argmin() for i in range(initial_sample.shape[0])]

    # for each initial sample, find the closest x3 = x30 layer it can help de-alias
    # x3_closest_index[i] = initial sample[i]'s dealiasing x30 layer
    x3_closest_index = [(np.abs(x3 - initial_sample[i, 2])).argmin() for i in range(initial_sample.shape[0])]

    #############################################################################

    te_to_data = {
        "keys" : ts,
        "grid" : np.meshgrid(x1,x2,x3,copy=False),
        "N" : N
    }

    do_dealiasing = False

    for t_e in ts:
        start = time.time()

        if (t_e < 1e-8) or (np.abs(j1 - j2) > 1e-8) or ignore_symmetry:
            '''
            for t = 0 we ignore the inverse flow map
            also for the non-axissymmetric case
            '''
            if (t_e > 1e-8):
                print("NOT axis-symmetric, NOT using inverse flow map")

            x10 = X1
            x20 = X2
            x30 = X3
        else:
            print("axis-symmetric, using inverse flow map")
            # axis-symmetric inverse flow map
            # implemented with numpy broadcasting
            omegas = (alpha2 * X3)

            tans = np.tan((omegas*t_e) % (2*np.pi))

            # where arctan(x2 / x1) > 0, x20 / x10 > 0
            gammas = (X2 - X1 * tans) / (X1 + X2*tans)
            gammas = np.nan_to_num(gammas, copy=False)

            x10 = np.sqrt((X1**2 + X2**2) / (1 + gammas))
            x20 = gammas * np.sqrt((X1**2 + X2**2) / (1 + gammas**2))
            x30 = X3

            do_dealiasing = True

        ################### compute the gaussian given the corresponding x0

        x10_diff = x10 - mu_0[0]
        x20_diff = x20 - mu_0[1]
        x30_diff = x30 - mu_0[2]

        # 3 x NxNxN to N**3 x 3
        x10_x20_x30 = np.vstack([x10_diff.reshape(-1), x20_diff.reshape(-1), x30_diff.reshape(-1)])

        # N**3 x 1
        probs = np.exp(np.einsum('i...,ij,j...',x10_x20_x30,cov_inv,x10_x20_x30)/(-2)) / den

        probs_reshape = probs.reshape(N,N,N)
        probs_reshape = np.nan_to_num(probs_reshape, copy=False)

        total_time = time.time() - start
        print("compute %s\n" % str(total_time))

        #############################################################################

        if t_e > 0 and ((np.abs(j1 - j2) > 1e-8) or ignore_symmetry):
            print("NOT axis-symmetric, integrating initial pdf")

            initial_probs = probs_reshape[x1_closest_index, x2_closest_index, x3_closest_index]

            initial_sample = np.column_stack((initial_sample, initial_probs))

        #############################################################################

        all_time_data = None
        unforced_all_time_data = None

        if t_e > 0:
            A = np.sqrt(initial_sample[:, 0]**2 + initial_sample[:, 1]**2)
            phi = np.arctan(initial_sample[:, 1] / initial_sample[:, 0])

            print("distribution_samples", distribution_samples)
            t_samples = np.linspace(0, t_e, 20)

            print(t_samples)

            all_time_data = np.empty(
                (
                    initial_sample.shape[0],
                    initial_sample.shape[1],
                    len(t_samples))
                )
            # x/y slice is all samples at that time, 1 x/y slice per z time initial_sample

            # x[i] is sample [i]
            # y[i] is state dim [i]
            # z[i] is time [i]

            print("integrating forced", initial_sample.shape[0])
            for sample_i in range(initial_sample.shape[0]):
                print("sample_i", sample_i)
                print("initial_sample[sample_i, :]", initial_sample[sample_i, :])
                sample_states = integrate.odeint(
                    dynamics,
                    initial_sample[sample_i, :],
                    t_samples,
                    args=(j1, j2, j3, None))
                all_time_data[sample_i, :, :] = sample_states.T

            # print("initial_sample", initial_sample[0])

            # i = 0
            # state = initial_sample[0]
            # while i < len(t_samples) - 1:
            #     # solve differential equation, take final result only
            #     print("i", i, t_samples[i:i+2])
            #     state = integrate.odeint(
            #         dynamics,
            #         state,
            #         t_samples[i:i+2],
            #         args=(j1, j2, j3, None))[-1]
            #     all_time_data[0, :, i] = state
            #     i = i + 1


            # sample_states = integrate.odeint(
            #     dynamics,
            #     initial_sample[0, :],
            #     t_samples,
            #     args=(j1, j2, j3, None)
            # )
            # all_time_data[0, :, :] = sample_states.T

            t_samples = np.linspace(0, t_e, 50)

            unforced_all_time_data = np.empty(
                (
                    initial_sample.shape[0],
                    initial_sample.shape[1],
                    len(t_samples))
                )

            # import ipdb; ipdb.set_trace()

            print("integrating unforced")
            # for sample_i in range(initial_sample.shape[0]):
            #     print("sample_i", sample_i)
            #     sample_states = integrate.odeint(
            #         unforced_dynamics_with_args,
            #         initial_sample[sample_i, :],
            #         t_samples)
            #     unforced_all_time_data[sample_i, :, :] = sample_states.T

            sample_states = integrate.odeint(
                dynamics,
                initial_sample[0, :],
                t_samples,
                args=(3, 2, 1, None)
            )
            unforced_all_time_data[0, :, :] = sample_states.T

        #############################################################################

        if t_e > 0 and (do_dealiasing):
            '''
            deal with the aliasing issue here for axissymmetric case / inverse flow map
            for each integrated endpoint, create a decision boundary
            +-90 deg on either side
            and all probabilities on the side where the endpoint lives
            scaled by 1, otherwise scaled by 0
            '''
            print("axis-symmetric, fixing aliasing")

            ends = all_time_data[:, :, -1]

            atan2s = np.arctan2(ends[:, 1], ends[:, 0])
            xa = np.cos(atan2s + np.pi / 2)
            ya = np.sin(atan2s + np.pi / 2)
            xb = np.cos(atan2s - np.pi / 2)
            yb = np.sin(atan2s - np.pi / 2)
            slopes = (yb - ya) / (xb - xa)

            switches = np.where(ends[:, 1] > ends[:, 0] * slopes, 1, 0)
            not_switches = np.where(ends[:, 1] <= ends[:, 0] * slopes, 1, 0)

            # x3_closest_index[i] = initial sample[i]'s dealiasing x30 layer
            for i, x3_i in enumerate(x3_closest_index):
                slope = slopes[i]
                X2_layer = X2[:, :, x3_i]
                X1_layer = X1[:, :, x3_i]

                scale = np.where(X2_layer > X1_layer * slope, switches[i], not_switches[i])
                probs_reshape[:, :, x3_i] = probs_reshape[:, :, x3_i] * scale

        #############################################################################

        if t_e > 0 and ((np.abs(j1 - j2) > 1e-8) or ignore_symmetry):
            print("NOT axis-symmetric, assigning and interpolating integrated pdf")

            # probs_reshape = np.zeros

            latest_slice = all_time_data[:, :, -1] # this will be distribution_samples x 4

            closest_1 = [(np.abs(x1 - latest_slice[i, 0])).argmin() for i in range(latest_slice.shape[0])]
            closest_2 = [(np.abs(x2 - latest_slice[i, 1])).argmin() for i in range(latest_slice.shape[0])]
            closest_3 = [(np.abs(x3 - latest_slice[i, 2])).argmin() for i in range(latest_slice.shape[0])]

            probs_reshape = np.zeros_like(probs_reshape)
            # some transposing going on in some reshape
            # swapping closest_1/2 works well
            probs_reshape[closest_2, closest_1, closest_3] = latest_slice[:, 3] # 1.0

            probs_reshape = gd(
                latest_slice[:, :3],
                latest_slice[:, 3],
                (X1, X2, X3),
                method='linear')
            probs_reshape = np.nan_to_num(probs_reshape, copy=False)

            # import ipdb; ipdb.set_trace();

        #############################################################################

        # NxNxN back to N**3 x 1
        probs = probs_reshape.reshape(-1)

        te_to_data[t_e] = {
            "probs" : probs,
            "all_time_data" : all_time_data,
            "unforced_all_time_data" : unforced_all_time_data,
        }

    return initial_sample, te_to_data, X1, X2, X3

#############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--times',
        type=str,
        default="0,1.0,2.0,3.0,4.0,5.0",
        required=False)

    parser.add_argument('--mu_0',
        type=float,
        default=2.0,
        required=False)

    parser.add_argument('--sampling',
        type=str,
        default="15,15,15,15,15,15,30,100",
        required=False)

    parser.add_argument('--system',
        type=str,
        default="1,1,2", # 3,2,1
        required=False)

    parser.add_argument('--ignore_symmetry',
        type=int,
        default=0,
        required=False)

    parser.add_argument('--control_data',
        type=str,
        default="",
        required=False)

    args = parser.parse_args()

    # distribution
    # mu_0 = np.array([args.mu_0]*3)
    # cov_0 = np.eye(3)

    # sampling
    # ts = [float(x) for x in args.times.split(",")]
    ts = np.linspace(T_0, 20, 2)
    print("ts", ts)

    sampling = [int(x) for x in args.sampling.split(",")]
    windows = sampling[:6]
    N = sampling[6]
    print("N", N)
    distribution_samples = sampling[7]

    # j1, j2, j3 = [float(x) for x in args.system.split(",")]
    # j1 = float(j1)
    # j2 = float(j2)
    # j3 = float(j3)
    print("j1, j2, j3", j1, j2, j3)

    #############################################################################

    control_data = None
    if len(args.control_data) > 0:
        control_data = np.load(
            args.control_data,
            allow_pickle=True).item()

    #############################################################################

    initial_sample, te_to_data, X1, X2, X3 = init_data(
        [mu_0]*3, np.eye(3)*sigma_0,
        # mu_0, cov_0,
        windows, distribution_samples, N, ts,
        j1, j2, j3,
        control_data,
        args.ignore_symmetry)

    #############################################################################

    ## Create a GL View widget to display data
    # app = QtGui.QApplication([])
    app = pg.mkQApp("GLScatterPlotItem Example")

    point_size = np.ones(distribution_samples) * 0.08

    initial_pdf_sample = gl.GLScatterPlotItem(
        pos=initial_sample[:, :3],
        size=point_size,
        color=green,
        pxMode=False)

    # import ipdb; ipdb.set_trace();

    w = MyGLViewWidget(initial_pdf_sample, te_to_data, point_size, distribution_samples)

    w.setWindowTitle('snapshots')
    w.setCameraPosition(distance=20)

    #############################################################################

    ## Add a grid to the view
    g = gl.GLGridItem()
    g.scale(2,2,1)
    g.setDepthValue(10)  # draw grid after surfaces since they may be translucent
    # w.addItem(g)

    pyqtgraph_plot_gnomon(w, np.eye(4), 1.0, 1.0)

    #############################################################################

    ## Start Qt event loop unless running in interactive mode.
    w.show()

    pg.exec()

#!/usr/bin/env python3

'''
#!/usr/bin/python3

USAGE: ./distribution0.py

n: next
p: prev

k: hide lines
l: show lines

a: hide blue
b: show blue
'''

import argparse

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import os, pickle, time, sys
from matplotlib import cm

import scipy.integrate as integrate
from scipy.interpolate import griddata as gd

import scipy.io

def normal_dist_array(x , mean , cov_matrix):
    '''
    x: vector
    mean: vector
    sd: matrix
    '''
    prob_density = np.exp(
        np.linalg.multi_dot([x-mean, np.linalg.inv(cov_matrix), x-mean]) * -0.5) \
        / (np.sqrt(2*np.pi*np.linalg.norm(cov_matrix)))
    # this is a SCALAR value

    return prob_density

def inverse_flow(x1, x2, x3, t, alpha2):
    '''
    inverse flow map [x1, x2, x3] -> [x10, x20, x30]
    '''
    if (np.abs(x1) < 1e-8 and np.abs(x2) < 1e-8):

        print("noo")
        raise Exception("singularity")
    elif (t < 1e-8):
        return np.array([x1, x2, x3])

    alpha2 = 1.0

    omega = alpha2 * x3
    gamma = (x2 - x1 * np.tan(omega*t)) / (x1 + x2*np.tan(omega*t))

    x10 = np.sqrt((x1**2 + x2**2) / (1 + gamma))
    x20 = gamma * np.sqrt((x1**2 + x2**2) / (1 + gamma**2))
    x30 = x3

    return np.array([x10, x20, x30])

def composited(x1, x2, x3, t, alpha2, mean, covariance):
    try:
        inversed_state = inverse_flow(x1, x2, x3, t, alpha2)
        return normal_dist_array(inversed_state, mean, covariance)
    except Exception as e:
        print(str(e))
        return 0.0

def Sphere(rows, cols, func, args=None):
    verts = np.empty((rows+1, cols, 3), dtype=float)
    phi = (np.arange(rows+1) * 2*np.pi *(1+2/rows)/ rows).reshape(rows+1, 1)
    th = ((np.arange(cols) * np.pi / cols).reshape(1, cols)) 

    # if args is not None:
    #     r = func(th, phi, *args)
    # else:
    #     r = func(th, phi)

    r = 1.0
    s =  r* np.sin(th)
    verts[...,0] = s * np.cos(phi)
    verts[...,1] = s * np.sin(phi)
    verts[...,2] = r * np.cos(th)

    verts = verts.reshape((rows+1)*cols, 3)[cols-1:-(cols-1)]  ## remove redundant vertexes from top and bottom
    faces = np.empty((rows*cols*2, 3), dtype=np.uint)
    rowtemplate1 = ((np.arange(cols).reshape(cols, 1) + np.array([[0, 1, 0]])) % cols) + np.array([[0, 0, cols]])
    rowtemplate2 = ((np.arange(cols).reshape(cols, 1) + np.array([[0, 1, 1]])) % cols) + np.array([[cols, 0, cols]])
    for row in range(rows):
        start = row * cols * 2 
        faces[start:start+cols] = rowtemplate1 + row * cols
        faces[start+cols:start+(cols*2)] = rowtemplate2 + row * cols
    faces = faces[cols:-cols]  ## cut off zero-area triangles at top and bottom

    ## adjust for redundant vertexes that were removed from top and bottom
    vmin = cols-1
    faces[faces<vmin] = vmin
    faces -= vmin  
    vmax = verts.shape[0]-1
    faces[faces>vmax] = vmax

    return gl.MeshData(vertexes=verts, faces=faces)

X1_index = 0
X2_index = 1
X3_index = 2
def dynamics(state, t, j1, j2, j3, control_data):
    statedot = np.zeros_like(state)
    # implicit is that all state dimension NOT set
    # have 0 dynamics == do not change in value

    alpha1 = (j2 - j3) / j1
    alpha2 = (j3 - j1) / j2
    alpha3 = (j1 - j2) / j3

    ########################################

    statedot[X1_index] = alpha1 * state[X2_index] * state[X3_index]
    statedot[X2_index] = alpha2 * state[X3_index] * state[X1_index]
    statedot[X3_index] = alpha3 * state[X1_index] * state[X2_index]

    ########################################

    if control_data is None:
        return statedot

    V1 = control_data["mid_V1"]
    V2 = control_data["mid_V2"]
    V3 = control_data["mid_V3"]

    v1 = V1[closest_1, closest_2, closest_3, closest_t]
    v2 = V2[closest_1, closest_2, closest_3, closest_t]
    v3 = V3[closest_1, closest_2, closest_3, closest_t]

    statedot[X1_index] = statedot[X1_index] + v1
    statedot[X2_index] = statedot[X2_index] + v2
    statedot[X3_index] = statedot[X3_index] + v3

    return statedot

colors = [
    (0, 0, 0),
    (45, 5, 61),
    (84, 42, 55),
    (150, 87, 60),
    (208, 171, 141),
    (255, 255, 255)
]
cmap = pg.ColorMap(pos=np.linspace(0.0, 0.25, len(colors)), color=colors)
cmap = cm.get_cmap('gist_heat') # you want a colormap that for 0 is close to clearColor (black)

red = (1, 0, 0, 1)
green = (0, 1, 0, 1)
gray = (0.5, 0.5, 0.5, 0.3)
blue = (0, 0, 1, 1)
yellow = (1, 1, 0, 1)

def pyqtgraph_plot_line(
    view,
    line_points_row_xyz,
    mode = 'lines', # 'line_strip' = all points are one line
    color = red,
    linewidth = 5.0):
    plt = gl.GLLinePlotItem(
        pos = line_points_row_xyz,
        mode = mode,
        color = color,
        width = linewidth
    )
    view.addItem(plt)

def pyqtgraph_plot_gnomon(view, g, length = 0.5, linewidth = 5):
    o = g.dot(np.array([0.0, 0.0, 0.0, 1.0]))
    x = g.dot(np.array([length, 0.0, 0.0, 1.0]))
    y = g.dot(np.array([0.0, length, 0.0, 1.0]))
    z = g.dot(np.array([0.0, 0.0, length, 1.0]))

    # import ipdb; ipdb.set_trace();

    pyqtgraph_plot_line(view, np.vstack([o, x])[:, :-1], color = red, linewidth = linewidth)
    pyqtgraph_plot_line(view, np.vstack([o, y])[:, :-1], color = green, linewidth = linewidth)
    pyqtgraph_plot_line(view, np.vstack([o, z])[:, :-1], color = blue, linewidth = linewidth)

class MyGLViewWidget(gl.GLViewWidget):
    def __init__(self, initial_pdf, data, point_size, distribution_samples, parent=None, devicePixelRatio=None, rotationMethod='euler'):
        super(MyGLViewWidget, self).__init__(parent, devicePixelRatio)

        self.initial_pdf = initial_pdf
        self.addItem(self.initial_pdf)
        self.showing_initial_pdf = True

        self._data = data
        self.distribution_samples = distribution_samples

        self.X1, self.X2, self.X3 = te_to_data["grid"]
        self.N = te_to_data["N"]
        self.pdf_pos = np.vstack([self.X1.reshape(-1), self.X2.reshape(-1), self.X3.reshape(-1)]).T

        self.i = 0

        data = self._data[self._data["keys"][self.i]]

        self.pdf_scale = 5.0

        self.pdf = gl.GLScatterPlotItem(
            pos=self.pdf_pos,
            size=np.ones((self.N**3)) * 0.05,
            color=cmap(data["probs"])*self.pdf_scale,
            pxMode=False)
        self.addItem(self.pdf)

        # import ipdb; ipdb.set_trace();

        self.lines = []
        for i in range(distribution_samples):
            lines = np.zeros((self.distribution_samples, 3))
            if data["all_time_data"] is not None:
                lines = data["all_time_data"][i, :3, :].T

            self.lines.append(gl.GLLinePlotItem(
                pos = lines,
                width = 0.05,
                color = gray))
            self.addItem(self.lines[-1])
        self.showing_lines = True

        if data["all_time_data"] is not None:
            ends = data["all_time_data"][:, :3, -1]
        else:
            ends = np.zeros((self.distribution_samples, 3))

        self.endpoints = gl.GLScatterPlotItem(
            pos=ends,
            size=point_size,
            color=blue,
            pxMode=False)
        self.addItem(self.endpoints)
        self.showing_endpoints = True

        if data["unforced_all_time_data"] is not None:
            unforced_ends = data["unforced_all_time_data"][:, :3, -1]
        else:
            unforced_ends = np.zeros((self.distribution_samples, 3))

        self.unforced_endpoints = gl.GLScatterPlotItem(
            pos=unforced_ends,
            size=point_size,
            color=yellow,
            pxMode=False)
        self.addItem(self.unforced_endpoints)

    def keyPressEvent(self, ev):
        print("keyPressEvent",
            str(ev.text()), str(ev.key()))
        super(MyGLViewWidget, self).keyPressEvent(ev)

        if ev.text() == "n":
            self.i = min(self.i + 1, len(self._data["keys"])-1)

            data = self._data[self._data["keys"][self.i]]

            self.pdf.setData(
                color=cmap(data["probs"])*self.pdf_scale)

            if data["all_time_data"] is not None:
                ends = data["all_time_data"][:, :3, -1]
            else:
                ends = np.zeros((self.distribution_samples, 3))
            self.endpoints.setData(pos=ends)

            if data["unforced_all_time_data"] is not None:
                unforced_ends = data["unforced_all_time_data"][:, :3, -1]
            else:
                unforced_ends = np.zeros((self.distribution_samples, 3))
            self.unforced_endpoints.setData(pos=unforced_ends)

            for i in range(self.distribution_samples):
                lines = np.zeros((self.distribution_samples, 3))
                if data["all_time_data"] is not None:
                    lines = data["all_time_data"][i, :3, :].T

                self.lines[i].setData(
                    pos=lines)

        elif ev.text() == "p":
            self.i = max(self.i - 1, 0)

            data = self._data[self._data["keys"][self.i]]

            self.pdf.setData(
                color=cmap(data["probs"])*self.pdf_scale)

            if data["all_time_data"] is not None:
                ends = data["all_time_data"][:, :3, -1]
            else:
                ends = np.zeros((self.distribution_samples, 3))
            self.endpoints.setData(pos=ends)

            if data["unforced_all_time_data"] is not None:
                unforced_ends = data["unforced_all_time_data"][:, :3, -1]
            else:
                unforced_ends = np.zeros((self.distribution_samples, 3))
            self.unforced_endpoints.setData(pos=unforced_ends)

            for i in range(self.distribution_samples):
                lines = np.zeros((self.distribution_samples, 3))
                if data["all_time_data"] is not None:
                    lines = data["all_time_data"][i, :3, :].T

                self.lines[i].setData(
                    pos=lines)

        elif ev.text() == "k":
            if self.showing_lines:
                for i in range(self.distribution_samples):
                    self.removeItem(self.lines[i])
                self.showing_lines = False

        elif ev.text() == "l":
            if not self.showing_lines:
                for i in range(self.distribution_samples):
                    self.addItem(self.lines[i])
                self.showing_lines = True

        elif ev.text() == "a":
            if self.showing_endpoints:
                self.removeItem(self.endpoints)
                self.showing_endpoints = False

        elif ev.text() == "b":
            if not self.showing_endpoints:
                self.addItem(self.endpoints)
                self.showing_endpoints = True

        elif ev.text() == "c":
            if not self.showing_initial_pdf:
                self.addItem(self.initial_pdf)
                self.showing_initial_pdf = True
            else:
                self.removeItem(self.initial_pdf)
                self.showing_initial_pdf = False

def init_data(
    mu_0, cov_0,
    windows, distribution_samples, N, ts,
    j1, j2, j3,
    control_data,
    ignore_symmetry=False):

    alpha1 = (j2 - j3) / j1
    alpha2 = (j3 - j1) / j2
    alpha3 = (j1 - j2) / j3
    # if j1 = j2 != j3
    # alpha1 = j1 - j3 / j1
    # alpha2 = j3 - j1 / j1 = -alpha1
    # alpha3 = 0

    #############################################################################

    x1 = np.linspace(mu_0[0] - windows[0], mu_0[0] + windows[1], N)
    x2 = np.linspace(mu_0[1] - windows[2], mu_0[1] + windows[3], N)
    x3 = np.linspace(mu_0[2] - windows[4], mu_0[2] + windows[5], N)

    X1, X2, X3 = np.meshgrid(x1,x2,x3,copy=False) # each is NxNxN

    #############################################################################
    # using broadcasting: 0.12s

    # den = np.sqrt(np.linalg.det(2*np.pi*cov_0))
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html
    den = (2*np.pi)**(len(mu_0)/2) * np.linalg.det(cov_0)**(1/2)
    cov_inv = np.linalg.inv(cov_0)

    initial_sample = np.random.multivariate_normal(
        mu_0, cov_0, distribution_samples) # 100 x 3

    #############################################################################

    x1_closest_index = [(np.abs(x1 - initial_sample[i, 0])).argmin() for i in range(initial_sample.shape[0])]
    x2_closest_index = [(np.abs(x2 - initial_sample[i, 1])).argmin() for i in range(initial_sample.shape[0])]

    # for each initial sample, find the closest x3 = x30 layer it can help de-alias
    # x3_closest_index[i] = initial sample[i]'s dealiasing x30 layer
    x3_closest_index = [(np.abs(x3 - initial_sample[i, 2])).argmin() for i in range(initial_sample.shape[0])]

    #############################################################################

    te_to_data = {
        "keys" : ts,
        "grid" : np.meshgrid(x1,x2,x3,copy=False),
        "N" : N
    }

    do_dealiasing = False

    for t_e in ts:
        start = time.time()

        if (t_e < 1e-8) or (np.abs(j1 - j2) > 1e-8) or ignore_symmetry:
            '''
            for t = 0 we ignore the inverse flow map
            also for the non-axissymmetric case
            '''
            if (t_e > 1e-8):
                print("NOT axis-symmetric, NOT using inverse flow map")

            x10 = X1
            x20 = X2
            x30 = X3
        else:
            print("axis-symmetric, using inverse flow map")
            # axis-symmetric inverse flow map
            # implemented with numpy broadcasting
            omegas = (alpha2 * X3)

            tans = np.tan((omegas*t_e) % (2*np.pi))

            # where arctan(x2 / x1) > 0, x20 / x10 > 0
            gammas = (X2 - X1 * tans) / (X1 + X2*tans)
            gammas = np.nan_to_num(gammas, copy=False)

            x10 = np.sqrt((X1**2 + X2**2) / (1 + gammas))
            x20 = gammas * np.sqrt((X1**2 + X2**2) / (1 + gammas**2))
            x30 = X3

            do_dealiasing = True

        ################### compute the gaussian given the corresponding x0

        x10_diff = x10 - mu_0[0]
        x20_diff = x20 - mu_0[1]
        x30_diff = x30 - mu_0[2]

        # 3 x NxNxN to N**3 x 3
        x10_x20_x30 = np.vstack([x10_diff.reshape(-1), x20_diff.reshape(-1), x30_diff.reshape(-1)])

        # N**3 x 1
        probs = np.exp(np.einsum('i...,ij,j...',x10_x20_x30,cov_inv,x10_x20_x30)/(-2)) / den

        probs_reshape = probs.reshape(N,N,N)
        probs_reshape = np.nan_to_num(probs_reshape, copy=False)

        total_time = time.time() - start
        print("compute %s\n" % str(total_time))

        #############################################################################

        if t_e > 0 and ((np.abs(j1 - j2) > 1e-8) or ignore_symmetry):
            print("NOT axis-symmetric, integrating initial pdf")

            initial_probs = probs_reshape[x1_closest_index, x2_closest_index, x3_closest_index]

            initial_sample = np.column_stack((initial_sample, initial_probs))

        #############################################################################

        all_time_data = None
        unforced_all_time_data = None

        if t_e > 0:
            A = np.sqrt(initial_sample[:, 0]**2 + initial_sample[:, 1]**2)
            phi = np.arctan(initial_sample[:, 1] / initial_sample[:, 0])

            t_samples = np.linspace(0, t_e, distribution_samples)

            all_time_data = np.empty(
                (
                    initial_sample.shape[0],
                    initial_sample.shape[1],
                    len(t_samples))
                )
            # x/y slice is all samples at that time, 1 x/y slice per z time initial_sample

            # x[i] is sample [i]
            # y[i] is state dim [i]
            # z[i] is time [i]

            for sample_i in range(initial_sample.shape[0]):
                sample_states = integrate.odeint(
                    dynamics,
                    initial_sample[sample_i, :],
                    t_samples,
                    args=(j1, j2, j3, control_data))
                all_time_data[sample_i, :, :] = sample_states.T

            unforced_all_time_data = np.empty(
                (
                    initial_sample.shape[0],
                    initial_sample.shape[1],
                    len(t_samples))
                )

            for sample_i in range(initial_sample.shape[0]):
                sample_states = integrate.odeint(
                    dynamics,
                    initial_sample[sample_i, :],
                    t_samples,
                    args=(j1, j2, j3, None))
                unforced_all_time_data[sample_i, :, :] = sample_states.T

        #############################################################################

        if t_e > 0 and (do_dealiasing):
            '''
            deal with the aliasing issue here for axissymmetric case / inverse flow map
            for each integrated endpoint, create a decision boundary
            +-90 deg on either side
            and all probabilities on the side where the endpoint lives
            scaled by 1, otherwise scaled by 0
            '''
            print("axis-symmetric, fixing aliasing")

            ends = all_time_data[:, :, -1]

            atan2s = np.arctan2(ends[:, 1], ends[:, 0])
            xa = np.cos(atan2s + np.pi / 2)
            ya = np.sin(atan2s + np.pi / 2)
            xb = np.cos(atan2s - np.pi / 2)
            yb = np.sin(atan2s - np.pi / 2)
            slopes = (yb - ya) / (xb - xa)

            switches = np.where(ends[:, 1] > ends[:, 0] * slopes, 1, 0)
            not_switches = np.where(ends[:, 1] <= ends[:, 0] * slopes, 1, 0)

            # x3_closest_index[i] = initial sample[i]'s dealiasing x30 layer
            for i, x3_i in enumerate(x3_closest_index):
                slope = slopes[i]
                X2_layer = X2[:, :, x3_i]
                X1_layer = X1[:, :, x3_i]

                scale = np.where(X2_layer > X1_layer * slope, switches[i], not_switches[i])
                probs_reshape[:, :, x3_i] = probs_reshape[:, :, x3_i] * scale

        #############################################################################

        if t_e > 0 and ((np.abs(j1 - j2) > 1e-8) or ignore_symmetry):
            print("NOT axis-symmetric, assigning and interpolating integrated pdf")

            # probs_reshape = np.zeros

            latest_slice = all_time_data[:, :, -1] # this will be distribution_samples x 4

            closest_1 = [(np.abs(x1 - latest_slice[i, 0])).argmin() for i in range(latest_slice.shape[0])]
            closest_2 = [(np.abs(x2 - latest_slice[i, 1])).argmin() for i in range(latest_slice.shape[0])]
            closest_3 = [(np.abs(x3 - latest_slice[i, 2])).argmin() for i in range(latest_slice.shape[0])]

            probs_reshape = np.zeros_like(probs_reshape)
            # some transposing going on in some reshape
            # swapping closest_1/2 works well
            probs_reshape[closest_2, closest_1, closest_3] = latest_slice[:, 3] # 1.0

            probs_reshape = gd(
                latest_slice[:, :3],
                latest_slice[:, 3],
                (X1, X2, X3),
                method='linear')
            probs_reshape = np.nan_to_num(probs_reshape, copy=False)

            # import ipdb; ipdb.set_trace();

        #############################################################################

        # NxNxN back to N**3 x 1
        probs = probs_reshape.reshape(-1)

        te_to_data[t_e] = {
            "probs" : probs,
            "all_time_data" : all_time_data,
            "unforced_all_time_data" : unforced_all_time_data,
        }

    return initial_sample, te_to_data, X1, X2, X3

#############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--times',
        type=str,
        default="0,1.0,2.0,3.0,4.0,5.0",
        required=False)

    parser.add_argument('--mu_0',
        type=float,
        default=2.0,
        required=False)

    parser.add_argument('--sampling',
        type=str,
        default="15,15,15,15,15,15,100,200",
        required=False)

    parser.add_argument('--system',
        type=str,
        default="1,1,2", # 3,2,1
        required=False)

    parser.add_argument('--ignore_symmetry',
        type=int,
        default=0,
        required=False)

    parser.add_argument('--control_data',
        type=str,
        default="",
        required=False)

    args = parser.parse_args()

    # distribution
    mu_0 = np.array([args.mu_0]*3)
    cov_0 = np.eye(3)

    # sampling
    ts = [float(x) for x in args.times.split(",")]

    sampling = [int(x) for x in args.sampling.split(",")]
    window0 = sampling[0]
    window1 = sampling[1]
    window2 = sampling[2]
    window3 = sampling[3]
    window4 = sampling[4]
    window5 = sampling[5]
    windows = sampling[:6]
    N = sampling[6]
    distribution_samples = sampling[7]

    j1, j2, j3 = [float(x) for x in args.system.split(",")]

    #############################################################################

    control_data = None

    #############################################################################

    initial_sample, te_to_data, X1, X2, X3 = init_data(
        mu_0, cov_0,
        windows, distribution_samples, N, ts,
        j1, j2, j3,
        control_data,
        args.ignore_symmetry)

    #############################################################################

    ## Create a GL View widget to display data
    app = pg.mkQApp("GLScatterPlotItem Example")

    point_size = np.ones(distribution_samples) * 0.08

    initial_pdf_sample = gl.GLScatterPlotItem(
        pos=initial_sample[:, :3],
        size=point_size,
        color=green,
        pxMode=False)

    # import ipdb; ipdb.set_trace();

    w = MyGLViewWidget(initial_pdf_sample, te_to_data, point_size, distribution_samples)

    w.setWindowTitle('snapshots')
    w.setCameraPosition(distance=20)

    #############################################################################

    ## Add a grid to the view
    g = gl.GLGridItem()
    g.scale(2,2,1)
    g.setDepthValue(10)  # draw grid after surfaces since they may be translucent
    # w.addItem(g)

    pyqtgraph_plot_gnomon(w, np.eye(4), 1.0, 1.0)

    #############################################################################

    ## Start Qt event loop unless running in interactive mode.
    w.show()

    pg.exec()






        if (t_e < 1e-8) or (np.abs(j1 - j2) > 1e-8) or ignore_symmetry:
            '''
            for t = 0 we ignore the inverse flow map
            also for the non-axissymmetric case
            '''
            if (t_e > 1e-8):
                print("NOT axis-symmetric, NOT using inverse flow map")

            x10 = X1
            x20 = X2
            # x30 = X3
        else:
            print("axis-symmetric, using inverse flow map")
            # axis-symmetric inverse flow map
            # implemented with numpy broadcasting
            omegas = (alpha2 * X3)

            tans = np.tan((omegas*t_e) % (2*np.pi))

            # where arctan(x2 / x1) > 0, x20 / x10 > 0
            gammas = (X2 - X1 * tans) / (X1 + X2*tans)
            gammas = np.nan_to_num(gammas, copy=False)

            x10 = np.sqrt((X1**2 + X2**2) / (1 + gammas))
            x20 = gammas * np.sqrt((X1**2 + X2**2) / (1 + gammas**2))
            x30 = X3

            do_dealiasing = True

        ################### compute the gaussian given the corresponding x0

        x10_diff = x10 - mu_0[0]
        x20_diff = x20 - mu_0[1]
        x30_diff = x30 - mu_0[2]

        # 3 x NxNxN to N**3 x 3
        x10_x20_x30 = np.vstack([x10_diff.reshape(-1), x20_diff.reshape(-1), x30_diff.reshape(-1)])

        # N**3 x 1
        probs = np.exp(np.einsum('i...,ij,j...',x10_x20_x30,cov_inv,x10_x20_x30)/(-2)) / den

        probs_reshape = probs.reshape(N,N,N)
        probs_reshape = np.nan_to_num(probs_reshape, copy=False)

        total_time = time.time() - start
        print("compute %s\n" % str(total_time))

        #############################################################################

        if t_e > 0 and ((np.abs(j1 - j2) > 1e-8) or ignore_symmetry):
            print("NOT axis-symmetric, integrating initial pdf")

            initial_probs = probs_reshape[x1_closest_index, x2_closest_index, x3_closest_index]

            initial_sample = np.column_stack((initial_sample, initial_probs))


        #############################################################################

        if t_e > 0 and (do_dealiasing):
            '''
            deal with the aliasing issue here for axissymmetric case / inverse flow map
            for each integrated endpoint, create a decision boundary
            +-90 deg on either side
            and all probabilities on the side where the endpoint lives
            scaled by 1, otherwise scaled by 0
            '''
            print("axis-symmetric, fixing aliasing")

            ends = all_time_data[:, :, -1]

            atan2s = np.arctan2(ends[:, 1], ends[:, 0])
            xa = np.cos(atan2s + np.pi / 2)
            ya = np.sin(atan2s + np.pi / 2)
            xb = np.cos(atan2s - np.pi / 2)
            yb = np.sin(atan2s - np.pi / 2)
            slopes = (yb - ya) / (xb - xa)

            switches = np.where(ends[:, 1] > ends[:, 0] * slopes, 1, 0)
            not_switches = np.where(ends[:, 1] <= ends[:, 0] * slopes, 1, 0)

            # x3_closest_index[i] = initial sample[i]'s dealiasing x30 layer
            for i, x3_i in enumerate(x3_closest_index):
                slope = slopes[i]
                X2_layer = X2[:, :, x3_i]
                X1_layer = X1[:, :, x3_i]

                scale = np.where(X2_layer > X1_layer * slope, switches[i], not_switches[i])
                probs_reshape[:, :, x3_i] = probs_reshape[:, :, x3_i] * scale



        #############################################################################

        if t_e > 0 and ((np.abs(j1 - j2) > 1e-8) or ignore_symmetry):
            print("NOT axis-symmetric, assigning and interpolating integrated pdf")

            # probs_reshape = np.zeros

            latest_slice = all_time_data[:, :, -1] # this will be distribution_samples x 4

            closest_1 = [(np.abs(x1 - latest_slice[i, 0])).argmin() for i in range(latest_slice.shape[0])]
            closest_2 = [(np.abs(x2 - latest_slice[i, 1])).argmin() for i in range(latest_slice.shape[0])]
            # closest_3 = [(np.abs(x3 - latest_slice[i, 2])).argmin() for i in range(latest_slice.shape[0])]

            probs_reshape = np.zeros_like(probs_reshape)
            # some transposing going on in some reshape
            # swapping closest_1/2 works well
            probs_reshape[closest_2, closest_1, closest_3] = latest_slice[:, 3] # 1.0

            probs_reshape = gd(
                latest_slice[:, :3],
                latest_slice[:, 3],
                (X1, X2, X3),
                method='linear')
            probs_reshape = np.nan_to_num(probs_reshape, copy=False)

            # import ipdb; ipdb.set_trace();


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

In [59]: history                                                                                                                                                                     
ls
import sympy
sympy.init_printing()
from sympy import *
x, y, z = symbols('x y z')
x
f = x**2 / y
f.diff(x).diff(x)
f.diff(x).diff(y)
f = x**2 / (y*z)
f
f.diff(x).diff(x)
f.diff(x).diff(y)
f.diff(x).diff(z)
from sympy.tensor.array import derive_by_array
derive_by_array(f, [x, y, z])
f
derive_by_array(f, [x, y, z])
derive_by_array(f, (x, y, z))
derive_by_array(derive_by_array(f, (x, y, z)), (x, y, z))
f2 = derive_by_array(derive_by_array(f, (x, y, z)), (x, y, z))
f2
print(latex(f2))
f2[0, 0]
f2[0, 0] / (y**3 * z**3)
f2[0, 0] / (1 / (y**3 * z**3))
f2[0, 1] / (1 / (y**3 * z**3))
f2[0, 2] / (1 / (y**3 * z**3))
f2[1, 0] / (1 / (y**3 * z**3))
f2[1, 1] / (1 / (y**3 * z**3))
f2[1, 2] / (1 / (y**3 * z**3))
f2[2, 0] / (1 / (y**3 * z**3))
f2 / (1 / (y**3 * z**3))
f2 / (2 / (y**3 * z**3))
t
t = [y*z, x*z, x*y]
from sympy.physics.quantum import TensorProduct
t1 = Matrix([y*z, x*z, x*y])
TensorProduct(t1, t1)
TensorProduct(t1, t1.T)
f
f2 = derive_by_array(derive_by_array(f, (x, y, z)), (x, y, z))
f2
f2 / (2 / (y**3 * z**3))
f2 / (1 / (y**3 * z**3))
TensorProduct(t1, t1.T)
TensorProduct(2*t1, 2*t1.T)
TensorProduct(t1, t1.T)
f2 / (1 / (y**3 * z**3))
t2 = Matrix([y*z, -x*z, x*y])
TensorProduct(t2, t2.T)
t2 = Matrix([y*z, x*z, -x*y])
TensorProduct(t2, t2.T)
t2 = Matrix([-y*z, x*z, x*y])
TensorProduct(t2, t2.T)
t2 = Matrix([-2*y*z, x*z, x*y])
TensorProduct(t2, t2.T)
f2 / (1 / (y**3 * z**3))
history
