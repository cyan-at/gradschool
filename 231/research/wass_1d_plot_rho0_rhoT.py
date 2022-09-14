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
from scipy.linalg import solve_discrete_are
from scipy.linalg import sqrtm

from cvxpylayers.torch import CvxpyLayer

import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--modelpt', type=str, required=True)
  parser.add_argument('--testdat', type=str, required=True)

  args = parser.parse_args()

  nb_name = "none"

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

  id_prefix = "empty"

  print(id_prefix)
  print(time.time())

  def pdf1d(x, mu, sigma):
      a, b = (state_min - mu) / sigma, (state_max - mu) / sigma
      rho_x=truncnorm.pdf(x, a, b, loc = mu, scale = sigma)

      # do NOT use gaussian norm, because it is only area=1
      # from -inf, inf, will not be for finite state/grid
      # rho_x = norm.pdf(x, mu, sigma)
      return rho_x

  def boundary(_, on_initial):
      return on_initial

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
  print(np.trapz(rho_0, axis=0)[0])
  print(np.sum(rho_0))
  rho_0_BC = dde.icbc.PointSetBC(time_0, rho_0, component=1)

  time_t=np.hstack((x_T,T_t*np.ones((len(x_T), 1))))
  rho_T=pdf1d(x_T, mu_T, sigma_T).reshape(len(x_T),1)
  rho_T = np.where(rho_T < 0, 0, rho_T)
  rho_T = rho_T / np.sum(np.abs(rho_T))
  print(np.trapz(rho_T, axis=0)[0])
  print(np.sum(rho_T))
  rho_T_BC = dde.icbc.PointSetBC(time_t, rho_T, component=1)

  data = dde.data.TimePDE(
      geomtime,
      pde,
      [rho_0_BC,rho_T_BC],
      num_domain=5000,
      num_initial=500)
  net = dde.nn.FNN([2] + [70] *3  + [3], "tanh", "Glorot normal")
  model = dde.Model(data, net)
  loss_func=["MSE","MSE","MSE", "MSE","MSE"]
  model.compile("adam", lr=1e-3,loss=loss_func)

  fig = plt.figure(1)
  ax1 = plt.subplot(111, frameon=False)

  X_IDX = 0
  T_IDX = 1
  EQ_IDX = 3

  test = np.loadtxt(args.testdat)

  #################################

  test_ti = test[:N, :] # first BC test data
  ind = np.lexsort((test_ti[:,X_IDX],test_ti[:,T_IDX]))
  test_ti = test_ti[ind]

  # post process
  test_ti[:,EQ_IDX] = np.where(test_ti[:,EQ_IDX] < 0, 0, test_ti[:,EQ_IDX])
  print("test_ti[:,EQ_IDX]", test_ti[:,EQ_IDX])
  s1 = np.trapz(test_ti[:,EQ_IDX], axis=0, x=test_ti[:,X_IDX])
  print(s1)
  test_ti[:, EQ_IDX] /= s1 # to pdf

  s1 = np.trapz(test_ti[:,EQ_IDX], axis=0, x=test_ti[:,X_IDX])
  s2 = np.sum(test_ti[:,EQ_IDX])

  ax1.plot(
      test_ti[:, 0],
      test_ti[:, EQ_IDX],
      linewidth=1,
      c='b',
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

  #################################

  test_ti = test[N:2*N, :] # first BC test data
  ind = np.lexsort((test_ti[:,X_IDX],test_ti[:,T_IDX]))
  test_ti = test_ti[ind]

  # instead of using test output, use model and 
  # generate / predict a new output
  inputs = test_ti[:, :2]
  model.restore(args.modelpt)
  output = model.predict(inputs)
  test_ti = np.hstack((inputs, output))
  ind = np.lexsort((test_ti[:,X_IDX],test_ti[:,T_IDX]))
  test_ti = test_ti[ind]

  # post process
  test_ti[:,EQ_IDX] = np.where(test_ti[:,EQ_IDX] < 0, 0, test_ti[:,EQ_IDX])
  # test_ti[:,EQ_IDX] = -1.0 * test_ti[:,EQ_IDX]
  print("test_ti[:,EQ_IDX]", test_ti[:,EQ_IDX])
  s3 = np.trapz(test_ti[:,EQ_IDX], axis=0, x=test_ti[:,X_IDX])
  print(s3)
  test_ti[:, EQ_IDX] /= s3 # to pdf

  s3 = np.trapz(test_ti[:,EQ_IDX], axis=0, x=test_ti[:,X_IDX])
  s4 = np.sum(test_ti[:,EQ_IDX])

  # ax1.set_aspect('equal')
  ax1.grid()
  ax1.set_title(
      'rho0: trapz=%.3f, sum=%.3f\nrhoT: trapz=%.3f, sum=%.3f' % (
          s1, s2, s3, s4))

  ax1.plot(
      test_ti[:, 0],
      test_ti[:, EQ_IDX],
      linewidth=1,
      c='c',
      label='test rho_T')

  test_rhoT=pdf1d(test_ti[:, 0], 4.0, 1.0).reshape(test_ti.shape[0],1)
  test_rhoT /= np.trapz(test_rhoT, axis=0, x=test_ti[:,X_IDX])
  # test_rhoT = test_rhoT / np.sum(np.abs(test_rhoT))
  ax1.plot(
      test_ti[:, 0],
      test_rhoT,
      c='g',
      linewidth=1,
      label='rho_T')

  ########################################

  ax1.plot(
      test_ti[:, 0],
      test_ti[:, 2],
      linewidth=1,
      c='k',
      label='y3')

  ax1.legend(loc='lower right')
  plt.show()


