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

from common import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--modelpt',
        type=str, default='')
    parser.add_argument('--testdat',
        type=str, required=True)

    args = parser.parse_args()

    test = np.loadtxt(args.testdat)

    if os.path.exists(args.modelpt):
        print("loading model")
        # instead of using test output, use model and 
        # generate / predict a new output
        inputs = test[:, :2]

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
        model = dde.Model(data, net)
        loss_func=["MSE","MSE","MSE", "MSE","MSE"]
        model.compile("adam", lr=1e-3,loss=loss_func)

        model.restore(args.modelpt)
        output = model.predict(inputs)
        test = np.hstack((inputs, output))
    else:
        print("no model, using test dat alone")

    ########################################################

    fig = plt.figure(1)
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

    plt.show()
