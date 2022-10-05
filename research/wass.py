#!/usr/bin/env python3
# coding: utf-8

# In[1]:


# 0 define backend

# get_ipython().run_line_magic('env', 'DDE_BACKEND=tensorflow.compat.v1')

# get_ipython().run_line_magic('env', 'XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/home/cyan3/miniforge/envs/tf')
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
from scipy.stats import truncnorm
# import tensorflow as tf
import sys

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    print(device)
    tf.config.experimental.set_memory_growth(device, True)

# sys.exit(0)

# import ipdb; ipdb.set_trace();


# In[3]:


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


# In[4]:


if dde.backend.backend_name == "pytorch":
    exp = dde.backend.torch.exp
else:
    from deepxde.backend import tf

    exp = tf.exp


# In[5]:


S3 = dde.Variable(1.0)


# In[6]:


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


# In[7]:


def boundary(_, on_initial):
    return on_initial


def pdf1d_T(x):
    mu = 5.
    sigma = .1
    a, b = (0. - mu) / sigma, (6. - mu) / sigma
    rho_T=truncnorm.pdf(x, a, b, loc = mu, scale = sigma)
    return rho_T


def pdf1d_0(x):
    sigma = 1
    mu=2
    a, b = (0. - mu) / sigma, (6. - mu) / sigma
    rho_0=truncnorm.pdf(x, a, b, loc = mu, scale = sigma)
    return rho_0


def modify_output(X, Y):
    y1, y2, y3  = Y[:, 0:1], Y[:, 1:2], Y[:, 2:3]
    u_new = tf.clip_by_value(y3, clip_value_min = .5, clip_value_max = 4)
    return tf.concat((y1, y2, u_new), axis=1)


# In[8]:

N = 1000
x_T = np.transpose(np.linspace(0., 6., N))
T_t=200. #Terminal time
T_0=0. #initial time
x_T=x_T.reshape(len(x_T),1)
time=T_t*np.ones(( 1,len(x_T))).reshape(len(x_T),1)
time_0=T_0*np.ones(( 1,len(x_T))).reshape(len(x_T),1)
rho_T=pdf1d_T(x_T).reshape(len(x_T),1)
rho_0=pdf1d_0(x_T).reshape(len(x_T),1)

# import ipdb; ipdb.set_trace();

terminal_time=np.hstack((x_T,time))
Initial_time=np.hstack((x_T,time_0))
rho_T_BC = dde.icbc.PointSetBC(terminal_time, rho_T, component=1)
rho_0_BC = dde.icbc.PointSetBC(Initial_time, rho_0, component=1)

a, b, c, d, f= 10., 2.1, 0.75, .0045, 0.0005

K, T=1.38066*10**-23, 293.

geom = dde.geometry.Interval(0., 6.)
timedomain = dde.geometry.TimeDomain(0., T_t)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)
# rho_0_BC = dde.icbc.IC(geomtime, lambda x: pdf1d_0(x[:,0:1]) , boundary, component=1)


# In[9]:


data = dde.data.TimePDE(geomtime, pde,  [rho_0_BC,rho_T_BC], num_domain=5000, num_initial=500)
net = dde.nn.FNN([2] + [70] *3  + [3], "tanh", "Glorot normal")
# net.apply_output_transform(modify_output)
# net.apply_output_transform(modify_output)
model = dde.Model(data, net)


# In[ ]:


# from scipy.optimize import linprog
# from scipy import sparse
# import numpy as np

# nSample=100
# x_grid = np.transpose(np.linspace(0., 6., nSample))
# y_grid = np.transpose(np.linspace(0., 6., nSample))


# [X,Y] = np.meshgrid(x_grid,y_grid)
# C = (X - Y)**2
# cvector = C.flatten('F')


# A = np.concatenate((np.kron(np.ones((1,nSample)), sparse.eye(nSample).toarray()), np.kron(sparse.eye(nSample).toarray(),np.ones((1,nSample)))), axis=0)
# bvector = np.concatenate((rho_T, rho_T), axis=0)
# res = linprog(cvector, A_eq=A, b_eq=bvector, options={"disp": True})

# print(res.fun)


# In[ ]:





# In[10]:


loss_func=["MSE","MSE","MSE","wass","wass"]
model.compile("adam", lr=1e-3,loss=loss_func)
losshistory, train_state = model.train(epochs=15000)


# In[ ]:


dde.saveplot(losshistory, train_state, issave=True, isplot=True)


# In[ ]:





# In[ ]:




