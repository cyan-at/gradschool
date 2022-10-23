# 1 import stuff

import os, json

# "tensorflow.compat.v1", "tensorflow", "pytorch"
def set_default_backend(backend_name):
    default_dir = os.path.join(os.path.expanduser("~"), ".deepxde")
    if not os.path.exists(default_dir):
        os.makedirs(default_dir)
    config_path = os.path.join(default_dir, "config.json")
    with open(config_path, "w") as config_file:
        json.dump({"backend": backend_name.lower()}, config_file)
    print(
        'Setting the default backend to "{}". You can change it in the '
        "~/.deepxde/config.json file or export the DDE_BACKEND environment variable. "
        "Valid options are: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle (all lowercase)".format(
            backend_name
        )
    )
    
import tensorflow as tf
set_default_backend("tensorflow")

# import tensorflow.compat.v1 as tf
# backend = "tensorflow_v1"
# set_default_backend("tensorflow.compat.v1")
# tf.disable_eager_execution()
# tf.enable_eager_execution()
# tf.compat.v1.disable_eager_execution()

tf.config.run_functions_eagerly(True)

# eager execution for tensorflow backend works, but is VERY slow
# disabling it leads to EagerFunction error

# import torch
# backend = "pytorch"
# set_default_backend("pytorch")

import deepxde as dde

dde.model.backend_name = "tensorflow"

dde.config.enable_xla_jit(True)
# this is key for speed in v2?
# happens by default for tensorflow v1?

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

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
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

import scipy.integrate as integrate

import time

import shutil

from scipy.interpolate import griddata as gd

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    print(device)
    tf.config.experimental.set_memory_growth(device, True)

# os.environ['XLA_FLAGS'] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
print(dde.model.backend_name)

##############################################################################################

# 3 define training parameters

state_min = -2.5
state_max = 2.5
# shrinking this helps a lot

######################################

N = 2000

T_t = 5.0

epsilon=.001

# j1, j2, j3=3,2,1
j1, j2, j3 =1,1,2 # axis-symmetric case

######################################

# q=1 # state cost

q_statepenalty_gain = 0 # 0.5

######################################

mu_0 = 2.0
sigma_0 = 1.0

mu_T = 0.0
sigma_T = 1.0

######################################

samples_between_initial_and_final = 12000 # 10^4 order, 20k = out of memory
initial_and_final_samples = 1000 # some 10^3 order

######################################

num_epochs = 20000

id_prefix = "run_%s_%.3f_%.3f__%.3f__%d__%.3f__%d_%d_%d__%.3f__%.3f_%.3f__%.3f_%.3f__%d_%d__%d" % (
    dde.model.backend_name,
    state_min,
    state_max,
    T_t,
    N,
    epsilon,
    j1, j2, j3,
    q_statepenalty_gain,
    mu_0, sigma_0,
    mu_T, sigma_T,
    samples_between_initial_and_final, initial_and_final_samples,
    num_epochs
)

print(T_t)
print(id_prefix)

# id_prefix is a id for this training


##############################################################################################

# 4 pde

def pde(x, y):
    """Euler system.
    dy1_t = g(x)-1/2||Dy1_x||^2-<Dy1_x,f>-epsilon*Dy1_xx
    dy2_t = -D.(y2*(f)+Dy1_x)+epsilon*Dy2_xx
    All collocation-based residuals are defined here
    """
    y1, y2 = y[:, 0:1], y[:, 1:]
    dy1_x = tf.gradients(y1, x)[0]
    dy1_x, dy1_y, dy1_z, dy1_t = dy1_x[:,0:1], dy1_x[:,1:2], dy1_x[:,2:3], dy1_x[:,3:]
    
    dy1_xx = tf.gradients(dy1_x, x)[0][:, 0:1]
    dy1_yy = tf.gradients(dy1_y, x)[0][:, 1:2]
    dy1_zz = tf.gradients(dy1_z, x)[0][:, 2:3] 
    
    dy2_x = tf.gradients(y2, x)[0]
    dy2_x, dy2_y, dy2_z, dy2_t = dy2_x[:,0:1], dy2_x[:,1:2], dy2_x[:,2:3], dy2_x[:,3:]    
    
    dy2_xx = tf.gradients(dy2_x, x)[0][:, 0:1]
    dy2_yy = tf.gradients(dy2_y, x)[0][:, 1:2]
    dy2_zz = tf.gradients(dy2_z, x)[0][:, 2:3]     

    f1=x[:, 1:2]*x[:, 2:3]*(j2-j3)/j1
    f2=x[:, 0:1]*x[:, 2:3]*(j3-j1)/j2
    f3=x[:, 0:1]*x[:, 1:2]*(j1-j2)/j3
    
    d_f1dy1_y2_x=tf.gradients((f1+dy1_x)*y2,x)[0][:, 0:1]
    d_f2dy1_y2_y=tf.gradients((f2+dy1_y)*y2,x)[0][:, 1:2]
    d_f3dy1_y2_z=tf.gradients((f3+dy1_x)*y2,x)[0][:, 2:3]

    # stay close to origin while searching, penalizes large state distance solutions
    q = q_statepenalty_gain*(x[:, 0:1] * x[:, 0:1] + x[:, 1:2] * x[:, 1:2] + x[:, 2:3] * x[:, 2:3])
    
    # also try
    # q = 0 # minimum effort control
    
    # TODO: verify this expression
    return [
        -dy1_t+q-.5*(dy1_x*dy1_x+dy1_y*dy1_y+dy1_z*dy1_z)-dy1_x*f1-dy1_y*f2-dy1_z*f3-epsilon*(dy1_xx+dy1_yy+dy1_zz),
        -dy2_t-(d_f1dy1_y2_x+d_f2dy1_y2_y+d_f3dy1_y2_z)+epsilon*(dy2_xx+dy2_yy+dy2_zz),
    ]

##############################################################################################

# 5 more pde, boundary

def boundary(_, on_initial):
    return on_initial

def pdf1d_0(x,y,z):
    '''
    a, b = (state_min - mu) / sigma, (state_max - mu) / sigma # must match sampling
    rho_x=truncnorm.pdf(x, a, b, loc = mu, scale = sigma)
    rho_y=truncnorm.pdf(y, a, b, loc = mu, scale = sigma)
    rho_z=truncnorm.pdf(z, a, b, loc = mu, scale = sigma) # replace with np.normal? gaussian
    '''

    rho_x = norm.pdf(x, mu_0, sigma_0)
    rho_y = norm.pdf(y, mu_0, sigma_0)
    rho_z = norm.pdf(z, mu_0, sigma_0)
    
#     rho_x = tf.random.normal(
#         [1],
#         mean=x,
#         stddev=mu_0,
#         dtype=tf.dtypes.float16,
#         seed=None,
#         name=None
#     )
#     rho_y = tf.random.normal(
#         [1],
#         mean=y,
#         stddev=mu_0,
#         dtype=tf.dtypes.float16,
#         seed=None,
#         name=None
#     )
#     rho_z = tf.random.normal(
#         shape,
#         mean=z,
#         stddev=mu_0,
#         dtype=tf.dtypes.float16,
#         seed=None,
#         name=None
#     )

    return rho_x*rho_y*rho_z

def pdf1d_T(x,y,z):
    '''
    a, b = (state_min - mu) / sigma, (state_max - mu) / sigma
    rho_x=truncnorm.pdf(x, a, b, loc = mu, scale = sigma)
    rho_y=truncnorm.pdf(y, a, b, loc = mu, scale = sigma)
    rho_z=truncnorm.pdf(z, a, b, loc = mu, scale = sigma)
    '''

    rho_x = norm.pdf(x, mu_T, sigma_T)
    rho_y = norm.pdf(y, mu_T, sigma_T)
    rho_z = norm.pdf(z, mu_T, sigma_T)
    
    return rho_x*rho_y*rho_z

##############################################################################################

# 6 define data and net

x_T = np.transpose(np.linspace(state_min, state_max, N))
y_T = np.transpose(np.linspace(state_min, state_max, N))
z_T = np.transpose(np.linspace(state_min, state_max, N))

x_T=x_T.reshape(len(x_T),1)
y_T=y_T.reshape(len(y_T),1)
z_T=z_T.reshape(len(z_T),1)

time=T_t*np.ones((len(x_T), 1))

terminal_time=np.hstack((x_T,y_T,z_T,time))

rho_T=pdf1d_T(x_T,y_T,z_T).reshape(len(x_T),1)

rho_T_BC = dde.icbc.PointSetBC(terminal_time, rho_T, component=1)

geom=dde.geometry.geometry_3d.Cuboid(
    [state_min, state_min, state_min],
    [state_max, state_max, state_max])
timedomain = dde.geometry.TimeDomain(0., T_t)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

rho_0_BC = dde.icbc.IC(
    geomtime,
    lambda x: pdf1d_0(x[:,0:1],x[:,1:2],x[:,2:3]),
    boundary,
    component=1)

data = dde.data.TimePDE(
    geomtime,
    pde, 
    [rho_0_BC,rho_T_BC],
    num_domain=samples_between_initial_and_final,
    num_initial=initial_and_final_samples)

net = dde.nn.FNN([4] + [70] *3  + [2], "tanh", "Glorot normal")

##############################################################################################

# 7 define model

model = dde.Model(data, net)

print(model)

##############################################################################################

# 9 stop as soon as loss is low enough

ck_path = "%s/%s/model" % (os.path.abspath("./"), id_prefix)

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
# no patience otherwise keeps going to improve but bounces around

# modelcheckpt_cb = dde.callbacks.ModelCheckpoint(ck_path, verbose=True, save_better_only=True, period=1000)

# model.compile("adam", lr=1e-3,external_trainable_variables=[])
# losshistory, train_state = model.train(epochs=num_epochs, callbacks=[])

##############################################################################################

_ = input("hit enter")

model.compile(optimizer="adam", lr=1e-3)

##############################################################################################

_ = input("hit enter")

# 11 compile and train model

losshistory, train_state = model.train(
    display_every=1,
    iterations=num_epochs,
    callbacks=[earlystop_cb])

# variable = dde.callbacks.VariableValue([S], period=600, filename="variables.dat")
# losshistory, train_state = model.train(epochs=20000, callbacks=[variable])