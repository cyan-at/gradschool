#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import sys, os, time
import matplotlib.pyplot as plt
from scipy import stats

from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from scipy import interpolate
from mpl_toolkits import mplot3d
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mat4py import loadmat

from pylab import figure, show
from mpl_toolkits.mplot3d import Axes3D

from scipy.stats import truncnorm
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colors import LightSource
from scipy.stats import gaussian_kde
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt


# In[18]:


def get_u2(test, output_tensor, inputs_tensor, args, batchsize):
    # import ipdb; ipdb.set_trace()

    t0_u = test[:batchsize, inputs.shape[1] + 3 - 1:inputs.shape[1] + 5 - 1]
    tT_u = test[batchsize:2*batchsize, inputs.shape[1] + 3 - 1:inputs.shape[1] + 5 - 1]
    tt_u = test[2*batchsize:, inputs.shape[1] + 3 - 1:inputs.shape[1] + 5 - 1]
    print(
        np.max(t0_u),
        np.max(tT_u),
        np.max(tt_u)
    )

    return t0_u, tT_u, tt_u


# In[78]:


def do_integration2(control_data, d, T_0, T_t, mu_0, sigma_0, args, sde, sde2):
    # dt = (T_t - T_0)/(args.bif)
    # ts = np.arange(T_0, T_t + dt, dt)
    ts = torch.linspace(T_0, T_t, int(T_t * 500), device=cuda0)
    # ts = torch.linspace(T_0, 1, int(1 * 500), device=cuda0)

    # import ipdb; ipdb.set_trace()

    initial_sample = np.random.multivariate_normal(
        np.array(mu_0), np.eye(d)*sigma_0, args.M) # 100 x 3

    print(sigma_0)

    v_scales = [float(x) for x in args.v_scale.split(",")]
    biases = [float(x) for x in args.bias.split(",")]

    ##############################

    all_results = {}

    mus = np.zeros(d*2)
    variances = np.zeros(d*2)

    without_control = np.empty(
        (
            initial_sample.shape[0],
            initial_sample.shape[1],
            len(ts),
        ))

    with_control = np.empty(
        (
            initial_sample.shape[0],
            initial_sample.shape[1],
            len(ts),
        ))

    initial_sample_tensor = torch.tensor(initial_sample,
        dtype=torch.float32, device=cuda0)

    # import ipdb; ipdb.set_trace()

    ts = ts.to(cuda0)

    for i in range(initial_sample_tensor.shape[0]):
        y0 = initial_sample_tensor[i, :]
        y0 = torch.reshape(y0, [1, -1])

        # import ipdb; ipdb.set_trace()

        bm = torchsde.BrownianInterval(
            t0=float(T_0),
            t1=float(T_t),
            size=y0.shape,
            device=cuda0,
        )  # We need space-time Levy area to use the SRK solver

        y_pred = torchsde.sdeint(sde, y0, ts, method='euler', bm=bm, dt=1/(T_t*500)).squeeze()
        # calculate predictions
        without_control[i, :, :] = y_pred.detach().cpu().numpy().T

        y_pred = torchsde.sdeint(sde2, y0, ts, method='euler', bm=bm, dt=1/(T_t*500)).squeeze()
        with_control[i, :, :] = y_pred.detach().cpu().numpy().T

        print(i)
        print(y0)
        print(y_pred[-1, :])

    for d_i in range(d):
        mus[2*d_i] = np.mean(with_control[:, d_i, -1])
        variances[2*d_i] = np.var(with_control[:, d_i, -1])

        mus[2*d_i+1] = np.mean(without_control[:, d_i, -1])
        variances[2*d_i+1] = np.var(without_control[:, d_i, -1])

    ts = ts.detach().cpu().numpy()

    return ts, initial_sample, with_control, without_control,\
        None, mus, variances


# In[84]:


sys.path.insert(0,'./training/iman/')
from train import *

d = 2
batchsize = None

d = 2
N = 15
batchsize = None

mu_0 = [0.35, 0.35]

sigma = 0.1
T_t = 200.0
bcc = np.array([0.41235, 0.37605])

class Container(Container):
    state_bound_min = 0.1
    state_bound_max = 0.6
    bound_u = 0

    bif = 100000
    batchsize2 = "5000"
    batch2_period = 5000
    batchsize = ""
    
    diff_on_cpu = 1
    grid_n = 30
    interp_mode = "nearest"
    
    M = 10
    v_scale = "1.0"
    bias = "1.0"
    
    testdat = './training/iman/test.dat'
    N = 15
    modelpt = './training/iman/tt200_2d_mse-15000.pt'
    
    sigma = sigma
    T_t = T_t
args = Container()

N = args.N


# In[42]:


test = np.loadtxt(args.testdat)

print("test.shape", test.shape)
d = test.shape[1] - 1 - 4 # one for time, 4 for pinn output
print("found %d dimension state space" % (d))
M = N**d


# In[41]:


sys.path.insert(0, './sde/T_t200_2D')
from trained_sde_model import SDE

sde = SDE()

files = glob.glob(
    sde_path + "/*.pt", 
    recursive = False)
assert(len(files) == 1)
print("using model: ", files[0])
sde.load_state_dict(torch.load(files[0]))

if torch.cuda.is_available():
    print("Using GPU.")
    sde = sde.to(cuda0)
# set model to evaluation mode
sde.eval()
    
sde.r = torch.tensor(np.array([0.0]*2), dtype=torch.float32)
sde.r = sde.r.reshape([-1, 2])test = np.loadtxt(args.testdat)

print("test.shape", test.shape)
d = test.shape[1] - 1 - 4 # one for time, 4 for pinn output
print("found %d dimension state space" % (d))
M = N**d


# In[ ]:


data = np.genfromtxt('./training/iman/test.dat')
points=data[:,0:2]
inputs=data[:,4].reshape(-1,1)
psi=data[:,2].reshape(-1,1)
print(inputs.shape)

fig, ax = plt.subplots(1,1)
ax.plot(points[:,1:2], points[:,0:1], '.')
ax.set_xlabel(r'$t$ [s]', fontsize=16)
ax.set_ylabel(r"$\langle C_{6} \rangle$ ", fontsize=16,rotation='horizontal')
ax.yaxis.set_label_coords(-0.07, 0.5)
ax.xaxis.set_label_coords(0.5, -.1)
plt.tight_layout()
fig.set_size_inches(8,4)
plt.show()


# In[54]:


model, meshes = get_model(
    d,
    N,
    batchsize,
    0,
    "tanh",

    mu_0,
    sigma,

    bcc,
    sigma,

    T_t,
    args,
    sde.network_f,
    sde.network_g,
)
model.restore(args.modelpt)


# In[55]:


inputs = np.float32(test[:, :d+1])

print("inputs.shape", inputs.shape)

inputs = np.vstack((
    model.data.bc_points(),
    model.data.train_x_all
))

print("inputs.shape", inputs.shape)


# In[62]:


test, T_t,\
rho0, rhoT,\
bc_grids, domain_grids, grid_n_meshes,\
control_data = make_control_data(
    model, inputs, N, d, meshes, args, get_u2)


# In[ ]:


class SDE2(SDE):
    def __init__(self, control_data):
        super(SDE2, self).__init__()

        self.control_data = control_data

    def query_u(self, t, y):
        if torch.abs(t - 0) < 1e-8:
            t_key = 't0'
        elif torch.abs(t - 1.0) < 1e-8:
            t_key = 'tT'
        else:
            t_key = 'tt'
        t_control_data = self.control_data[t_key]

        query = y[0].detach().cpu().numpy()

        if t_control_data['grid'].shape[1] == 2 + 1:
            t2 = t.detach().cpu().numpy()
            query = np.append(query, t2)

        if 'grid_tree' in t_control_data:
            _, closest_grid_idx = t_control_data['grid_tree'].query(
                np.expand_dims(query, axis=0),
                k=1)
        else:
            closest_grid_idx = np.linalg.norm(
                query - t_control_data['grid'], ord=1, axis=1).argmin()

        u1 = t_control_data['0'][closest_grid_idx]
        u2 = t_control_data['1'][closest_grid_idx]

        u_tensor = torch.tensor(np.array([u1, u2]), dtype=torch.float32)
        u_tensor = u_tensor.reshape([-1, 2])

        return u_tensor

    # drift
    def f(self, t, y): # ~D1
        u_tensor = self.query_u(t, y)

        t = torch.reshape(t, [-1, 1])
        # need to cat the ramp rates on the input vector for y
        input_vec = torch.cat([y, u_tensor, t], axis=1)

        # print(self.network_f.forward(input_vec).shape)

        return self.network_f.forward(input_vec)

    # diffusion
    def g(self, t, y): # ~D2
        """
        Output of g: should be a single tensor of size
        (batch_size, d)
        """
        u_tensor = self.query_u(t, y)

        t = torch.reshape(t, [-1, 1])

        # need to cat the ramp rates on the input vector for g
        input_vec = torch.cat([y, u_tensor, t], axis=1)

        # print("g", self.network_g.forward(input_vec))

        return self.network_g.forward(input_vec)

sde2 = SDE2(control_data)

sde2.load_state_dict(torch.load(files[0]))
if torch.cuda.is_available():
    print("Using GPU.")
    sde2 = sde2.to(cuda0)
# set model to evaluation mode
sde2.eval()


# In[85]:


ts, initial_sample, with_control, without_control,\
    all_results, mus, variances = do_integration2(
        control_data, d, T_0, T_t, mu_0, args.sigma,
        args, sde, sde2)


# print(ts)
