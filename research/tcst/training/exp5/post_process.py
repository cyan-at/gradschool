#!/usr/bin/env python
# coding: utf-8

'''
USAGE:
./plot_bc.py --testdat ./wass_2d_model-99875-99875.dat --modelpt ./wass_2d_model-99875.pt
'''

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
from mpl_toolkits import mplot3d
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm

class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)
    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)
        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)
        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)
def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''
    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)
setattr(Axes3D, 'arrow3D', _arrow3D)
def show_assignments_3d(state_a, state_b, P): 
    np_p = P.cpu().detach().numpy()
    norm_P = np_p/np_p.max()
    for i in range(state_a.shape[0]):
        for j in range(state_b.shape[0]):
            ax.arrow3D(
                state_a[i, 0], state_a[i, 1], z1,
                state_b[j, 0] - state_a[i, 0],
                state_b[j, 1] - state_a[i, 1],
                z2 - z1,
                alpha=norm_P[i,j].item(),
                mutation_scale=20,
                arrowstyle="-|>",
                linestyle='dashed')

from os.path import dirname, join as pjoin

from scipy import stats
import scipy.io
from scipy.stats import truncnorm, norm
from scipy.optimize import linprog
from scipy import sparse
from scipy.stats import multivariate_normal
from scipy.interpolate import griddata as gd

if dde.backend.backend_name == "pytorch":
    exp = dde.backend.torch.exp
else:
    from deepxde.backend import tf

    exp = tf.exp
    
import cvxpy as cp
import numpy as np

import argparse

from common import *

sys.path.insert(0, "..")

call_dir = os.getcwd()
sys.path.insert(0,call_dir)
print("expect train.py in %s" % (call_dir))
from train import *

###patch start###
from mpl_toolkits.mplot3d.axis3d import Axis
if not hasattr(Axis, "_get_coord_info_old"):
    def _get_coord_info_new(self, renderer):
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs
    Axis._get_coord_info_old = Axis._get_coord_info  
    Axis._get_coord_info = _get_coord_info_new
###patch end###

def do_integration2(control_data, d, T_0, T_t, mu_0, sigma_0, args, sde, sde2):
    # dt = (T_t - T_0)/(args.bif)
    # ts = np.arange(T_0, T_t + dt, dt)
    ts = torch.linspace(T_0, T_t, int(T_t * 500), device=cuda0)
    # ts = torch.linspace(T_0, 1, int(1 * 500), device=cuda0)

    import ipdb; ipdb.set_trace()

    initial_sample = np.random.multivariate_normal(
        np.array(mu_0), np.eye(d)*sigma_0, args.M) # 100 x 3

    print(sigma_0)

    v_scales = [float(x) for x in args.v_scale.split(",")]
    biases = [float(x) for x in args.bias.split(",")]

    ##############################

    all_results = {}

    mus = np.zeros(d*2)
    variances = np.zeros(d*2)

    pde_key = d
    if len(args.pde_key) > 0:
        pde_key = int(args.pde_key)
    print("pde_key", pde_key)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--modelpt',
        type=str, default='')
    parser.add_argument('--testdat',
        type=str, required=True)
    parser.add_argument('--plot',
        type=int, default=0)

    parser.add_argument('--diff_on_cpu',
        type=int, default=1)
    parser.add_argument('--fullstate',
        type=int, default=1)
    parser.add_argument('--interp_mode',
        type=str, default="nearest")
    parser.add_argument('--grid_n',
        type=int, default=30)
    parser.add_argument('--control_strategy',
        type=int,
        default=0)

    parser.add_argument('--optimizer',
        type=str, default="adam", help='')

    parser.add_argument('--mu_0',
        type=str, required=True, help='')
    parser.add_argument('--mu_T',
        type=str, default="", help='')
    parser.add_argument('--T_t',
        type=float, default=5.0, help='')
    parser.add_argument('--a', type=str, default="-1,1,2", help='')
    parser.add_argument('--N', type=int, default=22, help='')
    parser.add_argument('--state_bound_min', type=float, default=0.1, help='')
    parser.add_argument('--state_bound_max', type=float, default=0.5, help='')

    parser.add_argument('--train_distribution', type=str, default="Hammersley", help='')
    parser.add_argument('--timemode', type=int, default=0, help='')
    # timemode  0 = linspace, 1 = even time samples
    parser.add_argument('--bif', type=int, default=1000, help='')
    parser.add_argument('--ni', type=int, default=-1, help='')
    parser.add_argument('--loss_func', type=str, default="wass3", help='')
    parser.add_argument('--pde_key', type=str, default="", help='')
    parser.add_argument('--batchsize',
        type=str,
        default="")
    parser.add_argument('--batchsize2',
        type=str,
        default="")
    parser.add_argument('--batch2_period',
        type=int,
        default=5)

    parser.add_argument('--do_integration',
        type=int,
        default=1)

    parser.add_argument('--integrate_N',
        type=int,
        default=2000,
        required=False)
    parser.add_argument('--M',
        type=int,
        default=100,
        required=False)
    parser.add_argument('--workers',
        type=int,
        default=4)
    parser.add_argument('--noise',
        action='store_true')
    parser.add_argument('--v_scale',
        type=str,
        default="1.0")
    parser.add_argument('--bias',
        type=str,
        default="0.0")

    parser.add_argument('--headless',
        type=int,
        default=0)

    parser.add_argument('--sdept',
        type=str,
        default="../../../sde/4fold_3_2_layer_model.pt",
        help='')
    parser.add_argument('--sigma',
        type=float,
        default=0.001)
    parser.add_argument('--crystal',
        type=str,
        required=True)

    parser.add_argument('--plot_samples',
        type=int,
        default=4)

    args, _ = parser.parse_known_args()

    ################################################

    N = args.N

    test = np.loadtxt(args.testdat)

    ################################################

    print("test.shape", test.shape)
    d = test.shape[1] - 1 - 4 # one for time, 4 for pinn output
    print("found %d dimension state space" % (d))
    M = N**d
    batchsize = M

    if len(args.batchsize) > 0:
        batchsize = int(args.batchsize)

    Ns = tuple([N]*d)

    ################################################

    model = None
    meshes = None
    if os.path.exists(args.modelpt):
        print("loading model")

        ni = initial_samples
        if args.ni >= 0:
            ni = args.ni

        sde = SDE_3()
        # state path to model information file
        # load model parameters

        files = glob.glob(
            sde_path + "/27hyperparameter_num_7_model.pt", 
            recursive = False)
        assert(len(files) == 1)
        print("using model: ", files[0])
        sde.load_state_dict(torch.load(files[0]))

        if torch.cuda.is_available():
            print("Using GPU.")
            sde = sde.to(cuda0)
        # set model to evaluation mode
        sde.eval()

        mu_0 = [float(x) for x in args.mu_0.strip().split(",")]

        target = tcst_map[args.crystal]

        model, meshes = get_model(
            d,
            N,
            batchsize,
            0,
            "tanh",
            mu_0,
            args.sigma,
            target,
            args.sigma,
            T_t,
            args,
            sde.network_f,
            sde.network_g,
            args.optimizer,
            train_distribution=args.train_distribution,
            timemode=args.timemode,
            ni=ni
            )
        model.restore(args.modelpt)
    else:
        print("no model")
        sys.exit(0)

    ########################################

    inputs = np.float32(test[:, :d+1])

    print("inputs.shape", inputs.shape)

    test, T_t,\
    rho0, rhoT,\
    bc_grids, domain_grids, grid_n_meshes,\
    control_data = make_control_data(
        model, inputs, N, d, meshes, args, get_tcst)

    # import ipdb; ipdb.set_trace()

    fig = plt.figure()

    axs = []
    ax_count = 1

    ax1 = fig.add_subplot(1, ax_count, 1, projection='3d')
    axs.append(ax1)
    # ax2 = fig.add_subplot(1, ax_count, 2, projection='3d')
    # ax3 = fig.add_subplot(1, ax_count, 3, projection='3d')
    # axs = [ax1, ax2, ax3]

    ax_i = 0

    z1 = T_0
    z2 = T_t
    p = 1.0

    ########################################################

    rho0_contour = axs[ax_i].contourf(
        meshes[0],
        meshes[1],
        rho0.reshape(*Ns),
        50, zdir='z',
        cmap=cm.jet,
        offset=z1,
        alpha=0.4
    )

    rhoT_contour = axs[ax_i].contourf(
        meshes[0],
        meshes[1],
        rhoT.reshape(*Ns),
        50, zdir='z',
        cmap=cm.jet,
        offset=z2,
        alpha=0.4,
    )

    axs[ax_i].set_xlim(
        args.state_bound_min, args.state_bound_max)
    axs[ax_i].set_zlim(T_0 - 0.1, T_t + 0.1)

    axs[ax_i].set_xlabel('x')
    axs[ax_i].set_ylabel('y')
    axs[ax_i].set_zlabel('t')
    axs[ax_i].set_title('rho_opt')

    fig.colorbar(rho0_contour, shrink=0.25)
    fig.colorbar(rhoT_contour, shrink=0.25)


    manager = plt.get_current_fig_manager()

    c = Counter()
    fig.canvas.mpl_connect('key_press_event', lambda e: c.on_press_saveplot(e,
            '%s_Tt=%.3f_rho_opt_bc_batch=%d_%d_%s_%d' % (
                args.modelpt.replace(".pt", ""),
                T_t,
                batchsize,
                args.diff_on_cpu,
                args.interp_mode,
                args.grid_n
            )
        )
    )

    plt.show()