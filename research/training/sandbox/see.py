#!/usr/bin/env python
# coding: utf-8

'''
USAGE:
./plot_bc.py --testdat ./wass_3d_model-60970-60970.dat --modelpt ./wass_3d_model-60970.pt --interp_mode nearest --grid_n 15
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

class Util:
    # constants
    NUM0S = 7
    VID_PREFIX = "vid"
    VID_SUFFIX = "avi"
    IMG_PREFIX = "img"
    IMG_SUFFIX = "jpg"

    # methods
    @staticmethod
    def make_name(path, prefix, n, suffix, extension, numZeros=7):
        """ makes a filename in a certain formatting
        i.e., make_name('./', 'test', 52, 'experiment1', 'png')
        returns ./test_0000052_experiment1.png

        Parameters
        ----------
        path : str
            the folder/path/dir to search in
        prefix : str
            the prefix string
        n: int
            the number between prefix and suffix
        suffix : str
            the suffix string
        extension : str
            the extension string, assumed to be valid
        numZeros : int
            the number of digits in the number between prefix and suffix

        Returns
        -------
        name : str
            the filename except the extension
        fname : str
            the entire filename including extension
        """
        tokens = []
        if (prefix != ''):
            tokens.append(prefix)
        tokens.append(str(n).zfill(numZeros))
        if (suffix != ''):
            tokens.append(suffix)
        name = path + "/" + '_'.join(tokens)
        fname = ".".join([name, extension])
        return name, fname

    @staticmethod
    def get_next_valid_name_increment(path, prefix, n, suffix, extension, numZeros=7):
        """ get the next 'valid' name in a path given a certain formatting
        i.e., make_name('./', 'test', 52, 'experiment1', 'png')
        if ./ contains ./test_0000052_experiment1.png
        will return ./test_0000053_experiment1.png, 53

        Notes
        -----
        'valid' in this sense means the file doesn't already exist
        function does not allow overwriting!

        Parameters
        ----------
        path : str
            the folder/path/dir to search in
        prefix : str
            the prefix string
        n: int
            the number between prefix and suffix
        suffix : str
            the suffix string
        extension : str
            the extension string, assumed to be valid
        numZeros : int
            the number of digits in the number between prefix and suffix

        Returns
        -------
        fname : str
            the entire filename including extension
        n : int
            the count at which the valid file was found
        """

        if (not os.path.isdir(path)):
            raise ValueError('get_next_valid_name:no_such_path', path)

        name, fname = Util.make_name(path, prefix, n, suffix, extension, numZeros)
        while (os.path.isfile(fname)):
            n = n + 1
            name, fname = Util.make_name(path, prefix, n, suffix, extension, numZeros)
        return fname, n

class Counter(object):
    def __init__(self):
        self.count = 0

    def on_press_saveplot(self, event, png_name):
        print('press', event.key)
        sys.stdout.flush()
        if event.key == 'x':
            # visible = xl.get_visible()
            # xl.set_visible(not visible)
            # fig.canvas.draw()

            dir_name = os.path.dirname(png_name)
            print(dir_name)
            bname = os.path.basename(png_name)
            fname, _ = Util.get_next_valid_name_increment(
                dir_name, bname, 0, '', 'png')

            # fname = png_name.replace(".png", "_%d.png" % (
            #     self.count))

            plt.savefig(
                fname,
                dpi=500,
                bbox_inches='tight')
            print("saved figure", fname)

            self.count += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dat',
        type=str, required=True)

    parser.add_argument('--indices',
        type=str, default="")

    parser.add_argument('--n',
        type=int, default=6)

    args = parser.parse_args()

    if not os.path.exists(args.dat):
        print("bad dat")
        sys.exit(0)

    x = None
    try:
        x = np.load(args.dat, allow_pickle=True)
    except Exception as e:
        print(str(e))

    if x is None:
        try:
            x = np.loadtxt(args.dat)
            print("loaded as txt")
        except Exception as e:
            print("unable to load", str(e))
            sys.exit(1)
    else:
        print("loaded numpy data")

    try:
        x = x.item()
    except Exception as e:
        print(e)

    ########################################################

    all_data_to_plot = []
    k_tokens = []

    if type(x) == dict:
        while type(x) == dict:
            t = [type(y) for y in x.keys()]
            print(t)

            s = [str(y) for y in x.keys()]

            s_to_t = {}
            for s_i, q in enumerate(s):
                s_to_t[q] = t[s_i]

            k = input('found dict, enter keys (%s): ' % (",".join(s)))
            k_tokens = k.strip().split(",")

            pending_level = x[s_to_t[k_tokens[0]](k_tokens[0])]

            for k in k_tokens:
                if type(pending_level) != dict:
                    print('k', k, type(pending_level))
                    all_data_to_plot.append(x[s_to_t[k](k)])

            x = pending_level
    else:
        all_data_to_plot.append(x)

    ########################################################

    colors = 'rgbymck'
    fig = plt.figure()

    tmp = np.array(all_data_to_plot)
    ax = fig.add_subplot(autoscale_on=False,
        xlim=(np.min(tmp) - 0.1, np.max(tmp) + 0.1),
        ylim=(np.min(tmp) - 0.1, np.max(tmp) + 0.1)
    )
    # ax.set_aspect('equal')

    ax = fig.add_subplot()

    for x_i, x in enumerate(all_data_to_plot):
        # import ipdb; ipdb.set_trace()
        c = colors[x_i % len(colors)]

        print("x.shape", x.shape)

        x = np.float32(x)

        if len(x.shape) == 1:
            x = x[:, np.newaxis]

        ########################################################

        if len(args.indices) > 0:
            indices = [int(x) for x in args.indices.split(",")]
        else:
            indices = list(range(x.shape[1]))

        print(indices)

        if len(indices) > 4:
            print("bad indices")
            sys.exit(0)

        ########################################################

        if len(indices) == 3:
            ax1 = fig.add_subplot(1, 1, 1, projection='3d')

            sc1=ax1.scatter(
                x[:, 0],
                x[:, 1],
                x[:, 2],
                c=0.5*np.ones(x.shape[0]),
                s=1.0*np.ones(x.shape[0]),
                cmap=cm.jet,
                alpha=1.0,
                # label=k_tokens[x_i]
                )
            plt.colorbar(sc1, shrink=0.25)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('z')
        elif len(indices) == 4:
            ax1 = fig.add_subplot(1, 1, 1, projection='3d')

            p = 5
            a = 1

            alphas = x[:, 3] / np.max(x[:, 3])

            # import ipdb; ipdb.set_trace()

            contours = []

            for s in np.linspace(0, 59, args.n):
                s = int(s)

                if (s == 0):
                    q = 0
                    r = 484
                    x_slice = x[q:r, :]
                    n2 = 22
                elif (s == 59):
                    print('last one')
                    x_slice = x[-484:, :]
                    n2 = 22
                else:
                    q = 484+s*900
                    r = 484+(s+1)*900
                    x_slice = x[q:r, :]

                    n2 = 30

                print(s)

                # import ipdb; ipdb.set_trace()

                x_slice = x_slice[np.lexsort((x_slice[:,0], x_slice[:,1]))]

                z = np.min(x_slice[:, 2])
                print(z)
                rho0_contour = ax1.contourf(
                    x_slice[:, 0].reshape(n2, n2),
                    x_slice[:, 1].reshape(n2, n2),
                    x_slice[:, 3].reshape(n2, n2),
                    zdir='z',
                    cmap=cm.jet,
                    offset=z,
                    alpha=0.4
                )

                contours.append(rho0_contour)

            # sc1=ax1.scatter(
            #     x[:, 0],
            #     x[:, 1],
            #     x[:, 2],
            #     c=x[:, 3],
            #     s=p*x[:, 3],
            #     cmap=cm.jet,
            #     alpha=0.0, # a*alphas,
            #     # label=k_tokens[x_i]
            #     )

            plt.colorbar(contours[-1], shrink=0.25)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('z')

            ax1.set_zlim(0, 201)

        elif len(indices) == 2:
            sizes = 2*np.linspace(5, 100, x.shape[0])
            ax.scatter(x[:, indices[0]], x[:, indices[1]],
                c=c,
                marker='^',
                s=sizes,
                alpha=0.5)
            ax.plot(x[:, indices[0]], x[:, indices[1]],
                c=c,
                alpha=0.5,
                label=k_tokens[x_i])
        elif len(indices) == 1:
            # import ipdb; ipdb.set_trace()
            plt.plot(
                np.linspace(0, x.shape[0]-1, x.shape[0]),
                x[:, indices[0]],
                c=c,
                alpha=1/len(all_data_to_plot),
                )

    ########################################################

    plt.suptitle(args.dat)
    plt.legend()

    c = Counter()
    fig.canvas.mpl_connect('key_press_event', lambda e: c.on_press_saveplot(e,
            '%s'  %(
                args.dat.replace(".npy", ""),
            )
        )
    )

    ########################################################

    plt.grid(True)
    plt.tight_layout()
    plt.show()
