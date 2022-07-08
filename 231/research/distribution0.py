#!/usr/bin/python3

import argparse

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import os, pickle, time, sys
from matplotlib import cm

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

colors = [
    (0, 0, 0),
    (45, 5, 61),
    (84, 42, 55),
    (150, 87, 60),
    (208, 171, 141),
    (255, 255, 255)
]
cmap = pg.ColorMap(pos=np.linspace(0.0, 0.5, len(colors)), color=colors)
cmap = cm.get_cmap('gist_heat')

#############################################################################

parser = argparse.ArgumentParser()

parser.add_argument('--times',
    type=str,
    default="0,1,5",
    required=False)

parser.add_argument('--mu_0',
    type=float,
    default=4.5,
    required=False)

parser.add_argument('--sampling',
    type=str,
    default="15,5,15,5,5,5,100",
    required=False)

args = parser.parse_args()

# system
alpha2 = 1.0

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

N = sampling[6]

#############################################################################

## Create a GL View widget to display data
app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()
w.setWindowTitle('snapshots')
w.setCameraPosition(distance=20)

#############################################################################

## Add a grid to the view
g = gl.GLGridItem()
g.scale(2,2,1)
g.setDepthValue(10)  # draw grid after surfaces since they may be translucent
w.addItem(g)

#############################################################################

x1 = np.linspace(mu_0[0] - window0, mu_0[0] + window1, N)
x2 = np.linspace(mu_0[1] - window2, mu_0[1] + window3, N)
x3 = np.linspace(mu_0[2] - window4, mu_0[2] + window5, N)

with_args = lambda x, y, z: composited(x, y, z, 0, alpha2, mu_0, cov_0)

start = time.time()

#############################################################################
# for loop: 5s

# temp = np.zeros((N, N, N, 1))
# for i in range(N):
#     for j in range(N):
#         for k in range(N):
#             # pos = inverse_flow(x1[i], x2[j], x3[k], 0, alpha2)
#             # temp[i, j, k, 3] = normal_dist_array(pos, mu_0 , cov_0)
#             temp[i, j, k, 0] = with_args(x1[i], x2[j], x3[k])

#############################################################################
# bad vectorize: 5s

# with_args_vec = np.vectorize(with_args)
# temp = with_args_vec(X1, X2, X3)

#############################################################################
# using broadcasting: 0.12s

X1, X2, X3 = np.meshgrid(x1,x2,x3,copy=False) # each is NxNxN

t_e = 0
omegas = alpha2 * X3
tans = np.tan(omegas*t_e)
gammas = (X2 - X1 * tans) / (X1 + X2*tans)

x10 = np.sqrt((X1**2 + X2**2) / (1 + gammas))
x20 = gammas * np.sqrt((X1**2 + X2**2) / (1 + gammas**2))
x30 = X3

###################

den = (np.sqrt(2*np.pi*np.linalg.norm(cov_0)))
cov_inv = np.linalg.inv(cov_0)

x10_diff = x10 - mu_0[0]
x20_diff = x20 - mu_0[1]
x30_diff = x30 - mu_0[2]

# 3 x NxNxN to N**3 x 3
x10_x20_x30 = np.vstack([x10_diff.reshape(-1), x20_diff.reshape(-1), x30_diff.reshape(-1)])

probs = np.exp(np.einsum('i...,ij,j...',x10_x20_x30,cov_inv,x10_x20_x30)/(-2)) / den

# probs = probs.reshape(N, N, N)

#############################################################################

sample = np.random.multivariate_normal(mu_0, cov_0, 100) # 100 x 3
print(sample.shape)

size = np.ones(100) * 0.2
g = (0, 255, 0, 255)
gray = (0.5, 0.5, 0.5, 0.1)
b = (0, 0, 255, 255)

# import ipdb; ipdb.set_trace();

if t_e > 0:
    A = np.sqrt(sample[:, 0]**2 + sample[:, 1]**2)
    phi = np.arctan2(sample[:, 1], sample[:, 0])

    n_time_samples = 100
    t_samples = np.linspace(0, t_e, n_time_samples)

    all_time_data = np.empty((sample.shape[0], sample.shape[1], n_time_samples))
    # x/y slice is all samples at that time, 1 x/y slice per z time sample

    # import ipdb; ipdb.set_trace();

    for i, t_i in enumerate(t_samples):
        x_1 = A * np.cos(alpha2 * sample[:, 2] * t_samples[i] + phi)
        x_2 = A * np.sin(alpha2 * sample[:, 2] * t_samples[i] + phi)
        x_3 = sample[:, 2]
        # all are nx1 for t_i

        # stack it up
        all_time_data[:, :, i] = np.vstack([x_1, x_2, x_3]).T

    # generate the line from looking at x / z slices

    for i in range(sample.shape[0]):
        item = gl.GLLinePlotItem(
            pos = all_time_data[i, :, :].T,
            width = 0.5,
            color = gray)

        w.addItem(item)

    ends = all_time_data[:, :, -1] # the top most time slice is the end
    endpoints = gl.GLScatterPlotItem(pos=ends, size=size, color=b, pxMode=False)
    w.addItem(endpoints)

sp0 = gl.GLScatterPlotItem(pos=sample, size=size, color=g, pxMode=False)

#############################################################################

end = time.time()

total_time = end - start
print("compute %s\n" % str(total_time))

#############################################################################

start = time.time()

size = np.ones((N**3)) * 0.1
pos = np.vstack([X1.reshape(-1), X2.reshape(-1), X3.reshape(-1)]).T
color0 = cmap(probs)

# # color0[:, 3] = color0[:, 3] / 10

# color0[:, 3] /= 10
print(color0[:, 3])

sp1 = gl.GLScatterPlotItem(pos=pos, size=size, color=color0, pxMode=False)

end = time.time()

total_time = end - start
print("render %s\n" % str(total_time))

#############################################################################

w.addItem(sp1)
w.addItem(sp0)

#############################################################################

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()