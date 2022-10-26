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

T_t = 5.0

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

    statedot[X1_index] = alpha1 * state[X2_index]
    statedot[X2_index] = alpha2 * state[X1_index]

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
    if t_control_data['grid'].shape[1] == 3:
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
        t_control_data['1'][closest_grid_idx])

    k = 1

    statedot[X1_index] = statedot[X1_index] + t_control_data['0'][closest_grid_idx] * k
    statedot[X2_index] = statedot[X2_index] + t_control_data['1'][closest_grid_idx] * k
    # statedot[X3_index] = statedot[X3_index] + t_control_data['2'][closest_grid_idx]

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
    def __init__(self, initial_pdf, data, point_size, distribution_samples, d, parent=None, devicePixelRatio=None, rotationMethod='euler'):
        super(MyGLViewWidget, self).__init__(parent, devicePixelRatio)

        self.initial_pdf = initial_pdf
        self.addItem(self.initial_pdf)
        self.showing_initial_pdf = True

        self._data = data
        self.distribution_samples = distribution_samples

        self.X1, self.X2, self.X3 = te_to_data["grid"]
        self.N = te_to_data["N"]

        self.i = 0

        data = self._data[self._data["keys"][self.i]]

        self.pdf_scale = 5.0

        self.pdf = gl.GLScatterPlotItem(
            pos=self.pdf_pos,
            size=np.ones((self.N**d)) * 0.05,
            color=cmap(data["probs"])*self.pdf_scale,
            pxMode=False)
        self.addItem(self.pdf)

        # import ipdb; ipdb.set_trace();

        self.lines = []
        for i in range(distribution_samples):
            lines = np.zeros((self.distribution_samples, d))
            if data["all_time_data"] is not None:
                lines = data["all_time_data"][i, :d, :].T

            self.lines.append(gl.GLLinePlotItem(
                pos = lines,
                width = 0.05,
                color = gray))
            self.addItem(self.lines[-1])
        self.showing_lines = True

        if data["all_time_data"] is not None:
            ends = data["all_time_data"][:, :d, -1]
        else:
            ends = np.zeros((self.distribution_samples, d))

        self.endpoints = gl.GLScatterPlotItem(
            pos=ends,
            size=point_size,
            color=blue,
            pxMode=False)
        self.addItem(self.endpoints)
        self.showing_endpoints = True

        if data["unforced_all_time_data"] is not None:
            unforced_ends = data["unforced_all_time_data"][:, :d, -1]
        else:
            unforced_ends = np.zeros((self.distribution_samples, d))

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

            # self.pdf.setData(
            #     color=cmap(data["probs"])*self.pdf_scale)

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

            # self.pdf.setData(
            #     color=cmap(data["probs"])*self.pdf_scale)

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
        mu_0[:2], cov_0[:2, :2], distribution_samples) # 100 x 3

    temp = np.zeros((initial_sample.shape[0], 1))
    initial_sample = np.hstack((initial_sample, temp))

    #############################################################################

    # x1_closest_index = [(np.abs(x1 - initial_sample[i, 0])).argmin() for i in range(initial_sample.shape[0])]
    # x2_closest_index = [(np.abs(x2 - initial_sample[i, 1])).argmin() for i in range(initial_sample.shape[0])]

    # for each initial sample, find the closest x3 = x30 layer it can help de-alias
    # x3_closest_index[i] = initial sample[i]'s dealiasing x30 layer
    # x3_closest_index = [(np.abs(x3 - initial_sample[i, 2])).argmin() for i in range(initial_sample.shape[0])]

    #############################################################################

    te_to_data = {
        "keys" : ts,
        "grid" : np.meshgrid(x1,x2,x3,copy=False),
        # "grid" : np.meshgrid(x1,x2,copy=False),
        "N" : N
    }

    do_dealiasing = False

    for t_e in ts:
        start = time.time()

        #############################################################################

        all_time_data = None
        unforced_all_time_data = None

        if t_e > 0:
            t_samples = np.linspace(0, t_e, 30)

            all_time_data = np.empty(
                (
                    initial_sample.shape[0],
                    3,
                    len(t_samples))
                )
            # x/y slice is all samples at that time, 1 x/y slice per z time initial_sample

            # x[i] is sample [i]
            # y[i] is state dim [i]
            # z[i] is time [i]

            for sample_i in range(initial_sample.shape[0]):
                print("forcing sample_i", sample_i)
                sample_states = integrate.odeint(
                    dynamics,
                    initial_sample[sample_i, :2],
                    t_samples,
                    args=(j1, j2, j3, control_data))

                temp = np.zeros((sample_states.shape[0], 1))
                sample_states = np.hstack((sample_states, temp))

                all_time_data[sample_i, :, :] = sample_states.T

            unforced_all_time_data = np.empty(
                (
                    initial_sample.shape[0],
                    3,
                    len(t_samples))
                )

            for sample_i in range(initial_sample.shape[0]):
                print("unforced sample_i", sample_i)
                sample_states = integrate.odeint(
                    dynamics,
                    initial_sample[sample_i, :2],
                    t_samples,
                    args=(j1, j2, j3, None))

                temp = np.zeros((sample_states.shape[0], 1))
                sample_states = np.hstack((sample_states, temp))

                unforced_all_time_data[sample_i, :, :] = sample_states.T

        #############################################################################

        te_to_data[t_e] = {
            "all_time_data" : all_time_data,
            "unforced_all_time_data" : unforced_all_time_data,
        }

    return initial_sample, te_to_data, X1, X2, X3

#############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--times',
        type=str,
        default="0, 2.5, 5.0",
        required=False)

    parser.add_argument('--mu_0',
        type=float,
        default=2.0,
        required=False)

    parser.add_argument('--sampling',
        type=str,
        default="15,15,15,15,15,15,30,10",
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

    d = 2

    # distribution
    mu_0 = np.array([mu_0]*d)
    cov_0 = np.eye(d)*sigma_0

    # sampling
    # ts = [float(x) for x in args.times.split(",")]
    ts = np.linspace(T_0, T_t, 2)

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
    if len(args.control_data) > 0:
        control_data = np.load(
            args.control_data,
            allow_pickle=True).item()

    #############################################################################

    initial_sample, te_to_data, X1, X2, X3 = init_data(
        mu_0, cov_0,
        windows, distribution_samples, N, ts,
        j1, j2, j3,
        control_data,
        args.ignore_symmetry)

    #############################################################################

    ## Create a GL View widget to display data
    app = pg.mkQApp("")

    point_size = np.ones(distribution_samples) * 0.08

    initial_pdf_sample = gl.GLScatterPlotItem(
        pos=initial_sample[:, :3],
        size=point_size,
        color=green,
        pxMode=False)

    # import ipdb; ipdb.set_trace();

    w = MyGLViewWidget(initial_pdf_sample, te_to_data, point_size, distribution_samples, d)

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
