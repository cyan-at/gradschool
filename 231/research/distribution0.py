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

    ########################################

    closest_1 = (np.abs(control_data["x_1_"] - state[X1_index])).argmin()
    closest_2 = (np.abs(control_data["x_2_"] - state[X2_index])).argmin()
    closest_3 = (np.abs(control_data["x_3_"] - state[X3_index])).argmin()
    closest_t = (np.abs(control_data["t_"] - t)).argmin()

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

    unforced_dynamics_with_args = lambda state, t: dynamics(state, t, j1, j2, j3, None)
    dynamics_with_args = lambda state, t: dynamics(state, t, j1, j2, j3, control_data)

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
                    dynamics_with_args,
                    initial_sample[sample_i, :],
                    t_samples)
                all_time_data[sample_i, :, :] = sample_states.T

            unforced_all_time_data = np.empty(
                (
                    initial_sample.shape[0],
                    initial_sample.shape[1],
                    len(t_samples))
                )

            for sample_i in range(initial_sample.shape[0]):
                sample_states = integrate.odeint(
                    unforced_dynamics_with_args,
                    initial_sample[sample_i, :],
                    t_samples)
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

    parser.add_argument('--control_prefix',
        type=str,
        default="run_tensorflow_-2.500_2.500__5.000__2000__0.001__1_1_2__0.000__2.000_1.000__0.000_1.000__12000_1000__20000",
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
    if len(args.control_prefix) > 0:
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

    #############################################################################

    initial_sample, te_to_data, X1, X2, X3 = init_data(
        mu_0, cov_0,
        windows, distribution_samples, N, ts,
        j1, j2, j3,
        control_data,
        args.ignore_symmetry)

    #############################################################################

    ## Create a GL View widget to display data
    app = QtGui.QApplication([])

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

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()