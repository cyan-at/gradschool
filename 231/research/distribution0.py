#!/usr/bin/python3

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import os, pickle

alpha2 = 1.0

# time sample t = 0
t = 0.0

# p0 is distribution with mean at 0.5
window = 5.0
N = 50
mu_0 = np.array([5.5]*3)
cov_0 = np.eye(3)

colors = [
    (0, 0, 0),
    (45, 5, 61),
    (84, 42, 55),
    (150, 87, 60),
    (208, 171, 141),
    (255, 255, 255)
]

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
    # x1 = state[0]
    # x2 = state[1]
    # x3 = state[2]

    alpha2 = 1.0

    omega = alpha2 * x3
    gamma = (x2 - x1 * np.tan(omega*t)) / (x1 + x2*np.tan(omega*t))

    x10 = np.sqrt((x1**2 + x2**2) / (1 + gamma))
    x20 = gamma * np.sqrt((x1**2 + x2**2) / (1 + gamma**2))
    x30 = x3

    return np.array([x10, x20, x30])

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

#############################################################################

## Create a GL View widget to display data
app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()
w.setWindowTitle('pyqtgraph example: GLSurfacePlot')
w.setCameraPosition(distance=20)

#############################################################################

## Add a grid to the view
g = gl.GLGridItem()
g.scale(2,2,1)
g.setDepthValue(10)  # draw grid after surfaces since they may be translucent
w.addItem(g)

#############################################################################

x1 = np.linspace(mu_0[0] - window, mu_0[0] + window, N)
x2 = np.linspace(mu_0[1] - window, mu_0[1] + window, N)
x3 = np.linspace(mu_0[2] - window, mu_0[2] + window, N)

fname = 'inversed_%d_%d_%d.pkl' % (N, int(window), int(t))
inversed0 = None
inversed1 = None
inversed2 = None
if os.path.exists(fname):
    print("%s found" % (fname))
    with open(fname, 'rb') as handle:
        data = pickle.load(handle)
        inversed0 = data["inversed0"]
        inversed1 = data["inversed1"]
        inversed2 = data["inversed2"]
else:
    '''
    X1, X2, X3 = np.meshgrid(x1,x2,x3) # each is NxNxN

    pos = np.empty(X1.shape + (4,)) # NxNxN, for x10, x20, x30, p0

    t = 0
    omega = alpha2 * X3
    gamma = (X2 - X1 * np.tan(omega*t)) / (X1 + X2*np.tan(omega*t))

    pos[:, :, :, 0] = np.sqrt((X1**2 + X2**2) / (1 + gamma))
    pos[:, :, :, 1] = gamma * np.sqrt((X1**2 + X2**2) / (1 + gamma**2))
    pos[:, :, :, 2] = X3

    pos[:, :, :, 3] = 

    # this takes the linspace at that axis
    # and duplicates it across the other axii
    # so instead of iterating, we sample
    # taking more space to take less time

    inversed = inverse_flow(X1, X2, X3) # 500x500x500, 3D matrix

    inversed1 = inverse_flow_t1(X1, X2, X3) # 500x500x500, 3D matrix

    '''

    inversed0 = np.zeros((N, N, N, 4))
    inversed1 = np.zeros((N, N, N, 4))
    inversed2 = np.zeros((N, N, N, 4))

    for i in range(N):
        for j in range(N):
            for k in range(N):
                inversed0[i, j, k, :3] = inverse_flow(x1[i], x2[j], x3[k], 0, alpha2)
                inversed0[i, j, k, 3] = normal_dist_array(inversed0[i, j, k, :3], mu_0 , cov_0)

                inversed1[i, j, k, :3] = inverse_flow(x1[i], x2[j], x3[k], 1, alpha2)
                inversed1[i, j, k, 3] = normal_dist_array(inversed1[i, j, k, :3], mu_0 , cov_0)

                inversed2[i, j, k, :3] = inverse_flow(x1[i], x2[j], x3[k], 5, alpha2)
                inversed2[i, j, k, 3] = normal_dist_array(inversed2[i, j, k, :3], mu_0 , cov_0)
    with open(fname, 'wb') as handle:
        pickle.dump(
            {
                "inversed0" : inversed0,
                "inversed1" : inversed1,
                "inversed2" : inversed2,
            },
            handle, protocol=pickle.HIGHEST_PROTOCOL)

#############################################################################

size = np.ones((N**3)) * 0.2

pos0 = np.empty((N**3, 3))
color0 = np.zeros((N**3, 4))

pos1 = np.empty((N**3, 3))
color1 = np.zeros((N**3, 4))

pos2 = np.empty((N**3, 3))
color2 = np.zeros((N**3, 4))


max_0 = np.max(inversed0[:, :, :, 3])
max_1 = np.max(inversed1[:, :, :, 3])
max_2 = np.max(inversed2[:, :, :, 3])

print("max_0", max_0)
print("max_1", max_1)
print("max_2", max_2)

m = max([max_0, max_1, max_2])

cmap = pg.ColorMap(pos=np.linspace(0.0, max_0, len(colors)), color=colors)

for i in range(N):
    for j in range(N):
        for k in range(N):
            pos0[i*N*N+j*N+k] = inversed0[i, j, k, :3]
            # color0[i*N*N+j*N+k] = (inversed0[i, j, k, 3] * 5, 0.0, 0.0, 0.5)

            t = cmap.mapToQColor(inversed0[i, j, k, 3])
            color0[i*N*N+j*N+k] = t.getRgbF()

            pos1[i*N*N+j*N+k] = inversed1[i, j, k, :3]            
            # color1[i*N*N+j*N+k] = (0.0, inversed1[i, j, k, 3] * 5 , 0.0, 0.5)

            t = cmap.mapToQColor(inversed1[i, j, k, 3])
            color1[i*N*N+j*N+k] = t.getRgbF()

            pos2[i*N*N+j*N+k] = inversed2[i, j, k, :3]            
            # color2[i*N*N+j*N+k] = (0.0, 0.0, inversed2[i, j, k, 3] * 5, 0.5)

            t = cmap.mapToQColor(inversed2[i, j, k, 3])
            color2[i*N*N+j*N+k] = t.getRgbF()

sp0 = gl.GLScatterPlotItem(pos=pos0, size=size, color=color0, pxMode=False)
w.addItem(sp0)

sp1 = gl.GLScatterPlotItem(pos=pos1, size=size, color=color1, pxMode=False)
sp1.translate(5,5,0)
w.addItem(sp1)

sp2 = gl.GLScatterPlotItem(pos=pos2, size=size, color=color2, pxMode=False)
sp2.translate(10,10,0)
w.addItem(sp2)

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()