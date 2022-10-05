#!/usr/bin/env python3

'''
#!/usr/bin/python3
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata as gd
import time
from numpy.random import default_rng
import argparse

# import tensorflow as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--control',
        type=str,
        default="",
        required=False)

    args = parser.parse_args()

    test = np.genfromtxt('./%s/%s_test.dat' % (args.control, args.control))
    # x1, x2, x3, t, y1 (psi), y2 (rho)

    test_timesorted = test[test[:, 3].argsort()]
    source_t = test_timesorted[:, 3]

    state_min = -2.0
    state_max = 2.0
    state_N = 50j

    T_t = 5.0
    time_N = 50j

    ################################################################

    '''

    target_t = 5.0
    target_thresh = 1e-8

    test_t5 = test_timesorted[np.where(np.abs(source_t-target_t) < target_thresh), :] # 2k
    test_t5 = test_t5[0]

    test_t0 = test_timesorted[np.where(np.abs(source_t) < 1e-8), :] # 2k
    test_t0 = test_t0[0]

    source_x1 = test_t0[:, 0]
    source_x2 = test_t0[:, 1]
    source_x3 = test_t0[:, 2]
    source_psi = test_t0[:, 4]

    grid_x1, grid_x2, grid_x3 = np.mgrid[
      state_min:state_max:state_N,
      state_min:state_max:state_N,
      state_min:state_max:state_N]

    PSI = gd(
      (source_x1, source_x2, source_x3),
      source_psi,
      (grid_x1, grid_x2, grid_x3),
      method='nearest')

    #Plot original values
    fig1 = plt.figure()
    ax1=fig1.gca(projection='3d')
    sc1=ax1.scatter(source_x1, source_x2, source_x3, c=source_psi, cmap=plt.hot())
    plt.colorbar(sc1)
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('x3')
    ax1.set_title('source data t=%.3f' % (target_t))

    #Plot interpolated values
    fig2 = plt.figure()
    ax2=fig2.gca(projection='3d')
    sc2=ax2.scatter(grid_x1, grid_x2, grid_x3, c=PSI, cmap=plt.hot())
    plt.colorbar(sc2)
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_zlabel('x3')
    ax2.set_title('grid data t=%.3f' % (target_t))

    plt.show()
    '''

    ################################################################

    '''
    middle = test_timesorted[(source_t < 5.0) & (source_t > 0.0), :] # 16k - 4k = 12k

    N = 50
    x_1_ = np.linspace(state_min, state_max, N)
    x_2_ = np.linspace(state_min, state_max, N)
    x_3_ = np.linspace(state_min, state_max, N)
    t_ = np.linspace(0, T_t, N)
    grid_x1, grid_x2, grid_x3, grid_t = np.meshgrid(
      x_1_,
      x_2_,
      x_3_,
      t_, copy=False) # each is NxNxN

    source_x1 = middle[:, 0]
    source_x2 = middle[:, 1]
    source_x3 = middle[:, 2]
    source_t = middle[:, 3]
    source_psi = middle[:, 4]

    PSI = gd(
      (source_x1, source_x2, source_x3, source_t),
      source_psi,
      (grid_x1, grid_x2, grid_x3, grid_t),
      method='nearest')
    '''



    # x1_closest_index = [(np.abs(x_1_ - initial_sample[i, 0])).argmin() for i in range(initial_sample.shape[0])]
    # x2_closest_index = [(np.abs(x_2_ - initial_sample[i, 0])).argmin() for i in range(initial_sample.shape[0])]
    # x3_closest_index = [(np.abs(x_3_ - initial_sample[i, 0])).argmin() for i in range(initial_sample.shape[0])]
    # t_closest_index = [(np.abs(t_ - initial_sample[i, 0])).argmin() for i in range(initial_sample.shape[0])]

    import ipdb; ipdb.set_trace();

    '''
    source_x1 = test_timesorted[:, 0]
    source_x2 = test_timesorted[:, 1]
    source_x3 = test_timesorted[:, 2]
    source_psi = test_timesorted[:, 4]
    source_t = test_timesorted[:, 3]

    # source_v1 = np.gradient(psi, x1)
    # source_v2 = np.gradient(psi, x2)
    # source_v3 = np.gradient(psi, x3)

    ######################

    import ipdb; ipdb.set_trace();

    state_min = -2.0
    state_max = 2.0
    state_N = 50j

    T_t = 5.0
    T_N = 10j

    grid_x1, grid_x2, grid_x3, grid_t = np.mgrid[
      state_min:state_max:state_N,
      state_min:state_max:state_N,
      state_min:state_max:state_N,
      0:T_t:N]

    # state_min = -2.0
    # state_max = 2.0
    # T_t = 5.0
    # N = 100

    # x_1 = np.linspace(state_min, state_max, N)
    # x_2 = np.linspace(state_min, state_max, N)
    # x_3 = np.linspace(state_min, state_max, N)
    # t_ = np.linspace(0, T_t, N)
    # X1, X2, X3, T = np.meshgrid(x_1, x_2, x_3, t_, copy=False) # each is NxNxN

    PSI = gd(
      (source_x1, source_x2, source_x3, source_t),
      source_psi,
      (grid_x1, grid_x2, grid_x3, grid_t),
      method='nearest')

    import ipdb; ipdb.set_trace();

    ########################################################################

    #read values
    sampl = np.random.uniform(low=-5.0, high=5.0, size=(50,3))
    x = sampl[:, 0]
    y = sampl[:, 1]
    z = sampl[:, 2]
    # v = sampl[:, 3]

    def func(x,y,z):
        return 0.5*(3)**(1/2)-((x-0.5)**2+(y-0.5)**2+(z-0.5)**2)**(1/2)
    x = np.random.rand(100)
    y = np.random.rand(100)
    z = np.random.rand(100)
    v = func(x,y,z)
    print("")

    #generate new grid X,Y,Z
    print("Generate new grid...")
    # xi,yi,zi=np.ogrid[0:1:11j, 0:1:11j, 0:1:11j]

    # X1=xi.reshape(xi.shape[0],)
    # Y1=yi.reshape(yi.shape[1],)
    # Z1=zi.reshape(zi.shape[2],)

    # ar_len=len(X1)*len(Y1)*len(Z1)
    # X=np.arange(ar_len,dtype=float)
    # Y=np.arange(ar_len,dtype=float)
    # Z=np.arange(ar_len,dtype=float)
    # l=0
    # for i in range(0,len(X1)):
    #     for j in range(0,len(Y1)):
    #         for k in range(0,len(Z1)):
    #             X[l]=X1[i]
    #             Y[l]=Y1[j]
    #             Z[l]=Z1[k]
    #             l=l+1
    # print("")
    X, Y, Z = np.mgrid[0:1:11j, 0:1:11j, 0:1:11j]

    N = 10
    x1 = np.linspace(0, 1, N)
    x2 = np.linspace(0, 1, N)
    x3 = np.linspace(0, 1, N)
    X1, X2, X3 = np.meshgrid(x1,x2,x3,copy=False) # each is NxNxN

    #interpolate "data.v" on new grid "X,Y,Z"
    print("Interpolate...")
    V = gd((x,y,z), v, (X1, X2, X3), method='nearest')
    print("")

    #Plot original values
    fig1 = plt.figure()
    ax1=fig1.gca(projection='3d')
    sc1=ax1.scatter(x, y, z, c=v, cmap=plt.hot())
    plt.colorbar(sc1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    #Plot interpolated values
    fig2 = plt.figure()
    ax2=fig2.gca(projection='3d')
    sc2=ax2.scatter(X1, X2, X3, c=V, cmap=plt.hot())
    plt.colorbar(sc2)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    plt.show()
    '''