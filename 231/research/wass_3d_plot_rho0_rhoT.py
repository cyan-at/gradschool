#!/usr/bin/env python
# coding: utf-8

# 0 define backend
import sys, os, time

import numpy as np
from numpy import linalg as LA
import math

import matplotlib.pyplot as plt
import pylab

from os.path import dirname, join as pjoin

from scipy import stats
import scipy.io
from scipy.stats import truncnorm, norm
from scipy.optimize import linprog
from scipy import sparse
from scipy.stats import multivariate_normal

from params import *

import argparse

def slice(matrix_3d, i, j, mode):
  if mode == 0:
    return matrix_3d[j, i, :]
  elif mode == 1:
    return matrix_3d[i, j, :]
  else:
    return matrix_3d[i, :, j]

def get_trapznormd_marginal(matrix_3d, xs, mode):
  marginal = np.array([
      np.trapz(
          np.array([
              np.trapz(
                slice(matrix_3d, i, j, mode),
                x=xs[2]) # x3 slices for one x2 => R
              for i in range(len(xs[1]))]) # x3 slices across all x2 => Rn
          , x=xs[1]) # x2 slice for one x1 => R
  for j in range(len(xs[0]))])
  marginal /= np.trapz(marginal, x=xs[0])
  return marginal

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--modelpt', type=str, required=True)
  parser.add_argument('--testdat', type=str, required=True)

  parser.add_argument('--N', type=int, default=50, help='')
  parser.add_argument('--js', type=str, default="1,1,2", help='')
  parser.add_argument('--q', type=float, default=0.0, help='')
  args = parser.parse_args()

  N = 10
  j1, j2, j3 = 3,2,1
  q_statepenalty_gain = 0.0

  x1 = np.transpose(np.linspace(state_min, state_max, N))
  x2 = np.transpose(np.linspace(state_min, state_max, N))
  x3 = np.transpose(np.linspace(state_min, state_max, N))
  [X,Y,Z] = np.meshgrid(x1,x2,x3)

  ########################################################

  test = np.loadtxt(args.testdat)

  ####################################################################

  test_rho0 = test[:N**3, :]

  test_rho0_rhoopt = test_rho0[:, 5]

  test_rho0_rhoopt = np.where(test_rho0_rhoopt<0, 0, test_rho0_rhoopt)
  test_rho0_rhoopt /= np.trapz(
    test_rho0_rhoopt,
    axis=0,
    x=test_rho0[:,0])

  ####################################################################

  rho_opt = np.zeros((N,N,N))

  closest_1 = [(np.abs(x1 - test_rho0[i, 0])).argmin() for i in range(test_rho0.shape[0])]
  closest_2 = [(np.abs(x2 - test_rho0[i, 1])).argmin() for i in range(test_rho0.shape[0])]
  closest_3 = [(np.abs(x3 - test_rho0[i, 2])).argmin() for i in range(test_rho0.shape[0])]

  rho_opt[closest_1, closest_2, closest_3] = test_rho0_rhoopt

  ####################################################################

  rho0_x1_marginal = get_trapznormd_marginal(
    rho_opt, [x1, x2, x3], 0)

  rho0_x2_marginal = get_trapznormd_marginal(
    rho_opt, [x2, x1, x3], 1)

  rho0_x3_marginal = get_trapznormd_marginal(
    rho_opt, [x3, x1, x2], 2)

  ####################################################################

  test_rhoT = test[N**3:2*N**3, :]

  test_rhoT_rhoopt = test_rhoT[:, 5]

  test_rhoT_rhoopt = np.where(test_rhoT_rhoopt<0, 0, test_rhoT_rhoopt)
  test_rhoT_rhoopt /= np.trapz(
    test_rhoT_rhoopt,
    axis=0,
    x=test_rhoT[:,0])

  ####################################################################

  rho_opt = np.zeros((N,N,N))

  closest_1 = [(np.abs(x1 - test_rhoT[i, 0])).argmin() for i in range(test_rhoT.shape[0])]
  closest_2 = [(np.abs(x2 - test_rhoT[i, 1])).argmin() for i in range(test_rhoT.shape[0])]
  closest_3 = [(np.abs(x3 - test_rhoT[i, 2])).argmin() for i in range(test_rhoT.shape[0])]

  rho_opt[closest_1, closest_2, closest_3] = test_rhoT_rhoopt

  ####################################################################

  rhoT_x1_marginal = get_trapznormd_marginal(
    rho_opt, [x1, x2, x3], 0)

  rhoT_x2_marginal = get_trapznormd_marginal(
    rho_opt, [x2, x1, x3], 1)

  rhoT_x3_marginal = get_trapznormd_marginal(
    rho_opt, [x3, x1, x2], 2)

  ####################################################################

  fig = plt.figure(1)
  ax1 = plt.subplot(131, frameon=False)
  # ax1.set_aspect('equal')
  ax1.grid()
  ax1.set_title('x1 marginal')
  ax2 = plt.subplot(132, frameon=False)
  # ax2.set_aspect('equal')
  ax2.grid()
  ax2.set_title('x2 marginal')
  ax3 = plt.subplot(133, frameon=False)
  # ax3.set_aspect('equal')
  ax3.grid()
  ax3.set_title('x3 marginal')

  ####################################################################

  colors="rgbymkc"

  i = 0
  ax1.plot(x1,
      rho0_x1_marginal,
      colors[i % len(colors)],
      linewidth=1,
      label=T_0)
  ax1.legend(loc='lower right')

  ax2.plot(x2,
      rho0_x2_marginal,
      colors[i % len(colors)],
      linewidth=1,
      label=T_0)
  ax2.legend(loc='lower right')

  ax3.plot(x3,
      rho0_x3_marginal,
      colors[i % len(colors)],
      linewidth=1,
      label=T_0)
  ax3.legend(loc='lower right')

  ###################################################################

  i = 1
  ax1.plot(x1,
      rhoT_x1_marginal,
      colors[i % len(colors)],
      linewidth=1,
      label=T_t)
  ax1.legend(loc='lower right')

  ax2.plot(x2,
      rhoT_x2_marginal,
      colors[i % len(colors)],
      linewidth=1,
      label=T_t)
  ax2.legend(loc='lower right')

  ax3.plot(x3,
      rhoT_x3_marginal,
      colors[i % len(colors)],
      linewidth=1,
      label=T_t)
  ax3.legend(loc='lower right')


  ###################################################################

  fig.suptitle('rho0 / rhoT', fontsize=16)

  plt.show()