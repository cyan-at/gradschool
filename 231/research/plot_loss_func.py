#!/usr/bin/env python3

import argparse
import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt

import scipy.integrate as integrate

import time

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--t_stop', type=int, default=10, help='')
  parser.add_argument('--dt', type=float, default=0.02, help='')

  args = parser.parse_args()

  loss_loaded = np.genfromtxt('loss.dat')

  # import ipdb; ipdb.set_trace();

  # [0] epoch
  # [1] y1, psi, hjb
  # [2] y2, rho, plank pde
  # [3] rho0, initial
  # [4] rhoT, terminal

  epoch = loss_loaded[:, 0]
  y1_psi_hjb = loss_loaded[:, 1]
  y2_rho_plankpde = loss_loaded[:, 2]
  rho0_initial = loss_loaded[:, 3]
  rhoT_terminal = loss_loaded[:, 4]

  fig, ax = plt.subplots()
  ax.set_yscale('log')

  line1, = ax.plot(epoch, y1_psi_hjb, color='orange', lw=1, label='HJB PDE')
  line2, = ax.plot(epoch, y2_rho_plankpde, color='blue', lw=1, label='Controlled Fokker-Planck PDE')
  line3, = ax.plot(epoch, rho0_initial, color='red', lw=1, label='p0 boundary condition')
  line4, = ax.plot(epoch, rhoT_terminal, color='purple', lw=1, label='pT boundary condition')

  ax.grid()
  ax.legend(loc="upper right")
  ax.set_title('training error/residual plots: mu0=2 -> muT=5')

  plt.show()