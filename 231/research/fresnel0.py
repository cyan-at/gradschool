#!/usr/bin/env python3

import argparse
import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt

import scipy.integrate as integrate


S_INDEX = 0
C_INDEX = 1

def fresnel(state, t):
  '''
  sin(t**2)
  cos(t**2)
  '''
  statedot = np.zeros_like(state)

  statedot[S_INDEX] = np.sin(t**2)
  statedot[C_INDEX] = np.cos(t**2)

  return statedot

def S(x):
  pass

def C(x):
  pass

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--t_stop', type=int, default=10, help='')
  parser.add_argument('--dt', type=float, default=0.02, help='')

  args = parser.parse_args()

  state = np.array([0.0]*2)
  times = np.arange(0, args.t_stop, args.dt)
  states = np.zeros((len(times), state.shape[0]))

  i = 0
  while i < len(times) - 1:
    # solve differential equation, take final result only
    state = integrate.odeint(
      fresnel,
      state,
      times[i:i+2])[-1]
    states[i+1, :] = state
    i += 1
  print("done integrating")


  fig = plt.figure(1)
  ax1 = plt.subplot(111, frameon=False)
  ax1.set_aspect('equal')
  ax1.grid()

  ax1.plot(
      states[:, C_INDEX],
      states[:, S_INDEX],
      'r',
      linewidth=1,
      label='euler')
  ax1.legend(loc='lower right')

  plt.show()