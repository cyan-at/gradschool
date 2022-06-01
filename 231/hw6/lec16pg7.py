#!/usr/bin/env python3

from numpy import sin, cos, exp
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import scipy.integrate as integrate
from collections import deque
import argparse

# system parameters
J1 = 1
F1 = 1
K = 1
N = 2

J2 = 0.5
F2 = 1.6

m = M = 1
g = G = 9.8
d = 1.7

# controller params
k1 = -24
k2 = -50
k3 = -35
k4 = -10

# k1 = -14
# k2 = -71
# k3 = -154
# k4 = -120

# initial state
x0 = np.array([np.pi / 6, np.pi / 3, 1])

class System(object):
  def alpha(self, x):
    return -2*x[0]*x[1]/(2*x[1]*exp(x[1]) + exp(x[1])) - 2*x[1]**3/(2*x[1]*exp(x[1]) + exp(x[1]))


  def beta(self, x):
    return 1/(-2*x[1]*exp(x[1]) - exp(x[1]))

  def tau(self, x):
    return np.array([
      x[2],
      x[0] - x[1],
      -x[0] - x[1]**2,
    ])

  ############################################################################

  def derivs_alphabetau(self, x, t):
    xdot = np.zeros_like(x)
    # state = [q1,  q2,  q1*,  q2*]
    # xdot =  [q1*, q2*, q1**, q2**]

    # feedback control
    z = self.tau(x)
    v = np.dot(np.array([k1, k2, k3]), z)
    u = self.alpha(x) + self.beta(x) * v

    # plant
    xdot[0] = 0
    xdot[1] = x[0] + x[1]**2
    xdot[2] = x[0] - x[1]

    xdot[0] += exp(x[1]) * u
    xdot[1] += exp(x[1]) * u 

    return xdot

  ############################################################################

  def __init__(self,
    args,
    initial_state,
    sampletimes):
    self._args = args

    self.initial_state = initial_state
    self.sampletimes = sampletimes

    self.state = None
    self.us = []

  def init_data(self):
    self.states = np.zeros((len(self.sampletimes), self.initial_state.shape[0] + 1))

    self.states[0, :self.initial_state.shape[0]] = self.initial_state

    i = 0
    state = self.initial_state

    '''
    # self.state = integrate.odeint(
    #   self._modes[0],
    #   self.initial_state,
    #   self.sampletimes)
    https://stackoverflow.com/a/63189903
    use this integration method instead of
    passthrough odeint for systems with
    'state'
    '''

    if self._args.mode == 0:
      print("derivs_alphabetau")

      while i < len(self.sampletimes) - 1:
        x = state
        z = self.tau(x)
        v = np.dot(np.array([k1, k2, k3]), z)
        self.states[i+1, self.initial_state.shape[0]] = self.alpha(x) + self.beta(x) * v

        # solve differential equation, take final result only
        state = integrate.odeint(
          self.derivs_alphabetau,
          state,
          self.sampletimes[i:i+2])[-1]
        self.states[i+1, :self.initial_state.shape[0]] = state

        #############################################

        i += 1

    print("done integrating")

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description="")
  parser.add_argument('--playback', type=int, default=1, help='')
  parser.add_argument('--history', type=int, default=500, help='')
  parser.add_argument('--plot', type=str, default="animation", help='')

  parser.add_argument('--dt', type=float, default=0.01, help='')
  parser.add_argument('--t_stop', type=int, default=5, help='')

  parser.add_argument('--mode', type=int, default=0, help='')

  args = parser.parse_args()

  times = np.arange(0, args.t_stop, args.dt)
  system = System(args, x0, times)
  system.init_data()

  # plot

  fig = plt.figure(figsize=(12, 4))
  ax = fig.add_subplot()
  ax.grid()

  _, = ax.plot(times, system.states[:, 0], 'r', linewidth=1, label="x1") # x1 = q1
  _, = ax.plot(times, system.states[:, 1], 'g', linewidth=1, label="x2") # x2 = q2
  _, = ax.plot(times, system.states[:, 2], 'b', linewidth=1, label="x3") # x3 = q1dot
  _, = ax.plot(times, system.states[:, 3], 'k', linewidth=2, label="u") # x4 = q2dot
  ax.legend()

  # title = "r = theta1, b = theta2, green = amplitude (hilbert transform + denoise)"
  # title += "\n"
  # title += "pumped = %d, stable = %d, rise time = %.1fs, effort = %.1f" % (
  #     pumped, stable, rise_time, effort)

  modes = [
    "derivs_alphabetau",
  ]

  plt.title('Feedback linearization, mode = %s' % (modes[args.mode]))
  plt.ylabel("state")
  plt.xlabel("time (s)")
  plt.tight_layout()

  plt.show()