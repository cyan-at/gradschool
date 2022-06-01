#!/usr/bin/env python3

from numpy import sin, cos
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

m = 1
g = 9.8
d = 1.7

# controller params
k1 = -24
k2 = -50
k3 = -35
k4 = -10
k = np.array([k1, k2, k3, k4])

# initial state
x0 = np.array([np.pi / 6, np.pi / 3, 1, 2])
Q1 = 0
Q2 = 1
Q1DOT = 2
Q2DOT = 3

class System(object):
  def alpha(self, x):
    return (F2*J1*J2*K*N**2*x[Q1DOT] + J1*J2*N**3*(-F2*K + F2*d*g*m*sin(x[Q2]) - J2*d*g*m*cos(x[Q2])*x[Q2DOT])*x[Q2DOT] + J1*N**2*(F2**2 + J2*(-K + d*g*m*sin(x[Q2])))*(K*(N*x[Q2] - x[Q1]) + N*(F2*x[Q2DOT] + d*g*m*cos(x[Q2]))) + J2**2*K*(F1*N**2*x[Q1DOT] + K*(N*x[Q2] - x[Q1])))/(J2**2*K*N**2)

  def beta(self, x):
    return J1 * J2 * N / K

  def tau(self, x):
    l2f_lambda = (-F2*x[Q2DOT] - K * (x[Q2] - x[Q1] / N) - d * m * g * np.cos(x[Q2])) / J2
    l3f_lambda = -F2*(-F2*x[Q2DOT] - K*(x[Q2] - x[Q1]/N) - d*g*m*cos(x[Q2]))/J2**2 + K*x[Q1DOT]/(J2*N) + (-K + d*g*m*sin(x[Q2]))*x[Q2DOT]/J2
    return np.array([
        x[Q2],
        x[Q2DOT],
        l2f_lambda,
        l3f_lambda,
      ])

  def derivs(self, x, t):
    xdot = np.zeros_like(x)
    # state = [q1,  q2,  q1*,  q2*]
    # xdot =  [q1*, q2*, q1**, q2**]

    # feedback control
    z = self.tau(x)
    v = np.dot(k, z)
    u = self.alpha(x) + np.dot(self.beta(x), v)
    # u = 1 / self.beta(x) * (-self.alpha(x) + v)
    self.us.append(u)

    # plant
    xdot[0] = x[Q1DOT]
    xdot[1] = x[Q2DOT]

    xdot[2] = (-F1*x[Q1DOT] - K*(x[Q2] - x[Q1]/N)/N)/J1
    xdot[2] += (u / J1)

    xdot[3] = (-F2*x[Q2DOT] - K*(x[Q2] - x[Q1]/N) - d*g*m*cos(x[Q2]))/J2

    return xdot

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
    self.states = np.zeros((len(self.sampletimes), self.initial_state.shape[0]))

    self.states[0, :] = self.initial_state

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
    while i < len(self.sampletimes) - 1:
      # solve differential equation, take final result only
      state = integrate.odeint(
        self.derivs,
        state,
        self.sampletimes[i:i+2])[-1]
      self.states[i+1, :] = state

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
  _, = ax.plot(times, system.states[:, 3], 'm', linewidth=1, label="x4") # x4 = q2dot
  # _, = ax.plot(times, system.us, 'm', linewidth=2, label="u") # x4 = q2dot
  ax.legend()

  # title = "r = theta1, b = theta2, green = amplitude (hilbert transform + denoise)"
  # title += "\n"
  # title += "pumped = %d, stable = %d, rise time = %.1fs, effort = %.1f" % (
  #     pumped, stable, rise_time, effort)
  # plt.title(title)
  plt.ylabel("state")
  plt.xlabel("time (s)")
  plt.tight_layout()

  plt.show()