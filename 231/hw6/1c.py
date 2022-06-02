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

m = M = 1
g = G = 9.8
d = 1.7

# controller params
k1 = -24
k2 = -50
k3 = -35
k4 = -10

# initial state
x0 = np.array([np.pi / 6, np.pi / 3, 1, 2])

class System(object):
  def alpha(self, x):
    return F1*x[2] + F2**3*J1*N*x[3]/(J2**2*K) + F2**2*G*J1*N*d*M*cos(x[1])/(J2**2*K) + F2**2*J1*N*x[1]/J2**2 - F2**2*J1*x[0]/J2**2 + 2*F2*G*J1*N*d*M*x[3]*sin(x[1])/(J2*K) - 2*F2*J1*N*x[3]/J2 + F2*J1*x[2]/J2 + G**2*J1*N*d**2*M**2*sin(x[1])*cos(x[1])/(J2*K) - G*J1*N*d*M*x[3]**2*cos(x[1])/K + G*J1*N*d*M*x[1]*sin(x[1])/J2 - G*J1*N*d*M*cos(x[1])/J2 - G*J1*d*M*x[0]*sin(x[1])/J2 - J1*K*N*x[1]/J2 + J1*K*x[0]/J2 + K*x[1]/N - K*x[0]/N**2

  def beta(self, x):
    return J1 * J2 * N / K

  def tau(self, x):
    return np.array([
        x[1],
        x[3],
        -F2*x[3]/J2 - G*d*M*cos(x[1])/J2 - K*x[1]/J2 + K*x[0]/(J2*N),
        -F2*(-F2*x[3]/J2 - G*d*M*cos(x[1])/J2 - K*x[1]/J2 + K*x[0]/(J2*N))/J2 + (G*d*M*sin(x[1])/J2 - K/J2)*x[3] + K*x[2]/(J2*N),
      ])

  def tau_inv(self, z):
    return np.array([
      N*(F2*z[1] + G*d*M*cos(z[0]) + J2*z[2] + K*z[0])/K,
      z[0],
      N*(F2*z[2] - G*d*M*z[1]*sin(z[0]) + J2*z[3] + K*z[1])/K,
      z[1]])

  def f(self, x):
    return np.array([
      x[2],
      x[3],
      -F1*x[2]/J1 - K*x[1]/(J1*N) + K*x[0]/(J1*N**2),
      -F2*x[3]/J2 - G*d*M*cos(x[1])/J2 - K*x[1]/J2 + K*x[0]/(J2*N),
    ])

  def g(self, x):
    return np.array([
        0,
        0,
        1/J1,
        0
      ])

  ############################################################################

  def derivs_alphabetau(self, x, t):
    xdot = np.zeros_like(x)
    # state = [q1,  q2,  q1*,  q2*]
    # xdot =  [q1*, q2*, q1**, q2**]

    # feedback control
    z = self.tau(x)
    v = np.dot(np.array([k1, k2, k3, k4]), z)
    u = self.alpha(x) + self.beta(x) * v

    # plant
    xdot = self.f(x) + self.g(x) * u

    return xdot

  def derivs_z(self, z, t):
    zdot = np.zeros_like(z)

    zdot[0] = z[1]
    zdot[1] = z[2]
    zdot[2] = z[3]
    zdot[3] = np.dot(np.array([k1, k2, k3, k4]), z)

    return zdot

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
        v = np.dot(np.array([k1, k2, k3, k4]), z)
        self.states[i+1, self.initial_state.shape[0]] = self.alpha(x) + self.beta(x) * v

        # solve differential equation, take final result only
        state = integrate.odeint(
          self.derivs_alphabetau,
          state,
          self.sampletimes[i:i+2])[-1]
        self.states[i+1, :self.initial_state.shape[0]] = state

        #############################################

        i += 1

    elif self._args.mode == 1:
      print("derivs_z")

      z = self.tau(self.initial_state)
      while i < len(self.sampletimes) - 1:
        x = self.tau_inv(z)

        z = integrate.odeint(
          self.derivs_z,
          z,
          self.sampletimes[i:i+2])[-1]

        self.states[i+1, :self.initial_state.shape[0]] = self.tau_inv(z)

        v = np.dot(np.array([k1, k2, k3, k4]), z)
        self.states[i+1, self.initial_state.shape[0]] = self.alpha(x) + self.beta(x) * v

        i += 1

      '''
      while i < len(self.sampletimes) - 1:
        x = self.tau_inv(z)

        zdot = self.derivs_z(z, self.sampletimes[i:i+2])
        z += zdot * (self.sampletimes[i+1] - self.sampletimes[i])

        self.states[i+1, :self.initial_state.shape[0]] = self.tau_inv(z)

        v = np.dot(np.array([k1, k2, k3, k4]), z)
        self.states[i+1, self.initial_state.shape[0]] = self.alpha(x) + self.beta(x) * v

        i += 1
      '''

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
  _, = ax.plot(times, system.states[:, 3], 'm', linewidth=1, label="x4") # x4 = q2dot
  _, = ax.plot(times, system.states[:, 4], 'k', linewidth=2, label="u") # x4 = q2dot
  ax.legend()

  modes = [
    "derivs_alphabetau",
    "derivs_z",
  ]

  plt.title('Feedback linearization, mode = %s' % (modes[args.mode]))
  plt.ylabel("state")
  plt.xlabel("time (s)")
  plt.tight_layout()

  plt.show()