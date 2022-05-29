#!/usr/bin/env python3

from numpy import sin, cos
import numpy as np, math

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import scipy.integrate as integrate
from collections import deque
import argparse
import copy

import time

X, Y, XDOT, YDOT, R, THETA, RDOT, THETADOT = range(8)

def deserialize(args):
  global system_params
  global k, L0, M, G
  k, L0, M, G = system_params = [float(x) for x in args.system.strip().split(",")]

  global controller_params
  controller_params = np.radians([float(x) for x in args.control1.strip().split(",")])
  global alpha
  alpha = controller_params[0]

  initial_state = np.array([float(x) for x in args.initial.split(',')])
  initial_state[THETA] = np.radians(initial_state[THETA])

  return system_params, controller_params, initial_state

def two_d_make_x_y_theta_hom(x, y, theta):
  hom = np.eye(3)

  theta = theta % (2 * np.pi)
  # 2019-08-02 parentheses!!!

  hom[0, 0] = np.cos(theta)
  hom[0, 1] = -np.sin(theta)
  hom[1, 0] = np.sin(theta)
  hom[1, 1] = np.cos(theta)

  hom[0, 2] = x
  hom[1, 2] = y
  return hom

class SimpleSLIP(object):
  def foot_height(self, state):
    # theta = w.r.t vertical line
    # return state[Y] - L0 * np.cos(state[THETA])
    return state[Y]

  def dynamx_flying(self, state, t):
    statedot = xdot = np.zeros_like(state)

    statedot[X] = state[XDOT]
    statedot[Y] = state[YDOT]
    statedot[XDOT] = 0
    statedot[YDOT] = -G

    # statedot[R] = state[RDOT]
    # statedot[THETA] = state[THETADOT]
    # statedot[RDOT] = 0
    # statedot[THETADOT] = 0

    return statedot

  def takeoff(self, state, t):
    '''
    y = r * cos(theta)
    x = -r * sin(theta)

    x.diff(t)
    y.diff(t)
    '''
    print("TAKEOFF!")
    state[Y] = 0 # L0 * np.cos(state[THETA])
    state[XDOT] = -state[RDOT] * np.sin(state[THETA]) - L0 * state[THETADOT] * np.cos(state[THETA])
    state[YDOT] = state[RDOT] * np.cos(state[THETA]) - L0 * state[THETADOT] * np.sin(state[THETA])

    # Update theta to commanded leg angle.
    # this is the 'controller'
    xdelta1 = L0 * np.sin(-state[THETA])
    print("state[XDOT]", state[XDOT])
    if state[THETA] < 0:
      # taking off 'right', so angle leg 'right' / +
      # we don't want too high otherwise we bounce 'left' too far
      # too low and it will not 'damp' the rightward acceleration enough
      state[THETA] = min(-state[THETA], 20 / 180 * np.pi)
    else:
      # taking off 'left', from a bounce back
      # so angle leg 'left' / -
      state[THETA] = max(-state[THETA], -15 / 180 * np.pi) 
      # state[THETA] = min(state[THETA], -10 / 180 * np.pi) 
    state[X] += (xdelta1 + L0 * np.sin(state[THETA])) # because we render [X] = feet
    print("hello")

    state[THETADOT] = 0
    state[R] = L0
    state[RDOT] = 0

    return state

  def touchdown(self, state, t):
    # Update rdot and thetadot to match xdot and ydot, using
    # x = -r*sin(theta), z = r*cos(theta)
    #  => xdot = -rdot*s - r*c*thetadot, zdot = rdot*c - r*s*thetadot
    #  => xdot*c + zdot*s = -r*thetadot
    # r^2 = x^2 + z^2
    #  => 2r*rdot = 2x*xdot + 2z*zdot
    #  => rdot = -xdot*sin(theta) + zdot*cos(theta)
    # (matches Geyer05 Eq. 2.24 up to the symbol changes)
    state[R] = L0
    state[RDOT] = -np.sin(state[THETA]) * state[XDOT] + np.cos(state[THETA]) * state[YDOT]
    state[THETADOT] = -(state[XDOT] * cos(state[THETA]) + state[YDOT] * sin(state[THETA])) / L0

    state[XDOT] = 0
    state[YDOT] = 0

    print("touchdown!", state[THETADOT], state[RDOT])

    return state

  def dynamx_stance(self, state, t):
    statedot = xdot = np.zeros_like(state)

    statedot[R] = state[RDOT]
    statedot[THETA] = state[THETADOT]
    statedot[RDOT] = (-G*M*cos(state[THETA]) + k*(L0 - state[R]) + M*state[R]*state[THETADOT]**2)/M
    statedot[THETADOT] = (G*sin(state[THETA]) - 2*state[RDOT]*state[THETADOT])/state[R]
    # print("statedot[RDOT]", statedot[RDOT], statedot[THETADOT], k*(L0 - state[R]))

    statedot[X] = state[XDOT]
    statedot[Y] = state[YDOT]
    # statedot[XDOT] = -state[R] * sin(state[THETA]) * state[THETADOT] + cos(state[THETA]) * state[RDOT]
    # statedot[YDOT] = state[R] * cos(state[THETA]) * state[THETADOT] + sin(state[THETA]) * state[RDOT]

    # statedot[RDOT] = (k / M * (L0 - state[R]) +
    #              state[R] * state[THETADOT]**2 - G * np.cos(state[THETA]))
    # statedot[THETADOT] = (G / state[R] * np.sin(state[THETA]) -
    #                  2 * state[RDOT] * state[THETADOT] / state[R])

    # # statedot[RDOT] = (-G*M*cos(state[THETA]) + k*(L0 - state[R]) + M*state[R]*state[THETADOT]**2)/M
    # # statedot[THETADOT] = (G*sin(state[THETA]) - 2*state[RDOT]*state[THETADOT])/state[R]

    # statedot[X] = state[XDOT]
    # statedot[Y] = state[YDOT]
    # statedot[XDOT] = (-statedot[RDOT] * np.sin(state[THETA]) -
    #              2 * state[RDOT] * state[THETADOT] * np.cos(state[THETA]) +
    #              state[R] * state[THETADOT]**2 * np.sin(state[THETA]) -
    #              state[R] * statedot[THETADOT] * np.cos(state[THETA]))
    # statedot[YDOT] = (statedot[RDOT] * np.cos(state[THETA]) -
    #              2 * state[RDOT] * state[THETADOT] * np.sin(state[THETA]) -
    #              state[R] * statedot[THETADOT] * np.sin(state[THETA]) -
    #              state[R] * state[THETADOT]**2 * np.cos(state[THETA]))

    return statedot

  def __init__(self,
    args,
    system_params,
    controller_params,
    initial_state,
    sampletimes):
    self._args = args

    self.system_params = system_params
    self.controller_params = controller_params
    self.initial_state = initial_state
    self.sampletimes = sampletimes

    self._data_gen_cb = None
    self.states = []
    self.dynamx_handler = None

  def init_data(self):
    self.states = np.zeros((len(self.sampletimes), self.initial_state.shape[0]))

    i = 0
    state = self.initial_state

    # decide the initial dynamx_handler
    if self.foot_height(state) <= 0:
      print("stancing")
      self.dynamx_handler = self.dynamx_stance
    else:
      print("flying")
      self.dynamx_handler = self.dynamx_flying

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
      # time.sleep(self.sampletimes[i+1] - self.sampletimes[i])

      # solve differential equation, take final result only
      state = integrate.odeint(
        self.dynamx_handler,
        state,
        self.sampletimes[i:i+2])[-1]
      self.states[i+1, :] = state

      #############################################

      if self.dynamx_handler == self.dynamx_flying:
        if self.foot_height(state) <= 0:
          self.touchdown(state, self.sampletimes[i:i+2])
          self.dynamx_handler = self.dynamx_stance
      elif self.dynamx_handler == self.dynamx_stance:
        if state[R] > L0:
          self.takeoff(state, self.sampletimes[i:i+2])
          self.dynamx_handler = self.dynamx_flying

      #############################################

      i += 1
    print("done integrating")


    # import ipdb; ipdb.set_trace();

    # # extra data
    # aux = np.zeros((self.state.shape[0], 5))
    # self.state = np.hstack([self.state, aux])

    # self.state[:, 4] = L1*sin(self.state[:, 0]) # x1
    # self.state[:, 5] = -L1*cos(self.state[:, 0]) # y1
    # self.state[:, 6] = L2*sin(self.state[:, 2]) + self.state[:, 4] # x2
    # self.state[:, 7] = -L2*cos(self.state[:, 2]) + self.state[:, 5] # y2

    print("done")

  def init_plot(self, fig, ax, texts):
    # ground
    self.ground = ax.plot([-50, 50], [0, 0], "k")

    # leg
    # since theta = angle relative to +x axis
    # at theta = 0, it is 'lying down' from 0 to L0
    self.leg_line = [ax.plot([0, L0], [0, 0], "k")[0]]
    self.leg_data = [self.leg_line[0].get_xydata().T]

    # leg spring
    for i in range(1, 13):
      self.leg_line.append(
          ax.plot(
            .2 + .7 / 13 * np.array([i - 1, i]), # notches
            0.1 * np.array([np.sin((i - 1) * np.pi / 2.), np.sin(i * np.pi / 2.)]), # amplitude
            "k")[0])
      self.leg_data.append(self.leg_line[i].get_xydata().T)

    # mass / hip
    a = np.linspace(0, 2 * np.pi, 50)
    radius = 0.1
    self.hip_fill = ax.fill(
      radius * np.sin(a),
      radius * np.cos(a),
      zorder=1,
      edgecolor="k",
      facecolor=[.6, .6, .6])
    self.hip = copy.copy(self.hip_fill[0].get_path().vertices)

    self.render_state(self.initial_state)

    # self.trace, = ax.plot([], [], '.-', lw=1, ms=2)
    # self.history_x = deque(maxlen=self._args.history)
    # self.history_y = deque(maxlen=self._args.history)

  def data_gen(self):
    i = 0
    while True:
      i = (i + 1) % self.states.shape[0]

      if self._data_gen_cb is not None:
        self._data_gen_cb(i)

      yield i

  def render_state(self, state):
    g_world_foot = two_d_make_x_y_theta_hom(
      state[X],
      state[Y],
      0)

    g_foot_theta = two_d_make_x_y_theta_hom(
      0,
      0,
      state[THETA] + np.pi / 2) # in plot space, theta = w.r.t. x axis

    g_world_theta = np.dot(g_world_foot, g_foot_theta)

    g_theta_head = two_d_make_x_y_theta_hom(
      state[R],
      0,
      0) # in plot space, theta = w.r.t. x axis

    g_foot_head = np.dot(g_foot_theta, g_theta_head)

    g_world_head = np.dot(g_world_foot, g_foot_head)

    for i in range(13):
      temp = np.vstack([
        self.leg_data[i],
        np.array([1, 1])
      ]) # 3 x 2

      temp[0, :] = temp[0, :] * state[R] / L0

      g_world_leg = np.dot(g_world_theta, temp) # 3 x 2: by row: xs, ys, 1s

      self.leg_line[i].set_xdata(g_world_leg[0, :])
      self.leg_line[i].set_ydata(g_world_leg[1, :])

    self.hip_fill[0].get_path().vertices[:, 0] = g_world_head[0, 2] + self.hip[:, 0]
    self.hip_fill[0].get_path().vertices[:, 1] = g_world_head[1, 2] + self.hip[:, 1]

  def render(self, i):
    state = self.states[i]
    # x,y,x*,y*,r,theta,r*,theta*

    self.render_state(state)

    return *self.leg_line, self.hip_fill[0]

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description="")
  parser.add_argument('--playback', type=float, default=1.0, help='')
  parser.add_argument('--history', type=int, default=500, help='')
  parser.add_argument('--plot', type=str, default="sim", help='')
  parser.add_argument('--dt', type=float, default=0.02, help='')
  parser.add_argument('--t_stop', type=int, default=300, help='')

  parser.add_argument(
    '--system',
    type=str,
    default="190, 1, 1.25, 9.8",
    help='k, L0, M, G')
  parser.add_argument('--initial',
    type=str,
    default="0,0.5,0,0,1,-10,0,0",
    help="x,y,x*,y*,r,theta,r*,theta*")

  parser.add_argument(
      '--control1',
      type=str,
      default="90") # alpha = angle of attack

  args = parser.parse_args()
  system_params, controller_params, initial_state = deserialize(args)

  times = np.arange(0, args.t_stop, args.dt)
  system = SimpleSLIP(args,
    system_params,
    controller_params,
    initial_state,
    times)
  system.init_data()

  ####################################

  if args.plot == "sim":
    fig = plt.figure()

    # a 'viewport'
    center = [19,0]
    dims = [40, 2]
    ax = fig.add_subplot(
      xlim=(center[0] - dims[0], center[0] + dims[0]),
      ylim=(center[1] - dims[1], center[1] + dims[1]))
    ax.set_aspect('equal')
    ax.grid()

    system.init_plot(fig, ax, [])

    ani = animation.FuncAnimation(
        fig,
        system.render,
        system.data_gen,
        interval=args.dt*1000/args.playback,
        blit=True)

    plt.show()