#!/usr/bin/env python3

'''
https://www.cs.cmu.edu/~hgeyer/Publications/Geyer05PhDThesis.pdf
http://underactuated.mit.edu/simple_legs.html#example5
file:///home/cyan3/Dev/jim/gradschool/216/4_Supplementary%20materials.pdf
file:///home/cyan3/Dev/jim/gradschool/216/4_1_Legs.pdf
https://github.com/RobotLocomotion/drake/blob/88fea982f7e7c10d1a809cf73c1c90a14719b8ba/systems/analysis/simulator.cc
file:///home/cyan3/Dev/jim/gradschool/216/dual_slip.pdf

intuition

k=200
too stiff, so energy is lost quickly and robot stops hopping 'rightward'

k=175
loose enough, marches right but no way to control for speed

if k is loose and energy stays in the system, but control angles are not set correctly
x will converge and system starts to 'bounce-in-place'

the control part in 'takeoff' is how we 'introduce' energy into the system
and the dynamics 'stiffness' remove energy from the system.
it's actually not an explicit stiffness alone, but the angle of takeoff and touchdown that implicitly damp and introduce energy into the system
so it is about regulating and maintaining the amount of energy into the system, at the right timing
'''

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

class MatplotlibCamera(object):
  def __init__(self, center, dims, ax):
    self.center = center
    self.dims = dims
    self.ax = ax

    self.x_bounds = [self.center[0] - dims[0], self.center[0] + dims[0]]

  def update_cb(self, data):
    if (data[0] < self.x_bounds[0] or data[0] > self.x_bounds[1]):
      self.center[0] = data[0] + self.dims[0]
      print('self.center[0]', self.center[0])
      self.x_bounds = [self.center[0] - dims[0], self.center[0] + dims[0]]
      self.ax.set_xlim(self.x_bounds)

      xticks = np.linspace(self.x_bounds[0], self.x_bounds[1], 7)
      self.ax.set_xticks(xticks)
      pass

class SimpleSLIP(object):
  def foot_height(self, state):
    '''
    render_state plots from the feet up to mass
    '''
    return state[Y]

  def dynamx_flying(self, state, t):
    statedot = xdot = np.zeros_like(state)

    statedot[X] = state[XDOT]
    statedot[Y] = state[YDOT]
    statedot[XDOT] = 0
    statedot[YDOT] = -G

    return statedot

  def takeoff(self, state, t):
    '''
    y = r * cos(theta)
    x = -r * sin(theta)

    x.diff(t)
    y.diff(t)
    '''
    state[Y] = 0 # L0 * np.cos(state[THETA])
    state[XDOT] = -state[RDOT] * np.sin(state[THETA]) - L0 * state[THETADOT] * np.cos(state[THETA])
    state[YDOT] = state[RDOT] * np.cos(state[THETA]) - L0 * state[THETADOT] * np.sin(state[THETA])

    # Update theta to commanded leg angle.
    # this is the 'controller'
    xdelta1 = L0 * np.sin(-state[THETA])

    # these are tuned for initial state of -10
    K1 = 8.05
    K2 = -11

    if state[THETA] < 0:
      # taking off 'right', so angle leg 'right' / +
      # we don't want too high otherwise we bounce 'left' too far
      # too low and it will not 'damp' the rightward acceleration enough
      state[THETA] = min(-state[THETA], 20 / 180 * np.pi)

      state[THETA] = (K1*state[XDOT] + (1-state[YDOT])) / 180 * np.pi
      # adding the ydot is key, the higher it is, the higher you'll bounce
      # so the less you need to raise the leg. and the lower you'll bounce
      # the more 'stumbling you are', so raise your leg to damp yourself
    else:
      # taking off 'left', from a bounce back
      # so angle leg 'left' / -
      state[THETA] = max(-state[THETA], -20 / 180 * np.pi) 

      state[THETA] = (K2*state[XDOT]) / 180 * np.pi
      print("hi?") # this doesn't seem to ever run, unless bouncing straight up/down

    # because we render [X] = feet
    # to keep the mass in the same place
    # we need to shift the feet by the theta we shed
    # and the new theta we acquire
    state[X] += (xdelta1 + L0 * np.sin(state[THETA])) 

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

    return state

  def dynamx_stance(self, state, t):
    '''
    from hw1_derivation.py
    '''
    statedot = xdot = np.zeros_like(state)

    statedot[R] = state[RDOT]
    statedot[THETA] = state[THETADOT]
    statedot[RDOT] = (-G*M*cos(state[THETA]) + k*(L0 - state[R]) + M*state[R]*state[THETADOT]**2)/M
    statedot[THETADOT] = (G*sin(state[THETA]) - 2*state[RDOT]*state[THETADOT])/state[R]

    statedot[X] = state[XDOT]
    statedot[Y] = state[YDOT]

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

    self.render_cb = None

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

  def init_plot(self, fig, ax, texts, camera):
    # ground
    self.ground, = ax.plot([-50, 50], [0, 0], "k")

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

    self.trace, = ax.plot([], [], '.-', lw=1, ms=2)
    self.history_x = deque(maxlen=self._args.history)
    self.history_y = deque(maxlen=self._args.history)

    self.render_cb = camera.update_cb

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

    return g_world_head

  def render(self, i):
    state = self.states[i]
    # x,y,x*,y*,r,theta,r*,theta*

    g_world_head = self.render_state(state)

    if i == 0:
      self.history_x.clear()
      self.history_y.clear()

    self.history_x.appendleft(g_world_head[0, 2])
    self.history_y.appendleft(g_world_head[1, 2])
    self.trace.set_data(self.history_x, self.history_y)

    if (self.render_cb is not None):
      self.render_cb(state)

    return *self.leg_line, self.hip_fill[0], self.trace, self.ground

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
    default="175, 1, 1.41, 9.8",
    help='k, L0, M, G')
  parser.add_argument('--initial',
    type=str,
    default="-50,0.5,0,0,1,-10,0,0",
    help="x,y,x*,y*,r,theta,r*,theta*")
  # you can also start in stancing mode
  # there is less energy in the system
  # so it doesn't 'bounce' as much

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
    fig.tight_layout()

    # a 'viewport'
    center = [initial_state[0],0]
    dims = [6, 2]
    ax = fig.add_subplot(
      xlim=(center[0] - dims[0], center[0] + dims[0]),
      ylim=(center[1] - dims[1], center[1] + dims[1]))
    ax.set_aspect('equal')
    ax.grid()
    # ax.set_adjustable('box')

    camera = MatplotlibCamera(center, dims, ax)

    system.init_plot(fig, ax, [], camera)

    ani = animation.FuncAnimation(
      fig,
      system.render,
      system.data_gen,
      interval=args.dt*1000/args.playback,
      blit=True)

    plt.show()