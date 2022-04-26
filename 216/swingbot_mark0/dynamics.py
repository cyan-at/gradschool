#!/usr/bin/env python

import sys

import numpy as np
import scipy.integrate as integrate

if "darwin" == sys.platform:
  import matplotlib
  matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import time

import argparse

# from wall_borg.interaction.utils import globals_check_decorator

class NpRowRB(object):
    # not sparse
    def __init__(self, data):
        self._data = data

    def add(self, new_data):
        # slicing is the fastest way
        self._data[:-1, :] = self._data[1:, :]
        self._data[-1, :] = new_data

    def add_batch(self, new_data_batch):
        # slicing is the fastest way
        n = new_data_batch.shape[0]
        self._data[:-n, :] = self._data[n:, :]
        self._data[-n :] = new_data_batch

def globals_check_decorator(gs, globals_to_assert):
    def decorator(func):  # func should return void
        def wrapper(*args, **kwargs):
            for g in globals_to_assert:
                assert(g in gs)

            return func(*args, **kwargs)

        return wrapper
    return decorator

class Swingbot(object):
  """Swingbot Class

  init_state is [theta1, omega1, theta2, omega2] in degrees,
  where theta1, omega1 is the angular position and velocity of the first
  pendulum arm, and theta2, omega2 is that of the second pendulum arm
  """
  def __init__(self,
    init_state = [120, 0, -20, 0],
    L1=10.0,  # length of pendulum 1 in m
    L2=1.0,  # length of pendulum 2 in m
    M1=10.0,  # mass of pendulum 1 in kg
    M2=1.0,  # mass of pendulum 2 in kg
    G=9.8,  # acceleration due to gravity, in m/s^2
    B=1.0, # damping
    origin=(0, 0),
    buffer = 1000):
    self.params = (L1, L2, M1, M2, G, B)
    self.origin = origin
    self.time_elapsed = 0

    self.init_state = np.asarray(init_state, dtype='float')
    self.state = self.init_state * np.pi / 180.

    self.last_state = init_state

    # track sensed amplitude
    self.buffer = buffer
    self.amps = NpRowRB(np.zeros((buffer, 2)))
    self.amps.add([0, 0])

    self.ut = 0.0

  def position(self):
    """compute the current x,y positions of the pendulum arms"""
    (L1, L2, M1, M2, G, B) = self.params

    # x = np.cumsum([self.origin[0],
    #   L1 * np.sin(self.state[0]),
    #   L2 * np.sin(self.state[2])])
    # y = np.cumsum([self.origin[1],
    #   -L1 * np.cos(self.state[0]),
    #   -L2 * np.cos(self.state[2])])

    x = np.cumsum([self.origin[0],
      L1 * np.sin(self.state[0])])
    y = np.cumsum([self.origin[1],
      -L1 * np.cos(self.state[0])])
    return (x, y)

  def robot_pos(self):
    (L1, L2, M1, M2, G, B) = self.params

    x = np.cumsum([
      L1 * np.sin(self.state[0]),
      L2 * np.sin(self.state[2])
    ])
    y = np.cumsum([
      -L1 * np.cos(self.state[0]),
      -L2 * np.cos(self.state[2])
    ])
    return (x, y)

  def amplitude_history(self):
    return list(range(self.buffer)), self.amps._data[:, 1]

  def energy(self):
    """compute the energy of the current state"""
    (L1, L2, M1, M2, G, B) = self.params

    x = np.cumsum([L1 * np.sin(self.state[0]),
      L2 * np.sin(self.state[2])])
    y = np.cumsum([-L1 * np.cos(self.state[0]),
      -L2 * np.cos(self.state[2])])
    vx = np.cumsum([L1 * self.state[1] * np.cos(self.state[0]),
      L2 * self.state[3] * np.cos(self.state[2])])
    vy = np.cumsum([L1 * self.state[1] * np.sin(self.state[0]),
      L2 * self.state[3] * np.sin(self.state[2])])

    U = G * (M1 * y[0] + M2 * y[1])
    K = 0.5 * (M1 * np.dot(vx, vx) + M2 * np.dot(vy, vy))

    return U + K

  def a_matrix(self, state):
    '''
      nonlinear function
    '''

    # deserialize
    (L1, L2, M1, M2, G, B) = self.params
    c = -G
    a = L1

    # finish
    return c / a * np.sin(state[0]) - B / a * state[1]

  def output_cd_and_control(self, reference, state, t):
    """
      we look at theta
    """
    c = 10.0
    y = c * state[0] # D = 0
    u = reference - y
    u = 0
    return u

  def loop(self, state, t):
    """
      state-space nonlinear plant dynamics:
      x = [
        x1 = theta
        x2 = theta*
      ]

      x* = [ 
        theta* = x2
        theta** = c / a * sin(x1) - b / a * x2 + 1 / a * u(t)
      ]

      y = Cx + Du
    """
    # deserialize
    (L1, L2, M1, M2, G, B) = self.params
    c = -M1 * G * L1
    a = M1*L1**2

    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    # # dispatch
    # # track amplitude
    # if state[1] < 0:
    #   # 'left'

    # normalized = state[0] - 2*np.pi * np.floor((state[0] + np.pi) / 2*np.pi)


    self.amps.add([self.amps._data[-1, 0] + 1, state[0]])

    self.last_state = state

    # classical controls approach:
    reference = np.pi / 4 # semantically, desired amplitude of swing
    # self.ut = self.output_cd_and_control(reference, state, t)
    dydx[1] = self.a_matrix(state) + 1 * self.ut
    # self.ut = 0.0

    # this is a controlled revolute joint
    dydx[2] = 0
    dydx[3] = 0

    return dydx

  # def loop(self, state, t):
  #   """
  #     compute the derivative of the given state
  #     for a double pendulum undamped
  #   """
  #   (L1, L2, M1, M2, G, B) = self.params

  #   dydx = np.zeros_like(state)
  #   dydx[0] = state[1]
  #   dydx[2] = state[3]

  #   cos_delta = np.cos(state[2] - state[0])
  #   sin_delta = np.sin(state[2] - state[0])

  #   den1 = (M1 + M2) * L1 - M2 * L1 * cos_delta * cos_delta
  #   dydx[1] = (M2 * L1 * state[1] * state[1] * sin_delta * cos_delta
  #       + M2 * G * np.sin(state[2]) * cos_delta
  #       + M2 * L2 * state[3] * state[3] * sin_delta
  #       - (M1 + M2) * G * np.sin(state[0])) / den1

  #   den2 = (L2 / L1) * den1
  #   dydx[3] = (-M2 * L2 * state[3] * state[3] * sin_delta * cos_delta
  #       + (M1 + M2) * G * np.sin(state[0]) * cos_delta
  #       - (M1 + M2) * L1 * state[1] * state[1] * sin_delta
  #       - (M1 + M2) * G * np.sin(state[2])) / den2

  #   return dydx

  def step(self, dt):
    """execute one time step of length dt and update state"""
    self.state = integrate.odeint(
      self.loop, self.state, [0, dt])[1]
    self.time_elapsed += dt

@globals_check_decorator(globals(),
  ['line', 'line2', 'time_text', 'theta_text', 'ut_text'])
def init():
  """initialize animation"""
  time_text.set_text('')
  theta_text.set_text('')
  ut_text.set_text('')

  line.set_data(*pendulum.position())
  line2.set_data(*pendulum.robot_pos())

  return line, line2, time_text, theta_text, ut_text

# @globals_check_decorator(globals(),
#   ['pendulum', 'dt', 'line', 'line2', 'time_text', 'theta_text', 'ut_text'])
def animate(i, line, line2):
  """perform animation step"""
  global pendulum, dt
  pendulum.step(dt)

  time_text.set_text('time = %.1f' % pendulum.time_elapsed)
  theta_text.set_text('theta1 = %.3f rad' % pendulum.state[0])
  ut_text.set_text('ut = %.3f' % pendulum.ut)

  line.set_data(*pendulum.position())
  line2.set_data(*pendulum.robot_pos())

  return line, line2, time_text, theta_text, ut_text


@globals_check_decorator(globals(),
  ['line3', 'time_text', 'theta_text', 'ut_text'])
def init2():
  line3.set_data(*pendulum.amplitude_history())

  return line3,

@globals_check_decorator(globals(),
  ['pendulum', 'dt', 'line3', 'time_text', 'theta_text', 'ut_text'])
def animate2(i, line):
  """perform animation step"""
  # global pendulum, dt

  line3.set_data(*pendulum.amplitude_history())

  return line3,


def plot_conclusion(ax, xlim, ylim, title):
  ax.set_xlim(xlim[0], xlim[1])
  ax.set_ylim(ylim[0], ylim[1])
  ax.grid(True)
  # ax.gca().set_aspect(
  #     'equal', adjustable='box')
  ax.set_title(title)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description='')
  parser.add_argument('--buffer',
      type=int, default=5000, help='')
  args = parser.parse_args()

  pendulum = Swingbot(
    [90., 0.0, 0., 0.0], buffer = args.buffer)
  dt = 1./30 # 30 fps

  #------------------------------------------------------------
  # set up figure and animation

  fig = plt.figure()
  ax = plt.subplot(111)
  bounds = 15
  plot_conclusion(ax,
    [-bounds, bounds], [-bounds, bounds], "")

  time_text = ax.text(
    0.02, 0.95, '',
    transform=ax.transAxes)
  theta_text = ax.text(
    0.02, 0.90, '',
    transform=ax.transAxes)
  ut_text = ax.text(
    0.02, 0.85, '',
    transform=ax.transAxes)

  line, = ax.plot(
    [], [], 'o-', lw=2)
  line2, = ax.plot(
    [], [], 'ro-', lw=2)

  t0 = time.time()
  # animate(0)
  t1 = time.time()

  interval = 1000 * dt - (t1 - t0)
  ani = animation.FuncAnimation(
    fig, animate,
    frames=300,
    interval=interval,
    blit=True,
    init_func=init,
    fargs=[line, line2])

  def press(event):
    # if event.key == 'enter':
    #   print "entered"
    if event.key == 'u':
      print("u!")
      pendulum.ut += 2.0
    elif event.key == 'i':
      print("i!")
      pendulum.ut -= 2.0
  fig.canvas.mpl_connect(
      'key_press_event', press)

  fig2 = plt.figure()
  ax2 = plt.subplot(111)
  plot_conclusion(ax2,
    [0, args.buffer], [-2*np.pi, 2*np.pi], "")


  line3, = ax2.plot(
    [], [], 'b-', lw=2)

  ani2 = animation.FuncAnimation(
    fig2, animate2,
    frames=300,
    interval=interval,
    blit=True,
    init_func=init2,
    fargs=[line3])

  plt.show()

