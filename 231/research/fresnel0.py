#!/usr/bin/env python3

import argparse
import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt

import scipy.integrate as integrate

import time

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

def fresnel_s(state, t):
  return np.sin(t**2)

def fresnel_c(state, t):
  return np.cos(t**2)

def S_helper(x, dx=0.02):
  xs = np.arange(0, x, dx)
  trapz_x = np.cumsum([dx]*len(xs))
  return np.trapz(np.sin(xs**2), x=trapz_x)

def S(x, dx=0.02):
  '''
  state = 0
  i = 0
  while i < x:
    state = integrate.odeint(
      fresnel_s,
      state,
      [i, i+dx])[-1]
    i += dx
  return state
  '''
  if type(x) == np.ndarray:
    print("S, array")
    return np.array([S_helper(x_, dx) for x_ in x])
  else:
    return S_helper(x, dx)

def C_helper(x, dx=0.02):
  xs = np.arange(0, x, dx)
  trapz_x = np.cumsum([dx]*len(xs))
  return np.trapz(np.cos(xs**2), x=trapz_x)

def C(x, dx=0.02):
  '''
  state = 0
  i = 0
  while i < x:
    state = integrate.odeint(
      fresnel_c,
      state,
      [i, i+dx])[-1]
    i += dx
  return state
  '''
  if type(x) == np.ndarray:
    print("C, array")
    return np.array([C_helper(x_, dx) for x_ in x])
  else:
    return C_helper(x, dx)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--t_stop', type=int, default=10, help='')
  parser.add_argument('--dt', type=float, default=0.02, help='')

  args = parser.parse_args()

  state = np.array([0.0]*2)
  times = np.arange(0, args.t_stop, args.dt)
  states = np.zeros((len(times), state.shape[0]))

  # i = 0
  # while i < len(times) - 1:
  #   # solve differential equation, take final result only
  #   state = integrate.odeint(
  #     fresnel,
  #     state,
  #     times[i:i+2])[-1]
  #   states[i+1, :] = state
  #   i += 1
  states[:, S_INDEX] = S(times)
  states[:, C_INDEX] = C(times)
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

  sc = ax1.scatter(
    [states[-1, 0]],
    [states[-1, 1]],
    c = [1])

  start = time.time()
  test_c = C(args.t_stop, args.dt)
  test_s = S(args.t_stop, args.dt)
  sc = ax1.scatter(
    [test_c],
    [test_s],
    c = [3])
  print("odeint dt: %.3f", time.time() - start)

  # trapz is the fastest way
  start = time.time()
  trapz_x = np.cumsum([args.dt]*len(times))
  test_s2 = np.trapz([np.sin(i**2) for i in times], x=trapz_x)
  test_c2 = np.trapz([np.cos(i**2) for i in times], x=trapz_x)
  print(test_c2)

  print("trapz dt: %.3f", time.time() - start)

  sc = ax1.scatter(
    [test_c2],
    [test_s2],
    c = [5])

  plt.show()