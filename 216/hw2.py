#!/usr/bin/env python3

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

def grashof(a, b, c, d):
  sorted_abcd = sorted([a, b, d, c])

  s = sorted_abcd[0]
  l = sorted_abcd[-1]

  p = sorted_abcd[1]
  q = sorted_abcd[2]

  if (s + l) <= (p + q):
    return True
  else:
    return False

def outputposition(input_phi, a, b, c, d):
  k1 = d / c
  k2 = d / a
  k3 = (a**2 - b**2 + c**2 + d**2) / (2*a*c)

  A = np.sin(input_phi)
  B = np.cos(input_phi) - k2
  C = k1 * np.cos(input_phi) - k3

  root = np.sqrt(A**2 + B**2 - C**2)

  psi1 = 2 * np.arctan2(A - np.sqrt(A**2 + B**2 - C**2), B + C)

  num = np.sin(input_phi - psi1) - k1 * np.sin(input_phi)
  den = np.sin(input_phi - psi1) - k2 * np.sin(input_phi)
  ratio1 = num / den

  psi2 = 2 * np.arctan2(A + np.sqrt(A**2 + B**2 - C**2), B + C)

  num = np.sin(input_phi - psi2) - k1 * np.sin(input_phi)
  den = np.sin(input_phi - psi2) - k2 * np.sin(input_phi)
  ratio2 = num / den

  return psi1, psi2, ratio1, ratio2

def deg2rad(deg):
  return np.pi * deg / 180.0

def rad2deg(rad):
  return rad / np.pi * 180.0

def normalize(rad):
  # rad = rad % 2*np.pi
  # rad = (rad + 2*np.pi) % 2*np.pi
  # if (rad > np.pi):
  #   rad -= 2*np.pi

  if (rad < -np.pi):
    rad += 2*np.pi
  elif (rad > np.pi):
    rad -= 2*np.pi

  return rad

# problem 1
print("problem 1")
d = 150
a = 0.6 * d
b = 1.4 * d
c = 0.5 * d

print(grashof(a, b, d, c))

# from the diagram
# it looks like as phi DEcreases
# psi INcreases
# and vice-versa
# as phi INcreases
# psi DEcreases
# print(outputposition(deg2rad(135), a, b, d, c))
# print(outputposition(deg2rad(90), a, b, d, c))
# print(outputposition(deg2rad(45), a, b, d, c))

phi = np.linspace(135, 45, 181)
psi = [outputposition(deg2rad(x), a, b, c, d) for x in phi]
psi1 = [normalize(x[0]) for x in psi]
psi2 = [normalize(x[1]) for x in psi]

cand1, cand2, ratio1, ratio2 = outputposition(deg2rad(90), a, b, c, d)
s = "psi(phi=90deg)=%.3f, ratio=%.3f" % (normalize(rad2deg(cand2)), ratio2)
print(s)

fig, ax = plt.subplots()
ax.grid()

'''
def data_gen():
  t = data_gen.t
  cnt = 0
  while t < len(psi)-1:
    t += 1
    yield t, psi1[t-1], psi2[t-1]
data_gen.t = 0

xdata = []
data1, data2 = [], []

l1, = ax.plot([], [], lw=1, color='b', label='psi1')
l2, = ax.plot([], [], lw=1, color='r', label='psi2')


def run(data):
  # update the data
  t, d1, d2 = data
  xdata.append(t)

  data1.append(d1)
  data2.append(d2)

  l1.set_data(xdata, data1)
  l2.set_data(xdata, data2)

  return l1, l2

ani = animation.FuncAnimation(
  fig, run, data_gen, blit=True, interval=10,
  repeat=False)
'''

'''
xdata = [x for x in range(len(psi))]
l1, = ax.plot(xdata, psi1, lw=1, color='b', label='psi1')
l2, = ax.plot(xdata, psi2, lw=1, color='r', label='psi2')

sc = ax.scatter(
  [90, 90],
  [normalize(cand1), normalize(cand2)],
  c = [1, 1],
  label=s)

ax.set_xlabel('phi')
ax.set_ylabel('psi')
ax.set_ylim(-2*np.pi, 2*np.pi)
ax.set_xlim(0, len(psi))
ax.legend()
plt.show()
'''

print("problem 2a")

a = 2
b = 3
c = 2.5
d = 4

print(grashof(a, b, d, c))

phi = np.linspace(135, 45, 181)
psi = [outputposition(deg2rad(x), a, b, c, d) for x in phi]
psi1 = [normalize(x[0]) for x in psi]
psi2 = [normalize(x[1]) for x in psi]

cand1, cand2, ratio1, ratio2 = outputposition(deg2rad(90), a, b, c, d)
s = "psi(phi=90deg)=%.3f, ratio=%.3f" % (rad2deg(normalize(cand1)), ratio1)
print(s)

'''
xdata = [x for x in range(len(psi))]
l1, = ax.plot(xdata, psi1, lw=1, color='b', label='psi1')
l2, = ax.plot(xdata, psi2, lw=1, color='r', label='psi2')

sc = ax.scatter(
  [90, 90],
  [normalize(cand1), normalize(cand2)],
  c = [1, 1],
  label=s)

ax.set_xlabel('phi')
ax.set_ylabel('psi')
ax.set_ylim(-2*np.pi, 2*np.pi)
ax.set_xlim(0, len(psi))
ax.legend()
plt.show()
'''

print("problem 2b")

a = 1.5
b = 3
c = 2.5
d = 3.5

print(grashof(a, b, d, c))

phi = np.linspace(135, 45, 181)
psi = [outputposition(deg2rad(x), a, b, c, d) for x in phi]
psi1 = [normalize(x[0]) for x in psi]
psi2 = [normalize(x[1]) for x in psi]

cand1, cand2, ratio1, ratio2 = outputposition(deg2rad(90), a, b, c, d)
s = "psi(phi=90deg)=%.3f, ratio=%.3f" % (rad2deg(normalize(cand1)), ratio1)
print(s)

xdata = [x for x in range(len(psi))]
l1, = ax.plot(xdata, psi1, lw=1, color='b', label='psi1')
l2, = ax.plot(xdata, psi2, lw=1, color='r', label='psi2')

sc = ax.scatter(
  [90, 90],
  [normalize(cand1), normalize(cand2)],
  c = [1, 1],
  label=s)

ax.set_xlabel('phi')
ax.set_ylabel('psi')
ax.set_ylim(-2*np.pi, 2*np.pi)
ax.set_xlim(0, len(psi))
ax.legend()
plt.show()