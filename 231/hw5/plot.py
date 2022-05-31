import numpy as np
import argparse
import matplotlib.pyplot as plt

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='')
  args = parser.parse_args()

  fig = plt.figure(1)
  ax = plt.subplot(111)

  x = np.linspace(-10, 10, 101)

  '''
  u_l = np.abs(-x)
  ax.plot(x, u_l, linewidth=1, color='b', label='u_l')

  u_fl = np.abs(x**3 - x)
  ax.plot(x, u_fl, linewidth=1, color='r', label='u_fl')
  '''
  
  u_l = -x
  ax.plot(x, u_l, linewidth=1, color='b', label='u_l')

  u_fl = x**3 - x
  ax.plot(x, u_fl, linewidth=1, color='r', label='u_fl')

  u_0 = x * 0 
  ax.plot(x, u_0, linewidth=1, color='g', label='u_0')

  u_s = x**3 - x * np.sqrt(x**4 + 1) 
  ax.plot(x, u_s, linewidth=1, color='m', label='u_s')

  a = 0
  b = 5
  c = 2
  plt.xlim(a-b, a+b)
  plt.ylim(c-b, c+b)
  ax.grid(True)
  ax.set_aspect('equal', adjustable='box')
  ax.set_title('controller output')
  ax.set_xlabel('x')
  ax.set_ylabel('u')
  fig.tight_layout()

  ax.legend()

  plt.show()