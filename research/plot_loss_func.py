#!/usr/bin/env python3

import argparse
import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt

import scipy.integrate as integrate

import time, os, sys

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--t_stop', type=int, default=10, help='')
  parser.add_argument('--dt', type=float, default=0.02, help='')
  parser.add_argument('--lossdat', type=str, required=True)

  args = parser.parse_args()

  loss_loaded = np.genfromtxt(args.lossdat)

  colors = 'rgbymck'

  fig, ax = plt.subplots()

  epoch = loss_loaded[:, 0]
  num_cols = loss_loaded.shape[1]
  print("num_cols", num_cols)

  num_cols = int((num_cols - 1) / 2) + 1

  for i in range(1, num_cols):
    data = loss_loaded[:, i]
    data = np.where(data < 1e-10, 10, data)
    ax.plot(epoch, data,
      color=colors[i % len(colors)],
      lw=1,
      label='eq %d' % (i))

  ax.grid()
  ax.legend(loc="lower left")
  ax.set_title('training error/residual plots')
  ax.set_yscale('log')
  ax.set_xscale('log')

  plot_fname = "%s/loss.png" % (os.path.abspath("./"))
  plt.savefig(plot_fname, dpi=300)
  print("saved plot")

  plt.show()