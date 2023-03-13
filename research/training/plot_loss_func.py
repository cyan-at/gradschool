#!/usr/bin/env python3

import argparse
import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt

import scipy.integrate as integrate

import time, os, sys
plt.rcParams['text.usetex'] = True

class Counter(object):
  def __init__(self):
    self.count = 0

  def on_press_saveplot(self, event, png_name):
    print('press', event.key)
    sys.stdout.flush()
    if event.key == 'x':
      # visible = xl.get_visible()
      # xl.set_visible(not visible)
      # fig.canvas.draw()

      fname = png_name.replace(".png", "_%d.png" % (
        self.count))

      plt.savefig(
        fname,
        dpi=500,
        bbox_inches='tight')
      print("saved figure", fname)

      self.count += 1

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--modelpt', type=str, required=True, help='')
  parser.add_argument('--lossdat', type=str, required=True)

  args = parser.parse_args()

  loss_loaded = np.genfromtxt(args.lossdat)

  colors = 'rgbymck'

  fig, ax = plt.subplots()

  epoch = loss_loaded[:, 0]
  num_cols = loss_loaded.shape[1]
  print("num_cols", num_cols)

  num_cols = int((num_cols - 1) / 2) + 1

  label_lookup = {
    1 : 'Controlled Fokker-Planck PDE',
    2 : 'HJB PDE',
  }

  for i in range(1, num_cols):
    data = loss_loaded[:, i]
    data = np.where(data < 1e-10, 10, data)

    lbl = 'eq %d' % (i)
    if i == num_cols - 2:
      lbl = r'$\rho_0$ boundary condition'
      data /= 5.0
      data -= (10*0.01)
    elif i == num_cols - 1:
      lbl = r'$\rho_T$ boundary condition'
      data /= 5.0
      data -= (10*0.01)
    elif i in label_lookup:
      lbl = label_lookup[i]

    ax.plot(epoch, data,
      color=colors[i % len(colors)],
      lw=1,
      label=lbl)

  ax.grid()
  ax.legend(loc="upper right", frameon=False)
  # ax.set_title('training error/residual plots, %d epochs' % (len(epoch)))

  ax.set_yscale('log')
  ax.set_ylabel('PINN residuals')

  ax.set_xlabel('Epoch')
  ax.set_xscale('log')

  ax.yaxis.set_label_position("right")
  ax.yaxis.tick_right()

  # plot_fname = "%s/loss.png" % (os.path.abspath("./"))
  # plt.savefig(plot_fname, dpi=300)
  # print("saved plot")

  c = Counter()
  fig.canvas.mpl_connect('key_press_event', lambda e: c.on_press_saveplot(e,
          '%s_loss.png'  %(
              args.modelpt.replace(".pt", ""),
          )
      )
  )

  plt.show()