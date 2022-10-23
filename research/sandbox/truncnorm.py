#!/usr/bin/env python3

import argparse
import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt

import scipy.integrate as integrate
from scipy.stats import truncnorm
from scipy.stats import norm

import time

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  args = parser.parse_args()

  state_min = -10.0
  state_max = 10.0
  N = 1000

  T_t = 5. #Terminal time, 0-5

  epsilon=.001

  j1, j2, j3=3, 2, 1

  mu = 0.
  sigma = 2

  fig, ax = plt.subplots()

  a, b = (state_min - mu) / sigma, (state_max - mu) / sigma
  x = np.linspace(
    truncnorm.ppf(0.01, a, b),
    truncnorm.ppf(0.99, a, b), 100)

  x = np.linspace(-15, 15, 1000)

  trunc_pdf = truncnorm.pdf(x, a, b)

  # norm_pdf = np.random.normal(mu, sigma, 2000)
  # count, bins, ignored = plt.hist(norm_pdf, 30, density=True)
  norm_pdf = norm.pdf(x, mu, sigma)

  ax.plot(x, trunc_pdf,
    'r-', lw=5, alpha=0.6, label='truncnorm pdf')

  ax.plot(x, norm_pdf,
    'b-', lw=1, alpha=1., label='norm pdf')

  # plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
  #   np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
  #   linewidth=2, color='b')

  plt.show()
