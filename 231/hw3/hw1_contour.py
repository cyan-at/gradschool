#!/usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
bound = 2
X1 = np.arange(-bound, bound + 0.1, 0.1)
X2 = np.arange(-bound, bound + 0.1, 0.1)

X1, X2 = np.meshgrid(X1, X2)

Z = X1**2 / 2 + X2**2 / 2 - X1**4 / 4

# Plot the surface.
surf = ax.plot_surface(
  X1,
  X2,
  Z,
  cmap=cm.coolwarm,
  alpha=0.25,
  linewidth=0,
  antialiased=False)

# Customize the z axis.
vmin = 0.0
vmax = 2.0
vn = 20
ax.set_zlim(vmin - 0.5, vmax)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.1, aspect=5)


# plot the fixed points
ax.scatter([0, 1, -1], [0, 0, 0], [0, 0, 0], marker='o')

# v_range = np.linspace(0, 0.5, 10)

v_range = np.linspace(vmin, vmax, vn)
# t = np.sqrt(2) / 2
# x = t ** 2 / 2 - t**4 / 4.0
# v_range = np.linspace(x - 0.1, x + 0.1, 11)
# print(v_range)
ax.contour(X1, X2, Z, v_range, zdir='z')



# plot contours / level sets
# print(X1)
# import ipdb; ipdb.set_trace();
# x1_range = X1[0]
# for v in v_range:
#   print(v)
#   x1_x2s = []
#   for x1 in x1_range:
#     x2 = np.sqrt(2 * (v + x1**4 / 4 - x1**2 / 2))
#     # print(np.abs(x2))

#     if (np.abs(x2) <= 2):
#       x1_x2s.append([x1, x2])
#       x1_x2s.append([x1, -x2])

  # ax.scatter(
  #   [x[0] for x in x1_x2s],
  #   [x[1] for x in x1_x2s],
  #   [v for x in x1_x2s],
  #   marker='*',
  #   alpha=1.0)

plt.title("V, level sets V = [%.3f, %.3f, %d samples]" % (vmin, vmax, vn))

plt.show()
