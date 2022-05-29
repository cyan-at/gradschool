#!/usr/bin/env python3

import sympy
from sympy import *
from sympy.physics.mechanics import *
sympy.init_printing()

from sympy_utils import *

# stance phase

t = Symbol('t')

m, l0, k, g, torsol =\
  symbols('m l0 k g torsol', nonzero=True, positive=True)

rf, thetaf, rh, thetah, phif, phih =\
  dynamicsymbols('rf thetaf rh thetah phif phih') # state vars
# theta(f, h) = angle w.r.t. 'vertical line'
# phi(f, h) = angle w.r.t. 'horizontal'

#################################################################
'''
primitives
'''

yh = rh * cos(thetah)
xh = -rh * sin(thetah)

# yf = rf * cos(thetaf)
# xf = -rf * sin(thetaf)

#################################################################
'''
flight dynamics for 'front' while 'hind' stance
'''

T_hindmotion = m / 2 * (xh**2 + yh**2)
T_frontpolar = m / 2 * (torsol**2 + torsol**2 * phih.diff(t)**2)
# rotational velocity from front mass swinging about 'waist'
T = T_hindmotion + T_frontpolar
U = m * g * (yh + torsol * sin(phih)) # total height 
L = simplify(T - U)

# in the above lagrange, we treat at each timestep
# the xh, yh, torsol, m, g as constants
# the only dynamics we see are phih
el1 = L.diff(phih) - (L.diff(phih.diff(t))).diff(t)

#################################################################

try:
  M, C, tau_g = pull_out_manipulator_matrices([el1], [phih.diff(t)], t)
except:
  print("failed to pull_out_manipulator_matrices")

