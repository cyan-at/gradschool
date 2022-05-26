#!/usr/bin/env python3

import sympy
from sympy import *
from sympy.physics.mechanics import *
sympy.init_printing()

from sympy_utils import *

# stance phase

t = Symbol('t')

m, l0, a, k, g =\
  symbols('m l0 a k g', nonzero=True, positive=True)

r, theta, x, y, phi = dynamicsymbols('r theta x y phi') # state vars

rdot = r.diff(t)
thetadot = theta.diff(t)
xdot = x.diff(t)
ydot = y.diff(t)

T = m / 2 * (rdot**2 + r**2 * thetadot**2)
U = m*g*r*cos(theta) + k / 2 * (l0 - r)**2

L = simplify(T - U)

# state for stance involves r and theta

el1 = L.diff(r) - (L.diff(r.diff(t))).diff(t)
el2 = L.diff(theta) - (L.diff(theta.diff(t))).diff(t)

el1_cp = L.diff(r) - (L.diff(r.diff(t))).diff(t)
el2_cp = L.diff(theta) - (L.diff(theta.diff(t))).diff(t)

try:
  el1 = simplify_eq_with_assumptions(Eq(el1, 0)).lhs
  el2 = simplify_eq_with_assumptions(Eq(el2, 0)).lhs

  el1_cp = simplify_eq_with_assumptions(Eq(el1_cp, 0)).lhs
  el2_cp = simplify_eq_with_assumptions(Eq(el2_cp, 0)).lhs
except:
  pass

# mass-matrix M
# note that here the term right-multiplied is q** exactly
M11, el1 = pull_out_term(el1, rdot.diff(t))
M12, el1 = pull_out_term(el1, thetadot.diff(t))
M21, el2 = pull_out_term(el2, rdot.diff(t))
M22, el2 = pull_out_term(el2, thetadot.diff(t))
M = Matrix([[M11, M12], [M21, M22]])

C11, el1 = pull_out_term(el1, rdot, [rdot, thetadot])
C12, el1 = pull_out_term(el1, thetadot, [rdot, thetadot])
C21, el2 = pull_out_term(el2, rdot, [rdot, thetadot])
C22, el2 = pull_out_term(el2, thetadot, [rdot, thetadot])
C = Matrix([[C11, C12], [C21, C22]])

tau_g = Matrix([[el1], [el2]])

try:
  M2, C2, tau_g2 = pull_out_manipulator_matrices([el1_cp, el2_cp], [rdot, thetadot], t)
except:
  print("failed to pull_out_manipulator_matrices")

Q = Matrix([[0], [0]])
tau = rhs = simplify(Q - tau_g - C * Matrix([[rdot], [thetadot]]))

M_inv = simplify(M.inv())
M_inv * tau
qdotdot = simplify(M_inv * tau)
for i in range(len(qdotdot)):
    expr = sympy_to_expression(
        qdotdot[i])

    expr = expr.replace("r(t)", "r")
    expr = expr.replace("theta(t)", "theta")
    expr = expr.replace("Derivative(theta, t)", "thetadot")
    expr = expr.replace("Derivative(r, t)", "rdot")

    print("qdotdot[%d]" % (i))
    print(expr)

'''
2.2.2 Apex return map
To investigate periodicity for this running model, it suffices to consider the apex
height yi of two subsequent flight phases. This holds since (i) at apex the vertical
velocity ˙yi equals zero, (ii) the forward velocity ˙xi can be expressed in terms of the
apex height due to the constant system energy Es, and (iii) the forward position xi
has no influence on the further system dynamics.
Consequently, the stability of spring-mass running can be analyzed with a onedimensional return map yi+1(yi) of the apex height of two subsequent flight phases
(single step analysis). In terms of the apex return map, a periodic movement trajectory in spring-mass running is represented by a fixed point yi+1(yi) = yi
. Moreover,
as a sufficient condition, a slope dyi+1(yi)/dyi within a range of (−1, 1) in the neighborhood of the fixed point indicates the stability of the movement pattern (higher
than period 1 stability, which corresponds to symmetric contacts with time reflection symmetry about midstance, is not considered). The size of the neighborhood
defines the basin of attraction of the stable trajectory.
'''