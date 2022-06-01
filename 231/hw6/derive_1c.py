#!/usr/bin/env python3

from sympy_utils import *

t = Symbol('t')

q1, q2 = dynamicsymbols('q1 q2')

F1, F2, J1, J2, K, N, m, G, d, u, v = symbols('F1 F2 J1 J2 K N m G d u v')

x1, x2, x3, x4 = dynamicsymbols('x1 x2 x3 x4')
xnew = [x1, x2, x3, x4]

########################################################

f2 = Matrix([0, 0, 0, 0])
f2[0] = x3
f2[1] = x4
q1dotdot = simplify(solve(
  Eq(J1*q1.diff(t).diff(t) + F1*q1.diff(t) + K / N *(q2 - q1 / N), u), q1.diff(t).diff(t))
  [0]).replace(q1.diff(t), x3).replace(q2.diff(t), x4).replace(q1, x1).replace(q2, x2)
f2[2] = expand(q1dotdot.replace(u, 0))
f2[3] = expand(solve(
  Eq(J2*q2.diff(t).diff(t) + F2*q2.diff(t) + K*(q2 - q1 / N), -m*G*d*cos(q2)), q2.diff(t).diff(t))
  [0].replace(q1.diff(t), x3).replace(q2.diff(t), x4).replace(q1, x1).replace(q2, x2))

print("f python:")
for i in range(len(f2)):
  print(sympy_to_expression(f2[i]).replace('x3(t)', 'x[2]').replace('x1(t)', 'x[0]').replace('x2(t)', 'x[1]').replace('x4(t)', 'x[3]'))
print("\n")

########################################################

g = Matrix([[0], [0], [1/J1], [0]])

########################################################

h = x2

########################################################

alpha = expand(simplify(
  -lie_derivative(f2, h, xnew, order=4) / lie_derivative(g, lie_derivative(f2, h, xnew, order=3),
  xnew)))

print("alpha python:")
alpha_py = sympy_to_expression(alpha).replace('x3(t)', 'x[2]').replace('x1(t)', 'x[0]').replace('x2(t)', 'x[1]').replace('x4(t)', 'x[3]')
print(alpha_py)
print("\n")

########################################################

beta = 1 / lie_derivative(g, lie_derivative(f2, h, xnew, order=3), xnew)

print("beta python:")
print(sympy_to_expression(beta).replace('x3(t)', 'x[2]').replace('x1(t)', 'x[0]').replace('x2(t)', 'x[1]').replace('x4(t)', 'x[3]'))
print("\n")

########################################################

z = tau = Matrix([
  h,
  lie_derivative(f2, h, xnew, order=1),
  lie_derivative(f2, h, xnew, order=2),
  lie_derivative(f2, h, xnew, order=3)
])

print("tau python:")
for i in range(len(tau)):
  print(sympy_to_expression(tau[i]).replace('x3(t)', 'x[2]').replace('x1(t)', 'x[0]').replace('x2(t)', 'x[1]').replace('x4(t)', 'x[3]'))
print("\n")

########################################################

z1, z2, z3, z4 = symbols('z_1 z_2 z_3 z_4')

tau_inv = Matrix([0, 0, 0, 0])
tau_inv[1] = z1
tau_inv[3] = z2

x1_z3 = solve(Eq(z[2], z3), x1)[0]
tau_inv[0] = x1_z3.replace(x2, z1).replace(x4, z2)

x3_z4 = solve(Eq(z[3], z4), x3)[0]
tau_inv[2] = simplify(x3_z4.replace(x4, z2).replace(x2, z1).replace(x1, x1_z3).replace(x2, z1).replace(x4, z2))

print("tau_inv python:")
for i in range(len(tau_inv)):
  print(sympy_to_expression(tau_inv[i]).replace('z_4', 'z[3]').replace('z_3', 'z[2]').replace('z_2', 'z[1]').replace('z_1', 'z[0]'))
print("\n")
