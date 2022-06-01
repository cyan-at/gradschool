#!/usr/bin/env python3

from sympy_utils import *

t = Symbol('t')

q1, q2 = dynamicsymbols('q1 q2')

F1, F2, J1, J2, K, N, m, G, d, u, v = symbols('F1 F2 J1 J2 K N m G d u v')

x1, x2, x3, x4 = dynamicsymbols('x1 x2 x3 x4')
xnew = [x1, x2, x3]

x_replacements = [
    ['x3(t)', 'x[2]'],
    ['x1(t)', 'x[0]'],
    ['x2(t)', 'x[1]'],
    ['x4(t)', 'x[3]'],
]

########################################################

f2 = Matrix([0, x1 + x2**2, x1-x2])

print("f python:")
for i in range(len(f2)):
  f_python = sympy_to_expression(f2[i], x_replacements)
  print(f_python)
print("\n")

########################################################

g = Matrix([[exp(x2)], [exp(x2)], [0]])

########################################################

h = x3
n = 3

########################################################

alpha = expand(simplify(
  -lie_derivative(f2, h, xnew, order=n) / lie_derivative(g, lie_derivative(f2, h, xnew, order=n-1), xnew)))

print("alpha python:")
alpha_py = sympy_to_expression(alpha, x_replacements)
print(alpha_py)
print("\n")

########################################################

beta = 1 / lie_derivative(g, lie_derivative(f2, h, xnew, order=n-1), xnew)

print("beta python:")
print(sympy_to_expression(beta, x_replacements))
print("\n")

########################################################

z = tau = Matrix([lie_derivative(f2, h, xnew, order=i) for i in range(n)])

print("tau python:")
for i in range(len(tau)):
  print(sympy_to_expression(tau[i], x_replacements))
print("\n")

########################################################

z1, z2, z3, z4 = symbols('z_1 z_2 z_3 z_4')

z_replacements = [
    ['z_4', 'z[3]'],
    ['z_3', 'z[2]'],
    ['z_2', 'z[1]'],
    ['z_1', 'z[0]'],
]

tau_inv = Matrix([0]*len(tau))

# tau_inv[1] = z1 # z1 = x2 => x2 = z1
# tau_inv[3] = z2 # z2 = x4 => x4 = z2

# x1_z3 = solve(Eq(z[2], z3), x1)[0]
# tau_inv[0] = x1_z3.replace(x2, z1).replace(x4, z2)

# x3_z4 = solve(Eq(z[3], z4), x3)[0]
# tau_inv[2] = simplify(x3_z4.replace(x4, z2).replace(x2, z1).replace(x1, x1_z3).replace(x2, z1).replace(x4, z2))

# print("tau_inv python:")
# for i in range(len(tau_inv)):
#   print(sympy_to_expression(tau_inv[i]).replace('z_4', 'z[3]').replace('z_3', 'z[2]').replace('z_2', 'z[1]').replace('z_1', 'z[0]'))
# print("\n")
