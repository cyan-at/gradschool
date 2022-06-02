#!/usr/bin/env python3

from sympy_utils import *

t = Symbol('t')

q1, q2 = dynamicsymbols('q1 q2')

F1, F2, J1, J2, K, N, m, G, d, u, v = symbols('F1 F2 J1 J2 K N m G d u v')

x1, x2, x3, x4 = dynamicsymbols('x1 x2 x3 x4')
xnew = [x1, x2, x3, x4]

x_replacements = [
    ['x3(t)', 'x[2]'],
    ['x1(t)', 'x[0]'],
    ['x2(t)', 'x[1]'],
    ['x4(t)', 'x[3]'],
]

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
  f_python = sympy_to_expression(f2[i], x_replacements)
  print(f_python)
print("\n")

########################################################

g = Matrix([[0], [0], [1/J1], [0]])

########################################################

h = x2
n = 4

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

z = tau = Matrix(
  [lie_derivative(f2, h, xnew, order=i) for i in range(n)])

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

tau_inv[1] = z1 # z1 = x2 => x2 = z1
tau_inv[3] = z2 # z2 = x4 => x4 = z2

x1_z3 = solve(Eq(z[2], z3), x1)[0]
tau_inv[0] = x1_z3.replace(x2, z1).replace(x4, z2)

x3_z4 = solve(Eq(z[3], z4), x3)[0]
tau_inv[2] = simplify(x3_z4.replace(x4, z2).replace(x2, z1).replace(x1, x1_z3).replace(x2, z1).replace(x4, z2))

print("tau_inv python:")
for i in range(len(tau_inv)):
  print(sympy_to_expression(tau_inv[i], z_replacements))
print("\n")

print("sanity checking tau_inv, tau")
for i in range(4):
  print(simplify(tau_inv[i].replace(z1, tau[0]).replace(z2, tau[1]).replace(z3, tau[2]).replace(z4, tau[3]))) 

'''
sanity checking tau_inv and tau are inverses of each other:

[tau_inv(tau, z=x), x=z] = x?

In [22]: simplify(tau_inv[0].replace(z1, tau[0]).replace(z2, tau[1]).replace(z3, tau[2]).replace(z4, tau[3]))                        
Out[22]: x1(t)

In [23]: simplify(tau_inv[1].replace(z1, tau[0]).replace(z2, tau[1]).replace(z3, tau[2]).replace(z4, tau[3]))                        
Out[23]: x2(t)

In [24]: simplify(tau_inv[2].replace(z1, tau[0]).replace(z2, tau[1]).replace(z3, tau[2]).replace(z4, tau[3]))                        
Out[24]: x3(t)

In [25]: simplify(tau_inv[3].replace(z1, tau[0]).replace(z2, tau[1]).replace(z3, tau[2]).replace(z4, tau[3]))                        
Out[25]: x4(t)

you see that tau_inv is the inverse of tau

and tau_inv[0] has a dgmcos(z_1) term, which if z_1 => 0, will converge to Ndgm / K = (1.7*9.8*1 / 1 ) * 2 = 33 which is what se see in the data:

z [-0.01231327  0.01200814 -0.0113987   0.01018241]
x [ 3.33198749e+01 -1.23132732e-02  2.64938006e-03  1.20081351e-02]
[33.32  0.    0.    0.  ]

furthermore, since tau_inv[0] = x1 is based on tau[2] = z3 mostly
and tau[2] is Lfh(2) = f[3], this is something I mostly derived
without much risk of making a mistake?
'''