import sympy; from sympy import *; sympy.init_printing();
x10, x20, x30, x11, x21, x31 = symbols('x_10 x_20 x_30 x_11 x_21 x_31');
x10
a1, a2, a3 = symbols('alpha_1 alpha_2 alpha_3');
a1
b1, b2, b3 = symbols('b_1 b_2 b_3');
b1
b1, b2, b3 = symbols('beta_1 beta_2 beta_3');
b1
l1, l2, l3 = symbols('lambda_1 lambda_2 lambda_3');
t1, t2, t3 = symbols('theta_1 theta_2 theta_3');
t1
b1_b2 = Matrix([[-1/(b1**2)], [-1/(b2**2)]])
b1_b2
x11_x21 = Matrix([[x_11], [x_21]])
x11_x21 = Matrix([[x11], [x21]])
x11_x21
x10_x20 = Matrix([[x10], [x20]])
x10_x20
tau = symbols('tau')
tau
k3
k1, k2, k3 = symbols('k_1 k_2 k_3')
k1
t = symbols('t')
d1 = cos(a2*k3*(t**2 - tau**2) / 2 + a2 * x30*(t - tau))
c = cos(a2*k3*(t**2 - tau**2) / 2 + a2 * x30*(t - tau))
d = a2*k3*(t**2 - tau**2) / 2 + a2 * x30*(t - tau)
d
c = cos(d)
c
s = sin(d)
s
history
phi = Matrix([[c, -s], [s, c]])
phi
phi
phi_t_tau = Matrix([[c, -s], [s, c]])
phi_t_tau.replace(t, 1).replace(tau, 0)
simplify(phi_t_tau.replace(t, 1).replace(tau, 0))
simplify(phi_t_tau.replace(t, 1).replace(tau, 0)).replace(k3, x31 - x30)
simplify(phi_t_tau.replace(t, 1).replace(tau, 0).replace(k3, x31 - x30))
lambda_expr_1 = simplify(phi_t_tau.replace(t, 1).replace(tau, 0).replace(k3, x31 - x30))
lambda_expr_1 * x10_x20
x11_x21 - lambda_expr_1 * x10_x20
simplify(x11_x21 - lambda_expr_1 * x10_x20)
lambda_expr_2 = simplify(x11_x21 - lambda_expr_1 * x10_x20)
b1_b2
sympy.matrices.dense.matrix_multiply_elementwise(b1_b2, lambda_expr_2)
lambda_expr_3 = simplify(sympy.matrices.dense.matrix_multiply_elementwise(b1_b2, lambda_expr_2))
lambda_expr_3
a
c
s
integrate(c, (tau, 0, 1))
c
s
e, f, g, h = symbols('e f g h')
c
e, f = symbols('e f')
1 / (e**2 + f**2)
c
d = c
d
d
s
1 / (e**2 + f**2) * Matrix([[e, f], [-f, e]])
1 / (e**2 + f**2) * Matrix([[e, f], [-f, e]])
lambda_expr_3
1 / (e**2 + f**2) * Matrix([[e, f], [-f, e]]) * lambda_expr_3
lambda_expr_4 = 1 / (e**2 + f**2) * Matrix([[e, f], [-f, e]]) * lambda_expr_3
lambda_expr_4
lambda_expr_4[0]
lambda_expr_4[1]
simplify(lambda_expr_4[1])
lambda_expr_5 = simplify(lambda_expr_4)
lambda_expr_5
lambda_expr_5[0]
print(latex(lambda_expr_5[0]))
print(latex(lambda_expr_5))
lambda_expr_5
lambda_expr_5[0]**2 * b1**2
simplify(lambda_expr_5[0]**2 * b1**2)
lambda_expr_5[0]**2
lambda_expr_5[0]**2 * b1**2
lambda_expr_5[0]**2 * b1**2
lambda_expr_5[0]**2 * b1**2 + lambda_expr_5[1]**2 * b2**2
lambda_expr_6 = (x30 - x31) / (b3**2)
lambda_expr_6
lambda_expr_6**2 * b3**2
lambda_expr_5[0]**2 * b1**2 + lambda_expr_5[1]**2 * b2**2 + lambda_expr_6**2 * b3**2
1/2 * lambda_expr_5[0]**2 * b1**2 + lambda_expr_5[1]**2 * b2**2 + lambda_expr_6**2 * b3**2
simplify(1/2 * lambda_expr_5[0]**2 * b1**2 + lambda_expr_5[1]**2 * b2**2 + lambda_expr_6**2 * b3**2)
history
simplify((lambda_expr_5[0]**2 * b1**2 + lambda_expr_5[1]**2 * b2**2 + lambda_expr_6**2 * b3**2) / 2)
lambda_expr_5
lambda_expr_5[0]
lambda_expr_5[0]**2 * b1**2
lambda_expr_5[0]**2 * b1**2
lambda_expr_5[0]**2 * b1**2 + lambda_expr_5[1]**2 * b2**2
simplify(lambda_expr_5[0]**2 * b1**2 + lambda_expr_5[1]**2 * b2**2)
lamba_expr_5
lambda_expr_5
history
lambda_expr_5
lambda_expr_5[0]
lambda_expr_5[0]**2
expand(lambda_expr_5[0]**2)
simplify(expand(lambda_expr_5[0]**2))
history
(lambda_expr_5[0]**2 * b1**2 + lambda_expr_5[1]**2 * b2**2 + lambda_expr_6**2 * b3**2) / 2
print(latex((lambda_expr_5[0]**2 * b1**2 + lambda_expr_5[1]**2 * b2**2 + lambda_expr_6**2 * b3**2) / 2))
c_expr_1 = (lambda_expr_5[0]**2 * b1**2 + lambda_expr_5[1]**2 * b2**2 + lambda_expr_6**2 * b3**2) / 2
print(latex(c_expr_1))
a
a, b, g, aprime, delta, v = symbols('a b g aprime delta v')
a
b
g
aprime
delta
a, b, g, aprime, delta, v = symbols('a b g aprime Delta v')
delta
t1
t2
a = (t1 / 2 + t2)
b = -t2
g = -t1 / 2
v = sqrt(g) * (tau + b / (2*g))
delta = 4*a*g - b**2
v**2 + delta / (4*g)
simplify(v**2 + delta / (4*g))
aprime = simplify(a - delta / (4*g))
aprime
simplify(aprime + b + g)
simplify(aprime + b + g).replace(t1, a2*k3).replace(t2, a2*x30)
simplify(simplify(aprime + b + g).replace(t1, a2*k3).replace(t2, a2*x30))
simplify(simplify(aprime + b + g).replace(t1, a2*k3).replace(t2, a2*x30).replace(k3, x31 - x30))
delta / (4*g)
simplify(simplify(delta / (4*g)).replace(t1, a2*k3).replace(t2, a2*x30).replace(k3, x31 - x30))
print(latex(simplify(simplify(delta / (4*g)).replace(t1, a2*k3).replace(t2, a2*x30).replace(k3, x31 - x30))))
aprime + b + g + delta / (4*g)
simplify(aprime + b + g + delta / (4*g))
g
g.replace(t1, a2*k3).replace(k3, x31 - x30)
simplify(g.replace(t1, a2*k3).replace(k3, x31 - x30))
simplify(sqrt(g).replace(t1, a2*k3).replace(k3, x31 - x30))
print(latex(simplify(sqrt(g).replace(t1, a2*k3).replace(k3, x31 - x30))))
simplify(simplify(aprime + b + g).replace(t1, a2*k3).replace(t2, a2*x30).replace(k3, x31 - x30))
print(latex(simplify(simplify(aprime + b + g).replace(t1, a2*k3).replace(t2, a2*x30).replace(k3, x31 - x30))))
aprime
print(latex(simplify(simplify(aprime).replace(t1, a2*k3).replace(t2, a2*x30).replace(k3, x31 - x30))))
simplify(simplify(aprime).replace(t1, a2*k3).replace(t2, a2*x30).replace(k3, x31 - x30))
aprime
simplify(simplify(aprime).replace(t1, a2*k3).replace(t2, a2*x30).replace(k3, x31 - x30))
simplify(simplify(aprime + b + g).replace(t1, a2*k3).replace(t2, a2*x30).replace(k3, x31 - x30))
simplify(simplify(sqrt(g)).replace(t1, a2*k3).replace(t2, a2*x30).replace(k3, x31 - x30))
g
g.replace(t1, a2*k3).replace(k3, x31-x30)
sin(-a2)
cos(a2)
cos(-a2)
phi
phi.replace(t, 1).replace(tau, 0)
phi_1_0 = phi.replace(t, 1).replace(tau, 0)
phi_1_0
print(latex(phi_1_0))
phi_1_0
phi_1_0_k30 = phi_1_0.replace(k3, 0)
phi_1_0_k30 = simplify(phi_1_0.replace(k3, 0))
phi_1_0_k30
phi_1_0_k30 * x10_x20
x11_x21 - phi_1_0_k30 * x10_x20
simplify(x11_x21 - phi_1_0_k30 * x10_x20)
lambda_expr_5
lambda_expr_4
lambda_expr_3
lambda_expr_2
lambda_expr_1
simplify(x11_x21 - phi_1_0_k30 * x10_x20)
phi_1_0_k30
lambda_expr_k30_1 = phi_1_0_k30
lambda_expr_k30_2 = simplify(x11_x21 - phi_1_0_k30 * x10_x20)
lambda_expr_k30_2
lambda_expr_2
lambda_expr_3
lambda_expr_k30_3
lambda_expr_k30_3 = simplify(sympy.matrices.dense.matrix_multiply_elementwise(b1_b2, lambda_expr_k30_2))
lambda_expr_k30_3
lambda_expr_4
lambda_expr_k30_3
phi_1_0_k30
integrate(phi_1_0_k30, (tau, 0, 1))
integrate(phi_1_0_k30[0, 0], (tau, 0, 1))
phi_1_0
phi
phi_k30 = Matrix([[cos(a2*x30*(1-tau)), -sin(a2*x30*(1-tau))], [sin(a2*x30*(1-tau)), cos(a2*x30*(1-tau))]])
phi_k30
phi_k30.integrate(tau, 0, 1)
integrate(phi_k30[0, 0], (tau, 0, 1))
integrate(phi_k30, (tau, 0, 1))
simplify(integrate(phi_k30, (tau, 0, 1)))
phi_k30_int = simplify(integrate(phi_k30, (tau, 0, 1)))
phi_k30_int.inv()
simplify(phi_k30_int.inv())
phi_k30_int_inv = simplify(phi_k30_int.inv())
lambda_expr_k30_3
phi_k30_int_inv * lambda_expr_k30_3
lambda_expr_k30_4 = simplify(phi_k30_int_inv * lambda_expr_k30_3)
lambda_expr_k30_4
phi_k30_int_inv * lambda_expr_k30_3
lambda_expr_k30_3
lambda_expr_k30_2
phi_k30_int_inv * lambda_expr_k30_2
simplify(phi_k30_int_inv * lambda_expr_k30_2)
lambda_expr_k30_3 = simplify(phi_k30_int_inv * lambda_expr_k30_2)
lambda_expr_k30_4 = sympy.matrices.dense.matrix_multiply_elementwise(b1_b2, lambda_expr_k30_3)
lambda_expr_k30_4 = simplify(sympy.matrices.dense.matrix_multiply_elementwise(b1_b2, lambda_expr_k30_3))
lambda_expr_k30_4
b1_b2
lambda_expr_k30_4
lambda_expr_k30_3
sympy.matrices.dense.matrix_multiply_elementwise(b1_b2, lambda_expr_k30_3)
lambda_expr_k30_4
print(latex(lambda_expr_k30_4))
lambda_expr_k30_4
lambda_expr_k30_4[0]**2 * b1**2 + lambda_expr_k30_4[1]**2 * b2**2 + ((x30 - x31) / (b3**2))**2 * b3**2
simplify(lambda_expr_k30_4[0]**2 * b1**2 + lambda_expr_k30_4[1]**2 * b2**2 + ((x30 - x31) / (b3**2))**2 * b3**2)
history
simplify(lambda_expr_k30_4[0]**2 * b1**2 + lambda_expr_k30_4[1]**2 * b2**2 + ((x30 - x31) / (b3**2))**2 * b3**2)
c_expr_k30 = simplify(lambda_expr_k30_4[0]**2 * b1**2 + lambda_expr_k30_4[1]**2 * b2**2 + ((x30 - x31) / (b3**2))**2 * b3**2)
print(latex(c_expr_k30))
history
