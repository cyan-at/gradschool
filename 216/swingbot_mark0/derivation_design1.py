#!/usr/bin/env python3

import sympy
from sympy import *
from sympy.physics.mechanics import *
sympy.init_printing()

from sympy_utils import *

'''
---------------------------------------------------------------
# define symbols
'''

t = Symbol('t')
m1, l1, m2, l2, m3, l3, g, p1_damping =\
    symbols('m1 l1 m2 l2 m3 l3 g p1_damping', nonzero=True, positive=True)
t1, t2, t3 = dynamicsymbols('t1 t2 t3') # state vars

'''
---------------------------------------------------------------
# free-body-diagram is expressed here
'''

x1 = l1*sin(t1)
y1 = -l1*cos(t1)
x2 = x1 + l2*sin(t2)
y2 = y1 - l2*cos(t2)

# convienence

dt1 = t1.diff(t)
dt2 = t2.diff(t)

x1dot = x1.diff(t)
y1dot = y1.diff(t)
x2dot = x2.diff(t)
y2dot = y2.diff(t)

v1_2 = simplify(x1dot**2 + y1dot**2)
v2_2 = simplify(x2dot**2 + y2dot**2)

'''
---------------------------------------------------------------
# Lagrange
'''

T = m1 / 2 * v1_2 + m2 / 2 * v2_2

U = m1*g*y1 + m2*g*y2

L = simplify(T - U)

'''
---------------------------------------------------------------
# Euler-Lagrange Eq., do one per state var
# do NOT add any damping here, this is the 'LHS'
# and it is assumed for now there is NOT Q input force to the system
# so right now RHS is still = 0
'''

# unactuated states

p1 = simplify(L.diff(dt1))
el1 = simplify(p1.diff(t) - L.diff(t1))

# actuated states

p2 = simplify(L.diff(dt2))
el2 = simplify(p2.diff(t) - L.diff(t2))

# simplify

try:
    el1 = simplify_eq_with_assumptions(Eq(el1, 0)).lhs
except:
    pass

el1_latex = latex(Eq(el1, 0))
el1_latex = el1_latex.replace("t_{1}", "\\theta_{1}")
el1_latex = el1_latex.replace("t_{2}", "\\theta_{2}")
print("el1_latex")
print(el1_latex)

try:
    el2 = simplify_eq_with_assumptions(Eq(el2, 0)).lhs
except:
    pass

el2_latex = latex(Eq(el2, 0))
el2_latex = el2_latex.replace("t_{1}", "\\theta_{1}")
el2_latex = el2_latex.replace("t_{2}", "\\theta_{2}")
print("el2_latex")
print(el2_latex)

'''
---------------------------------------------------------------
# construct (the manipulator matrices.)
'''

# mass-matrix M
# note that here the term right-multiplied is q** exactly
M11, el1 = pull_out_term(el1, dt1.diff(t))
M12, el1 = pull_out_term(el1, dt2.diff(t))
M21, el2 = pull_out_term(el2, dt1.diff(t))
M22, el2 = pull_out_term(el2, dt2.diff(t))
M = Matrix([[M11, M12], [M21, M22]])

M_latex = latex(M)
M_latex = M_latex.replace("t_{1}", "\\theta_{1}")
M_latex = M_latex.replace("t_{2}", "\\theta_{2}")
print("M_latex")
print(M_latex)

# velocity-cross-product matrix C
# note that here the term right-multiplied is q*
# and we are pulling out velocity products
C11, el1 = pull_out_term(el1, dt1, [dt1, dt2])
C12, el1 = pull_out_term(el1, dt2, [dt1, dt2])
C21, el2 = pull_out_term(el2, dt1, [dt1, dt2])
C22, el2 = pull_out_term(el2, dt2, [dt1, dt2])
C = Matrix([[C11, C12], [C21, C22]])

C_latex = latex(C)
C_latex = C_latex.replace("t_{1}", "\\theta_{1}")
C_latex = C_latex.replace("t_{2}", "\\theta_{2}")
print("C_latex")
print(C_latex)


# gravity matrix tau_g
tau_g = Matrix([[el1], [el2]])

tau_g_latex = latex(tau_g)
tau_g_latex = tau_g_latex.replace("t_{1}", "\\theta_{1}")
tau_g_latex = tau_g_latex.replace("t_{2}", "\\theta_{2}")
print("tau_g_latex")
print(tau_g_latex)

'''
---------------------------------------------------------------
now we have Mq** + Cq* + tau_g = 0
we can add external forces Q to the RHS
and then solve for Mq** = Q - Cq* - tau_g
'''

Q = Matrix([[-p1_damping * dt1], [0]])
tau = rhs = simplify(Q - tau_g - C * Matrix([[dt1], [dt2]]))

Q_latex = latex(Q)
Q_latex = Q_latex.replace("t_{1}", "\\theta_{1}")
Q_latex = Q_latex.replace("t_{2}", "\\theta_{2}")
print("Q_latex")
print(Q_latex)

tau_latex = latex(tau)
tau_latex = tau_latex.replace("t_{1}", "\\theta_{1}")
tau_latex = tau_latex.replace("t_{2}", "\\theta_{2}")
print("tau_latex")
print(tau_latex)

'''
------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------
'''

'''
---------------------------------------------------------------
we can solve for dynamics of the system in free-fall with no external control
'''
M_inv = simplify(M.inv())
M_inv * tau
qdotdot = simplify(M_inv * tau)

qdotdot_latex = latex(qdotdot)
qdotdot_latex = qdotdot_latex.replace("t_{1}", "\\theta_{1}")
qdotdot_latex = qdotdot_latex.replace("t_{2}", "\\theta_{2}")
print("qdotdot_latex")
print(qdotdot_latex)

for i in range(len(qdotdot)):
    expr = sympy_to_expression(
        qdotdot[i])
    print("qdotdot[%d]" % (i))
    print(expr)

'''
---------------------------------------------------------------
we can do partial feedback linearization expressions

for example to get collocated pfl we solve for unactuated = f(actuated)
then eliminate unactuated term from actuated eq of motion
so then we have actuated eq of motion = f(actuated q**) = f(external force input)
and if say our external force input = f(actuated q**)
then we've equated / linearized the measurement of q**
y = q2 
'''
q1dotdot = 1 / M[0, 0] * (tau[0] - M[0, 1] * dt2.diff(t))

q1dotdot_latex = latex(q1dotdot)
q1dotdot_latex = q1dotdot_latex.replace("t_{1}", "\\theta_{1}")
q1dotdot_latex = q1dotdot_latex.replace("t_{2}", "\\theta_{2}")
print(q1dotdot_latex)

expr = sympy_to_expression(q1dotdot)
print(expr)
# put this in code

expr_expected = "(-G*M1*sin(t1) - G*M2*sin(t1) - L2*M2*sin(t1 - t2)*t2_dot**2 - L2*M2*cos(t1 - t2)*Derivative(t2, (t, 2)) - Q1_DAMPING*t1_dot)/(L1*M1 + L1*M2)"
assert(expr, expr_expected)