import sympy
from sympy import *
from sympy.physics.mechanics import *

sympy.init_printing()
from IPython.display import display, Math

t = Symbol('t')
t1, t2, t3 = dynamicsymbols('t1 t2 t3')
m1, l1, m2, l2, m3, l3, g, p1_damping = symbols('m1 l1 m2 l2 m3 l3 g p1_damping');
dt1 = t1.diff(t)
dt2 = t2.diff(t)
dt3 = t3.diff(t)

x1 = l1*sin(t1)
y1 = -l1*cos(t1)
x2 = x1 - l2*sin(t2) # key is (-)
y2 = y1 - l2*cos(t2)
x3 = x1 + l3*sin(t3)
y3 = y1 - l3*cos(t3)

x1dot = x1.diff(t)
y1dot = y1.diff(t)
x2dot = x2.diff(t)
y2dot = y2.diff(t)
x3dot = x3.diff(t)
y3dot = y3.diff(t)

v1_2 = simplify(x1dot**2 + y1dot**2)
v2_2 = simplify(x2dot**2 + y2dot**2)
v3_2 = simplify(x3dot**2 + y3dot**2)

T = m1 / 2 * v1_2 + m2 / 2 * v2_2 + m3 / 2 * v3_2

V = m1*g*y1 + m2*g*y2 + m3*g*y3

L = simplify(T - V)

# sympy trick: verify sympy's equation matches yours by hand
# In [137]: simplify(simplify(L.diff(dt1)) - p1_byhand)                                                       
# Out[137]: 0

p1 = simplify(L.diff(dt1))
el1 = simplify(p1.diff(t) - L.diff(t1))
el1 = simplify(el1 / l1)

# p1 = -damping * dt1 => p1 + damping * dt1 = 0
p1 = simplify(p1 + p1_damping * dt1)

p2 = simplify(L.diff(dt2))
el2 = simplify(p2.diff(t) - L.diff(t2))

p3 = simplify(L.diff(dt3))
el3 = simplify(p3.diff(t) - L.diff(t3))

#############################################################
# 2. RE-write equations of motion to <manipulator matrices: M, C, tau_g>

import ipdb; ipdb.set_trace();


# import ipdb; ipdb.set_trace();

# T = m1 / 2 * l1**2 * dt1**2 + m2 / 2 * (l1**2 * dt1**2 + l2**2*dt2**2 - 2 * l1 * l2 * dt1 * dt2 * cos(t1 + t2)) + m3 / 2 * (l1**2 * dt1**2 + l3 * dt3**2 + 2*l1*l3*dt1*dt3*cos(t1 - t3))

# import sympy
# sympy.init_printing()
# import sympy
# sympy.init_printing()
# f.diff(t)
# f = cos(x - y)
# f.diff(t)
# from sympy import *; import sympy;

# V = (m1 + m2 + m3)*g*(-l1)*cos(t1) + m2*g*(-l2)*cos(t2) + m3*g*(-l3)*cos(t3)
# g = Symbol('g')
# V = (m1 + m2 + m3)*g*(-l1)*cos(t1) + m2*g*(-l2)*cos(t2) + m3*g*(-l3)*cos(t3)
# L = T - V
# L
# L.diff(dt1)
# simplify(L.diff(dt1))
# p1 = L.diff(dt1)
# p1
# p1.diff(t)
# p1
# p1_byhand = l1**2*(m1+m2+m3)*dt1 - m2*l1*l2*dt2*cos(t1+t2) + m3*l1*l3*dt3*cos(t1-t3)
# p1
# p1_byhand
# p1_byhand.diff(t)
# L
# L.diff(t1)
# L.diff(t2)
# L.diff(t3)
# p1_byhand.diff(t) - L.diff(t1)
# el1 = p1_byhand.diff(t) - L.diff(t1)
# el1 / l1
# simplify(el1 / l1)
# x
# y
# temp = cos(x + y)
# temp.diff(t)
# temp = cos(x - y)
# temp.diff(t)
# p2_byhand = m2*l2**2*dt2 - m2*l1*l2*dt1*cos(t1+t2)
# p2_byhand.diff(t)
# p3_byhand = m3*l3**2*dt3 + m3*l1*l3*dt1*cos(t1-t3)
# p3_byhand.diff(t)
# el1 = (p1_byhand.diff(t) - L.diff(t1)) / l1
# el1
# el1 = simplify((p1_byhand.diff(t) - L.diff(t1)) / l1)
# el1
# el2 = p2_byhand.diff(t) - L.diff(t2)
# el2
# el3 = p3_byhand.diff(t) - L.diff(t3)
# el3
# el2 = (p2_byhand.diff(t) - L.diff(t2)) / l2
# el2
# el2 = simplify((p2_byhand.diff(t) - L.diff(t2)) / l2)
# el2
# el2 = simplify((p2_byhand.diff(t) - L.diff(t2)) / (m2*l2))
# el2
# el3 = simplify((p3_byhand.diff(t) - L.diff(t3)) / (m3*l3))
# el3
# el3
# el2
# el1
