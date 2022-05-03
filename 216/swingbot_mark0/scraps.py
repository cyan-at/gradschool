

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
