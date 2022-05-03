import sympy
from sympy import *
from sympy.physics.mechanics import *

sympy.init_printing()
from IPython.display import display, Math

#############################################################
'''
1. use Lagrange physics to create <equations of motion>
this is the unique part of this, the plant
for a different plant, change this section
'''

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
# p1 = -damping * dt1 => p1 + damping * dt1 = 0
p1 = simplify(p1 + p1_damping * dt1)
el1 = simplify(p1.diff(t) - L.diff(t1))
el1 = simplify(el1 / l1)

p2 = simplify(L.diff(dt2))
el2 = simplify(p2.diff(t) - L.diff(t2))

p3 = simplify(L.diff(dt3))
el3 = simplify(p3.diff(t) - L.diff(t3))

#############################################################
# 2. RE-write equations of motion to <manipulator matrices: M, C, tau_g>
# manipulator form:
# M(q)q** + C(q, q*)q* = tau_g(q) + Bu

# repeat this procedure to extract out into matrix elements from system of equations:
# #trick
# collect(expand(el2), t3, evaluate=False)[t3.diff(t).diff(t)]

# M:
# [
#   [l1*m1 + l1*m2 + l1*m3 + p1_damping / l1, -l2*m2*cos(t1 + t2), l3*m3*cos(t1 - t3)],
#   [-l1*l2*m2*cos(t1 + t2),            l2**2*m2,                  0],
#   [l1*l3*m3*cos(t1 - t3),             0,                         l3**2*m3],
# ]
M = Matrix([ 
  [l1*m1 + l1*m2 + l1*m3 + p1_damping / l1, -l2*m2*cos(t1 + t2), l3*m3*cos(t1 - t3)], 
  [-l1*l2*m2*cos(t1 + t2),            l2**2*m2,                  0], 
  [l1*l3*m3*cos(t1 - t3),             0,                         l3**2*m3], 
])

# C:
# [
#   [0,  -l2*m2*cos(t1 + t2)*dt2, l3*m3*sin(t1 - t3)*dt3],
#   [l1*l2*m2*sin(t1 + t2)*dt1, 0, 0],
#   [-l1*l3*m3*sin(t1 - t3)*dt1, 0, 0],
# ]
C = Matrix([ 
  [0,  l2*m2*sin(t1 + t2)*dt2, l3*m3*sin(t1 - t3)*dt3], 
  [l1*l2*m2*sin(t1 + t2)*dt1, 0, 0], 
  [-l1*l3*m3*sin(t1 - t3)*dt1, 0, 0], 
])

# to strip away all over bits recurse and pick out [1] #trick
# collect(collect(collect(el1, t1, evaluate=False)[1], t2, evaluate=False)[1], t3, evaluate=False)[1]
# collect(collect(collect(expand(el2), t1, evaluate=False)[1], t2, evaluate=False)[1], t3, evaluate=False)[1]
# collect(collect(collect(expand(el3), t1, evaluate=False)[1], t2, evaluate=False)[1], t3, evaluate=False)[1]

# tau:
tau_g = Matrix([
  [g*m1*sin(t1) + g*m2*sin(t1) + g*m3*sin(t1)],
  [g*l2*m2*sin(t2)],
  [g*l3*m3*sin(t3)]
])

# to sanity check above:
qdotdot = Matrix([[t1.diff(t).diff(t)],[t2.diff(t).diff(t)],[t3.diff(t).diff(t)]])
qdot = Matrix([[t1.diff(t)], [t2.diff(t)], [t3.diff(t)]])
test = expand(M*qdotdot + C3*qdot + tau_g)
el1 - test[0]
el2 - test[1]
el3 - test[2]
simplify(el2 - test[1])
simplify(el3 - test[2])

# but remember that this is from the LHS, and tau is on the RHS!
# so you need to NEGATE this
tau_g =-tau_g

##########################################################

# 3. collated linearize to get:
# dynamics of the NONactuated joints = f(actuated joints)

# linearizing actuated joints = collated linearization
# collated linearize means to rewrite the manipulator matrice expressions
# to solve for non-actuated dynamics q** and the actuated q** in terms of something

# here is also where we add in + u control input (which can be 0)
u2, u3 = dynamicsymbols('u2 u3')
u = Matrix([[u2],[u3]])

# in our robot, unactuated is [0:0]
M11 = M[0, 0]

# actuated is [1:2]
M12 = M[0, 1:]
M21 = M[1:, 0]
M22 = M[1:, 1:]
q2dotdot = qdotdot[1:]

###################################

tau = expand(tau_g - C*qdot)

M_blob = M22 - M21 * M12 / M11
M_blob_inv = M_blob.inv()

###################################
'''
semantically, this expresses how the nonactuated joint moves
to solve for actuated q2** = f(u)
and solve for non-actuated dynamics q1** = f(q2**)

sanity checking:
In [252]: collect(expand(q1dotdot), t3, evaluate=False).keys()                                                                                                                                                    
Out[252]: dict_keys([Derivative(t3(t), t)**2, Derivative(t3(t), (t, 2)), 1])

In [253]: collect(expand(q1dotdot), t2, evaluate=False).keys()                                                                                                                                                    
Out[253]: dict_keys([Derivative(t2(t), (t, 2)), Derivative(t2(t), t)**2, 1])

notice it is NOT a function of t1
In [254]: collect(expand(q1dotdot), t1, evaluate=False).keys()                                                                                                                                                    
Out[254]: dict_keys([1])
'''

q1dotdot_derived =\
  simplify(1 / M11 * (tau[0] - M12[0] * q2dotdot[0] + M12[1] * q2dotdot[1]))

###################################

rhs = u + Matrix(tau[1:]) - M21 / M11 * tau[0]
q2dotdot_derived = simplify(M_blob_inv * rhs)

###################################

##########################################################

'''
4.
we can also solve for the dynamics of the system without linearizing
so that we can get x* = Ax.
The advantage here is that you solve for the change in state (q**) given state (q)
so you can integrate given u = 0, no control input

M * q** = tau + Bu
q** = Minv * (tau + Bu)
'''

M_inv = simplify(M.inv())
Bu = Matrix([[0], [u2], [u3]])
qdotdot_derived = M_inv * (tau + Bu)

'''
sanity check
In [284]: collect(expand(qdotdot_derived[0]), t1, evaluate=False).keys()                                                                                                                                          
Out[284]: dict_keys([Derivative(t1(t), t)**2, 1])

In [285]: collect(expand(qdotdot_derived[0]), t2, evaluate=False).keys()                                                                                                                                          
Out[285]: dict_keys([Derivative(t2(t), t)**2, 1])

In [286]: collect(expand(qdotdot_derived[0]), t3, evaluate=False).keys()                                                                                                                                          
Out[286]: dict_keys([Derivative(t3(t), t)**2, 1])

In [287]: collect(expand(qdotdot_derived[1]), t1, evaluate=False).keys()                                                                                                                                          
Out[287]: dict_keys([Derivative(t1(t), t)**2, 1])

In [288]: collect(expand(qdotdot_derived[1]), t2, evaluate=False).keys()                                                                                                                                          
Out[288]: dict_keys([Derivative(t2(t), t)**2, 1])

In [289]: collect(expand(qdotdot_derived[1]), t3, evaluate=False).keys()                                                                                                                                          
Out[289]: dict_keys([Derivative(t3(t), t)**2, 1])

we find no part of the q** derives from q**, which is good
'''

