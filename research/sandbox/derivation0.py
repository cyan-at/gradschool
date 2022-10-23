#! /usr/bin/env python3

'''
w_initial to w_final
that solution depends only depends on w_initial, w_final

suppose you can solve it, analytical or numerical

this answer is only function of w_initial / w_final

(the trick, related to control grammian)

23, c(x, y), rhs is what we desire, some kind of nonlinear thing
this means we have 11, for nonlinear
solve static optimization problem 11, done

write down L^, follow notation (x, y, v)
then derive L^, surely we can do, we don't have pseudo-inverse

L^ lagrangian is new
can we do c(x, y)?

LTV stochastic control + change of variable = classical OMT problem

we add noise for computational regularization

####################################################

11 + 


c(x, y) is solution to optimal control problem

'''

from sympy_utils import *

################################################

J1, J2, J3 = symbols('J1 J2 J3')

###############################
t = Symbol('t')
w1, w2, w3 = dynamicsymbols('w1 w2 w3')
w1dot = w1.diff(t)
w2dot = w2.diff(t)
w3dot = w3.diff(t)

###############################

t1, t2, t3 = symbols('t1 t2 t3')

################################################

t1 = J1*w1dot - (J2 - J3) * w2 * w3
t2 = J2*w2dot - (J3 - J1) * w3 * w1
t3 = J3*w3dot - (J1 - J2) * w1 * w2

################################################

u = Matrix([t1, t2, t3])

L = u.dot(u.T) / 2
