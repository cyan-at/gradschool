#!/usr/bin/env python3

from sympy_utils import *

t = Symbol('t')

q1, q2 = dynamicsymbols('q1 q2')

F1, F2, J1, J2, K, N, m, G, d, u, v = symbols('F1 F2 J1 J2 K N m G d u v')

x1, x2, x3, x4 = dynamicsymbols('x1 x2 x3 x4')
xnew = [x1, x2, x3, x4]


'''
from sympy_utils import *
g = Matrix([[0], [0], [1/J1], [0]])
f0, f1 = symbols('f0 f1')
t = Symbol('t')
t1, t2 = dynamicsymbols('t1 t2')
t0, t1 = dynamicsymbols('t0 t1')
g1 = Matrix([[f0*cos(t0)], [f0*sin(t0)],[f1*sin(t1-t0)],[0]])
g1
g1 = Matrix([[f0*cos(t0)], [f0*sin(t0)],[f1*sin(t1-t0)],[0]])
g1
g1 = Matrix([[f0*cos(t0)], [f0*sin(t0)],[simplify(f1*sin(t1-t0))],[0]])
g1
f1
g2 = Matrix([[0],[0],[0],[1]])
g2
lie_bracket(g1, g2)
x, y = dynamicsymbols('x y')
state = [x, y, t0, t1]
lie_bracket(g1, g2, state)
g2.jacobian(state)
g1.jacobian(state)
g1
g1 = Matrix([[f0*cos(t0)], [f0*sin(t0)],[simplify(f1*sin(t1-t0))],[0]])
g1
g1 = Matrix([[f0*cos(t0)], [f0*sin(t0)],[f1*sin(t1-t0)],[0]])
g1
g1[0]
g1[0].diff(t0)
lie_bracket(g1, g2, state)
lie_bracket(g1, g2, state, order=2)
lie_bracket(g1, g2, state, order=3)
lie_bracket(g1, g2, state, order=4)
lie_bracket(g1, g2, state, order=5)
lie_bracket(g1, g2, state, order=6)
test_g1 = Matrix([[cos(t0)], [sin(t0)],[0]])
test_g2 = Matrix([[0],[0],[1]])
lie_bracket(test_g1, test_g2, [x, y, t0])
g1
g2
lie_bracket(g1, g2, state)
adj_g1_g2 = lie_bracket(g1, g2, state)
Matrix([g1, g2, adj_g1_g2])
Matrix([[g1], [g2], [adj_g1_g2]])
g1
g2
adj_g1_g2
Matrix([g1, g2]
)
Matrix([[g1], [g2]])
Matrix([g1.T g2.T])
Matrix([g1.T,g2.T])
Matrix([g1.T,g2])
Matrix([g1,g2])
Matrix([g1.T,g2.T])
Matrix([g1.T,g2.T]).T
Matrix([g1.T,g2.T,adj_g1_g2.T]).T
rank(Matrix([g1.T,g2.T,adj_g1_g2.T]).T)
augmented_matrix = Matrix([g1.T,g2.T,adj_g1_g2.T]).T
augmented_matrix.rank()
augmented_matrix.det()
lie_bracket(g1, lie_bracket(g1, g2, state), state)
lie_bracket(g1, lie_bracket(g2, g1, state), state)
adj_g1_g2_2 = lie_bracket(g1, lie_bracket(g2, g1, state), state)
augmented_matrix = Matrix([g1.T,g2.T,adj_g1_g2.T, adj_g1_g2_2.T]).T
augmented_matrix.rank()
augmented_matrix.det()
simplify(augmented_matrix.det())
lie_bracket(g1, g2, state, order=2)
adj_g1_g2_2 = lie_bracket(g1, g2, state, order=2)
augmented_matrix = Matrix([g1.T,g2.T,adj_g1_g2.T, adj_g1_g2_2.T]).T
augmented_matrix.rank()
lie_bracket(g1, lie_bracket(g1, g2, state), state)
lie_bracket(g1, lie_bracket(g2, g1, state), state)
lie_bracket(g2, g1, state)
lie_bracket(g2, g1, state).T
augmented_matrix = Matrix([g1.T,g2.T, lie_bracket(g2, g1, state).T, lie_bracket(g1, lie_bracket(g2, g1, state), state).T]).T
augmented_matrix
augmented_matrix.rank()
simplify(augmented_matrix.det())
lie_bracket(g1, g2, state) - lie_bracket(g2, g1, state)
lie_bracket(g2, g1, state)
print(latex(lie_bracket(g2, g1, state)))
g2.jacobian
g2.jacobian()
g2.jacobian(state)
g2.jacobian(state) - g1.jacobian(state) * g2
g2.jacobian(state) * g1 - g1.jacobian(state) * g2
g1.jacobian(state) * g2 - g2.jacobian(state) * g1
lie_bracket(g1, lie_bracket(g2, g1, state), state)
print(latex(lie_bracket(g1, lie_bracket(g2, g1, state), state)))
lie_bracket(g1, lie_bracket(g2, g1, state), state)
lie_bracket(lie_bracket(g1, g2, state), g2, state)
lie_bracket(g1, lie_bracket(lie_bracket(g1, g2, state), g2, state), state)
lie_bracket(g2, lie_bracket(lie_bracket(g1, g2, state), g2, state), state)
lie_bracket(lie_bracket(lie_bracket(g1, g2, state), g2, state), g1, state)
augmented_matrix = Matrix([g1.T,g2.T, lie_bracket(g2, g1, state).T, lie_bracket(g1, lie_bracket(g2, g1, state), state).T]).T
augmented_matrix
augmented_matrix.det()
simplify(augmented_matrix.det())
print(latex(simplify(augmented_matrix.det())))
history
'''