from sympy_utils import *
x1, x2, x3, x4 = dynamicsymbols('x1 x2 x3 x4')
t = Symbol('t')
x = [x1, x2, x3, x4]
q1, q2 = dynamicsymbols('q1 q2')
x = [q1, q2, q1.diff(t), q2.diff(t)]
F1, F2, J1, J2, K, N, m, g, d = symbols('F1 F2 J1 J2 K N m g d')
f = Matrix([[q1.diff(t)], [q2.diff(t)], [(-F1 * q1.diff(t) - K / N * (q2 - q1 / N)) / J1], [(-F2*q2.diff(t) - K*(q2 - q1 / N) - m*g*d*cos(q2)) / J2]])
g = Matrix([[0], [0], [1/J1], [0]])
adj_f_g = lie_bracket(f, g, x)
adj_f2_g = lie_bracket(f, adj_f_g, x)
adj_f3_g = lie_bracket(f, adj_f2_g, x)

M = Matrix([g, adj_f_g, adj_f2_g, adj_f3_g])


M = Matrix([g, adj_f_g, adj_f2_g, adj_f3_g])
adj_f_g = lie_bracket(f, g, x)
adj_f2_g = lie_bracket(f, adj_f_g, x)
adj_f3_g = lie_bracket(f, adj_f2_g, x)

M = Matrix([g, adj_f_g, adj_f2_g, adj_f3_g])
M
M = Matrix([[g], [adj_f_g], [adj_f2_g], [adj_f3_g]])
M
g
[g, adj_f_g]
[g, adj_f_g, adj_f2_g, adj_f3_g]
Matrix([g, adj_f_g, adj_f2_g, adj_f3_g])
g
g.T
[g, adj_f_g]
Matrix([g, adj_f_g])
Matrix([g])
g.row_join(adj_f_g)
temp = g.row_join(adj_f_g)
temp = temp.row_join(adj_f2_g)
temp = temp.row_join(adj_f3_g)
temp
temp.rank()
latex(temp)
print(latex(temp))
lie_bracket(g, adj_f_g, x)
lie_bracket(g, adj_f2_g, x)
lie_bracket(adj_f_g, adj_f2_g, x)
f
print(latex(f))
latex(x)
print(latex(x))
print(latex(x.T))
print(latex(Matrix(x)))
print(latex(f))
g
print(latex(g))
adj_f_g.T
print(latex(adj_f_g.T))
print(latex(adj_f2_g.T))
print(latex(adj_f3_g.T))
print(latex(adj_f3_g[1]))
f
f[1]
f[1].diff(q2)
f[1]
(3*q2).diff(q2)
(q2.diff(t)).diff(q2)
(q2.diff(t)).diff(t)
f
f[3]
f[3].diff(q1)
f[3].diff(q2)
f[3].diff(q1.diff(t))
f[3].diff(q2.diff(t))
[f[3].diff(q1), f[3].diff(q2), f[3].diff(q1.diff(t)), f[3].diff(q2.diff(t))]
Matrix([f[3].diff(q1), f[3].diff(q2), f[3].diff(q1.diff(t)), f[3].diff(q2.diff(t))])
temp = Matrix([f[3].diff(q1), f[3].diff(q2), f[3].diff(q1.diff(t)), f[3].diff(q2.diff(t))])
latex(temp.T)
print(latex(temp.T))
f3_diffx = temp
f3_diffx
f3_diffx.dot(f)
f
f[3].diff([q1, q2, q1.diff(t), q2.diff(t)]
)
f3_diffx.dot(f)
f.dot(f3_diffx)
temp = f.dot(f3_diffx)
temp.diff(q1)
temp.diff(q2)
temp.diff(q1.diff(t))
temp.diff(q2.diff(t))
temp
print(latex(temp))
Lf3h = temp
Lf3h.diff(q1)
[Lf3h.diff(q1), Lf3h.diff(q2), Lf3h.diff(q1.diff(t)), Lf3h.diff(q2.diff(t))]
Matrix([Lf3h.diff(q1), Lf3h.diff(q2), Lf3h.diff(q1.diff(t)), Lf3h.diff(q2.diff(t))])
g
Lf3hdiffx = Matrix([Lf3h.diff(q1), Lf3h.diff(q2), Lf3h.diff(q1.diff(t)), Lf3h.diff(q2.diff(t))])
Lf3hdiffx.dot(g)
print(latex(Lf3hdiffx.dot(g)))
from sympy.diffgeom import LieDerivative
print(latex(Lf3hdiffx.dot(f)))
Lf3hdiffx.dot(f)
Lf3hdiffx.dot(f)
Lf3hdiffx.dot(f) / Lf3hdiffx.dot(g)
Lf3hdiffx.dot(f) / Lf3hdiffx.dot(g)
simplify(Lf3hdiffx.dot(f) / Lf3hdiffx.dot(g))
Lf3hdiffx
Lf3h
Lf3hdiffx.dot(f)
- Lf3hdiffx.dot(f) / Lf3hdiffx.dot(g)
simplify(- Lf3hdiffx.dot(f) / Lf3hdiffx.dot(g))
print(latex(simplify(- Lf3hdiffx.dot(f) / Lf3hdiffx.dot(g))))
f
g
LieDerivative(f, g)
g
f
history
def lie_derivative(f, l, x):
  '''
    L_f l
    f: vector
    l: scalar func
    x: state vector
  '''
  temp = Matrix([l.diff(y) for y in x])
  return temp.dot(f)
x
lie_derivative(g, f, x)
l
lie_derivative(g, f[3], x)
lie_derivative(g, q2, x)
lie_derivative(f, q2, x)
lie_derivative(g, lie_derivative(f, q2, x))
lie_derivative(g, lie_derivative(f, q2, x), x)
lie_derivative(g, lie_derivative(f, lie_derivative(f, q2, x), x), x)
lie_derivative(g, lie_derivative(f, lie_derivative(f, lie_derivative(f, q2, x), x), x), x)
lie_derivative(f, lie_derivative(f, lie_derivative(f, lie_derivative(f, q2, x), x), x), x)
def lie_derivative(f, l, x, order = 1):
  '''
    L_f l
    f: vector
    l: scalar func
    x: state vector
  '''
  res = l
  while order > 0:
    res = Matrix([res.diff(y) for y in x])
    res = res.dot(f)
  return res
lie_derivative(g, lie_derivative(f, q2, x), x)
def lie_derivative(f, l, x, order = 1):
  '''
    L_f l
    f: vector
    l: scalar func
    x: state vector
  '''
  res = l
  while order > 0:
    res = Matrix([res.diff(y) for y in x])
    res = res.dot(f)
    order -= 1
  return res
lie_derivative(g, lie_derivative(f, q2, x), x)
lie_derivative(g, lie_derivative(f, q2, x, order=1), x)
lie_derivative(g, lie_derivative(f, q2, x, order=2), x)
lie_derivative(g, lie_derivative(f, q2, x, order=3), x)
lie_derivative(g, lie_derivative(f, q2, x, order=0), x)
lie_derivative(f, q2, x, order=4)
-lie_derivative(f, q2, x, order=4) / lie_derivative(g, lie_derivative(f, q2, x, order=3), x)
simplify(-lie_derivative(f, q2, x, order=4) / lie_derivative(g, lie_derivative(f, q2, x, order=3), x))
alpha = simplify(-lie_derivative(f, q2, x, order=4) / lie_derivative(g, lie_derivative(f, q2, x, order=3), x))
latex(alpha)
print(latex(alpha))
lie_derivative(f, q2, x, order=3)
lie_derivative(g, lie_derivative(f, q2, x, order=3), x)
1 / lie_derivative(g, lie_derivative(f, q2, x, order=3), x)
beta = 1 / lie_derivative(g, lie_derivative(f, q2, x, order=3), x)
print(latex(beta)
)
lie_derivative(f, q2, x, order=1)
lie_derivative(f, q2, x, order=2)
print(latex(lie_derivative(f, q2, x, order=2)))
print(latex(lie_derivative(f, q2, x, order=3)))
lie_derivative(f, q2, x, order=3)
lie_derivative(f, q2, x, order=4)
f
lie_derivative(f, q2, x, order=0)
lie_derivative(f, q2, x, order=1)
g
lie_derivative(g, lie_derivative(f, q2, x, order=1), x)
lie_derivative(g, lie_derivative(f, q2, x, order=2), x)
lie_derivative(g, lie_derivative(f, q2, x, order=3), x)
lie_derivative(f, q2, x, order=0)
lie_derivative(f, q2, x, order=1)
lie_derivative(f, q2, x, order=2)
lie_derivative(f, q2, x, order=3)


In [187]: history
from sympy_utils import *
f = Matrix([[0], [0]])
f
f = Matrix([[0], [0], [0], [0]])
f
g = Matrix([[0], [0], [0], [0]])
f = Matrix([[0], [0], [0], [0]])
x1, x2, x3, x4 = dynamicsymbols('x1 x2 x3 x4')
t = Symbol('t')
x = [x1, x2, x3, x4]
x
q1, q2 = dynamicsymbols('q1 q2')
x = [q1, q2, q1.diff(t), q2.diff(t)]
x
F1, F2, J1, J2, K, N, m, g, d = Symbols('F1 F2 J1 J2 K N m g d')
F1, F2, J1, J2, K, N, m, g, d = Symbol('F1 F2 J1 J2 K N m g d')
F1, F2, J1, J2, K, N, m, g, d = symbols('F1 F2 J1 J2 K N m g d')
f = Matrix([[q1.diff(t)], [q2.diff(t)], [(-F1 * q1.diff(t) - K / N * (q2 - q1 / N)) / J1], [(-F2*q2.diff(t) - K*(q2 - q1 / N) - m*g*d*cos(q2)) / J2]])
f
g = Matrix([[0], [0], [1/J1], [0]])
g
x
lie_bracket(f, g, x)
adj_f_g = lie_bracket(f, g, x)
adj_f_g = lie_bracket(f, adj_f_g, x)
adj_f_g = lie_bracket(f, g, x)
lie_bracket(f, adj_f_g, x)
lie_bracket(f, lie_bracket(f, adj_f_g, x), x)
history
M = Matrix([g, adj_f_g, adj_f2_g, adj_f3_g])
adj_f_g = lie_bracket(f, g, x)
adj_f2_g = lie_bracket(f, adj_f_g, x)
adj_f3_g = lie_bracket(f, adj_f2_g, x)

M = Matrix([g, adj_f_g, adj_f2_g, adj_f3_g])
M
M = Matrix([[g], [adj_f_g], [adj_f2_g], [adj_f3_g]])
M
g
[g, adj_f_g]
[g, adj_f_g, adj_f2_g, adj_f3_g]
Matrix([g, adj_f_g, adj_f2_g, adj_f3_g])
g
g.T
[g, adj_f_g]
Matrix([g, adj_f_g])
Matrix([g])
g.row_join(adj_f_g)
temp = g.row_join(adj_f_g)
temp = temp.row_join(adj_f2_g)
temp = temp.row_join(adj_f3_g)
temp
temp.rank()
latex(temp)
print(latex(temp))
lie_bracket(g, adj_f_g, x)
lie_bracket(g, adj_f2_g, x)
lie_bracket(adj_f_g, adj_f2_g, x)
f
print(latex(f))
latex(x)
print(latex(x))
print(latex(x.T))
print(latex(Matrix(x)))
print(latex(f))
g
print(latex(g))
adj_f_g.T
print(latex(adj_f_g.T))
print(latex(adj_f2_g.T))
print(latex(adj_f3_g.T))
print(latex(adj_f3_g[1]))
f
f[1]
f[1].diff(q2)
f[1]
(3*q2).diff(q2)
(q2.diff(t)).diff(q2)
(q2.diff(t)).diff(t)
f
f[3]
f[3].diff(q1)
f[3].diff(q2)
f[3].diff(q1.diff(t))
f[3].diff(q2.diff(t))
[f[3].diff(q1), f[3].diff(q2), f[3].diff(q1.diff(t)), f[3].diff(q2.diff(t))]
Matrix([f[3].diff(q1), f[3].diff(q2), f[3].diff(q1.diff(t)), f[3].diff(q2.diff(t))])
temp = Matrix([f[3].diff(q1), f[3].diff(q2), f[3].diff(q1.diff(t)), f[3].diff(q2.diff(t))])
latex(temp.T)
print(latex(temp.T))
f3_diffx = temp
f3_diffx
f3_diffx.dot(f)
f
f[3].diff([q1, q2, q1.diff(t), q2.diff(t)]
)
f3_diffx.dot(f)
f.dot(f3_diffx)
temp = f.dot(f3_diffx)
temp.diff(q1)
temp.diff(q2)
temp.diff(q1.diff(t))
temp.diff(q2.diff(t))
temp
print(latex(temp))
Lf3h = temp
Lf3h.diff(q1)
[Lf3h.diff(q1), Lf3h.diff(q2), Lf3h.diff(q1.diff(t)), Lf3h.diff(q2.diff(t))]
Matrix([Lf3h.diff(q1), Lf3h.diff(q2), Lf3h.diff(q1.diff(t)), Lf3h.diff(q2.diff(t))])
g
Lf3hdiffx = Matrix([Lf3h.diff(q1), Lf3h.diff(q2), Lf3h.diff(q1.diff(t)), Lf3h.diff(q2.diff(t))])
Lf3hdiffx.dot(g)
print(latex(Lf3hdiffx.dot(g)))
from sympy.diffgeom import LieDerivative
print(latex(Lf3hdiffx.dot(f)))
Lf3hdiffx.dot(f)
Lf3hdiffx.dot(f)
Lf3hdiffx.dot(f) / Lf3hdiffx.dot(g)
Lf3hdiffx.dot(f) / Lf3hdiffx.dot(g)
simplify(Lf3hdiffx.dot(f) / Lf3hdiffx.dot(g))
Lf3hdiffx
Lf3h
Lf3hdiffx.dot(f)
- Lf3hdiffx.dot(f) / Lf3hdiffx.dot(g)
simplify(- Lf3hdiffx.dot(f) / Lf3hdiffx.dot(g))
print(latex(simplify(- Lf3hdiffx.dot(f) / Lf3hdiffx.dot(g))))
f
g
LieDerivative(f, g)
g
f
history
def lie_derivative(f, l, x):
  '''
    L_f l
    f: vector
    l: scalar func
    x: state vector
  '''
  temp = Matrix([l.diff(y) for y in x])
  return temp.dot(f)
x
lie_derivative(g, f, x)
l
lie_derivative(g, f[3], x)
lie_derivative(g, q2, x)
lie_derivative(f, q2, x)
lie_derivative(g, lie_derivative(f, q2, x))
lie_derivative(g, lie_derivative(f, q2, x), x)
lie_derivative(g, lie_derivative(f, lie_derivative(f, q2, x), x), x)
lie_derivative(g, lie_derivative(f, lie_derivative(f, lie_derivative(f, q2, x), x), x), x)
lie_derivative(f, lie_derivative(f, lie_derivative(f, lie_derivative(f, q2, x), x), x), x)
def lie_derivative(f, l, x, order = 1):
  '''
    L_f l
    f: vector
    l: scalar func
    x: state vector
  '''
  res = l
  while order > 0:
    res = Matrix([res.diff(y) for y in x])
    res = res.dot(f)
  return res
lie_derivative(g, lie_derivative(f, q2, x), x)
def lie_derivative(f, l, x, order = 1):
  '''
    L_f l
    f: vector
    l: scalar func
    x: state vector
  '''
  res = l
  while order > 0:
    res = Matrix([res.diff(y) for y in x])
    res = res.dot(f)
    order -= 1
  return res
lie_derivative(g, lie_derivative(f, q2, x), x)
lie_derivative(g, lie_derivative(f, q2, x, order=1), x)
lie_derivative(g, lie_derivative(f, q2, x, order=2), x)
lie_derivative(g, lie_derivative(f, q2, x, order=3), x)
lie_derivative(g, lie_derivative(f, q2, x, order=0), x)
lie_derivative(f, q2, x, order=4)
-lie_derivative(f, q2, x, order=4) / lie_derivative(g, lie_derivative(f, q2, x, order=3), x)
simplify(-lie_derivative(f, q2, x, order=4) / lie_derivative(g, lie_derivative(f, q2, x, order=3), x))
alpha = simplify(-lie_derivative(f, q2, x, order=4) / lie_derivative(g, lie_derivative(f, q2, x, order=3), x))
latex(alpha)
print(latex(alpha))
lie_derivative(f, q2, x, order=3)
lie_derivative(g, lie_derivative(f, q2, x, order=3), x)
1 / lie_derivative(g, lie_derivative(f, q2, x, order=3), x)
beta = 1 / lie_derivative(g, lie_derivative(f, q2, x, order=3), x)
print(latex(beta)
)
lie_derivative(f, q2, x, order=1)
lie_derivative(f, q2, x, order=2)
print(latex(lie_derivative(f, q2, x, order=2)))
print(latex(lie_derivative(f, q2, x, order=3)))
lie_derivative(f, q2, x, order=3)
lie_derivative(f, q2, x, order=4)
f
lie_derivative(f, q2, x, order=0)
lie_derivative(f, q2, x, order=1)
g
lie_derivative(g, lie_derivative(f, q2, x, order=1), x)
lie_derivative(g, lie_derivative(f, q2, x, order=2), x)
lie_derivative(g, lie_derivative(f, q2, x, order=3), x)
lie_derivative(f, q2, x, order=0)
lie_derivative(f, q2, x, order=1)
lie_derivative(f, q2, x, order=2)
lie_derivative(f, q2, x, order=3)
history
f
f[0]
lie_derivative(f, q2, x, order=3)
print(python(lie_derivative(f, q2, x, order=3)))
alpha
print(python(alpha))
f
f[2]
print(python(f[2]))
print(python(f[3]))
history
