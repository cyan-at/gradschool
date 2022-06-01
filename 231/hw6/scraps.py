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

v = dynamicsymbols('v')
temp = simplify(alpha + beta * v)
simplify(lie_derivative(f, q2, x, order=4) + lie_derivative(g, lie_derivative(f, q2, x, order=3), x) * temp)
# Out[224]: v(t)

https://maths-people.anu.edu.au/~andrews/DG/DG_chap7.pdf
https://math.stackexchange.com/questions/2927059/what-does-involutive-span-means
https://www.egr.msu.edu/~khalil/NonlinearSystems/Sample/Lect_23.pdf
https://math.stackexchange.com/questions/435801/calculate-the-lie-derivative
https://abrarhashmi.files.wordpress.com/2017/03/lecture_feedbacklinearization.pdf
https://www.cse.sc.edu/~gatzke/cache/npc-Chapter4-scan.pdf
http://ele.aut.ac.ir/~abdollahi/Lec_8_N11.pdf
https://www.youtube.com/watch?v=ECqr0oUfZz4
https://www.lehigh.edu/~eus204/teaching/ME450_NSC/lectures/lecture06.pdf
http://users.isr.ist.utl.pt/~pedro/NCS2012/07_FeedbackLinearization.pdf

https://nbviewer.org/github/abhishekhalder/AM231-S22/blob/main/S22-AM231-HW6.ipynb
http://localhost:8888/notebooks/YanC_Sol_HW6.ipynb


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
lambda
h
q2
q2.diff(t)
f
f[3]
 f[3].diff(t)
 f[3].diff(t).diff(t)
f
lie_derivative(f, q2, order=3)
lie_derivative(f, q2, x, order=3)
x
lie_derivative(f, q2, x, order=3)
lie_derivative(f, q2, x, order=4)
lie_derivative(g, lie_derivative(f, q2, x, order=3), x, order=1)
f
lie_derivative(g, lie_derivative(f, q2, x, order=3), x, order=1)
lie_derivative(f, q2, x, order=4)
lie_derivative(f, q2, x, order=4)
alpha
lie_derivative(f, q2, x, order=4) / lie_derivative(g, lie_derivative(f, q2, x, order=3), x, order=1)
simplify(lie_derivative(f, q2, x, order=4) / lie_derivative(g, lie_derivative(f, q2, x, order=3), x, order=1))
simplify(-lie_derivative(f, q2, x, order=4) / lie_derivative(g, lie_derivative(f, q2, x, order=3), x, order=1))
lie_derivative(f, q2, x, order=2)
lie_derivative(f, q2, x, order=3)
v = dynamicsymbols('v')
beta * v
alpha + beta * v
simplify(alpha + beta * v)
g
g.dot(simplify(alpha + beta * v))
temp = simplify(alpha + beta * v)
alpha + beta * temp
lie_derivative(g, lie_derivative(f, q2, x, order=3), x)
lie_derivative(g, lie_derivative(f, q2, x, order=3), x) * temp
lie_derivative(f, q2, x, order=4) + lie_derivative(g, lie_derivative(f, q2, x, order=3), x) * temp
simplify(lie_derivative(f, q2, x, order=4) + lie_derivative(g, lie_derivative(f, q2, x, order=3), x) * temp)
x1
x2
x3
x4
f
f[2]
f[1]
f[1].replace(q2.diff(t), x3)
f[2].replace(q2.diff(t), x4).replace(q1.diff(t), x3).replace(q1, x1).replace(q2, x2)
f[3].replace(q2.diff(t), x4).replace(q1.diff(t), x3).replace(q1, x1).replace(q2, x2)
type(f)
f2 = Matrix([0, 0, 0, 0])
f2[0] = f[0].replace(q2.diff(t), x4).replace(q1.diff(t), x3).replace(q1, x1).replace(q2, x2)
f2[1] = f[1].replace(q2.diff(t), x4).replace(q1.diff(t), x3).replace(q1, x1).replace(q2, x2)
f2[2] = f[2].replace(q2.diff(t), x4).replace(q1.diff(t), x3).replace(q1, x1).replace(q2, x2)
f2[3] = f[3].replace(q2.diff(t), x4).replace(q1.diff(t), x3).replace(q1, x1).replace(q2, x2)
f2
x
x2 = [x1, x2, x3, x4]
f2[3].diff(x2)
temp = Matrix([f2[3].diff(x1), f2[3].diff(x2), f2[3].diff(x3), f2[3].diff(x4)])
f2[3].diff(x1)
f2[3].diff(x2)
f2[3]
f2[3].diff(x4)
f2[3].diff(x2)
f2[3].diff(x4)
f2[3].diff(x2)
f2[3].diff(x2)
f2[3]
f2[3].diff(x4)
f2[3].diff(x3)
f2[3].diff(x2)
f2[3].diff(x1)
f2[3]
f2[3].diff(x2)
test = (-F2*x4 - K*(x2 - x1 / N) - d*g*m*cos(x2)) / J2
x2
x2[1]
xnew = [x1, x2, x3, x4]
xnew
x2
x2 = dynamicsymbols('x2')
x2
xnew = [x1, x2, x3, x4]
xnew
f2[3].diff(x2)
def gradient(f, x):
  return [f.diff(y) for y in x]
gradient(f2[3], x)
gradient(f2[3], xnew)
lie_derivative(g, f2[3], xnew)
print(latex(Matrix(gradient(f2[3], xnew))))
print(latex(Matrix(gradient(f2[3], xnew)).T))
lie_derivative(g, f2[3], xnew, order=2)
lie_derivative(g, f2[3], xnew, order=3)
lie_derivative(g, f2[3], xnew, order=4)
lie_derivative(f, x2, xnew, order=1)
lie_derivative(f, x2, xnew, order=2)
lie_derivative(f, x2, xnew, order=3)
lie_derivative(f, x2, xnew, order=4)
lie_derivative(f2, x2, xnew, order=1)
lie_derivative(f2, x2, xnew, order=2)
lie_derivative(g, lie_derivative(f2, x2, xnew, order=2), xnew)
lie_derivative(g, lie_derivative(f2, x2, xnew, order=3), xnew)
lie_derivative(f2, x2, xnew, order=3)
lie_derivative(g, lie_derivative(f2, x2, xnew, order=3), xnew)
lie_derivative(f2, x2, xnew, order=3)
lie_derivative(f2, x2, xnew, order=2)
lie_derivative(f2, x2, xnew, order=1)
lie_derivative(f2, x2, xnew, order=0)
lie_derivative(f2, x2, xnew, order=4)
-lie_derivative(f2, x2, xnew, order=4) / lie_derivative(g, lie_derivative(f2, x2, xnew, order=3), xnew)
alpha = -lie_derivative(f2, x2, xnew, order=4) / lie_derivative(g, lie_derivative(f2, x2, xnew, order=3), xnew)
lie_derivative(f2, x2, xnew, order=1)
lie_derivative(f2, x2, xnew, order=0)
lie_derivative(f2, x2, xnew, order=1)
lie_derivative(f2, x2, xnew, order=2)
simplify(lie_derivative(f2, x2, xnew, order=2))
simplify(lie_derivative(f2, x2, xnew, order=3))
alpha = -lie_derivative(f2, x2, xnew, order=4) / lie_derivative(g, lie_derivative(f2, x2, xnew, order=3), xnew)
alpha =  simplify(-lie_derivative(f2, x2, xnew, order=4) / lie_derivative(g, lie_derivative(f2, x2, xnew, order=3), xnew))
alpha
f2[3]
gradient(f2[3], xnew)
gradient(f2[3], xnew).dot(f2)
Matrix(gradient(f2[3], xnew)).dot(f2)
simplify(lie_derivative(f2, x2, xnew, order=3))
lie_derivative(f2, x2, xnew, order=3)
lf2_3 = lie_derivative(f2, x2, xnew, order=3)
gradient(lf2_3, x2)
gradient(lf2_3, xnew)
Matrix(gradient(lf2_3, xnew)).dot(f)
Matrix(gradient(lf2_3, xnew)).dot(f) / (K / (J1*J2*N))
znew = Matrix([0, 0, 0, 0])
znew[0] = x2
znew[1] = lie_derivative(f2, x2, xnew)
znew
znew[2] = lie_derivative(f2, x2, xnew, order=2)
znew
Matrix(gradient(f2[3], xnew)).dot(f2)
znew[3] = lie_derivative(f2, x2, xnew, order=3)
znew
print(latex(znew))
alpha
alpha
print(latex(alpha))
lie_derivative(f2, x2, xnew, order=3)
gradient(lie_derivative(f2, x2, xnew, order=3), xnew)
Matrix(gradient(lie_derivative(f2, x2, xnew, order=3), xnew)).dot(f2)
znew
znew[2]
z3 = Symbol('z3')
Eq(znew[2], z3)
solve(Eq(znew[2], z3), x1)
solve(Eq(znew[2], z3), x1)[0]
z3 = Symbol('z_3')
Eq(znew[2], z3)
solve(Eq(znew[2], z3), x1)[0]
z4 = Symbol('z_4')
solve(Eq(znew[3], z4), x3)[0]
x1_z3 = solve(Eq(znew[2], z3), x1)[0]
x1_z3
x3_z4 = solve(Eq(znew[3], z4), x3)[0]
x3_z4
z1, z2, z3, z4 = Symbols('z_1 z_2 z_3 z_4')
z1, z2, z3, z4 = symbols('z_1 z_2 z_3 z_4')
x1
x1_z3
x3_z4.replace(x2, z1)
x1_z3.replace(x2, z1).replace(x4, z2)
print(latex(x1_z3.replace(x2, z1).replace(x4, z2)))
x3_z4.replace(x2, z1)
x3_z4.replace(x2, z1).replace(x4, z2)
x3_z4.replace(x2, z1).replace(x4, z2).replace(x1, x1_z3.replace(x2, z1).replace(x4, z2))
x3_z4.replace(x2, z1).replace(x4, z2).replace(x1, x1_z3.replace(x2, z1).replace(x4, z2))
simplify(x3_z4.replace(x2, z1).replace(x4, z2).replace(x1, x1_z3.replace(x2, z1).replace(x4, z2)))
tau_inv = Matrix([0, 0, 0, 0])
tau_inv[1] = z1
tau_inv[3] = z2
tau_inv
tau_inv[0] = x1_z3.replace(x2, z1).replace(x4, z2)
tau_inv
z3
tau
znew
x1_z3 = solve(Eq(znew[2], z3), x1)[0]
tau_inv[0] = solve(Eq(znew[2], z3), x1)[0].replace(x2, z1).replace(x4, z2)
tau_inv
x3_z4 = solve(Eq(znew[3], z4), x3)[0]
x3_z4
x3_z4 = solve(Eq(znew[3], z4), x3)[0].replace(x2, z1).replace(x4, z2).replace(x1, x1_z3)
x3_z4
x3_z4 = simplify(solve(Eq(znew[3], z4), x3)[0].replace(x2, z1).replace(x4, z2).replace(x1, x1_z3))
x3_z4
x3_z4 = solve(Eq(znew[3], z4), x3)[0]
x3_z4
x3_z4.replace(x4, z2)
x3_z4.replace(x4, z2).replace(x2, z1)
x3_z4.replace(x4, z2).replace(x2, z1).replace(x1, x1_z3)
simplify(x3_z4.replace(x4, z2).replace(x2, z1).replace(x1, x1_z3))
simplify(x3_z4.replace(x4, z2).replace(x2, z1).replace(x1, x1_z3)).replace(x2, z1).replace(x4, z2)
x3_z4.replace(x4, z2).replace(x2, z1).replace(x1, x1_z3).replace(x2, z1).replace(x4, z2)
simplify(x3_z4.replace(x4, z2).replace(x2, z1).replace(x1, x1_z3).replace(x2, z1).replace(x4, z2))
tau_inv
tau_inv[2] = simplify(x3_z4.replace(x4, z2).replace(x2, z1).replace(x1, x1_z3).replace(x2, z1).replace(x4, z2))
tau_inv
print(latex(tau_inv))
k1, k2, k3, k4 = symbols('k1 k2 k3 k4')
Matrix([k1, k2, k3, k4])
k = Matrix([k1, k2, k3, k4])
Matrix([0, 0, 0, 0 ])
Matrix([0, 0, 0, 0 ]).T
Matrix([0, 0, 0, 0 ]).dot(k.T)
Matrix([0, 0, 0, 1]).dot(k.T)
Matrix([0, 0, 0, 1]).dot(k)
Matrix([0, 0, 0, 1]).cross(k)
Matrix([0, 0, 0, 1]).cross(k.T)
Matrix([0, 0, 0, 1]).T.cross(k)
Matrix([0, 0, 0, 1]).cross(k)
Matrix([0, 0, 0, 1]).dot(k)
tau_inv
tau
tau_inv
znew
znew[2]
print(python(znew[2]))
print(python(znew[2]))[-1]
temp python(znew[2])
python(znew[2])
python(znew[2])[-1]
python(znew[2])
python(znew[2]).split("\n")
python(znew[2]).split("\n")[-1]
python(znew[2]).split("\n")[-1][4:]
python(znew[2]).split("\n")[-1][3:]
python(znew[2]).split("\n")[-1][3:].replace('x1(t)', 'x[0]')
python(znew[2]).split("\n")[-1][3:].replace('x1(t)', 'x[0]').replace('x2(t)', 'x[1]')
python(znew[2]).split("\n")[-1][3:].replace('x1(t)', 'x[0]').replace('x2(t)', 'x[1]').replace('x3(t)', 'x[2]').replace('x4(t)', 'x[3]')
python(znew[3]).split("\n")[-1][3:].replace('x1(t)', 'x[0]').replace('x2(t)', 'x[1]').replace('x3(t)', 'x[2]').replace('x4(t)', 'x[3]')
alpha
python(alpha).split("\n")[-1][3:].replace('x1(t)', 'x[0]').replace('x2(t)', 'x[1]').replace('x3(t)', 'x[2]').replace('x4(t)', 'x[3]')
f
f2
znew
lie_derivative(f, znew[3], xnew)
-lie_derivative(f, znew[3], xnew)
simplify(-lie_derivative(f, znew[3], xnew))
xnew
f
f2
J1
Eq(J1*q1.diff(t).diff(t) + F1*q1.diff(t) + K / N *(q2 - q1 / N), u)
u = Symbol('u')
Eq(J1*q1.diff(t).diff(t) + F1*q1.diff(t) + K / N *(q2 - q1 / N), u)
solve(Eq(J1*q1.diff(t).diff(t) + F1*q1.diff(t) + K / N *(q2 - q1 / N), u), q1.diff(t).diff(t))
solve(Eq(J1*q1.diff(t).diff(t) + F1*q1.diff(t) + K / N *(q2 - q1 / N), u), q1.diff(t).diff(t))[0]
simplify(solve(Eq(J1*q1.diff(t).diff(t) + F1*q1.diff(t) + K / N *(q2 - q1 / N), u), q1.diff(t).diff(t))[0])
simplify(solve(Eq(J2*q2.diff(t).diff(t) + F2*q2.diff(t) + K*(q2 - q1 / N), -m*g*d*cos(q2)), q2.diff(t).diff(t))[0])
simplify(solve(Eq(J2*q2.diff(t).diff(t) + F2*q2.diff(t) + K*(q2 - q1 / N), -m*g*d*cos(q2)), q2.diff(t).diff(t)))
solve(Eq(J2*q2.diff(t).diff(t) + F2*q2.diff(t) + K*(q2 - q1 / N), -m*g*d*cos(q2)), q2.diff(t).diff(t))
Eq(J2*q2.diff(t).diff(t) + F2*q2.diff(t) + K*(q2 - q1 / N), -m*g*d*cos(q2))
q1
q2
Eq(J2*q2.diff(t).diff(t) + F2*q2.diff(t) + K*(q2 - q1 / N), 0)
Eq(J2*q2.diff(t).diff(t) + F2*q2.diff(t) + K*(q2 - q1 / N), -m)
Eq(J2*q2.diff(t).diff(t) + F2*q2.diff(t) + K*(q2 - q1 / N), -m*cos(q2))
Eq(J2*q2.diff(t).diff(t) + F2*q2.diff(t) + K*(q2 - q1 / N), -m*g*d*cos(q2))
Eq(J2*q2.diff(t).diff(t) + F2*q2.diff(t) + K*(q2 - q1 / N), -m*g*cos(q2))
Eq(J2*q2.diff(t).diff(t) + F2*q2.diff(t) + K*(q2 - q1 / N), -m*cos(q2))
Eq(J2*q2.diff(t).diff(t) + F2*q2.diff(t) + K*(q2 - q1 / N), -m*g*cos(q2))
g
G = Symbol('G')
Eq(J2*q2.diff(t).diff(t) + F2*q2.diff(t) + K*(q2 - q1 / N), -m*G*cos(q2))
Eq(J2*q2.diff(t).diff(t) + F2*q2.diff(t) + K*(q2 - q1 / N), -m*G*d*cos(q2))
solve(Eq(J2*q2.diff(t).diff(t) + F2*q2.diff(t) + K*(q2 - q1 / N), -m*G*d*cos(q2)), q2.diff(t).diff(t))
solve(Eq(J2*q2.diff(t).diff(t) + F2*q2.diff(t) + K*(q2 - q1 / N), -m*G*d*cos(q2)), q2.diff(t).diff(t))[0]
solve(Eq(J2*q2.diff(t).diff(t) + F2*q2.diff(t) + K*(q2 - q1 / N), -m*G*d*cos(q2)), q2.diff(t).diff(t))[0].replace(q1, x1).replace(q2, x2).replace(q1.diff(t), x3).replace(q2.diff(t), x4)
simplify(solve(Eq(J1*q1.diff(t).diff(t) + F1*q1.diff(t) + K / N *(q2 - q1 / N), u), q1.diff(t).diff(t))[0])
simplify(solve(Eq(J1*q1.diff(t).diff(t) + F1*q1.diff(t) + K / N *(q2 - q1 / N), u), q1.diff(t).diff(t))[0]).replace(q1, x1).replace(q2, x2).replace(q1.diff(t), x3).replace(q2.diff(t), x4)
simplify(solve(Eq(J1*q1.diff(t).diff(t) + F1*q1.diff(t) + K / N *(q2 - q1 / N), u), q1.diff(t).diff(t))[0]).replace(q1, x1).replace(q2, x2).replace(q1.diff(t), x3).replace(q2.diff(t), x4).replace(u, 0)
simplify(solve(Eq(J1*q1.diff(t).diff(t) + F1*q1.diff(t) + K / N *(q2 - q1 / N), u), q1.diff(t).diff(t))[0]).replace(q1.diff(t), x3).replace(q2.diff(t), x4).replace(q1, x1).replace(q2, x2).replace(u, 0)
print(latex(simplify(solve(Eq(J1*q1.diff(t).diff(t) + F1*q1.diff(t) + K / N *(q2 - q1 / N), u), q1.diff(t).diff(t))[0]).replace(q1.diff(t), x3).replace(q2.diff(t), x4).replace(q1, x1).replace(q2, x2).replace(u, 0)))
f2
f2[2] = simplify(solve(Eq(J1*q1.diff(t).diff(t) + F1*q1.diff(t) + K / N *(q2 - q1 / N), u), q1.diff(t).diff(t))[0]).replace(q1.diff(t), x3).replace(q2.diff(t), x4).replace(q1, x1).replace(q2, x2).replace(u, 0)
f2
solve(Eq(J2*q2.diff(t).diff(t) + F2*q2.diff(t) + K*(q2 - q1 / N), -m*G*d*cos(q2)), q2.diff(t).diff(t))[0].replace(q1, x1).replace(q2, x2).replace(q1.diff(t), x3).replace(q2.diff(t), x4)
solve(Eq(J2*q2.diff(t).diff(t) + F2*q2.diff(t) + K*(q2 - q1 / N), -m*G*d*cos(q2)), q2.diff(t).diff(t))[0].replace(q1.diff(t), x3).replace(q2.diff(t), x4).replace(q1, x1).replace(q2, x2)
f2[2] = simplify(solve(Eq(J1*q1.diff(t).diff(t) + F1*q1.diff(t) + K / N *(q2 - q1 / N), u), q1.diff(t).diff(t))[0]).replace(q1.diff(t), x3).replace(q2.diff(t), x4).replace(q1, x1).replace(q2, x2).replace(u, 0)
solve(Eq(J2*q2.diff(t).diff(t) + F2*q2.diff(t) + K*(q2 - q1 / N), -m*G*d*cos(q2)), q2.diff(t).diff(t))[0].replace(q1.diff(t), x3).replace(q2.diff(t), x4).replace(q1, x1).replace(q2, x2)
solve(Eq(J2*q2.diff(t).diff(t) + F2*q2.diff(t) + K*(q2 - q1 / N), -m*G*d*cos(q2)), q2.diff(t).diff(t))[0].replace(q1.diff(t), x3).replace(q2.diff(t), x4).replace(q1, x1).replace(q2, x2)
temp = solve(Eq(J2*q2.diff(t).diff(t) + F2*q2.diff(t) + K*(q2 - q1 / N), -m*G*d*cos(q2)), q2.diff(t).diff(t))[0].replace(q1.diff(t), x3).replace(q2.diff(t), x4).replace(q1, x1).replace(q2, x2)
temp
f2[3] = solve(Eq(J2*q2.diff(t).diff(t) + F2*q2.diff(t) + K*(q2 - q1 / N), -m*G*d*cos(q2)), q2.diff(t).diff(t))[0].replace(q1.diff(t), x3).replace(q2.diff(t), x4).replace(q1, x1).replace(q2, x2)
f2
print(latex(f2[3]))
lie_derivative(f2, x2, xnew)
lie_derivative(f2, x2, xnew, order=0)
lie_derivative(f2, x2, xnew, order=1)
lie_derivative(f2, x2, xnew, order=2)
lie_derivative(f2, x2, xnew, order=3)
lie_derivative(f2, x2, xnew, order=4)
simplify(lie_derivative(f2, x2, xnew, order=4))
simplify(lie_derivative(f2, x2, xnew, order=4))
lie_derivative(f2, x2, xnew, order=3)
lie_derivative(g, lie_derivative(f2, x2, xnew, order=3), xnew)
f2
f2[2]
pull_out_term(f2[2], x3, [x3])
pull_out_term(f2[2], x3, [])
collect(f2[2], x3, evaluate=False)
x3
collect(f2[2], x2, evaluate=False)
expand(f2[2])
collect(expand(f2[2]), x2, evaluate=False)
pull_out_term(expand(f2[2], x2)
)
pull_out_term(expand(f2[2]), x2)
history

    z = self.tau(x)
    v = np.dot(k, z)
    u = self.alpha(x) + np.dot(self.beta(x), v)

    # plant
    xdot[0] = x[Q1DOT]
    xdot[1] = x[Q2DOT]

    xdot[2] = (-F1*x[Q1DOT] - K*(x[Q2] - x[Q1]/N)/N)/J1
    xdot[2] += (u / J1)

    xdot[3] = (-F2*x[Q2DOT] - K*(x[Q2] - x[Q1]/N) - d*g*m*cos(x[Q2]))/J2



  ############################################################################

  # def alpha(self, x):
  #   return F1**3*J2*N**4*x[2]/(F1**2*J2*N**4 + J1*J2*K*N**2) + F1**2*J2*K*N**3*x[1]/(F1**2*J2*N**4 + J1*J2*K*N**2) - F1**2*J2*K*N**2*x[0]/(F1**2*J2*N**4 + J1*J2*K*N**2) - F1*J1*J2*K*N**3*x[3]/(F1**2*J2*N**4 + J1*J2*K*N**2) + 2*F1*J1*J2*K*N**2*x[2]/(F1**2*J2*N**4 + J1*J2*K*N**2) - F2*J1**2*K*N**3*x[3]/(F1**2*J2*N**4 + J1*J2*K*N**2) - G*J1**2*K*N**3*d*M*cos(x[1])/(F1**2*J2*N**4 + J1*J2*K*N**2) - J1**2*K**2*N**3*x[1]/(F1**2*J2*N**4 + J1*J2*K*N**2) + J1**2*K**2*N**2*x[0]/(F1**2*J2*N**4 + J1*J2*K*N**2) + J1*J2*K**2*N*x[1]/(F1**2*J2*N**4 + J1*J2*K*N**2) - J1*J2*K**2*x[0]/(F1**2*J2*N**4 + J1*J2*K*N**2)

  # def beta(self, x):
  #   return J1/(F1**2/J1**2 + K/(J1*N**2))

  # def tau(self, x):
  #   return np.array([
  #     x[0],
  #     x[2],
  #     -F1*x[2]/J1 - K*x[1]/(J1*N) + K*x[0]/(J1*N**2),
  #     -F1*(-F1*x[2]/J1 - K*x[1]/(J1*N) + K*x[0]/(J1*N**2))/J1 - K*x[3]/(J1*N) + K*x[2]/(J1*N**2),
  #     ])

  # def tau_inv(self, z):
  #   return np.array([
  #   N*(F1*N*x3(t) + J1*N*z[2] + K*z[0])/K,
  #   z[0],
  #   N*(F1**2*N*x3(t) + F1*J1*N*z[2] + J1**2*N*z[3] + J1*K*z[1])/(F1**2*N**2 + J1*K),
  #   z[1],
  # ])