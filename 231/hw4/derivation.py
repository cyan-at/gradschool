import sympy
from sympy import *
from sympy.physics.mechanics import *
sympy.init_printing()

a, b, x1, x2, y, k, y = symbols('a b x1 x2 y k y')

v = integrate(a*y**3, (y, 0, x1)) + 1/2*x2**2

vdx = Matrix([v.diff(x1), v.diff(x2)])
vdx_t = transpose(vdx)

fx = Matrix([[x2],[-a*x1**3 - b*x2]])

g = Matrix([[0],[1]])
h = Matrix([[0],[1]])

exp = simplify((vdx_t * fx)[0]) + 1 / (2*y) * (vdx_t * g * transpose(g) * transpose(vdx_t))[0] + 1/2 * (transpose(h) * h)[0]
