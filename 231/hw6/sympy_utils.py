#!/usr/bin/env python3

import sympy
from sympy import *
from sympy.physics.mechanics import *
sympy.init_printing()

def simplify_eq_with_assumptions(eq):
    assert eq.rhs == 0  # assert that right-hand side is zero
    print(type(eq.lhs))
    assert type(eq.lhs) == Mul  # assert that left-hand side is a multipl.
    newargs = []  # define a list of new multiplication factors.
    for arg in eq.lhs.args:
        # print(arg)
        # print(type(arg))
        # print(arg.is_positive)
        if type(arg) == Symbol:
            if arg.is_positive:
                print("removing a constant", arg)
                continue  # arg is positive, let's skip it.
        newargs.append(arg)
    # rebuild the equality with the new arguments:
    return Eq(eq.lhs.func(*newargs), 0)

def pull_out_term(expr, term, product_candidates = [], default=0):
    ret = 0
    found = False

    expr_cp = expr

    product_candidates.append(1)
    for c in product_candidates:
        d = collect(expr_cp, term*c, evaluate=False)
        if term*c in d:
            found = True
            print("found")
            ret += c*d[term*c]
            expr_cp = d[1]

    if not found:
        print("not found")
        ret = default

    return ret, expr_cp

def pull_out_manipulator_matrices(euler_lagrange_eqs, statevar_dots, t):
  zeros = [[0] * len(euler_lagrange_eqs)] * len(euler_lagrange_eqs)

  num_eqs = len(euler_lagrange_eqs)
  num_states = len(statevar_dots)

  # mass matrix M
  M = Matrix(zeros)

  for row in range(num_eqs):
    for col in range(num_states):
      statevar_dot = statevar_dots[col]
      M[row, col], euler_lagrange_eqs[row] = pull_out_term(
        euler_lagrange_eqs[row],
        statevar_dot.diff(t))

  # velocity cross-products C
  C = Matrix(zeros)
  for row in range(num_eqs):
    for col in range(num_states):
      statevar_dot = statevar_dots[col]
      C[row, col], euler_lagrange_eqs[row] = pull_out_term(
        euler_lagrange_eqs[row],
        statevar_dot,
        statevar_dots)

  # leftover gravity matrix
  tau_g = Matrix([[0]] * num_eqs)
  for row in range(num_eqs):
    tau_g[row] = euler_lagrange_eqs[row]

  return M, C, tau_g

def sympy_to_expression(sympy_expr, replacements = []):
    expr = python(sympy_expr).split("\n")[-1].split(" = ")[-1]

    expr = expr.replace("p1_damping", "Q1_DAMPING")

    expr = expr.replace("t2(t)", "t2")
    expr = expr.replace("t1(t)", "t1")
    expr = expr.replace("Derivative(t2, t)", "t2_dot")
    expr = expr.replace("Derivative(t1, t)", "t1_dot")
    expr = expr.replace("g", "G")
    expr = expr.replace("m", "M")
    expr = expr.replace("l", "L")

    for r in replacements:
      expr = expr.replace(r[0], r[1])

    return expr

def lie_bracket(f, g, x, order=1):
  '''
    L_f g
    f: vector
    g: vector
    x: state vector
  '''
  res = g
  while order > 0:
    res = simplify(res.jacobian(x) * f - f.jacobian(x) * res)
    order -= 1

  return res

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

def gradient(f, x):
  return [f.diff(y) for y in x]

def latex_cleanup(expr, dynamicvars):
  expr = expr
  for var in dynamicvars:
    expr = expr.replace('\operatorname{'+latex(Symbol(var))+'}{\\left(t \\right)}', latex(Symbol(var)))
  return expr