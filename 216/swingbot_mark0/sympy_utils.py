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

def sympy_to_expression(sympy_expr):
    expr = python(sympy_expr).split("\n")[-1].split(" = ")[-1]

    expr = expr.replace("p1_damping", "Q1_DAMPING")

    expr = expr.replace("t2(t)", "t2")
    expr = expr.replace("t1(t)", "t1")
    expr = expr.replace("Derivative(t2, t)", "t2_dot")
    expr = expr.replace("Derivative(t1, t)", "t1_dot")
    expr = expr.replace("g", "G")
    expr = expr.replace("m", "M")
    expr = expr.replace("l", "L")
    return expr