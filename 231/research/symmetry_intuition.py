#!/usr/bin/env python3

'''
Is A2 symmetric?
Is A1 symmetric?
'''

import argparse
import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt

def C(x, factorial_map, n=15):
    '''
    power series approximation of fresnel integral S(x)
    '''
    res = np.zeros_like(x)
    res = 0.0
    for i in range(n):
        d = ((-1)**i)/(factorial_map[2*i+1] * (4*i+1)) * x**(4*i+1) 
        print(d)
        res += d
    return res

def S(x, factorial_map, n=15):
    '''
    power series approximation of fresnel integral C(x)
    '''
    res = np.zeros_like(x)
    for i in range(n):
        res += ((-1)**i) / (factorial_map[2*i] * (4*i+3)) * x**(4*i+3)
    return res

def numpybroadcast_dotproduct():
    '''
    you have terms for a matrix in meshgrid form
    meaning meshgrid[i, j, k...] cell is a matrix element
    to a 
    '''
    pass

#############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--times',
        type=str,
        default="0,0.1,0.25,0.5,1.0,5.0,10.0",
        required=False)

    parser.add_argument('--mu_0',
        type=float,
        default=2.0,
        required=False)

    parser.add_argument('--sampling',
        type=str,
        default="15,15,15,15,15,15,100,200",
        required=False)

    args = parser.parse_args()

    # system
    alpha2 = 0.5

    '''
    # this linspace excludes 0, 1.09
    x03 = np.linspace(-12, 12, 12)
    x13 = np.linspace(-12, 12, 12)

    X03, X13 = np.meshgrid(x03, x13, copy=False)

    # use broadcasting to speed things up
    Y = alpha2 / 2 * (X03 + X13)
    CY = np.cos(Y)
    SY = np.sin(Y)

    # for E / F / L
    t1 = alpha2/ (2*(X03 - X13)) * X13**2
    t1 = np.nan_to_num(t1, copy=False, posinf=0.0)
    t0 = alpha2/ (2*(X03 - X13)) * X03**2
    t0 = np.nan_to_num(t0, copy=False, posinf=0.0)
    den = np.sqrt(alpha2*(X03 - X13)) / np.sqrt(2)
    den = np.nan_to_num(den, copy=False)

    # fresnel integrals
    
    a1 = np.cos(t1) / den
    a1 = np.nan_to_num(a1, copy=False, posinf=0.0, neginf=0.0)
    a2 = np.sin(t1) / den
    a2 = np.nan_to_num(a2, copy=False, posinf=0.0, neginf=0.0)

    e = a1 * (C(t1) - C(t0)) + a2 * (S(t1) - S(t0))
    f = -a2 * (C(t1) - C(t0)) + a1 * (S(t1) - S(t0))
    l = ((S(t1) - S(t0)) ** 2 + (C(t1) - C(t0)) ** 2) / den

    np.cos(term) / den
    '''


    # t = np.linspace(0, 1, 1024)
    # # t = t.reshape(12, 12)

    # print(t)


    n = 5
    factorial_map = {}
    for i in range(n):
        factorial_map[2*i+1] = np.math.factorial(2*i+1)
        factorial_map[2*i] = np.math.factorial(2*i)

    # x = C(t, factorial_map)
    # y = S(t, factorial_map)

    t = np.linspace(0, 10, 1024)
    x = [C(x, factorial_map, n) for x in t]
    y = S(t, factorial_map, n)

    # x = x.reshape(-1)
    # y = y.reshape(-1)

    fig = plt.figure(1)
    ax1 = plt.subplot(111)
    # ax1.set_aspect('equal')
    ax1.grid()

    ax1.plot(x, y,
        'r',
        linewidth=1,
        label='euler')
    ax1.legend(loc='lower right')


    # plt.gca().set_aspect(
    #     'equal', adjustable='box')

    plt.show()




def swap_symbol(expr, a, b, t=Symbol('q')):
    temp = expr.replace(a, t)
    temp = temp.replace(b, a)
    temp = temp.replace(t, b)
    return temp


def swap_symbols(expr, a, b, t=Symbol('q')):
    for i, a_ in enumerate(a):
        expr = swap_symbol(expr, a_, b[i], t)
    return expr


    a = c
    c = b
    b = a


'''
import numpy as np
np.linspace(-10, 10, 5)
np.linspace(-10, 10, 6)
np.linspace(-10, 10, 12)
np.linspace(-15, 15, 12)
np.linspace(-12, 12, 12)
x03 = np.linspace(-12, 12, 12)
x13 = np.linspace(-12, 12, 12)
X03, X13 = np.meshgrid(x03, x13, copy=False)
X03
X03 + X13
alpha2 = 0.5
alpha2 / 2 * (X03 + X13)
Y = alpha2 / 2 * (X03 + X13)
CY = np.cos(Y)
SY = np.sin(Y)
x03 = np.linspace(-12, 12, 12)
x02 = np.linspace(-12, 12, 12)
x01 = np.linspace(-12, 12, 12)
X01, X02, X03 = np.meshgrid(x01, x02, x03, copy=False)
X01
X01.shape
X03
X03.shape
X13
X13.shape
a . np.arange(25).reshape(5, 5)
np.arange(25).reshape(5, 5)
    den = np.sqrt(alpha2*(X03 - X13)) / np.sqrt(2)
den
den = np.sqrt(alpha2*(X03 - X13)) / np.sqrt(2)
np.nan_to_num(den)
den = np.nan_to_num(den)
    t1 = alpha2/ (2*(X03 - X13)) * X13**2
    t0 = alpha2/ (2*(X03 - X13)) * X03**2
    a1 = np.cos(t1) / den
    a1 = np.cos(t1) / den
    a1 = np.nan_to_num(a1, copy=False)
    a2 = np.sin(t1) / den
    a2 = np.nan_to_num(a2, copy=False)
a1
a2
t1
t1.shape
    t1 = np.nan_to_num(t1, copy=False, posinf=0.0)
t1
    t0 = np.nan_to_num(t0, copy=False, posinf=0.0)
    den = np.sqrt(alpha2*(X03 - X13)) / np.sqrt(2)
den
    den = np.nan_to_num(den, copy=False)
den
    a1 = np.cos(t1) / den
a1
    a1 = np.nan_to_num(a1, copy=False, posinf=0.0, neginf=0.0)
a1
    a2 = np.sin(t1) / den
    a2 = np.nan_to_num(a2, copy=False, posinf=0.0, neginf=0.0)
a2
np.math.factorial(3)
np.zeros_like(t1)
temp = np.zeros_like(t1)
temp.shape
-1**0
-1**1
-1**2
(-1)**2
(-1)**0
import sympy
from sympy import *
sympy.init_printing()
e, f, l, cy, sy = symbols('e f l cy sy')
e, f, l, cy, sy, g, h = symbols('e f l cy sy g h')
A1 = Matrix([[e/l, f/l, 0], [-f/l, e/l, 0], [0, 0, 1]])
A2 = Matrix([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
A1*A2
A1 = Matrix([[g, h, 0], [-h, g, 0], [0, 0, 1]])
A1
A1*A2
A1
x11, x12, x13 = symbols('x11 x12 x13')
x01, x02, x03 = symbols('x01 x02 x03')
x0 = Matrix([[x01], [x02], [x03]])
x1 = Matrix([[x11], [x12], [x13]])
x1
A1*x1
A1*x1
A1*x1 - A1*A2*x0
b1, b2, b3 = symbols('b1 b2 b3')
B = Matrix([[b1, 0, 0], [0, b2, 0], [0, 0, b3]])
B.inv * (A1*x1 - A1*A2*x0)
B.inv() * (A1*x1 - A1*A2*x0)
((A1*x1).transpose() - (A1*A2*x0).transpose()) * B.inv() * (A1*x1 - A1*A2*x0)
(A1*x1 - A1*A2*x0).transpose() * B.inv() * (A1*x1 - A1*A2*x0)
expand((A1*x1 - A1*A2*x0).transpose() * B.inv() * (A1*x1 - A1*A2*x0))
(A1*x1 - A1*A2*x0).transpose() * B.inv() * (A1*x1 - A1*A2*x0)
(A1*x1 - A1*A2*x0).transpose() * B.inv() * (A1*x1 - A1*A2*x0)[0]
c = (A1*x1 - A1*A2*x0).transpose() * B.inv() * (A1*x1 - A1*A2*x0)
c
c[0]
(g*x12 - h*x11 - x01*(-cy*h + g*sy) - x02*(cy*g+h*sy))**2
expand((g*x12 - h*x11 - x01*(-cy*h + g*sy) - x02*(cy*g+h*sy))**2)
(g*x12 - h*x11 - x01*(-cy*h + g*sy) - x02*(cy*g+h*sy))**2
t1 = (g*x12 - h*x11 - x01*(-cy*h + g*sy) - x02*(cy*g+h*sy))**2
c
c[0]
t2 = (g*x11 + h*x12 - x01*(cy*g + h*sy) - x02*(cy*h - g*sy))**2
c[0] - t1 - t2
simplify(c[0] - t1 - t2)
t1
c[0] - t1 / b2 - t2 / b1
t1 / b2 + t2 / b1
c[0] - t1 / b2 + t2 / b1
c[0] - t1 / b2
c[0] - t1 / b2 - t2 / b1
t1 / b2 + t2 / b1
t1
t1.replace(x12, x02).replace(x11, x01).replace(x13, x03)
t1 / b2 + t2 / b1
expand(t1 / b2 + t2 / b1)
Symbol('q')
def swap_symbols(expr, a, b, swap_symbol=Symbol('q')):
    temp = expr.replace(a, swap_symbol)
    temp = temp.replace(b, a)
    temp = temp.replace(swap_symbol, b)
    return temp
expand(t1 / b2 + t2 / b1)
exp = expand(t1 / b2 + t2 / b1)
swap_symbols(exp, x01, x11)
def swap_symbol(expr, a, b, swap_symbol=Symbol('q')):
    temp = expr.replace(a, swap_symbol)
    temp = temp.replace(b, a)
    temp = temp.replace(swap_symbol, b)
    return temp
def swap_symbols(expr, a, b, swap_symbol=Symbol('q')):
    for i, a_ in enumerate(a):
        expr = swap_symbol(expr, a_, b[i], swap_symbol)
    return expr
c
e
f
e + f
test = e + f
swap_symbol(test, e, f)
test = e + f / 2
swap_symbol(test, e, f)
test = e + f / 2 + g / 3 + h / 4
swap_symbol(test, e, f)
swap_symbols(test, [e, g], [f, h])
def swap_symbol(expr, a, b, t=Symbol('q')):
    temp = expr.replace(a, t)
    temp = temp.replace(b, a)
    temp = temp.replace(t, b)
    return temp


def swap_symbols(expr, a, b, t=Symbol('q')):
    for i, a_ in enumerate(a):
        expr = swap_symbol(expr, a_, b[i], t)
    return expr
swap_symbols(test, [e, g], [f, h])
test
exp = expand(t1 / b2 + t2 / b1)
exp
swap_symbols(exp, [x01, x02, x03], [x11, x12, x13])
exp2 = swap_symbols(exp, [x01, x02, x03], [x11, x12, x13])
exp2 - exp
simplify(exp2 - exp)
exp - exp2
simplify(exp - exp2)
t1
temp1 = Matrix([[1/b1, 0, 0], [0, 1/b2,0], [0, 0, 1/b3]])
temp1
B
B.inv()
A2
A1
A1.transpose() * B.inv() * A1
x1
x1 - x0
middle
mid = A1.transpose() * B.inv() * A1
mid
(x1 - x0).transpose() * mid * (x1 - x0)
(x1 - x0).transpose() * mid * (x1 - x0) - (x0 - x1).transpose() * mid * (x0 - x1)
simplify((x1 - x0).transpose() * mid * (x1 - x0) - (x0 - x1).transpose() * mid * (x0 - x1))
(x1 - x0).transpose() * mid * (x1 - x0) - (x0 - x1).transpose() * mid * (x0 - x1)
(x1 - x0).transpose() * mid * (x1 - x0) - (x0 - x1).transpose() * mid * (x0 - x1)
simplify((x1 - x0).transpose() * mid * (x1 - x0))
(x1 - x0).transpose() * mid * (x1 - x0)
t = (x1 - x0).transpose() * mid * (x1 - x0)
t[0]
t[0]
mid
A1
mid[0, 1]
mid[1, 0]
mid[0, 1] - mid[1, 0]
A1
A2
A1
  history

'''