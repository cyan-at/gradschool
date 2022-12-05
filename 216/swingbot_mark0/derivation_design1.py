#!/usr/bin/env python3

import sympy
from sympy import *
from sympy.physics.mechanics import *
sympy.init_printing()

from sympy_utils import *
import numpy as np
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

import argparse
import scipy.linalg as la

import scipy
import scipy.integrate as integrate

def solve_DARE_with_iteration(A, B, Q, R):
    """
    solve a discrete time_Algebraic Riccati equation (DARE)
    """
    X = Q
    maxiter = 150
    eps = 0.01

    for i in range(maxiter):
        Xn = A.T * X * A - A.T * X * B * \
            la.inv(R + B.T * X * B) * B.T * X * A + Q
        if (abs(Xn - X)).max() < eps:
            X = Xn
            break
        X = Xn

    return Xn

def dlqr_with_iteration(Ad, Bd, Q, R):
    """Solve the discrete time lqr controller.
    x[k+1] = Ad x[k] + Bd u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
    """

    # first, try to solve the ricatti equation
    # X = solve_DARE_with_iteration(Ad, Bd, Q, R)

    X = la.solve_discrete_are(Ad, Bd, Q, R)

    # compute the LQR gain
    K = np.matrix(la.inv(Bd.T * X * Bd + R) * (Bd.T * X * Ad))

    return K


def dlqr_with_arimoto_potter(Ad, Bd, Q, R, dt=0.1):
    """Solve the discrete time lqr controller.
    x[k+1] = Ad x[k] + Bd u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
    """

    n = len(Bd)

    # continuous
    Ac = (Ad - np.eye(n)) / dt
    Bc = Bd / dt

    # Hamiltonian
    Ham = np.vstack(
        (np.hstack((Ac, - Bc * la.inv(R) * Bc.T)),
         np.hstack((-Q, -Ac.T))))

    eigVals, eigVecs = la.eig(Ham)

    V1 = None
    V2 = None

    for i in range(2 * n):
        if eigVals[i].real < 0:
            if V1 is None:
                V1 = eigVecs[0:n, i]
                V2 = eigVecs[n:2 * n, i]
            else:
                V1 = np.vstack((V1, eigVecs[0:n, i]))
                V2 = np.vstack((V2, eigVecs[n:2 * n, i]))
    V1 = np.matrix(V1.T)
    V2 = np.matrix(V2.T)

    P = (V2 * la.inv(V1)).real

    K = la.inv(R) * Bc.T * P

    return K

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument('--playback', type=int, default=1, help='')
    parser.add_argument('--history', type=int, default=500, help='')
    parser.add_argument('--plot', type=str, default="animation", help='')
    parser.add_argument('--dt', type=float, default=0.02, help='')
    parser.add_argument('--t_stop', type=int, default=300, help='')

    parser.add_argument('--system', type=str, default="5,5,1,1,1", help='')

    parser.add_argument('--initial', type=str, default="0,0,1,0", help='')

    args = parser.parse_args()

    system_params = [float(x) for x in args.system.strip().split(",")]
    initial_state = np.radians([float(x) for x in args.initial.split(',')])
    L1_sub, M1_sub, L2_sub, M2_sub, Q1_DAMPING_sub = system_params
    G_sub = 9.8

    '''
    ---------------------------------------------------------------
    # define symbols
    '''

    t = Symbol('t')
    m1, l1, m2, l2, m3, l3, g, beta_damping, v =\
        symbols('M1 L1 M2 L2 M3 L3 G Q1_DAMPING v', nonzero=True, positive=True)
    t1, t2, t3 = dynamicsymbols('t1 t2 t3') # state vars

    t4, t5, t6, t7 = symbols('t1 t2 t1_dot t2_dot')
    '''
    ---------------------------------------------------------------
    # free-body-diagram is expressed here
    '''

    x1 = l1*sin(t1)
    y1 = -l1*cos(t1)
    x2 = x1 + l2*sin(t2)
    y2 = y1 - l2*cos(t2)

    # convienence

    dt1 = t1.diff(t)
    dt2 = t2.diff(t)

    x1dot = x1.diff(t)
    y1dot = y1.diff(t)
    x2dot = x2.diff(t)
    y2dot = y2.diff(t)

    v1_2 = simplify(x1dot**2 + y1dot**2)
    v2_2 = simplify(x2dot**2 + y2dot**2)

    '''
    ---------------------------------------------------------------
    # Lagrange mechanics: the Lagrangian
    '''

    T = m1 / 2 * v1_2 + m2 / 2 * v2_2

    U = m1*g*y1 + m2*g*y2

    L = simplify(T - U)

    '''
    ---------------------------------------------------------------
    # Euler-Lagrange Eq., do one per state var
    # do NOT add any damping here, this is the 'LHS'
    # and it is assumed for now there is NOT Q input force to the system
    # so right now RHS is still = 0
    '''

    # unactuated states

    p1 = simplify(L.diff(dt1))
    el1 = simplify(p1.diff(t) - L.diff(t1))

    # actuated states

    p2 = simplify(L.diff(dt2))
    el2 = simplify(p2.diff(t) - L.diff(t2))

    # simplify

    try:
        el1 = simplify_eq_with_assumptions(Eq(el1, 0)).lhs
    except:
        pass

    el1_latex = latex(Eq(el1, 0))
    el1_latex = el1_latex.replace("t_{1}", "\\theta_{1}")
    el1_latex = el1_latex.replace("t_{2}", "\\theta_{2}")
    print("el1_latex")
    print(el1_latex)

    try:
        el2 = simplify_eq_with_assumptions(Eq(el2, 0)).lhs
    except:
        pass

    el2_latex = latex(Eq(el2, 0))
    el2_latex = el2_latex.replace("t_{1}", "\\theta_{1}")
    el2_latex = el2_latex.replace("t_{2}", "\\theta_{2}")
    print("el2_latex")
    print(el2_latex)

    '''
    ---------------------------------------------------------------
    # construct (the manipulator matrices.)
    '''

    # mass-matrix M
    # note that here the term right-multiplied is q** exactly
    M11, el1 = pull_out_term(el1, dt1.diff(t))
    M12, el1 = pull_out_term(el1, dt2.diff(t))
    M21, el2 = pull_out_term(el2, dt1.diff(t))
    M22, el2 = pull_out_term(el2, dt2.diff(t))
    M = Matrix([[M11, M12], [M21, M22]])

    M_latex = latex(M)
    M_latex = M_latex.replace("t_{1}", "\\theta_{1}")
    M_latex = M_latex.replace("t_{2}", "\\theta_{2}")
    print("M_latex")
    print(M_latex)

    # velocity-cross-product matrix C
    # note that here the term right-multiplied is q*
    # and we are pulling out velocity products
    C11, el1 = pull_out_term(el1, dt1, [dt1, dt2])
    C12, el1 = pull_out_term(el1, dt2, [dt1, dt2])
    C21, el2 = pull_out_term(el2, dt1, [dt1, dt2])
    C22, el2 = pull_out_term(el2, dt2, [dt1, dt2])
    C = Matrix([[C11, C12], [C21, C22]])

    C_latex = latex(C)
    C_latex = C_latex.replace("t_{1}", "\\theta_{1}")
    C_latex = C_latex.replace("t_{2}", "\\theta_{2}")
    print("C_latex")
    print(C_latex)


    # gravity matrix tau_g
    tau_g = Matrix([[el1], [el2]])

    tau_g_latex = latex(tau_g)
    tau_g_latex = tau_g_latex.replace("t_{1}", "\\theta_{1}")
    tau_g_latex = tau_g_latex.replace("t_{2}", "\\theta_{2}")
    print("tau_g_latex")
    print(tau_g_latex)

    '''
    ---------------------------------------------------------------
    now we have Mq** + Cq* + tau_g = 0
    we can add external forces Q to the RHS
    and then solve for 
    Mq** = Q - Cq* - tau_g
    Mq** = tau (unrelated to tau_g, just a symbol for the RHS)
    '''

    Q = Matrix([[-beta_damping * dt1], [v]])
    # v is theoretical, we do PD control on theta2** directly
    # but we need a v so it is nonzero
    # B is nonzero, system symbolic dynamics is theoretically controllable
    tau = rhs = simplify(Q - tau_g - C * Matrix([[dt1], [dt2]]))

    Q_latex = latex(Q)
    Q_latex = Q_latex.replace("t_{1}", "\\theta_{1}")
    Q_latex = Q_latex.replace("t_{2}", "\\theta_{2}")
    print("Q_latex")
    print(Q_latex)

    tau_latex = latex(tau)
    tau_latex = tau_latex.replace("t_{1}", "\\theta_{1}")
    tau_latex = tau_latex.replace("t_{2}", "\\theta_{2}")
    print("tau_latex")
    print(tau_latex)

    '''
    ------------------------------------------------------------------------------------------------------------------------------
    ------------------------------------------------------------------------------------------------------------------------------
    ------------------------------------------------------------------------------------------------------------------------------
    '''

    '''
    ---------------------------------------------------------------
    we can solve for dynamics of the system in free-fall with no external control
    '''
    M_inv = simplify(M.inv())
    # M_inv * tau
    qdotdot = simplify(M_inv * tau)

    qdotdot_latex = latex(qdotdot)
    qdotdot_latex = qdotdot_latex.replace("t_{1}", "\\theta_{1}")
    qdotdot_latex = qdotdot_latex.replace("t_{2}", "\\theta_{2}")
    print("qdotdot_latex")
    print(qdotdot_latex)

    for i in range(len(qdotdot)):
        expr = sympy_to_expression(
            qdotdot[i])
        print("qdotdot[%d]" % (i))
        print(expr)

    '''
    ---------------------------------------------------------------
    we can do partial feedback linearization expressions

    for example to get collocated pfl we solve for unactuated = f(actuated)
    then eliminate unactuated term from actuated eq of motion
    so then we have actuated eq of motion = f(actuated q**) = f(external force input)
    and if say our external force input = f(actuated q**)
    then we've equated / linearized the measurement of q**
    y = q2 
    '''
    q1dotdot = 1 / M[0, 0] * (tau[0] - M[0, 1] * dt2.diff(t))

    M_blob2 = (M[1,1] - M[1,0]/M[0,0]*M[0,1])
    q2dotdot = simplify(1/M_blob2*(tau[1] - M[1,0]/M[0,0]*tau[0]))
    q2dotdot_expr = sympy_to_expression(q2dotdot)

    q1dotdot_latex = latex(q1dotdot)
    q1dotdot_latex = q1dotdot_latex.replace("t_{1}", "\\theta_{1}")
    q1dotdot_latex = q1dotdot_latex.replace("t_{2}", "\\theta_{2}")
    print(q1dotdot_latex)

    expr = sympy_to_expression(q1dotdot)
    print(expr)
    # put this in code

    expr_expected = "(-G*M1*sin(t1) - G*M2*sin(t1) - L2*M2*sin(t1 - t2)*t2_dot**2 - L2*M2*cos(t1 - t2)*Derivative(t2, (t, 2)) - Q1_DAMPING*t1_dot)/(L1*M1 + L1*M2)"
    assert(expr, expr_expected)

    '''
    ---------------------------------------------------------------
    we can do linearization around an (unstable) fixed point such as the upright position
    '''

    x = Matrix([t1, dt1, t2, dt2])
    xdot = f = Matrix([[dt1, qdotdot[0], dt2, qdotdot[1]]])

    dfdx = f.jacobian(x)
    dfdx = nsimplify(
        dfdx,
        tolerance=1e-8,
        rational=True)

    dfdv = f.jacobian(Matrix([v]))
    dfdv = nsimplify(
        dfdv,
        tolerance=1e-8,
        rational=True)

    #########################################################

    x_desiredfixedpt = Matrix([np.pi, 0, 0, 0])
    v_desiredfixedpt = 0.0

    dfdx_at_desiredfixedpt = simplify(dfdx.subs([
        (t1, x_desiredfixedpt[0]),
        (dt1, x_desiredfixedpt[1]),
        (t2, x_desiredfixedpt[2]),
        (dt2, x_desiredfixedpt[3]),
        (v, v_desiredfixedpt)
        ]))
    dfdx_at_desiredfixedpt = nsimplify(
        dfdx_at_desiredfixedpt,
        tolerance=1e-8,
        rational=True)

    dfdv_at_desiredfixedpt = simplify(dfdv.subs([
        (t1, x_desiredfixedpt[0]),
        (dt1, x_desiredfixedpt[1]),
        (t2, x_desiredfixedpt[2]),
        (dt2, x_desiredfixedpt[3]),
        (v, v_desiredfixedpt)
        ]))
    dfdv_at_desiredfixedpt = nsimplify(
        dfdv_at_desiredfixedpt,
        tolerance=1e-8,
        rational=True)

    Alin = dfdx_at_desiredfixedpt
    Blin = dfdv_at_desiredfixedpt

    #########################################################

    Alin_subbed = Alin.subs([
        (beta_damping, Q1_DAMPING_sub),
        (m1, M1_sub),
        (m2, M2_sub),
        (l1, L1_sub),
        (l2, L2_sub),
        (g, G_sub)
    ])
    Alin_subbed = np.array(Alin_subbed).astype(np.float64)

    Blin_subbed = Blin.subs([
        (beta_damping, Q1_DAMPING_sub),
        (m1, M1_sub),
        (m2, M2_sub),
        (l1, L1_sub),
        (l2, L2_sub),
        (g, G_sub)
    ])
    Blin_subbed = np.array(Blin_subbed).astype(np.float64)

    Alin_np = np.array(Alin_subbed).astype(np.float64)
    Blin_np = np.array(Blin_subbed).astype(np.float64)


    Kopt = dlqr_with_iteration(Alin_np, Blin_np, np.eye(4), np.eye(1))
    #  elapsed_time = time.time() - start
    #  print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    # u = -uref - Kopt * (x - xref)

    # x = np.array([np.pi * 0.95, 0, 0, 0])
    xref = np.array([np.pi, 0, 0, 0])
    # uref = 0.0
    # u = -uref - Kopt * (x - xref)
    # u = uref - np.dot(Kopt, (x - xref))[0, 0]

    # print(python(Alin_np))
    # print(python(Blin_np))

    # sympy_to_expr_vec = np.vectorize(sympy_to_expression)

    # Alin_np2 = sympy_to_expr_vec(Alin_np)
    # Blin_np2 = sympy_to_expr_vec(Blin_np)

    def derivs(state, t):
        u_opt = 0.0 - np.dot(Kopt, (state - xref))[0, 0]
        xdot =  Alin_subbed @ np.matrix(state - xref).T + Blin_subbed * u_opt
        return np.array(xdot).squeeze()

    times = np.arange(0, 10, 0.2)

    states = integrate.odeint(
        derivs,
        np.array([np.pi * 0.95, 0, 0, 0]),
        times)

    np.save(
        './A_B.npy',
        {
            'A' : Alin_np,
            'B' : Blin_np,
        })

    import ipdb; ipdb.set_trace();


