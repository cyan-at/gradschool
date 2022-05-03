"""
===========================
The double pendulum problem
===========================

This animation illustrates the double pendulum problem.

Double pendulum formula translated from the C code at
http://www.physics.usyd.edu.au/~wheat/dpend_html/solve_dpend.c

##########################################################

qualifications / modeling choices:
NOT compound pendulum, modeled as simple point-mass pendulums
modeled *not* a compound pendulum, but massless links and masses
this impacts the kinetic energy definitions (only translation, no rotational KE)
and therefore the derived equations of motion

from MIT underactuated you see at a meta level what you are doing

1. use Lagrange physics to create <equations of motion>
equations of motion equivalent to derivation here:
https://diego.assencio.com/?index=1500c66ae7ab27bb0106467c68feebc6
http://underactuated.mit.edu/acrobot.html#Spong96

##########################################################

2. RE-write equations of motion to <manipulator matrices: M, C, tau_g>

##########################################################

3. linearize to get dynamics of the system
linearizing actuated joints = collated linearization
linearizing unactuated joints = non-collated linearization

"
We'll use the term collocated partial feedback linearization to describe a controller which
linearizes the dynamics of the actuated joints.
What's more surprising is that it is often possible to achieve non-collocated
partial feedback linearization - a controller which linearizes the dynamics of the unactuated joints.
The treatment presented here follows from [9].
"

##########################################################

now that we have the equations of motion, and manipulator equation
how do we derive a control law w.r.t. what we want to achieve?

##########################################################


"""

from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
from collections import deque
import argparse

G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 10.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
L = L1 + L2  # maximal length of the combined pendulum
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg
t_stop = 60  # how many seconds to simulate
history_len = 500  # how many trajectory points to display

KD = 1.0
KP = 1.0
ALPHA = 1

def derivs_original(state, t):
    '''
        state = x = [
            theta1
            theta1*
            theta2
            theta2*
        ]

        state = x* = [
            theta1*
            theta1**
            theta2*
            theta2**
        ]
    '''
    dydx = np.zeros_like(state)

    dydx[0] = state[1]
    dydx[2] = state[3]

    delta = state[2] - state[0]

    den1 = (M1+M2) * L1 - M2 * L1 * cos(delta) * cos(delta)

    dydx[1] = ((M2 * L1 * state[1] * state[1] * sin(delta) * cos(delta)
                + M2 * G * sin(state[2]) * cos(delta)
                + M2 * L2 * state[3] * state[3] * sin(delta)
                - (M1+M2) * G * sin(state[0]))
               / den1)


    den2 = (L2/L1) * den1

    dydx[3] = ((- M2 * L2 * state[3] * state[3] * sin(delta) * cos(delta)
                + (M1+M2) * G * sin(state[0]) * cos(delta)
                - (M1+M2) * L1 * state[1] * state[1] * sin(delta)
                - (M1+M2) * G * sin(state[2]))
               / den2)

    return dydx

def insert_into_dict_of_arrays(d, k, v, mode="append"):
    """ Aggregates or assigns a dictionary with k, v

    Parameters
    ----------
    d : dict
        dict to modify
    k : any hashable type
        key into dict d
    v : any type
        value to modify at d[k]
    mode : str
        aggregation mode to modify d in, append, add, or override

    Notes
    -----
    side-effect: d is updated
    in 'append' mode, if v is a list it is extend'd into d[k]
    """

    if k in d.keys():
        if mode == "append":
            d[k].append(v)
        if mode == "extend":
            if type(v) == list:
                d[k].extend(v)
            else:
                d[k].append(v)
        elif mode == "add":
            d[k][0] += v
        elif mode == "override":
            d[k] = [v]
    else:
        d[k] = [v]

class Container(object):
    def __init__(self):
        self._data = {
        }

    def derivs_sympy(self, state, t):
        '''
            state = x = [
                theta1
                theta1*
                theta2
                theta2*
            ]

            state = x* = [
                theta1*
                theta1**
                theta2*
                theta2**
            ]
        '''
        dydx = np.zeros_like(state)

        dydx[0] = state[1]
        dydx[2] = state[3]

        t1 = state[0]
        t1_dot = state[1]
        t2 = state[2]
        t2_dot = state[3]
        t1_t2 = t1 - t2

        # both work below, are equivalent

        # this set is from matrix math
        # den1 = L1 * (M1 - M2 * cos(t1_t2)**2 + M2)
        # mine_dydx1 = -G * (M1 + M2) * sin(t1)\
        #     - L2*M2*sin(t1_t2)*t2_dot**2\
        #     + M2*(G*sin(t2) - L1*sin(t1_t2)*t1_dot**2)*cos(t1_t2)
        # dydx[1] = mine_dydx1 / den1

        # den2 = L2*(M1 - M2*cos(t1-t2)**2 + M2)
        # mine_dydx2 = -(M1 + M2)*(G*sin(t2) - L1*sin(t1_t2)*t1_dot**2)\
        #     + (G*(M1+M2)*sin(t1) + L2*M2*sin(t1_t2)*t2_dot**2)*cos(t1_t2)
        # dydx[3] = mine_dydx2 / den2

        '''
        # this set is from by hand simplification calculus
        den1 = L1*(-M1 - M2 + M2 * cos(t1_t2)**2)
        num1 = G*(M1 + M2)*sin(t1) \
            + M2*(L2*sin(t1_t2)*t2_dot**2 \
            - (G*sin(t2) - L1*sin(t1_t2)*t1_dot**2)*cos(t1_t2))
        dydx[1] = num1 / den1

        den2 = L2*(-M1 + M2*cos(t1_t2)**2 - M2)
        num2 = (-M1 - M2) * (-G*sin(t2) + L1*sin(t1_t2)*t1_dot**2) \
            - (G*(M1 + M2)*sin(t1) + L2*M2*sin(t1_t2)*t2_dot**2)*cos(t1_t2)
        dydx[3] = num2 / den2
        '''

        '''
            with a torque term u2 added in derivations
        # print(np.arctan(t1_dot))
        # U2 = 0.7 / (np.pi * np.arctan(t1_dot))
        U2 = 0

        num1 = -(G*(M1 + M2)*sin(t1) + L2*M2*sin(t1_t2)*t2_dot**2 + M2*(-G*sin(t2) + L1*sin(t1_t2)*t1_dot**2 + L2*U2)*cos(t1_t2))
        den1 = L1*(M1 - M2*cos(t1_t2)**2 + M2)
        dydx[1] = num1 / den1

        num2 = (M1 + M2)*(-G*sin(t2) + L1*sin(t1_t2)*t1_dot**2 + L2*U2)\
            + (G*(M1 + M2)*sin(t1) + L2*M2*sin(t1_t2)*t2_dot**2)*cos(t1_t2)
        den2 = L2*(M1 - M2*cos(t1_t2)**2 + M2)
        dydx[3] = num2/den2
        '''

        '''
            these are SEPARATE THINGS
            TRAJECTORY / GOAL, related artifacts
            ACTUATOR TORQUE 'U'
            Q**
        '''

        v = -KD*t2_dot + KP*(0.636619772367581*ALPHA*np.arctan(t1_dot) - t2)

        # u == input torque
        u = G*sin(t2)/L2 + L1*(-G*sin(t1)/L1 - L2*M2*sin(t1_t2)*t2_dot**2/(L1*(M1 + M2)))*cos(t1_t2)/L2 - L1*sin(t1_t2)*t1_dot**2/L2 + (-KD*t2 + KP*(0.63661977236758138*ALPHA*np.arctan(t1_dot) - t2))*(-M2*cos(t1_t2)**2/(M1 + M2) + 1)
        # print("u=",u)

        dydx[1] = -G*sin(t1)/L1 + L2*M2*cos(t1+t2)*v - L2*M2*sin(t1_t2)*t2_dot**2/(L1*(M1 + M2))
        dydx[3] = v

        # aux data
        q2_desired = 2*ALPHA/np.pi*np.arctan(t1_dot)
        data = np.append(dydx, q2_desired)
        
        insert_into_dict_of_arrays(self._data, "q2_desired", q2_desired)

        return dydx

# create a time array from 0..t_stop sampled at 0.02 second steps
dt = 0.02
t = np.arange(0, t_stop, dt)

# th1 and th2 are the initial angles (degrees)
# w10 and w20 are the initial angular velocities (degrees per second)
th1 = 0.0
w1 = 0.0
th2 = -10.0
w2 = 0.0

# initial state
state = np.radians([th1, w1, th2, w2])

# integrate your ODE using scipy.integrate.
c = Container()
y = integrate.odeint(c.derivs_sympy, state, t)
x1 = L1*sin(y[:, 0])
y1 = -L1*cos(y[:, 0])
x2 = L2*sin(y[:, 2]) + x1
y2 = -L2*cos(y[:, 2]) + y1

# state_orig = np.radians([th1, w1, th2, w2])
# y_original = integrate.odeint(derivs_original, state_orig, t)
# x1_orig = L1*sin(y_original[:, 0])
# y1_orig = -L1*cos(y_original[:, 0])
# x2_orig = L2*sin(y_original[:, 2]) + x1_orig
# y2_orig = -L2*cos(y_original[:, 2]) + y1_orig

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, L))
ax.set_aspect('equal')
ax.grid()
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], '.-', lw=1, ms=2)
history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)

# aux
q2_desired = c._data["q2_desired"]
print(q2_desired)

line1, = ax.plot([], [], 'o-', lw=2)
line2, = ax.plot([], [], 'o-', lw=2)

# line_orig, = ax.plot([], [], 'o-', lw=2)
# trace_orig, = ax.plot([], [], '.-', lw=1, ms=2)
# history_x_orig, history_y_orig = deque(maxlen=history_len), deque(maxlen=history_len)

def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    # thisx_orig = [0, x1_orig[i], x2_orig[i]]
    # thisy_orig = [0, y1_orig[i], y2_orig[i]]

    if i == 0:
        history_x.clear()
        history_y.clear()

        # history_x_orig.clear()
        # history_y_orig.clear()

    history_x.appendleft(thisx[2])
    history_y.appendleft(thisy[2])
    line.set_data(thisx, thisy)
    trace.set_data(history_x, history_y)

    # history_x_orig.appendleft(thisx_orig[2])
    # history_y_orig.appendleft(thisy_orig[2])
    # line_orig.set_data(thisx_orig, thisy_orig)
    # trace_orig.set_data(history_x_orig, history_y_orig)

    time_text.set_text(time_template % (i*dt))

    # return line, trace, line_orig, trace_orig, time_text
    return line, line1, line2, trace, time_text


ani = animation.FuncAnimation(
    fig, animate, len(y), interval=dt*1000, blit=True)
plt.show()

'''
alpha1 = l2 / l1 * (m2 / (m1 + m2))*cos(t1 - t2)
alpha2 = l1 / l2 * cos(t1 - t2)
A = Matrix([[1, alpha1], [alpha2, 1]])
Ainv = A.inv()
f1 = -l2 / l1 * (m2 / (m1 + m2)) * (t2.diff(t))**2 * sin(t1 - t2) - g / l1*sin(t1)
f2 = l1 / l2 * (t1.diff(t))**2 * sin(t1 - t2) - g / l2*sin(t2)
fs = Matrix([[f1], [f2]])
fs
Ainv * fs
test = Ainv * fs
test[0]
simplify(test[0])
(f1 - alpha1 * f2) / (1 - alpha1*alpha2)
simplify((f1 - alpha1 * f2) / (1 - alpha1*alpha2))
simplify(test[1])
clear
simplify(test[0])
clear
simplify(test[1])
simplify((f1 - alpha1 * f2) / (1 - alpha1*alpha2))
(f1 - alpha1 * f2) / (1 - alpha1*alpha2)
simplify((-alpha2*f1 + f2) / (1 - alpha1*alpha2))
simplify(test[0])

'''



'''
alpha1 = l2 / l1 * (m2 / (m1 + m2))*cos(t1 - t2)
alpha2 = l1 / l2 * cos(t1 - t2)
double_M = Matrix([[1, alpha1], [alpha2, 1]])

double_tau1 = -l2 / l1 * (m2 / (m1 + m2)) * (t2.diff(t))**2 * sin(t1 - t2) - g / l1*sin(t1)

# ADDING AN INPUT TORQUE ON Q2
double_tau2 = l1 / l2 * (t1.diff(t))**2 * sin(t1 - t2) - g / l2*sin(t2) + u2

double_tau = Matrix([[double_tau1], [double_tau2]])

double_M_inv = simplify(double_M.inv())

double_qdotdot_derived = simplify(double_M_inv * double_tau)

print(python(double_qdotdot_derived[0]))
print(python(double_qdotdot_derived[1]))
'''


'''
do PFL (partial feedback linearization) (collocated)
to decouple the influence of link 1 on link 2's dynamics

double_M * q** = double_tau

double_pfl_tau1 = -l2 / l1 * (m2 / (m1 + m2)) * (t2.diff(t))**2 * sin(t1 - t2) - g / l1*sin(t1)

# NO U
double_pfl_tau2 = l1 / l2 * (t1.diff(t))**2 * sin(t1 - t2) - g / l2*sin(t2)

double_pfl_tau = Matrix([[double_pfl_tau1], [double_pfl_tau2]])

double_q1dot = 1 / double_M[0, 0] * (double_pfl_tau[0] - M[0, 1] * dt2.diff(t))


double_mblob = (double_M[1, 1] - double_M[1, 0] * double_M[0, 1] / double_M[0, 0])

aaa, kp, kd = symbols('aaa kp kd')
q2d = (2 * aaa / np.pi) * atan(dt1)
v = kp * (q2d - t2) - kd * t2dot
u = double_mblob * v - double_pfl_tau[1] + double_M[1, 0] / double_M[0, 0] * double_pfl_tau[0]
double_q2dot = v

'''

'''
double_U = -(m1 + m2)*g*l1*cos(t1) - m2*g*l2*cos(t2)

double_E = simplify(transpose(qdot) * double_M * qdot / 2 + double_U)

t1_goal = symbols('t1_goal')

double_E_goal = double_U.subs({t1 : t1_goal, t2: 0})

double_E_err = simplify(double_E - double_E_goal)

'''