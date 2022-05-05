#!/usr/bin/env python3

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

'''
these are SEPARATE THINGS:
TRAJECTORY / GOAL, related artifacts (v = error derivative)
ACTUATOR TORQUE 'U' / 'tau'
Q**

###########################################################

one way to think decompose the problem:

control signals(q**, u / torque) = f1(reference, state(actuated q, q*, unactuated q, q*))
actuated q**   = f2(unactuated q**, actuated q**, control signals(q**, u / torque))
unactuated q** = f3(unactuated q**, actuated q**, control signals(q**, u / torque))

f1 comes from your CHOICE of
how you linearize (what you choose is input, what you choose is output)
your control law (what is reference -> what is error -> PD control, control signal 'v', and what the 'units' of that signal means aka what you feed that signal into)
    control output unit is a CHOICE, you can feed it to mean acceleration
    you can convert it to torque

    control signal can be converted to the 'u' input in the equations of motion
    if you are controlling for 'torque'
    or you can 

f2/f3 comes from dynamics equations and pfl / linearization choice

for example, if you do *collocated* pfl, you are CHOOSING to have a reference on
the ACTUATED joints 

i think dynamics is really interesting b/c you can try to 
reference on something UNACTUATED, something outside your plant but part of your system
and derive that into a control law on your plant
so you can control *outside* the robot
so you can swap out the plant and see if it can control the unactuated joints
better / worse
the unactuated joints are what is 'public', the 'interface', the 'what'
the actuated joints are what is 'private', the 'how', the 'implementation'
so 'underactuated' really is a term that encompasses this decomposition
of a dynamical system into what you 'can' and 'cannot' control and through their relation
how to 'control' what you 'cannot control'
#cool

###########################################################
'''

##########################################################


"""

from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
from collections import deque
import argparse

t_stop = 120 # np.pi * 10 # how many seconds to simulate
history_len = 500  # how many trajectory points to display
dt = 0.02
t = np.arange(0, t_stop, dt)

# constants
G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 10.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
L = 1.05*(L1 + L2)  # maximal length of the combined pendulum
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg
Q1_DAMPING = 0.0

ALPHA = 1.
K1 = 1
K2 = 1
K3 = 1.0
K4 = 5.0
K5 = 5.0

# derivs_pfl_collocated_strategy1
# ./double_pendulum.py --playback 20 --initial 10,0,1,0
# ALPHA <= 1 does NOT converges back, does not pump up energy
# gains influence how fast it can pump, but there is a qualitative limit to speed AND max height

G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 10.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
L = 1.05*(L1 + L2)  # maximal length of the combined pendulum
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg
Q1_DAMPING = 0.0

ALPHA = 10
K1 = L1*(M1+M2) * 10
K2 = L1*(M1+M2) * 10

ALPHA = 10
K1 = L1*(M1+M2) * 500
K2 = L1*(M1+M2) * 500

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
        self._data = {}

        self._t = 0.0

        self.total = 0

        self._state = None

    def derivs_freebody(self, state, t):
        dydx = np.zeros_like(state)

        dydx[0] = state[1]
        dydx[2] = state[3]

        t1 = state[0]
        t1_dot = state[1]
        t2 = state[2]
        t2_dot = state[3]
        t1_t2 = t1 - t2

        den1 = L1 * (M1 - M2 * cos(t1_t2)**2 + M2)
        mine_dydx1 = -G * (M1 + M2) * sin(t1)\
            - L2*M2*sin(t1_t2)*t2_dot**2\
            + M2*(G*sin(t2) - L1*sin(t1_t2)*t1_dot**2)*cos(t1_t2)
        dydx[1] = mine_dydx1 / den1

        den2 = L2*(M1 - M2*cos(t1-t2)**2 + M2)
        mine_dydx2 = -(M1 + M2)*(G*sin(t2) - L1*sin(t1_t2)*t1_dot**2)\
            + (G*(M1+M2)*sin(t1) + L2*M2*sin(t1_t2)*t2_dot**2)*cos(t1_t2)
        dydx[3] = mine_dydx2 / den2

        return dydx

    def derivs_pfl_collocated_strategy1(self, state, t):
        '''
        https://diego.assencio.com/?index=1500c66ae7ab27bb0106467c68feebc6
        https://underactuated.mit.edu/acrobot.html (collocated linearization derivation)

        import sympy
        from sympy import *
        from sympy.physics.mechanics import *
        sympy.init_printing()
        import numpy as np

        t = Symbol('t')
        t1, t2, t3 = dynamicsymbols('t1 t2 t3')
        m1, l1, m2, l2, m3, l3, g, p1_damping = symbols('M1 L1 M2 L2 M3 L3 G p1_damping');
        dt1 = t1.diff(t)
        dt2 = t2.diff(t)

        # double_M * q** = double_tau
        alpha1 = l2 / l1 * (m2 / (m1 + m2))*cos(t1 - t2)
        alpha2 = l1 / l2 * cos(t1 - t2)
        double_M = Matrix([[1, alpha1], [alpha2, 1]])

        double_pfl_tau1 = -l2 / l1 * (m2 / (m1 + m2)) * (t2.diff(t))**2 * sin(t1 - t2) - g / l1*sin(t1)
        double_pfl_tau2 = l1 / l2 * (t1.diff(t))**2 * sin(t1 - t2) - g / l2*sin(t2) # NO U
        double_pfl_tau = Matrix([[double_pfl_tau1], [double_pfl_tau2]])

        double_q1dot = 1 / double_M[0, 0] * (double_pfl_tau[0] - double_M[0, 1] * dt2.diff(t))

        # p1 =  M*t1** + tau1 = -damping * dt1 => M*t1** = -tau1 - damping * dt1
        double_pfl_tau1_with_damping = double_pfl_tau1 - p1_damping * dt1
        double_pfl_tau_with_damping = Matrix([[double_pfl_tau1_with_damping], [double_pfl_tau2]])

        double_q1dot_with_damping = 1 / double_M[0, 0] * (double_pfl_tau_with_damping[0] - double_M[0, 1] * dt2.diff(t))


        --------------------------

        alpha, k1, k2 = symbols('ALPHA K1 K2')
        q2d = (2 * alpha / np.pi) * atan(dt1) # THIS IS THE REFERENCE
        v = k1 * (q2d - t2) - k2 * dt2
        double_q2dot = v

        --------------------------

        double_mblob = (double_M[1, 1] - double_M[1, 0] * double_M[0, 1] / double_M[0, 0])
        u = simplify(double_mblob * v - double_pfl_tau_with_damping[1] + double_M[1, 0] / double_M[0, 0] * double_pfl_tau_with_damping[0])

        --------------------------

        sympy_to_expr(v)
        sympy_to_expr(double_q2dot) dydx[3]
        sympy_to_expr(double_q1dot_with_damping) dydx[1]
        '''
        dydx = np.zeros_like(state)

        dydx[0] = state[1]
        dydx[2] = state[3]

        t1 = state[0]
        t1_dot = state[1]
        t2 = state[2]
        t2_dot = state[3]
        t1_t2 = t1 - t2

        # ---PFL collocated control strategy 1
        v = -K2*t2_dot + K1*(0.636619772367581*ALPHA*np.arctan(t1_dot) - t2)
        # v is the output of the PD controller that tracks t2 to t2d = 2alpha/pi*atan(t1dot)
        # u == input torque that you would command to the physical motor
        # artifact not used for plotting
        # u = G*sin(t2)/L2 + L1*(-G*sin(t1)/L1 - L2*M2*sin(t1_t2)*t2_dot**2/(L1*(M1 + M2)))*cos(t1_t2)/L2 - L1*sin(t1_t2)*t1_dot**2/L2 + (-K2*t2 + K1*(0.63661977236758138*ALPHA*np.arctan(t1_dot) - t2))*(-M2*cos(t1_t2)**2/(M1 + M2) + 1)

        dydx[3] = v # our control law/strategy includes a choice that v is the acceleration(?)

        dydx[1] = -G*sin(t1)/L1 - L2*M2*cos(t1_t2)*v/(L1*(M1+M2)) - L2*M2*sin(t1_t2)*t2_dot**2/(L1*(M1 + M2)) - Q1_DAMPING*t1_dot

        return dydx

    def data_gen(self):
        i = 0
        while True:
            i = (i + 1) % self._state.shape[0]

            yield i

def animate(data):
    i = data

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

    aux1_text.set_text(aux1_template % (state[i, 0]))
    aux2_text.set_text(aux2_template % (state[i, 2]))

    aux2_text.set_text(aux2_template % (energy[i]))

    # aux3_text.set_text('hello')
    # aux4_text.set_text('hello')

    # return line, trace, line_orig, trace_orig, time_text
    return line, line1, line2, trace, aux1_text, aux2_text, aux3_text, aux4_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument('--playback', type=int, default=1, help='')
    parser.add_argument('--initial', type=str, default="0,0,1,0", help='')
    '''
    q2_dot [3] cannot be 0
    http://underactuated.mit.edu/pend.html#energy_shaping
    This is true for any  theta_dot, except for theta_dot = 0
    (so it will not actually swing us up from the downright fixed point...
    but if you nudge the system just a bit, then it will start
    pumping energy and will swing all of the way up).
    '''
    args = parser.parse_args()

    t1_goal = 1.0

    # create a time array from 0..t_stop sampled at 0.02 second steps

    # th1 and th2 are the initial angles (degrees)
    # w10 and w20 are the initial angular velocities (degrees per second)
    # th1 = 0.0
    # th1 = 100.0
    # w1 = 0.0
    # th2 = 1.0
    # w2 = 0.0

    # initial state
    state = np.radians([float(x) for x in args.initial.split(',')])

    # integrate your ODE using scipy.integrate.
    c = Container()
    # state = integrate.odeint(c.derivs_freebody, state, t)
    state = integrate.odeint(c.derivs_pfl_collocated_strategy1, state, t)
    # state = integrate.odeint(c.derivs_pfl_collocated_taskspace, state, t)
    c._state = state

    # state = integrate.odeint(c.derivs_pfl_noncollocated_strategy1, state, t)
    # state = integrate.odeint(c.derivs_pfl_collocated_taskspace_2, state, t)
    print("done")

    e = 1/2*(state[:,1]**2 + state[:,3]**2)
    u = M1*G*L1*(1 - cos(state[:,0])) +\
        M2*G*L1*(1 - cos(state[:,0])) + M2*G*L2*(1 - cos(state[:,2]))
    energy = e + u

    print(state.shape[0])
    print(len(energy))

    # state = np.zeros((len(t), 3))
    # state[:, 0] = t
    # state[:, 0] = np.sin(state[:, 0])

    x1 = L1*sin(state[:, 0])
    y1 = -L1*cos(state[:, 0])
    x2 = L2*sin(state[:, 2]) + x1
    y2 = -L2*cos(state[:, 2]) + y1

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
    aux1_template = 'aux1 = %.2f'
    aux1_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    aux2_template = 'aux2 = %.2f'
    aux2_text = ax.text(0.05, 0.8, '', transform=ax.transAxes)
    aux3_text = ax.text(0.05, 0.7, '', transform=ax.transAxes)
    aux4_text = ax.text(0.05, 0.6, '', transform=ax.transAxes)

    line, = ax.plot([], [], 'o-', lw=2)
    trace, = ax.plot([], [], '.-', lw=1, ms=2)
    history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)

    # aux
    # e = c._data["e"]

    line1, = ax.plot([], [], 'c--', lw=1, alpha=0.5)
    ref_x1 = L1*sin(np.pi / 3)
    ref_y1 = -L1*cos(np.pi / 3)
    line1.set_data([0, ref_x1], [0, ref_y1])

    line2, = ax.plot([], [], 'o-', lw=2)

    # line_orig, = ax.plot([], [], 'o-', lw=2)
    # trace_orig, = ax.plot([], [], '.-', lw=1, ms=2)
    # history_x_orig, history_y_orig = deque(maxlen=history_len), deque(maxlen=history_len)

    ani = animation.FuncAnimation(
        fig, animate, c.data_gen, interval=dt*1000/args.playback, blit=True)
    plt.title('playback speed %dx' % (args.playback))
    plt.show()

#####################################################################
