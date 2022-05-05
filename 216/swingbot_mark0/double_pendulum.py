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

# constants
G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 10.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
L = 1.05*(L1 + L2)  # maximal length of the combined pendulum
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg
t_stop = np.pi * 10 # how many seconds to simulate
history_len = 500  # how many trajectory points to display
dt = 0.02
t = np.arange(0, t_stop, dt)
Q1_DAMPING = 0.1

ALPHA = 1.
K1 = 1.0
K2 = 1.0
K3 = 1.0

K4 = 5.0
K5 = 5.0

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
        u = G*sin(t2)/L2 + L1*(-G*sin(t1)/L1 - L2*M2*sin(t1_t2)*t2_dot**2/(L1*(M1 + M2)))*cos(t1_t2)/L2 - L1*sin(t1_t2)*t1_dot**2/L2 + (-K2*t2 + K1*(0.63661977236758138*ALPHA*np.arctan(t1_dot) - t2))*(-M2*cos(t1_t2)**2/(M1 + M2) + 1)

        # err_diff = (G*L1*L2*(M1 + M2)*(L1*(M1 + M2)*cos(t1_goal) + L2*M2) - G*L1*L2*(M1 + M2)*(L1*(M1 + M2)*cos(t1) + L2*M2*cos(t2)) + L1*(M1 + M2)*(L1*cos(t1 - t2)*t2_dot + L2*t1_dot)*t1_dot/2 + L2*(L1*(M1 + M2)*t2_dot + L2*M2*cos(t1 - t2)*t1_dot)*t2_dot/2)/(L1*L2*(M1 + M2))
        # ubar = err_diff * t1_dot
        # v = -K1*t2 - K2*t2_dot + K3*ubar

        dydx[3] = v # our control law/strategy includes a choice that v is the acceleration(?)

        dydx[1] = -G*sin(t1)/L1 + L2*M2*cos(t1+t2)*v - L2*M2*sin(t1_t2)*t2_dot**2/(L1*(M1 + M2)) - Q1_DAMPING*t1_dot

        return dydx

    def derivs_pfl_collocated_taskspace(self, state, t):
        dydx = np.zeros_like(state)

        # q1_dot
        dydx[0] = state[1]

        # q2_dot
        dydx[2] = state[3]

        t1 = state[0]
        t1_dot = state[1]
        t2 = state[2]
        t2_dot = state[3]
        t1_t2 = t1 - t2

        # yd = 0.5 * sin(t)
        # yd_dot = cos(t)
        # yd_dotdot = -sin(t)

        # if t < np.pi * k:
        #     print("hi")
        #     yd = t / k * sin(t / k)
        #     yd_dot = t/(k**2)*cos(t/k) + sin(t/k) / k
        #     yd_dotdot = -t*sin(t/k)/(k**3) + 2*cos(t/k)/(k**2)
        # else:
        #     print("bye")
        #     yd = np.pi * sin(t / k)
        #     yd_dot = np.pi * cos(t / k) / k
        #     yd_dotdot = -np.pi * sin(t / k) / k**2

        # these 2 sets show it working on y = q2
        # this will converge to the task space goal
        # so NO need to ramp up
        k = 1
        a = np.pi / 4

        # k = 2
        # a = np.pi / 2

        k = 1
        a = np.pi / 3

        yd = a * sin(t / k)
        yd_dot = a * cos(t / k) / k
        yd_dotdot = -a * sin(t / k) / k**2

        v = yd_dotdot + K5 * (yd_dot - t2_dot) + K4 * (yd - t2)

        dydx[3] = v
        dydx[1] = -G*sin(t1)/L1 + L2*M2*cos(t1+t2)*v - L2*M2*sin(t1_t2)*t2_dot**2/(L1*(M1 + M2)) - Q1_DAMPING*t1_dot

        return dydx

    #####################################################################

    def derivs_pfl_noncollocated_strategy1(self, state, t):
        print(self._t)

        dydx = np.zeros_like(state)

        dydx[0] = state[1]
        dydx[2] = state[3]

        t1 = state[0]
        t1_dot = state[1]
        t2 = state[2]
        t2_dot = state[3]
        t1_t2 = t1 - t2

        target_q1_norm = np.pi / 2
        speed = 1
        freq = 1
        amp = min(target_q1_norm, self._t / speed)
        q1d = amp * np.sin(t * freq)

        # ---PFL noncollocated control strategy 1
        v = -K2*t1_dot + K1*(q1d - t1)

        dydx[1] = v
        den = (L2*M2*cos(t1_t2))
        print("den", den)
        dydx[3] = -(G*(M1 + M2)*sin(t1) + L1*(M1 + M2)*v + L2*M2*sin(t1_t2)*t2_dot**2)/den
        self._t += dt

        return dydx

    def derivs_pfl_collocated_taskspace_2(self, state, t):
        dydx = np.zeros_like(state)

        # q1_dot
        dydx[0] = state[1]

        # q2_dot
        dydx[2] = state[3]

        t1 = state[0]
        t1_dot = state[1]
        t2 = state[2]
        t2_dot = state[3]
        t1_t2 = t1 - t2


        # this will converge to the task space goal
        # so NO need to ramp up
        k = 1
        a = 0.1

        # k = 2
        # a = np.pi / 2

        yd = a * sin(t / k)
        yd_dot = a * cos(t / k) / k
        yd_dotdot = -a * sin(t / k) / k**2

        # control law stays the same
        v = 0 + K5 * (yd_dot - t1_dot) + K4 * (yd - t1)

        tau_blob = G*sin(t1)/L1 + Q1_DAMPING*t1_dot + L2*M2*sin(t1_t2)*t2_dot**2/(L1*(M1 + M2))

        '''
        It is state dependent; in the cart-pole example above  and drops rank exactly when .
        A system has Global Strong Inertial Coupling
        if it exhibits Strong Inertial Coupling in every state.
        '''
        if np.abs(cos(t1_t2)) < 1e-2:
            print("WARNNNNNINGGGG SINGULAR DROPPPP RANKKKK!!!!")

        if np.abs(cos(t1_t2)) > 1e-2:
            coupling_term = cos(t1_t2)
            print("coupling_term", coupling_term)
            gain = -L1*(M1 + M2)/(L2*M2*coupling_term)
            dydx[3] = gain*v - tau_blob


            # dydx[1] = L2*M2*cos(t1+t2)*v - G*sin(t1)/L1 - L2*M2*sin(t1_t2)*t2_dot**2/(L1*(M1 + M2)) - Q1_DAMPING*t1_dot

        else:
            dydx[3] = 0


        # q1_gain = -L2*M2*cos(t1_t2)/(L1*(M1 + M2))
        q1_gain = L2*M2*cos(t1+t2)
        dydx[1] = q1_gain * dydx[3] - tau_blob

        # print("dydx", dydx)

        # dydx[1] = -G*sin(t1)/L1 + L2*M2*cos(t1+t2)*dydx[3] - L2*M2*sin(t1_t2)*t2_dot**2/(L1*(M1 + M2)) - Q1_DAMPING*t1_dot

        return dydx

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

        # ---PFL collocated control strategy 1
        v = -K2*t2_dot + K1*(0.636619772367581*ALPHA*np.arctan(t1_dot) - t2)
        # v is the output of the PD controller that tracks t2 to t2d = 2alpha/pi*atan(t1dot)
        # u == input torque that you would command to the physical motor
        # artifact not used for plotting
        u = G*sin(t2)/L2 + L1*(-G*sin(t1)/L1 - L2*M2*sin(t1_t2)*t2_dot**2/(L1*(M1 + M2)))*cos(t1_t2)/L2 - L1*sin(t1_t2)*t1_dot**2/L2 + (-K2*t2 + K1*(0.63661977236758138*ALPHA*np.arctan(t1_dot) - t2))*(-M2*cos(t1_t2)**2/(M1 + M2) + 1)

        # err_diff = (G*L1*L2*(M1 + M2)*(L1*(M1 + M2)*cos(t1_goal) + L2*M2) - G*L1*L2*(M1 + M2)*(L1*(M1 + M2)*cos(t1) + L2*M2*cos(t2)) + L1*(M1 + M2)*(L1*cos(t1 - t2)*t2_dot + L2*t1_dot)*t1_dot/2 + L2*(L1*(M1 + M2)*t2_dot + L2*M2*cos(t1 - t2)*t1_dot)*t2_dot/2)/(L1*L2*(M1 + M2))
        # ubar = err_diff * t1_dot
        # v = -K1*t2 - K2*t2_dot + K3*ubar

        dydx[1] = -G*sin(t1)/L1 + L2*M2*cos(t1+t2)*v - L2*M2*sin(t1_t2)*t2_dot**2/(L1*(M1 + M2))
        dydx[3] = v

        # aux data
        # q2_desired = 2*ALPHA/np.pi*np.arctan(t1_dot)
        # data = np.append(dydx, q2_desired)
        
        # insert_into_dict_of_arrays(self._data, "q2_desired", q2_desired)
        # insert_into_dict_of_arrays(self._data, "e", e)

        return dydx

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

    aux1_text.set_text(aux1_template % (state[i, 0]))
    aux2_text.set_text(aux2_template % (state[i, 2]))
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
    # state = integrate.odeint(c.derivs_pfl_collocated_strategy1, state, t)
    # state = integrate.odeint(c.derivs_pfl_noncollocated_strategy1, state, t)
    state = integrate.odeint(c.derivs_pfl_collocated_taskspace, state, t)
    print("done")

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
        fig, animate, len(state), interval=dt*1000/args.playback, blit=True)
    plt.title('playback speed %dx' % (args.playback))
    plt.show()

#####################################################################

'''
derivs_freebody
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
derivs_pfl_collocated_strategy1
---PFL collocated control strategy 1
do PFL (partial feedback linearization) (collocated)
to decouple the influence of link 1 on link 2's dynamics

this is collocated, so we are commanding a q2d, joint 2 reference trajectory
so we are commanding the *actuated* joints aka we derive our control law based on q2d, not q1d
and we are *hoping* that the dynamics cause our real goal, q1, to change how we want it to

#############################

t = Symbol('t')
t1, t2, t3 = dynamicsymbols('t1 t2 t3')
m1, l1, m2, l2, m3, l3, g, p1_damping = symbols('M1 L1 M2 L2 M3 L3 G p1_damping');
dt1 = t1.diff(t)
dt2 = t2.diff(t)

alpha1 = l2 / l1 * (m2 / (m1 + m2))*cos(t1 - t2)
alpha2 = l1 / l2 * cos(t1 - t2)
double_M = Matrix([[1, alpha1], [alpha2, 1]])

# double_M * q** = double_tau

double_pfl_tau1 = -l2 / l1 * (m2 / (m1 + m2)) * (t2.diff(t))**2 * sin(t1 - t2) - g / l1*sin(t1)
double_pfl_tau2 = l1 / l2 * (t1.diff(t))**2 * sin(t1 - t2) - g / l2*sin(t2) # NO U
double_pfl_tau = Matrix([[double_pfl_tau1], [double_pfl_tau2]])

double_q1dot = 1 / double_M[0, 0] * (double_pfl_tau[0] - double_M[0, 1] * dt2.diff(t))

# p1 = -damping * dt1 => p1 + damping * dt1 = 0
double_pfl_tau1_with_damping = double_pfl_tau1 - p1_damping * dt1
double_pfl_tau_with_damping = Matrix([[double_pfl_tau1_with_damping], [double_pfl_tau2]])

double_q1dot_with_damping = 1 / double_M[0, 0] * (double_pfl_tau_with_damping[0] - double_M[0, 1] * dt2.diff(t))

double_mblob = (double_M[1, 1] - double_M[1, 0] * double_M[0, 1] / double_M[0, 0])

aaa, kp, kd = symbols('aaa kp kd')
q2d = (2 * aaa / np.pi) * atan(dt1) # THIS IS THE REFERENCE
v = kp * (q2d - t2) - kd * dt2
double_q2dot = v

# artifact for actuation
u = double_mblob * v - double_pfl_tau[1] + double_M[1, 0] / double_M[0, 0] * double_pfl_tau[0]

Using the same PFL and control law
You can also 'control' through motor torque directly
Without a control law, but do 'force' control

#############################

double_tau1 = -l2 / l1 * (m2 / (m1 + m2)) * (t2.diff(t))**2 * sin(t1 - t2) - g / l1*sin(t1)

# ADDING AN INPUT TORQUE ON Q2
double_tau2 = l1 / l2 * (t1.diff(t))**2 * sin(t1 - t2) - g / l2*sin(t2) + u2

double_tau = Matrix([[double_tau1], [double_tau2]])

double_M_inv = simplify(double_M.inv())

double_qdotdot_derived = simplify(double_M_inv * double_tau)

print(python(double_qdotdot_derived[0]))
print(python(double_qdotdot_derived[1]))
'''

#####################################################################

'''
double_U = -(m1 + m2)*g*l1*cos(t1) - m2*g*l2*cos(t2)

double_E = simplify(transpose(qdot) * double_M * qdot / 2 + double_U)

t1_goal = symbols('t1_goal')

double_E_goal = double_U.subs({t1 : t1_goal, t2: 0})

double_E_err = simplify(double_E - double_E_goal)
'''

#####################################################################

'''
double_M11 = double_M[0, 0]
double_M12 = double_M[0, 1]

h_bar = h2 - h1 / double_M11 * double_M12
h1, h2 = symbols('h1 h2')
double_M11 = double_M[0, 0]
double_M12 = double_M[0, 1]

h_bar = h2 - h1 / double_M11 * double_M12
h_bar
h_bar * h_bar
h_bar * (1 / h_bar * h_bar)
h_bar
1 / h_bar
simplify(1 / h_bar)
simplify(h_bar * (1 / (h_bar * h_bar)))
h_bar_pinv = simplify(h_bar * (1 / (h_bar * h_bar)))
h.diff(t)
h
h = Matrix([h1, h2])
h2
h1
h1 = t1.diff(t)
h1
q = t1
q.diff(t1)
h1 = q.diff(t1)
h2 = q.diff(t2)
h2
h1
h
h = Matrix([h1, h2])
h
h_bar_pinv = simplify(h_bar * (1 / (h_bar * h_bar)))
double_M11 = double_M[0, 0]
double_M12 = double_M[0, 1]


h_bar = h2 - h1 / double_M11 * double_M12
h_bar
h_bar_pinv = simplify(h_bar * (1 / (h_bar * h_bar)))
h_bar_pinv
h
h.diff(t)
h1 / double_M11
double_pfl_tau1_with_damping
simplify(h1 / double_M11 * double_pfl_tau1_with_damping)
yd_blob = simplify(h1 / double_M11 * double_pfl_tau1_with_damping)
h_bar_pinv
print(python(h_bar_pinv))
print(python(yd_blob))
'''