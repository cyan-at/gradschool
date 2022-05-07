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

control signals(q**, u / torque) = f1(reference, c.state(actuated q, q*, unactuated q, q*))
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
Q1_DAMPING = 0.0

'''
collocated pfl / task-spsace collocated pfl controller works well, intuitive to understand
  what is done in the literature
  and it does do what the literature uses it to do, which is reach the 'top'
  it does not respect the true goal which is a 'q1' target analytically
'''
ALPHA = 1.
K1 = 1
K2 = 1

K3 = 1.0
K4 = 1.0

'''
non-collocated pfl controller blows up as a matrix in system loses rank =>
divide by 0 => huge acc commands / integration errors
  this is what is reporte in literature
  easy to show as the cosine term goes to 0 in denominator
'''
K5 = 1.0
K6 = 1.0

'''
task-space / collocated pfl derivation, cascade controller based on 'energy', looks interesting
  it has the advantage that it converges to a 'set' q1
  it also has a nice property where low-energy => high actuator effort, high-energy => actuator chills out
    it's behavior at the same q1 state changes depending on how high of a q1 it achieves
    so it looks 'improviational'
  the K2 term chills out actuator, which at lower gains works 'too-hard' and gets in its own way
  it is also interesting with this controller how the actuator seems to behave 'asymmetrically' about q1 = 0
  because it leverages the dynamics from one half to be lazy in the other half
  downside is that q2 is high effort

  intuition: if you want higher q1 max, increase energy goal but also increase K8 to chill out actuator
'''
# this set works
K7 = 1.0
K8 = 1.0
K9 = 1.0
energy_goal = 10.0

K7 = 1.0
K8 = 1.0
K9 = 1.0
energy_goal = 20.0

# this does work and is stable around np.pi / 2
K7 = 5.0
K8 = 10.0 # this term has the effect of 'chilling out' the actuator
K9 = 1.0
energy_goal = 30.0

# this does work and is stable around np.pi / 2
K7 = 5.0
K8 = 8.0 # this term has the effect of 'chilling out' the actuator
K9 = 1.0
energy_goal = 60.0

# this does work and is stable around > np.pi / 2
K7 = 5.0
K8 = 5.0 # this term has the effect of 'chilling out' the actuator
K9 = 1.0
energy_goal = 100.0

# works
K10 = 1.0
K11 = 1.0 # this term has the effect of 'chilling out' the actuator
K12 = 1.0
energy_goal = 30.0

# this does work and is stable around > np.pi / 2
K10 = 1.0
K11 = 1.0 # this term has the effect of 'chilling out' the actuator
K12 = 1.0
energy_goal = 80.0

# this does work and is stable around > 2*np.pi / 3
K10 = 1.0
K11 = 1.0 # this term has the effect of 'chilling out' the actuator
K12 = 1.0
energy_goal = 120.0

# K7 = 5.0
# K8 = 20.0 # this term has the effect of 'chilling out' the actuator 30 is too high / slow, 20 is too low
# K9 = 1.0
# energy_goal = 60.0

# derivs_pfl_collocated_strategy1
# ./double_pendulum.py --playback 20 --initial 10,0,1,0
# ALPHA <= 1 does NOT converges back, does not pump up energy
# gains influence how fast it can pump, but there is a qualitative limit to speed AND max height
# regardless of what L1 is, ALPHA > 1 guarantees convergence of q1 up to q1 = np.pi (upright)
# a larger L1 does mean a larger K1, K2 to converge faster

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

# this shows just about reaching the top
L1 = 8.0  # length of pendulum 1 in m
ALPHA = 1.1
K1 = L1*(M1+M2) * 200
K2 = L1*(M1+M2) * 200

# this shows just about reaching the top
L1 = 10.0  # length of pendulum 1 in m
ALPHA = 1.1
K1 = L1*(M1+M2)
K2 = L1*(M1+M2)

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

class Acrobot(object):
    def __init__(self, args, initial_state, sampletimes):
        self._args = args

        self.initial_state = initial_state
        self.state = None
        self.sampletimes = sampletimes

        self._modes = [
            self.derivs_freebody,
            self.derivs_pfl_collocated_strategy1,
            self.derivs_pfl_collocated_taskspace,
            self.derivs_pfl_collocated_energy,
            self.derivs_pfl_collocated_energy2,

            self.derivs_pfl_noncollocated,
        ]

        self._data_gen_cb = None

    def init_data(self):
        self.state = integrate.odeint(
            self._modes[self._args.mode],
            self.initial_state,
            self.sampletimes)

        # extra data
        aux = np.zeros((self.state.shape[0], 5))
        self.state = np.hstack([self.state, aux])

        self.state[:, 4] = L1*sin(self.state[:, 0]) # x1
        self.state[:, 5] = -L1*cos(self.state[:, 0]) # y1
        self.state[:, 6] = L2*sin(self.state[:, 2]) + self.state[:, 4] # x2
        self.state[:, 7] = -L2*cos(self.state[:, 2]) + self.state[:, 5] # y2

        e = 1/2*(self.state[:,1]**2 + self.state[:,3]**2)
        u = M1*G*L1*(1 - cos(self.state[:,0])) +\
            M2*G*L1*(1 - cos(self.state[:,0])) +\
            M2*G*L2*(1 - cos(self.state[:,2]))
        self.state[:, 8] = e + u

        print("done")

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

        http://www2.ece.ohio-c.state.edu/~passino/PapersToPost/acrobot-JIRSTA.pdf
        LINEARIZATION
            matrix-forming, expressing as LHS and RHS
            PFL refers to the fact that:
                we solve for non-actuated joints q1**
                we substitute above to solve for q2** = f(input / tau / u)

                we re-write tau in the actuated joints
                q2** = f(v) is linear (q2** = v)

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

        '''
        http://www2.ece.ohio-c.state.edu/~passino/PapersToPost/acrobot-JIRSTA.pdf
        The arctangent function has the desirable characteristic of straightening out the
        second joint when q˙1 equals zero at the peak of each swing, allowing a balancing
        controller to catch the system in the approximately inverted position.

        - in my observation, the q1* = 0 at the tips means the q2d swings back to 0
        - which causes q2 to swing back the other way (with lag which allows for pumping upward)
        - which causes q1 to swing back the other way (with lag)

        we would define v to INCLUDE the q2d** but we are assuming that is not measurable realistically
        '''

        dydx[3] = v # our control law/strategy includes a choice that v is the acceleration(?)

        dydx[1] = -G*sin(t1)/L1 - L2*M2*cos(t1_t2)*v/(L1*(M1+M2)) - L2*M2*sin(t1_t2)*t2_dot**2/(L1*(M1 + M2)) - Q1_DAMPING*t1_dot/(L1*(M1 + M2))

        # dydx[1] = (-G*M1*sin(t1) - G*M2*sin(t1) - L2*M2*sin(t1 - t2)*t2_dot**2 - L2*M2*cos(t1 - t2)*v - Q1_DAMPING*t1_dot)/(L1*M1 + L1*M2)

        return dydx

    def derivs_pfl_collocated_taskspace(self, state, t):
        '''
        https://diego.assencio.com/?index=1500c66ae7ab27bb0106467c68feebc6
        https://underactuated.mit.edu/acrobot.html (collocated linearization derivation)

        SAME
        partial-feedback linearization as derivs_pfl_collocated_strategy1
        which guides the derivation of dydx[1] = nonlinearf(dydx[3] = v)
        '''
        dydx = np.zeros_like(state)

        dydx[0] = state[1]
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

        # let's say we can't measure this
        yd_dotdot = 0

        v = yd_dotdot + K3 * (yd_dot - t2_dot) + K4 * (yd - t2)

        dydx[3] = v
        dydx[1] = -G*sin(t1)/L1 - L2*M2*cos(t1_t2)*v/(L1*(M1+M2)) - L2*M2*sin(t1_t2)*t2_dot**2/(L1*(M1 + M2)) - Q1_DAMPING*t1_dot

        return dydx

    def derivs_pfl_collocated_energy(self, state, t):
        '''
        https://diego.assencio.com/?index=1500c66ae7ab27bb0106467c68feebc6
        https://underactuated.mit.edu/acrobot.html (collocated linearization derivation)

        SAME
        partial-feedback linearization as derivs_pfl_collocated_strategy1
        which guides the derivation of dydx[1] = nonlinearf(dydx[3] = v)

        # based on http://underactuated.mit.edu/acrobot.html#mjx-eqn-eq%3Asimple
        # which says use collocated pfl with this kind of energy func
        # it is different in that we set a energy_goal, not a reference q2 trajectory
        '''
        dydx = np.zeros_like(state)

        dydx[0] = state[1]
        dydx[2] = state[3]

        t1 = state[0]
        t1_dot = state[1]
        t2 = state[2]
        t2_dot = state[3]
        t1_t2 = t1 - t2

        e = 1/2*(state[0]**2 + state[2]**2)
        u = M1*G*L1*(1 - cos(state[0])) +\
            M2*G*L1*(1 - cos(state[0])) +\
            M2*G*L2*(1 - cos(state[2]))
        energy = u
        energy_err = energy - energy_goal

        u = t1_dot * energy_err

        v = K7 * u - K8 * (t2_dot) - K9 * (t2)

        dydx[3] = v
        dydx[1] = -G*sin(t1)/L1 - L2*M2*cos(t1_t2)*v/(L1*(M1+M2))\
            - L2*M2*sin(t1_t2)*t2_dot**2/(L1*(M1 + M2))\
            - Q1_DAMPING*t1_dot

        return dydx

    def derivs_pfl_collocated_energy2(self, state, t):
        '''
        https://diego.assencio.com/?index=1500c66ae7ab27bb0106467c68feebc6
        https://underactuated.mit.edu/acrobot.html (collocated linearization derivation)

        SAME
        partial-feedback linearization as derivs_pfl_collocated_strategy1
        which guides the derivation of dydx[1] = nonlinearf(dydx[3] = v)

        # based on http://underactuated.mit.edu/acrobot.html#mjx-eqn-eq%3Asimple
        # which says use collocated pfl with this kind of energy func
        # it is different in that we set a energy_goal, not a reference q2 trajectory
        '''
        dydx = np.zeros_like(state)

        dydx[0] = state[1]
        dydx[2] = state[3]

        t1 = state[0]
        t1_dot = state[1]
        t2 = state[2]
        t2_dot = state[3]
        t1_t2 = t1 - t2

        e = 1/2*(state[0]**2)
        u = M1*G*L1*(1 - cos(state[0])) +\
            M2*G*L1*(1 - cos(state[0])) +\
            M2*G*L2*(1 - cos(state[2]))
        energy = u
        energy_err = energy - energy_goal

        u = t1_dot * energy_err

        v = K10 * u - K11 * (t2_dot) - K12 * (t2)

        dydx[3] = v
        dydx[1] = -G*sin(t1)/L1 - L2*M2*cos(t1_t2)*v/(L1*(M1+M2))\
            - L2*M2*sin(t1_t2)*t2_dot**2/(L1*(M1 + M2))\
            - Q1_DAMPING*t1_dot

        return dydx

    def derivs_pfl_noncollocated(self, state, t):
        dydx = np.zeros_like(state)

        dydx[0] = state[1]
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
        # we route this to acceleration so that integration
        # v = yd_dotdot + K5 * (yd_dot - t1_dot) + K4 * (yd - t1)
        # yd_dotdot = 0 if we can't measure it
        v = 0.0 + K5 * (yd_dot - t1_dot) + K6 * (yd - t1)

        dydx[1] = v

        # if np.abs(cos(t1 - t2)) > 1e-2:
        #     den = (L2*M2*cos(t1 - t2))
        #     unsaturated = L1*(M1 + M2)*(-G*sin(t1)/L1 - dydx[1] - L2*M2*sin(t1 - t2)*t2_dot**2/(L1*(M1 + M2)))/den
        #     print(unsaturated)
        #     dydx[3] = min(unsaturated, 1)
        # else:
        #     den = (L2*M2*cos(t1 - t2))
        #     dydx[3] = np.sign(den)

        '''
        # task space pfl derivation
        h2 = 0
        h1 = 1
        h2 - h1 / double_M[0, 0] * double_M[0, 1]
        h_bar = h2 - h1 / double_M[0, 0] * double_M[0, 1]
        1 / h_bar
        hbar_pinv = sympy_to_expression(1 / h_bar)
        double_pfl_tau1_with_damping / double_M[0, 0]
        double_pfl_tau1_with_damping / (h_bar * double_M[0, 0])
        simplify(double_pfl_tau1_with_damping / (h_bar * double_M[0, 0]))
        q2dot = sympy_to_expression(simplify(double_pfl_tau1_with_damping / (h_bar * double_M[0, 0])))
        dydx[3] = q2dot
        '''
        hbar_pinv = -L1*(M1 + M2)/(L2*M2*cos(t1 - t2))
        den = (L2*M2*cos(t1 - t2))
        dydx[3] = hbar_pinv * v - (G*(M1 + M2)*sin(t1) + L1*Q1_DAMPING*(M1 + M2)*t1_dot + L2*M2*sin(t1 - t2)*t2_dot**2)/den
        # dydx[3] = 0
        # tau_blob = G*sin(t1)/L1 + Q1_DAMPING*t1_dot + L2*M2*sin(t1_t2)*t2_dot**2/(L1*(M1 + M2))

        # '''
        # It is c.state dependent; in the cart-pole example above  and drops rank exactly when .
        # A system has Global Strong Inertial Coupling
        # if it exhibits Strong Inertial Coupling in every c.state.
        # '''
        # if np.abs(cos(t1_t2)) > 1e-2:
        #     coupling_term = cos(t1_t2)
        #     print("coupling_term", coupling_term)
        #     gain = -L1*(M1 + M2)/(L2*M2*coupling_term)
        #     dydx[3] = gain*v - tau_blob


        #     # dydx[1] = L2*M2*cos(t1+t2)*v - G*sin(t1)/L1 - L2*M2*sin(t1_t2)*t2_dot**2/(L1*(M1 + M2)) - Q1_DAMPING*t1_dot

        # else:
        #     print("WARNNNNNINGGGG SINGULAR DROPPPP RANKKKK!!!!")
        #     dydx[3] = 0

        # q1_gain = -L2*M2*cos(t1_t2)/(L1*(M1 + M2))
        # # q1_gain = L2*M2*cos(t1+t2)
        # dydx[1] = q1_gain * dydx[3] - tau_blob

        # print("dydx", dydx)

        # dydx[1] = -G*sin(t1)/L1 + L2*M2*cos(t1+t2)*dydx[3] - L2*M2*sin(t1_t2)*t2_dot**2/(L1*(M1 + M2)) - Q1_DAMPING*t1_dot

        return dydx

    def init_plot(self, fig, ax, texts):
        self.line, = ax.plot([], [], 'o-', lw=2)
        self.trace, = ax.plot([], [], '.-', lw=1, ms=2)
        self.history_x = deque(maxlen=self._args.history)
        self.history_y = deque(maxlen=self._args.history)
        self.texts = texts

    def data_gen(self):
        i = 0
        while True:
            i = (i + 1) % self.state.shape[0]

            if self._data_gen_cb is not None:
                self._data_gen_cb(i)

            yield i

    def draw_func(self, data):
        i = data

        thisx = [0, self.state[i, 4], self.state[i, 6]]
        thisy = [0, self.state[i, 5], self.state[i, 7]]
        self.line.set_data(thisx, thisy)

        if i == 0:
            self.history_x.clear()
            self.history_y.clear()

        self.history_x.appendleft(thisx[2])
        self.history_y.appendleft(thisy[2])
        self.trace.set_data(self.history_x, self.history_y)

        self.texts[0].set_text("q1=%.3f" % (self.state[i, 0]))
        self.texts[1].set_text("energy=%.3f" % (self.state[i, 8]))
        # aux1_text.set_text(aux1_template % (c.state[i, 0]))
        # aux2_text.set_text(aux2_template % (c.state[i, 2]))

        # aux2_text.set_text(aux2_template % (energy[i]))

        # aux3_text.set_text('hello')
        # aux4_text.set_text('hello')

        # return line, trace, line_orig, trace_orig, time_text
        return self.line, self.trace, *self.texts

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument('--playback', type=int, default=1, help='')
    parser.add_argument('--history', type=int, default=500, help='')

    parser.add_argument('--mode', type=int, default=0, help='')
    parser.add_argument('--initial', type=str, default="0,0,1,0", help='')
    '''
    q2_dot [3] cannot be 0
    http://underactuated.mit.edu/pend.html#energy_shaping
    This is true for any  theta_dot, except for theta_dot = 0
    (so it will not actually swing us up from the downright fixed point...
    but if you nudge the system just a bit, then it will start
    pumping energy and will swing all of the way up).
    '''

    parser.add_argument('--dt', type=float, default=0.02, help='')
    parser.add_argument('--t_stop', type=int, default=300, help='')
    args = parser.parse_args()

    initial_state = np.radians([float(x) for x in args.initial.split(',')])
    c = Acrobot(args, initial_state, np.arange(0, args.t_stop, args.dt))
    c.init_data()

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, L))
    ax.set_aspect('equal')
    ax.grid()
    plt.title('playback speed %dx' % (args.playback))

    texts = [
        ax.text(0.05, 0.9, '', transform=ax.transAxes),
        ax.text(0.05, 0.8, '', transform=ax.transAxes),
        ax.text(0.05, 0.7, '', transform=ax.transAxes),
        ax.text(0.05, 0.6, '', transform=ax.transAxes),
    ]
    ref_x1 = L1*sin(np.pi / 3)
    ref_y1 = -L1*cos(np.pi / 3)
    line1, = ax.plot([], [], 'c--', lw=1, alpha=0.5)
    line1.set_data([0, ref_x1], [0, ref_y1])

    c.init_plot(fig, ax, texts)

    ani = animation.FuncAnimation(
        fig,
        c.draw_func,
        c.data_gen,
        interval=args.dt*1000/args.playback,
        blit=True)
    plt.show()

#####################################################################
