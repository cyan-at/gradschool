

# import ipdb; ipdb.set_trace();


# T = m1 / 2 * l1**2 * dt1**2 + m2 / 2 * (l1**2 * dt1**2 + l2**2*dt2**2 - 2 * l1 * l2 * dt1 * dt2 * cos(t1 + t2)) + m3 / 2 * (l1**2 * dt1**2 + l3 * dt3**2 + 2*l1*l3*dt1*dt3*cos(t1 - t3))

# import sympy
# sympy.init_printing()
# import sympy
# sympy.init_printing()
# f.diff(t)
# f = cos(x - y)
# f.diff(t)
# from sympy import *; import sympy;

# V = (m1 + m2 + m3)*g*(-l1)*cos(t1) + m2*g*(-l2)*cos(t2) + m3*g*(-l3)*cos(t3)
# g = Symbol('g')
# V = (m1 + m2 + m3)*g*(-l1)*cos(t1) + m2*g*(-l2)*cos(t2) + m3*g*(-l3)*cos(t3)
# L = T - V
# L
# L.diff(dt1)
# simplify(L.diff(dt1))
# p1 = L.diff(dt1)
# p1
# p1.diff(t)
# p1
# p1_byhand = l1**2*(m1+m2+m3)*dt1 - m2*l1*l2*dt2*cos(t1+t2) + m3*l1*l3*dt3*cos(t1-t3)
# p1
# p1_byhand
# p1_byhand.diff(t)
# L
# L.diff(t1)
# L.diff(t2)
# L.diff(t3)
# p1_byhand.diff(t) - L.diff(t1)
# el1 = p1_byhand.diff(t) - L.diff(t1)
# el1 / l1
# simplify(el1 / l1)
# x
# y
# temp = cos(x + y)
# temp.diff(t)
# temp = cos(x - y)
# temp.diff(t)
# p2_byhand = m2*l2**2*dt2 - m2*l1*l2*dt1*cos(t1+t2)
# p2_byhand.diff(t)
# p3_byhand = m3*l3**2*dt3 + m3*l1*l3*dt1*cos(t1-t3)
# p3_byhand.diff(t)
# el1 = (p1_byhand.diff(t) - L.diff(t1)) / l1
# el1
# el1 = simplify((p1_byhand.diff(t) - L.diff(t1)) / l1)
# el1
# el2 = p2_byhand.diff(t) - L.diff(t2)
# el2
# el3 = p3_byhand.diff(t) - L.diff(t3)
# el3
# el2 = (p2_byhand.diff(t) - L.diff(t2)) / l2
# el2
# el2 = simplify((p2_byhand.diff(t) - L.diff(t2)) / l2)
# el2
# el2 = simplify((p2_byhand.diff(t) - L.diff(t2)) / (m2*l2))
# el2
# el3 = simplify((p3_byhand.diff(t) - L.diff(t3)) / (m3*l3))
# el3
# el3
# el2
# el1



    def derivs_pfl_collocated_taskspace(self, state, t):
        dydx = np.zeros_like(state)

        # q1_dot
        dydx[0] = state[1] # state*[0] = q0* = state[1]

        # q2_dot
        dydx[2] = state[3] # state*[2] = q1* = state[1]

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
        dydx[1] = -G*sin(t1)/L1 + L2*M2*cos(t1+t2)*dydx[3] -\
            L2*M2*sin(t1_t2)*t2_dot**2/(L1*(M1 + M2))\
            - Q1_DAMPING*t1_dot

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

    def derivs_pfl_collocated_taskspace_3(self, state, t):
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
        k = 10
        a = 0.1

        # k = 2
        # a = np.pi / 2

        yd = a * sin(t / k)
        yd_dot = a * cos(t / k) / k
        yd_dotdot = -a * sin(t / k) / k**2

        # control law stays the same
        v = yd_dotdot + K5 * (yd_dot - t1_dot) + K4 * (yd - t1)

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


        q1_gain = -L2*M2*cos(t1_t2)/(L1*(M1 + M2))
        # q1_gain = L2*M2*cos(t1+t2)
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

expr = double_q1dot_with_damping
expr = python(expr).split("\n")[-1].split(" = ")[-1]
expr = expr.replace("t2(t)", "t2")
expr = expr.replace("t1(t)", "t1")
expr = expr.replace("Derivative(t1, t)", "t1_dot")
expr = expr.replace("Derivative(t2, t)", "t2_dot")
print(expr)

# artifact for actuation

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

G*sin(t1(t))/L1 + p1_damping*Derivative(t1(t), t) + L2*M2*sin(t1(t) - t2(t))*Derivative(t2(t), t)**2/(L1*(M1 + M2))


h_bar_pinv_blob = -L1*(M1 + M2)/(L2*M2*cos(t1_t2))

yd_blob = -(G*(M1 + M2)*sin(t1) + L1*p1_damping*(M1 + M2)*t1_dot + L2*M2*sin(t1 - t2)*t2_dot**2)/(L1*(M1 + M2))



yd_dotdot = -t⋅sin(t) + 2⋅cos(t)

h_bar_pinv_blob * (yd_dotdot - yd_blob)



def derivs_original(c.state, t):
    '''
        c.state = x = [
            theta1
            theta1*
            theta2
            theta2*
        ]

        c.state = x* = [
            theta1*
            theta1**
            theta2*
            theta2**
        ]
    '''
    dydx = np.zeros_like(c.state)

    dydx[0] = c.state[1]
    dydx[2] = c.state[3]

    delta = c.state[2] - c.state[0]

    den1 = (M1+M2) * L1 - M2 * L1 * cos(delta) * cos(delta)

    dydx[1] = ((M2 * L1 * c.state[1] * c.state[1] * sin(delta) * cos(delta)
                + M2 * G * sin(c.state[2]) * cos(delta)
                + M2 * L2 * c.state[3] * c.state[3] * sin(delta)
                - (M1+M2) * G * sin(c.state[0]))
               / den1)


    den2 = (L2/L1) * den1

    dydx[3] = ((- M2 * L2 * c.state[3] * c.state[3] * sin(delta) * cos(delta)
                + (M1+M2) * G * sin(c.state[0]) * cos(delta)
                - (M1+M2) * L1 * c.state[1] * c.state[1] * sin(delta)
                - (M1+M2) * G * sin(c.state[2]))
               / den2)

    return dydx


def pull_out_leftover(expr, terms, default=0):
    x = expr
    for t in terms:
        leftover_dict = collect(expr, t, evaluate=False)
        x = leftover_dict[1] if 1 in leftover_dict else 0
    return x


    # # remove noise
    # theta1_envelope = FilteredSignal(theta1_envelope, 1000., 10)

    # # hilbert fits better on more 'internally consistent data'
    # theta1_envelope = np.array([])
    # interval = 10000.0
    # samples = np.linspace(
    #     0,
    #     system.state.shape[0],
    #     int(system.state.shape[0] / interval + 1))

    # samples = []
    # i = 0
    # delta = 500
    # while i < system.state.shape[0]:
    #     samples.append(i)
    #     delta = delta * 2
    #     i += delta
    # samples.append(system.state.shape[0])

    # samples = [int(x) for x in samples]

    # i = 0
    # while i < len(samples) - 1:
    #     print(samples[i], samples[i+1])
    #     envelope_piece = np.abs(hilbert(system.state[samples[i]:samples[i+1], 0]))
    #     theta1_envelope = np.concatenate((theta1_envelope,envelope_piece))
    #     i += 1