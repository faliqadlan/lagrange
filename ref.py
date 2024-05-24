import logging

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a console handler
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

# Create a formatter
formatter = logging.Formatter(
    "%(asctime)s - %(lineno)d - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# Set the formatter for the handler
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)

# Log some messages
logger.debug("Debug message")

from sympy import symbols, diff, zeros, simplify, Function, lambdify
from sympy import symbols, zeros, simplify, det, Matrix
from scipy.integrate import odeint
import numpy as np


def LagrangeDynamicEqDeriver(L, q, Dq):
    # Define the symbol for time
    t = symbols("t")

    # Get the number of generalized coordinates
    N = len(q)

    # Initialize the derivatives of the Lagrangian with respect to q and Dq
    L_q = zeros(N, 1)
    L_Dq = zeros(N, 1)

    # Calculate the derivatives of the Lagrangian with respect to q and Dq
    for ii in range(N):
        L_q[ii] = diff(L, q[ii])
        L_Dq[ii] = diff(L, Dq[ii])

    # Initialize the time derivative of the derivative of the Lagrangian with respect to Dq
    L_Dq_dt = zeros(N, 1)

    logger.debug("L_Dq: %s", L_Dq)

    # Calculate the time derivative of the derivative of the Lagrangian with respect to Dq
    for ii in range(N):
        for jj in range(N):
            # Define the function for the generalized coordinate and its time derivative
            q_dst = Function(q[jj].name + "(t)")
            Dq_dst = diff(q_dst, t)

            # Substitute the generalized coordinate and its time derivative into the derivative of the Lagrangian
            L_Dq[ii] = L_Dq[ii].subs({q[jj]: q_dst, Dq[jj]: Dq_dst})

        # Define the function for the derivative of the Lagrangian
        L_Dq_fcn = L_Dq[ii].subs(t, t)

        # Calculate the time derivative of the derivative of the Lagrangian
        L_Dq_dt[ii] = diff(L_Dq_fcn, t)

        for jj in range(N):
            # Define the function for the generalized coordinate, its time derivative, and its second time derivative
            q_orig = Function(q[jj].name + "(t)")
            Dq_orig = diff(q_orig, t)
            DDq_orig = diff(q_orig, t, t)

            # Define the symbol for the second time derivative of the generalized coordinate
            DDq_dst = symbols("DD" + q[jj].name)

            # Substitute the generalized coordinate, its time derivative, and its second time derivative into the time derivative of the derivative of the Lagrangian
            L_Dq_dt[ii] = L_Dq_dt[ii].subs(
                {q_orig: q[jj], Dq_orig: Dq[jj], DDq_orig: DDq_dst}
            )

    # Initialize the Lagrange's equations of the second kind
    Eq = zeros(N, 1)

    # Calculate the Lagrange's equations of the second kind
    for ii in range(N):
        Eq[ii] = simplify(L_Dq_dt[ii] - L_q[ii])

    # Return the Lagrange's equations of the second kind
    return Eq


def DynamicEqSolver(Eq, q, Dq, ParamList, ParamVal, tspan, InitCnd):
    # Define the symbol for time
    t = symbols("t")

    # Get the number of equations
    N = len(Eq)

    # Initialize the second time derivatives of the generalized coordinates
    DDq = zeros(1, N)
    for ii in range(N):
        DDq[ii] = symbols("DD" + q[ii].name)

    # Define the left-hand side and right-hand side of the equations
    AA = Eq.jacobian(DDq)
    BB = -(simplify(Eq - AA * Matrix(DDq)))

    # Initialize the second time derivatives of the generalized coordinates
    DDQQ = zeros(N, 1)
    DET_AA = det(AA)

    # Solve for the second time derivatives of the generalized coordinates
    for ii in range(N):
        AAn = AA.copy()
        AAn[:, ii] = BB
        DDQQ[ii] = simplify(det(AAn) / DET_AA)

    # Initialize the state-space form of the equations
    SS = zeros(N, 1)

    # Define the state-space form of the equations
    for ii in range(N):
        SS[ii] = Dq[ii]
        SS[ii + N] = DDQQ[ii]

    # Change variables from q to x
    Q = q + Dq
    X = symbols("x:%d" % (2 * N))
    SS = SS.subs(dict(zip(Q, X)))

    # Substitute the parameter values into the state-space form of the equations
    SS_0 = SS.subs(dict(zip(ParamList, ParamVal)))

    # Convert the state-space form of the equations to a lambda function
    SS_ode0 = lambdify((X, t), SS_0, "numpy")

    # Define the function for the ODE solver
    def SS_ode(x, t):
        return np.array(SS_ode0(*x, t)).flatten()

    # Solve the ODEs
    xx = odeint(SS_ode, InitCnd, tspan)

    return SS, xx


from sympy import symbols, cos


# Define the symbols
th, Dth, x, Dx = symbols("th Dth x Dx")
m, l, k, g, t = symbols("m l k g t")

# Define the kinetic and potential energy
T = 1 / 2 * m * (Dx**2 + (l + x) ** 2 * Dth**2)
V = -m * g * (l + x) * cos(th) + 1 / 2 * k * x**2

# Define the Lagrangian
L = T - V

# Define the generalized coordinates and their time derivatives
q = [th, x]
Dq = [Dth, Dx]

# Derive the equations of motion
Eq = LagrangeDynamicEqDeriver(L, q, Dq)

# Define the time span
tt = np.linspace(0, 10, 300)

# Solve the equations of motion
SS, xx = DynamicEqSolver(
    Eq, q, Dq, [m, l, k, g], [1, 1, 10, 9.81], tt, [45 / 180 * np.pi, 0.1, 0, 0]
)
