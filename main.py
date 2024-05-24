import logging
from sympy import symbols, diff, zeros, simplify, Function, lambdify, cos, Matrix, det
from scipy.integrate import odeint
import numpy as np

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


def LagrangeDynamicEqDeriver(L, qArr):
    """
    This function derives the Lagrange's equations of motion from a given Lagrangian.

    Parameters:
    L: The Lagrangian of the system.
    qArr: An array of generalized coordinates.

    Returns:
    Eq: The Lagrange's equations of motion.
    """

    # Define the symbol for time
    t = symbols("t")

    # Get the number of generalized coordinates
    Nq = len(qArr)

    # Lagrangian equation
    L = T - V  # This line seems to be redefining the Lagrangian, it might be an error

    # Initialize the derivatives symbols
    Dq = zeros(Nq, 1)

    # Euler-Lagrange equation
    for ii in range(Nq):
        Dq[ii] = qArr[ii].diff()


    # Initialize the Lagrange's equations of the second kind
    Eq = zeros(Nq, 1)

    # Calculate the Lagrange's equations of the second kind
    for ii in range(Nq):
        # The Euler-Lagrange equation is given by d/dt(∂L/∂q̇) - ∂L/∂q = 0
        Eq[ii] = diff(diff(L, Dq[ii]), t) - diff(L, qArr[ii])

    return Eq


def DynamicEqSolver(Eq, qArr, paramSymbolList, paramVal, tSpan, initCnd):
    """
    This function solves the dynamic equations of motion derived from the Lagrangian.

    Parameters:
    Eq: The dynamic equations to be solved.
    qArr: An array of generalized coordinates.
    paramSymbolList: A list of symbols representing the parameters in the equations.
    paramVal: The values of the parameters.
    tSpan: The time span over which to solve the equations.
    initCnd: The initial conditions for the equations.

    Returns:
    SS: The state-space representation of the system.
    xx: The solution of the system over the given time span.
    """

    # Get the number of equations
    Nq = len(Eq)

    # Compute the derivative of the generalized coordinates
    Dq = [q.diff() for q in qArr]

    # Define the symbols for the second derivatives of the generalized coordinates
    DDq = [Dq[ii].diff() for ii in range(Nq)]

    # Compute the Jacobian of the equations with respect to the second derivatives
    print("Eq = ", Eq)
    print("DDq = ", DDq)
    AA = Eq.jacobian(DDq)
    print("AA = ",AA)

    # Compute the remaining terms in the equations
    BB = -simplify(Eq - AA * Matrix(DDq))
    print("BB = ", BB)

    # Initialize the second derivatives of the generalized coordinates
    DDQQ = zeros(Nq, 1)

    # Compute the determinant of the Jacobian
    DET_AA = det(AA)
    print("DET_AA = ", DET_AA)

    # Solve for the second derivatives of the generalized coordinates
    for ii in range(Nq):
        AAn = AA.copy()
        AAn[:, ii] = BB
        DDQQ[ii] = simplify(det(AAn) / DET_AA)

    # Initialize the state-space representation of the system
    SS = zeros(2 * Nq, 1)

    # Fill in the state-space representation
    # for ii in range(N):
    #     SS[ii] = Dq[ii]
    #     SS[ii + N] = DDQQ[ii]

    # Combine the generalized coordinates and their derivatives
    # Q = qArr + Dq

    # Define the symbols for the state variables
    # X = symbols("x:%d" % (2 * N))

    # Substitute the state variables into the state-space representation
    # SS = SS.subs(dict(zip(Q, X)))

    # Substitute the parameter values into the state-space representation
    # SS_0 = SS.subs(dict(zip(paramSymbolList, paramVal)))

    # Convert the symbolic state-space representation into a numerical function
    # SS_ode0 = lambdify((X, symbols("t")), SS_0, "numpy")

    # Define the function to be integrated
    # def SS_ode(t, x):
    #     x = np.atleast_1d(x)
    #     return np.array(SS_ode0(*x, t)).flatten()

    # Solve the system of equations over the given time span
    # xx = odeint(SS_ode, initCnd, tSpan)


# Define the symbols
m, k, l, g, t = symbols("m k l g t")

# Define the generalized coordinates
theta = Function("theta")(t)
x = Function("x")(t)

# Define the kinetic and potential energy
T = 1 / 2 * m * (x.diff(t) ** 2 + (l + x) ** 2 * theta.diff(t) ** 2)
V = -m * g * (l + x) * cos(theta) + 1 / 2 * k * x**2

# Define the Lagrangian
L = T - V

# Define the generalized coordinates and their time derivatives
qArr = [theta, x]

# Derive the equations of motion
Eq = LagrangeDynamicEqDeriver(L, qArr)

# Define the symbols for the parameters
paramSymbolList = [m, k, l, g]

# Define the values for the parameters
paramVal = [
    1,
    2,
    1,
    9.81,
]  # These are just example values, replace with your actual values

# Define the time span for which to solve the equations
tSpan = np.linspace(
    0, 10, 10
)  # Solve the equations from t=0 to t=10 with 10 points in between

# Define the initial conditions for the generalized coordinates and their derivatives
initCnd = [
    0,
    0,
    0,
    0,
]  # These are just example values, replace with your actual initial conditions

# Call the DynamicEqSolver function
DynamicEqSolver(Eq, qArr, paramSymbolList, paramVal, tSpan, initCnd)

# Print the state-space representation and the solution
# print(SS)
# print(xx)
