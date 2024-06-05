from sympy import symbols, Function, cos
import numpy as np
from lagrange import LagrangeSolver
from exact import HarmonicMotion
from logger import Logger
from animation import Animator2

# Create an instance of the Logger class
logger = Logger()

# Define the symbols
m, k, l, g, t = symbols("m k l g t")

# Define the generalized coordinates
theta = Function("theta")(t)
r = Function("r")(t)

# Define the kinetic and potential energy
T = 0.5 * m * (r.diff(t) ** 2 + (r * theta.diff(t)) ** 2)
V = 0.5 * k * (r - l) ** 2 - m * g * r * cos(theta)

# Define the generalized coordinates and their time derivatives
qArr = [theta, r]

# Create an instance of the LagrangeSolver class
lagrangeSolver = LagrangeSolver(T, V, qArr)
Eq = lagrangeSolver.LagrangeDynamicEqDeriver()

# Define the symbols for the parameters
paramSymbolList = [m, k, l, g]

# Define the values for the parameters
mVal, kVal, lVal, gVal = (
    2,
    8,
    0.5,
    9.81,
)  # These are just example values, replace with your actual values
paramVal = [
    mVal,
    kVal,
    lVal,
    gVal,
]

# Define the time span for which to solve the equations
t_range = 10  # Adjust time range as needed.
num_points = 100  # Adjust number of points as needed.
tSpan = np.linspace(
    0, t_range, num_points
)  # Solve the equations from t=0 to t=10 with 10 points in between

# Define the initial conditions for the generalized coordinates and their derivatives
r0 =  lVal + (mVal * gVal / kVal)  # Initial value of r, replace with the actual value.
theta0 = np.pi / 10  # Initial value of θ, replace with the actual value.

initCnd = [
    theta0,
    r0,
    0,
    0,
]  # These are just example values, replace with your actual initial conditions

# Call the DynamicEqSolver function
SS, xx = lagrangeSolver.DynamicEqSolver(Eq, paramSymbolList, paramVal, tSpan, initCnd)

# compare with the exact solution
# Constants
A = 0.05 * r0  # Amplitude for r, replace with the actual value.
B = theta0  # Amplitude for θ, replace with the actual value.
r0 = r0  # Initial value of r, replace with the actual value.
phi_1 = 0  # Phase constant for r, replace with the actual value.
phi_2 = 0  # Phase constant for θ, replace with the actual value.


motion = HarmonicMotion(
    lVal, mVal, gVal, kVal, A, B, r0, phi_1, phi_2, t_range, num_points
)
rExact = motion.calculate_r()
thetaExact = motion.calculate_theta()

# Plot the motion compared to the exact solution
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(tSpan, xx[:, 1], label="Numerical")
plt.plot(motion.t, rExact, label="Exact")
plt.title("Plot of r against t")
plt.xlabel("Time (t)")
plt.ylabel("r(t)")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(tSpan, xx[:, 0], label="Numerical")
plt.plot(motion.t, thetaExact, label="Exact")
plt.title("Plot of θ against t")
plt.xlabel("Time (t)")
plt.ylabel("θ(t)")
plt.legend()

plt.tight_layout()
plt.show()

# animate the motion
# Create an instance of the Animator2 class
animator = Animator2(xx, tSpan)
