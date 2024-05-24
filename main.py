import logging
from sympy import symbols, diff, zeros, simplify, Function, lambdify, cos, Matrix, det
from scipy.integrate import odeint
import numpy as np
import math
from lagrange import LagrangeSolver
from logger import Logger

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

# Define the Lagrangian
L = T - V

# Define the generalized coordinates and their time derivatives
qArr = [theta, r]

# Create an instance of the LagrangeSolver class
lagrangeSolver = LagrangeSolver(T, V, qArr)
Eq = lagrangeSolver.LagrangeDynamicEqDeriver()

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
    0, 10, 100
)  # Solve the equations from t=0 to t=10 with 10 points in between

# Define the initial conditions for the generalized coordinates and their derivatives
initCnd = [
    15 / 180 * math.pi,
    0.1,
    0,
    0,
]  # These are just example values, replace with your actual initial conditions

# Call the DynamicEqSolver function
SS, xx = lagrangeSolver.DynamicEqSolver(Eq, paramSymbolList, paramVal, tSpan, initCnd)

# Print the state-space representation and the solution
print(SS)
print("xx = ", len(xx), len(xx[0]), len(xx[1])  )

# Plot the solutions
import matplotlib.pyplot as plt

# Create a new figure
plt.figure()

# Plot the solution for theta
plt.plot(tSpan, [sol[0] for sol in xx], label="theta")

# Plot the solution for r
plt.plot(tSpan, [sol[1] for sol in xx], label="r")

# Add labels and title
plt.xlabel("Time (sec)")
plt.ylabel("Generalized coordinates")
plt.title("Solution of the dynamic equations")
plt.legend()

# Show the plot
plt.show()
