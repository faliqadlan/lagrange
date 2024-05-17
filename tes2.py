import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define parameters
m = 1.0  # mass in kg
k = 10.0  # spring constant in N/m
r0 = 0.5  # rest length of the spring in m
g = 9.81  # acceleration due to gravity in m/s^2

# Define Lagrangian and its partial derivatives
def lagrangian(theta, lmbda, dthetadt, dlambdadt):
    T = 0.5 * m * (dthetadt**2 * r0**2 + dlambdadt**2 * r0**2)  # Kinetic energy
    U = m * g * r0 * (1 - lmbda) + 0.5 * k * r0**2 * lmbda**2  # Potential energy
    return T - U

def dL_dthetadt(theta, lmbda, dthetadt, dlambdadt):
    return m * r0**2 * dthetadt

def dL_dlambdadt(theta, lmbda, dthetadt, dlambdadt):
    return m * r0**2 * dlambdadt

def dL_dtheta(theta, lmbda, dthetadt, dlambdadt):
    return 0

def dL_dlmbda(theta, lmbda, dthetadt, dlambdadt):
    return -m * g * r0 + k * r0**2 * lmbda

# Define the equations of motion using Euler-Lagrange equations
def euler_lagrange(t, y):
    theta, lmbda, dthetadt, dlambdadt = y
    dL_d_dthetadt = dL_dthetadt(theta, lmbda, dthetadt, dlambdadt)
    dL_d_dlambdadt = dL_dlambdadt(theta, lmbda, dthetadt, dlambdadt)
    dL_d_theta = dL_dtheta(theta, lmbda, dthetadt, dlambdadt)
    dL_d_dlmbda = dL_dlmbda(theta, lmbda, dthetadt, dlambdadt)
    
    epsilon = 1e-10  # Small epsilon value to avoid division by zero
    
    d2thetadt2 = (dL_d_dlmbda - dL_d_dthetadt) / (dL_d_theta + epsilon)
    d2lambdadt2 = dL_d_dlambdadt / dL_d_dlmbda
    
    return [dthetadt, dlambdadt, d2thetadt2, d2lambdadt2]

# Initial conditions
theta0 = 0.0  # initial angle
lmbda0 = 0.0  # initial displacement lambda
dthetadt0 = 0.0  # initial angular velocity
dlambdadt0 = 0.0  # initial velocity of lambda

# Time span for the simulation
t_span = (0, 10)  # 10 seconds
t_eval = np.linspace(t_span[0], t_span[1], 500)

# Solve the differential equation
sol = solve_ivp(euler_lagrange, t_span, [theta0, lmbda0, dthetadt0, dlambdadt0], t_eval=t_eval, method='RK45')

# Compute the equation of motion
equation_of_motion = sol.y[2] / (m * r0**2) - sol.y[0]

# Plot the results
plt.plot(sol.t, equation_of_motion, label='Equation of Motion')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (rad/s^2)')
plt.title('Equation of Motion for the Spring-Mass System')
plt.legend()
plt.grid()
plt.show()

