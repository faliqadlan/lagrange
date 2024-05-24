from sympy import symbols, diff, zeros, simplify, Function, lambdify, cos, Matrix, det
from scipy.integrate import odeint
import numpy as np
import math


class LagrangeSolver:
    def __init__(self, T, V, qArr):
        self.T = T
        self.V = V
        self.qArr = qArr

    def LagrangeDynamicEqDeriver(self):
        """
        This method derives the Lagrange's equations of motion from a given Lagrangian.

        Returns:
        Eq: The Lagrange's equations of motion.
        """

        # Define the symbol for time
        t = symbols("t")

        # Get the number of generalized coordinates
        Nq = len(self.qArr)

        # Lagrangian equation
        L = self.T - self.V

        # Initialize the derivatives symbols
        Dq = zeros(Nq, 1)

        # Euler-Lagrange equation
        for ii in range(Nq):
            Dq[ii] = self.qArr[ii].diff()

        # Initialize the Lagrange's equations of the second kind
        Eq = zeros(Nq, 1)

        # Calculate the Lagrange's equations of the second kind
        for ii in range(Nq):
            # The Euler-Lagrange equation is given by d/dt(∂L/∂q̇) - ∂L/∂q = 0
            Eq[ii] = diff(diff(L, Dq[ii]), t) - diff(L, self.qArr[ii])

        return Eq

    def DynamicEqSolver(self, Eq, paramSymbolList, paramVal, tSpan, initCnd):
        """
        This method solves the dynamic equations of motion derived from the Lagrangian.

        Parameters:
        Eq: The dynamic equations to be solved.
        paramSymbolList: A list of symbols representing the parameters in the equations.
        paramVal: The values of the parameters.
        tSpan: The time span over which to solve the equations.
        initCnd: The initial conditions for the equations.

        Returns:
        SS: The state-space representation of the system.
        xx: The solution of the system over the given time span.
        """

        # Define the symbol for time
        t = symbols("t")

        # Get the number of equations
        Nq = len(Eq)

        # Compute the derivative of the generalized coordinates
        Dq = [q.diff() for q in self.qArr]

        # Define the symbols for the second derivatives of the generalized coordinates
        DDq = [Dq[ii].diff() for ii in range(Nq)]

        # Compute the Jacobian of the equations with respect to the second derivatives
        AA = Eq.jacobian(DDq)

        # Compute the right-hand side of the equations
        BB = -simplify(Eq - AA * Matrix(DDq))

        # Initialize the second derivatives of the generalized coordinates
        DDQQ = zeros(Nq, 1)

        # Compute the determinant of the Jacobian
        DET_AA = det(AA)

        # Solve for the second derivatives of the generalized coordinates
        for ii in range(Nq):
            AAn = AA.copy()
            AAn[:, ii] = BB
            DDQQ[ii] = simplify(det(AAn) / DET_AA)

        # Initialize the state-space representation of the system
        SS = zeros(2 * Nq, 1)

        # Fill in the state-space representation
        for ii in range(Nq):
            SS[ii] = Dq[ii]
            SS[ii + Nq] = DDQQ[ii]

        # Define the symbols for the new variables
        Q = Matrix([self.qArr, Dq])

        # Define new symbols x
        X = symbols("x_:{}".format(2 * Nq))

        # Substitute q and Dq with x in SS
        SS = SS.subs(dict(zip(Q, X)))

        # Convert the symbolic state-space representation to a numerical function
        # Create a dictionary that maps parameter symbols to their values
        param_mapping = dict(zip(paramSymbolList, paramVal))

        # Substitute the parameter symbols in SS with their values
        SS_numerical = SS.subs(param_mapping)

        # Convert the symbolic SS_numerical to a numerical function
        # The function takes two arguments: X and t
        SS_func = lambdify((X, t), SS_numerical, "numpy")

        # Define the ODE system
        def SS_ode(x, t):
            return SS_func(x, t).flatten()

        # Solve the ODE system
        xx = odeint(func=SS_ode, y0=initCnd, t=tSpan)

        return SS, xx
