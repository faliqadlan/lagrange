import numpy as np
import matplotlib.pyplot as plt

class HarmonicMotion:
    """
    A class to represent the motion of a simple harmonic oscillator.

    Attributes:
    l (float): The length of the pendulum.
    m (float): The mass of the pend
    g (float): The acceleration due to gravity.
    k (float): The spring constant.
    A (float): The amplitude for r.
    B (float): The amplitude for θ.
    phi_1 (float): The phase constant for r.
    phi_2 (float): The phase constant for θ.
    t (ndarray): An array of time values.

    """
    def __init__(self, l, m, g, k, A, B, r0, phi_1, phi_2, t_range, num_points):
        self.l = l
        self.m = m
        self.g = g
        self.k = k
        self.A = A
        self.B = B
        self.r0 = r0
        self.phi_1 = phi_1
        self.phi_2 = phi_2
        self.t = np.linspace(0, t_range, num_points)

    def calculate_r(self):
        r =  self.r0 + self.A * np.cos((np.sqrt(self.k / self.m) * self.t) + self.phi_1)
        return r

    def calculate_theta(self):
        theta = self.B * np.cos((np.sqrt((self.k * self.g) / (self.k * self.l + self.m * self.g)) * self.t) + self.phi_2)
        return theta

    def plot_motion(self):
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(self.t, self.calculate_r())
        plt.title("Plot of r against t")
        plt.xlabel("Time (t)")
        plt.ylabel("r(t)")

        plt.subplot(2, 1, 2)
        plt.plot(self.t, self.calculate_theta())
        plt.title("Plot of θ against t")
        plt.xlabel("Time (t)")
        plt.ylabel("θ(t)")

        plt.tight_layout()
        plt.show()
