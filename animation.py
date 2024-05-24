import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image


# Define the Animator2 function
def Animator2(X, tt):
    # Create a new figure
    fig, ax = plt.subplots()

    # Initialize the plot objects
    (H1,) = ax.plot([], [], "b")
    (H2,) = ax.plot([], [], "ro", markersize=14, linewidth=3)

    # Set the axis limits
    ax.set_xlim(2, 8)
    ax.set_ylim(0, 6)

    # Remove the axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Define the rotation function
    def Rot(X, th):
        return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]).dot(X)

    # Animation function
    def animate(i):
        # Update the spring
        th1 = X[i, 0]
        dx = X[i, 1]
        x11 = 5
        y11 = 5
        x12 = x11 + (1 + dx) * np.sin(th1)
        y12 = y11 - (1 + dx) * np.cos(th1)
        a = 0.1
        t0 = np.linspace(0, (1 + dx), 1000)
        x = a * np.sin(100 / (1 + dx) * t0)
        y = -t0
        XX = np.array([x, y])
        Xr = Rot(XX, th1) + np.array([[x11], [y11]]).dot(np.ones((1, len(t0))))
        H1.set_data(Xr[0, :], Xr[1, :])

        # Update the mass
        H2.set_data(x12, y12)

        # Update the time text
        ax.set_title(f"Time: {tt[i]:.2f} sec")

        # Save the current frame as a GIF
        # plt.savefig(f"frame_{i:04d}.png")

    # Create the animation
    ani = animation.FuncAnimation(fig, animate, frames=len(tt))

    # Show the animation
    plt.show()

    # Create a GIF from the saved frames
    # frames = [Image.open(f"frame_{i:04d}.png") for i in range(len(tt))]
    # frames[0].save(
    #     "Anim2.gif",
    #     format="GIF",
    #     append_images=frames[1:],
    #     save_all=True,
    #     duration=50,
    #     loop=0,
    # )
