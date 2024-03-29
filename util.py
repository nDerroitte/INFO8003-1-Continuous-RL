import cv2
import os
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from constants import CST
from displaycaronthehill import save_caronthehill_image

###############################################################################
#                               Utils methods                                 #
###############################################################################


def createVideo(states, name):
    """
    Create a mp4 video of given name from a list of states given in
    argument. This method uses opencv.

    Parameters
    ----------
    states : list[State]
        List of the states to display
    name : str
        Name of the video to create
    """
    # Creating the first image to get the size of the image
    save_caronthehill_image(states[0].p, states[0].s, "tmpCOTH.png")
    frame = cv2.imread("tmpCOTH.png", 1)
    height, width, layers = frame.shape
    size = (width, height)

    # Opencv settings
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(name + '.avi', fourcc, CST.FPS, size)

    print("""Creating video file of the simulation """
          """({} frames)""".format(len(states)))
    # Only considerating 1 over CST.FRAME_STEP frame to gain performance
    for i in range(0, len(states), CST.FRAME_STEP):
        print("Currently working on frame {}".format(i), end='\r')

        # Creating the video from image and deleting temp image
        save_caronthehill_image(states[i].p, states[i].s, "tmpCOTH.png")
        frame = cv2.imread("tmpCOTH.png", 1)
        out.write(frame)
        os.remove("tmpCOTH.png")
    # Video creation
    out.release()
    print("\nVideo complete!")
    cv2.destroyAllWindows()


def plot3D(list):
    """
    Creating a 3D plot from the list of J.
    """
    # Getting X and Y
    y = np.linspace(-1, 1, 21)
    x = np.linspace(-3, 3, 61)

    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    # Creating Z from the list
    l = len(Z)
    w = len(Z[0])
    i = 0
    while i < l:
        j = 0
        while j < w:
            Z[i][j] = list[i][j]
            j = j + 1
        i = i + 1

    # Plot itself
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlabel("Speed")
    ax.set_ylabel("Position")
    ax.set_zlabel("Expected return")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis',
                    edgecolor='none')

    plt.show()


def getLSFirstIter(list_state):
    """
    Create a learning set for the Q fitted algorithm for the 1st iteration
    The "known" value is only the reward in this case

    Parameters
    ----------
    list_state : list[State]
        List of states according to our mesh size

    Returns
    -------
    list[list[float, float], float]
        Learning set
    """
    # Initialization
    learning_set = []
    for i in range(len(list_state)):
        for u in [-CST.BOUND_U, CST.BOUND_U]:
            # Getting the corresponding element
            p = list_state[i].p
            s = list_state[i].s
            y = list_state[i].getNextState(u).getReward()
            # Adding it to the LS
            learning_set.append([[p, s, u], y])
    array_LS = np.asarray(learning_set)
    return array_LS


def getLS(list_state, Q):
    """
    Create a learning set for the Q fitted algorithm

    Parameters
    ----------
    list_state : list[State]
        List of states according to our mesh size

    Q : skitlearn model
        Q(x, u)

    Returns
    -------
    list[list[float, float], float]
        Learning set
    """
    # Initialization
    learning_set = []
    for i in range(len(list_state)):
        for u in [-CST.BOUND_U, CST.BOUND_U]:
            # Computing the element needed
            p = list_state[i].p
            s = list_state[i].s
            next_x = list_state[i].getNextState(u)
            p_next = list_state[i].p
            s_next = list_state[i].s
            max_prevous_Q = max(Q.predict([[p_next, s_next, -CST.BOUND_U]]),
                                Q.predict([[p_next, s_next, CST.BOUND_U]]))
            r = next_x.getReward()
            y = r + CST.DISCOUT_FACTOR * max_prevous_Q
            # Adding them to the LS
            learning_set.append([[p, s, u], y])
    array_LS = np.asarray(learning_set)
    return array_LS
