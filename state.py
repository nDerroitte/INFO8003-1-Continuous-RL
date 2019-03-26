from constants import CST
from environment import *
###############################################################################
#                               State class                                   #
###############################################################################


class State:
    def __init__(self, p, s):
        self.p = p
        self.s = s

    def isFinal(self):
        """
        Check if the state is final or not.

        Returns
        -------
        bool
            True if the state is final. False otherwise
        """
        if abs(self.p) > CST.MAX_POS or \
           abs(self.s) > CST.MAX_SPEED:
            return True
        return False

    def discretize(self):
        """
        Discretize the state by step of 0.1

        Returns
        -------
        list [p, s]
            List of the position and speed approximated at the closest 0.1
        """
        return [round(self.p, 1), round(self.s, 1)]

    def print(self):
        """
        Print the state
        """
        print(self.p, self.s)

    def getReward(self):
        """
        Get the reward of the current state

        Returns
        -------
        int
            The reward of the state object
        """
        if self.p < -1 or abs(self.s) > 3:
            return -1
        elif self.p > 1 and abs(self.s) <= 3:
            return 1
        else:
            return 0

    def getNextState(self, action):
        """
        Get the next state from this one using the environment and the action
        given. In order to do so, this function need to define a tempory env
        since it is needed to create the next state.

        Parameters
        ----------
        action = {-4, 4}
            Action we want the agent to perform

        Returns
        -------
        State
            The next State from this State using the action given
        """
        e = Environment()
        next_p = self.p + CST.H * self.s
        next_s = self.s + (CST.H * e.DerivativeS(self.p, self.s, action))
        next_state = State(next_p, next_s)
        return next_state
