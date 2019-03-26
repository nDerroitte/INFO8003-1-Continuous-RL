from constants import CST
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
