from sklearn.linear_model import LinearRegression
###############################################################################
#                               SML Class                                     #
###############################################################################


class SML:
    def __init__(self, algo):
        """
        Parameters
        ----------
        algo : str
            The SML algorithm to use in the Q fitted iteration algo
        """
        self.algo = algo

    def getQ(self, learning_set):
        """
        Controller function responsible to call the correct SML algorithm
        depending on how the program has been set up.

        Parameters
        ----------
        learning_set : list[list[float, float], float]
            Learning set of the SML algorithm

        Returns
        -------
        list[list[list[float]]]
            Q(x, u)
        """
        if self.algo == "LR":
            return self.getQRL(learning_set)
        elif self.algo == "ERT":
            return self.getQERT(learning_set)
        elif self.algo == "NN":
            return self.getQNN(learning_set)
        else:
            # Should not happen
            print("Unknown algorithm.")

    def getQRL(self, learning_set):

        return LinearRegression(fit_intercept=True, normalize=False, n_jobs=1).fit(learning_set[0],
                                                                                   learning_set[1])
