from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from constants import CST
import numpy as np
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
        skitlearn model
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
        """
        Linear regression algorithm in charge of estimating Q based on the
        learning set

        Parameters
        ----------
        learning_set : list[list[float, float], float]
            Learning set of the SML algorithm

        Returns
        -------
        skitlearn model
            Q(x, u)
        """
        # Reshaping input
        x = learning_set[:, 0]
        X = np.array([np.array(i) for i in x])
        Y = learning_set[:, 1]
        # Using skitlearn package
        reg = LinearRegression(fit_intercept=True, normalize=False, n_jobs=-1)
        return reg.fit(X, Y)

    def getQERT(self, learning_set):
        """
        Extremely Randomized Trees algorithm in charge of estimating Q
        based on the learning set

        Parameters
        ----------
        learning_set : list[list[float, float], float]
            Learning set of the SML algorithm

        Returns
        -------
        skitlearn model
            Q(x, u)
        """
        # Reshaping input
        x = learning_set[:, 0]
        X = np.array([np.array(i) for i in x])
        Y = learning_set[:, 1]
        # Using skitlearn package
        reg = ExtraTreesRegressor(n_estimators=10, n_jobs=1,
                                  random_state=CST.RANDOM_SEED)
        return reg.fit(X, Y)

    def getQNN(self, learning_set):
        """
        Neural Networks algorithm in charge of estimating Q based on the
        learning set

        Parameters
        ----------
        learning_set : list[list[float, float], float]
            Learning set of the SML algorithm

        Returns
        -------
        skitlearn model
            Q(x, u)
        """
        # Reshaping input
        x = learning_set[:, 0]
        X = np.array([np.array(i) for i in x])
        Y = learning_set[:, 1]
        # Using skitlearn package
        reg = MLPRegressor(hidden_layer_sizes=(3,),
                                       activation='relu',
                                       solver='adam',
                                       learning_rate='adaptive',
                                       max_iter=1000,
                                       learning_rate_init=0.01,
                                       alpha=0.01)
        return reg.fit(X,Y)
