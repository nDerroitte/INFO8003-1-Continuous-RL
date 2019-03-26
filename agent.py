from state import *
from environment import *
from util import *
from constants import CST
from sml import *
import numpy as np

###############################################################################
#                               Agent Class                                   #
###############################################################################


class Agent:
    def __init__(self, start_p=0, start_s=0):
        """
        Parameters
        ----------
        start_p : float, optional
            Initial position
        start_s : float, optional
            Initial speed
        """
        self.__env = Environment()
        self.__state = State(start_p, start_s)
        # Discrete policy
        # Initialization of the policy to BOUND_U eveywhere
        # ie  action = 4 for all s, p.
        self.__disc_policy = [[CST.BOUND_U for i in range(self.__env.nb_s + 1)]
                              for j in range(self.__env.nb_p + 1)]

    def __getNextState(self, action, state=None):
        """
        Compute the next state with the next action from the state given
        in argument. If no state is given, the state of this object is used.

        Parameters
        ----------
        action : {-4, 4}
            Action taken
        state : State Object, optional
          Initial state from which we compute the next one

        Returns
        -------
        state : State Object
            the next state.
            If no state is given as parameters, no return but the self.__state
            object is modified accordindly
        """
        if state is None:
            x = self.__state
        else:
            x = state
        # Formula :
        # yn = yn-1 + h * y'
        # If y = p, y' = s
        # If y = s, y' = formula computed in self.__env.__getDerivativeS
        next_p = x.p + CST.H * x.s
        next_s = x.s + (CST.H * self.__env.DerivativeS(x.p, x.s, action))
        next_state = State(next_p, next_s)

        # Returning if state was given in argument
        if state is None:
            self.__state = next_state
            return
        else:
            return next_state

    def createSimplePolicy(self, direction=CST.DIR_LEFT):
        """
        Create a simple policy. Can be going always left (direction=
        CST.DIR_LEFT), going always right (direction=CST.DIR_RIGHT) or totally
        random ( direction=CST.DIR_RANDOM)

        Parameters
        ----------
        direction : {CST.DIR_LEFT, CST.DIR_RIGHT, CST.DIR_RANDOM}, optional
            Direction where the policy.

        Returns
        -------
        Nothing but update the self.__disc_policy of the object
        """
        # For each element of the policy
        for i in range(len(self.__disc_policy)):
            for j in range(len(self.__disc_policy[0])):
                if direction == CST.DIR_RANDOM:
                    # Random action
                    self.__disc_policy[i][j] = random.choice([-CST.BOUND_U,
                                                              CST.BOUND_U])
                else:
                    # real action
                    self.__disc_policy[i][j] = direction

    def visualisePolicy(self, name):
        """
        Allow to visualise the policy in the object for the time simulation
        given in argument of the main.

        Parameters
        ----------
        name : str
            Name of the video
        """
        states_to_display = []
        for t in range(CST.MAX_NB_FRAME):
            # Compute next state and append current one
            states_to_display.append(self.__state)
            next_p, next_s = self.__state.discretize()
            # Transform to index
            next_p_index = self.__env.pos_to_index(next_p)
            next_s_index = self.__env.speed_to_index(next_s)
            # Getting next state
            self.__getNextState(self.__disc_policy[next_p_index][next_s_index])
            # If final, stop the simulation
            if self.__state.isFinal():
                print("Final state reach before end of simulation!")
                break
        # Create the video
        createVideo(states_to_display, name)

    def __createRandomExp(self, initial_state=None, action=None):
        """
        Create a experience from the policy of the object from the initial
        state given in argument. If none is given, a random one is used

        Parameters
        ----------
        initial_state : state, optional
            Initial state from which we compute the experience
        action = {-4, 4}, optional
            The action taken by the agent

        Returns
        -------
        list : [xi, ui, ri, xi+1]
            The experience created
        """
        if initial_state is None:
            # If no initial state, create one randomly
            p = round(random.uniform(-CST.MAX_POS, CST.MAX_POS), 1)
            s = round(random.uniform(-CST.MAX_SPEED, CST.MAX_SPEED), 1)
            x = State(p, s)
        else:
            x = initial_state
        # Getting index of the state
        p_index = self.__env.pos_to_index(x.p)
        s_index = self.__env.speed_to_index(x.s)
        # Getting action of the state accoridng to the policy
        if action is None:
            u = self.__disc_policy[p_index][s_index]
        else:
            u = action
        # Computing the next state
        next_x = self.__getNextState(u, x)
        # Getting the reward of going to the next state
        r = next_x.getReward()
        return [x, u, r, next_x]

    def __createTrajectory(self, initial_state=None):
        """
        Create a trajectory from the policy of the object from the initial
        state given in argument. If none is given, a random one is used

        Parameters
        ----------
        initial_state : state, optional
            Initial state from which we compute the experience

        Returns
        -------
        list of list: [[xi, ui, ri, xi+1]]
            The trajectory created
        """
        trajectory = []
        # Getting the initial experience
        initial = self.__createRandomExp(initial_state)
        trajectory.append(initial)
        for t in range(1, CST.LENGTH_EPISODE):
            # Creating the following experiences
            x = trajectory[t-1][3]
            # Getting the corresponding indexes
            p_index = self.__env.pos_to_index(x.p)
            s_index = self.__env.speed_to_index(x.s)
            # Getting the action
            u = self.__disc_policy[p_index][s_index]
            # Getting the next state
            next_x = self.__getNextState(u, x)
            # Getting the reward
            r = next_x.getReward()
            trajectory.append([x, u, r, next_x])
            # Stop if final is arrived
            if next_x.isFinal():
                break

        return trajectory

    def evaluatePolicy(self, MonteCarlo=True):
        """
        Evaluate the policy by computing the expected return of the policy
        for all state.

        Parameters
        ----------
        MonteCarlo : bool, optional
            True if one wants to use the MonteCarlo principle (Only possible
            option for now)

        Returns
        -------
        list : [J for all x, mean of J]
            List containing the J for all the states and the mean of the J over
            all the states.
        """
        if MonteCarlo is True:
            return self.__MonteCarlo()

    def __MonteCarlo(self):
        """
        Evaluate the policy by computing the expected return of the policy
        for all state using the Monte Carlo principle. Since the problem is
        deterministic, the number of episode should be one.

        Returns
        -------
        list : [J for all x, mean of J]
            List containing the J for all the states and the mean of the J over
            all the states.
        """
        # Grid containing all the J computed.
        returns = [[0 for i in range(self.__env.nb_s + 1)]
                   for j in range(self.__env.nb_p + 1)]
        # Creating a grid to estimate J
        for i in range(self.__env.nb_p + 1):
            for k in range(self.__env.nb_s + 1):
                for n in range(CST.NB_EPISODES_MONTE_CARLO):
                    # Initial state of the trajectory
                    p_init = self.__env.index_to_pos(i)
                    s_init = self.__env.index_to_speed(k)
                    init_state = State(p_init, s_init)
                    # Getting the corresponding trajectory
                    cur_traj = self.__createTrajectory(init_state)
                    length_traj = len(cur_traj)
                    # Computing J_x
                    r = cur_traj[length_traj - 1][2]
                    # gamma to the power nb_experience in traj * time step
                    # by second to obtain t.
                    j_x = pow(CST.DISCOUT_FACTOR, len(cur_traj)*CST.H) * r
                    returns[i][k] = j_x

        return returns, np.mean(returns)

    def fittedQ(self, algorithm="ERT"):
        """
        Evaluate the Q using the Q fitted iteration algorithm with the SL algo
        given in argument. The experiences used are generated following a mesh
        on the possible states. The number of episodes used is the one defined
        in Constants, ie the given in parameter during the run of the filename

        Parameters
        ----------
        algorithm : str {"LR", "ERT", "NN"}, optional
            The SL algorithm used to estimate Q

        Returns
        -------
        list[list[list[float]]]
            Q(x, u)
        """
        # Initialization
        last_Q = np.zeros((self.__env.nb_p, self.__env.nb_s, 2))
        sml = SML(algorithm)
        # Initialization of the learning set
        list_sample = []
        # Using a mesh to get all the learning samples needed
        for i in range(self.__env.nb_p + 1):
            for k in range(self.__env.nb_s + 1):
                # Initial state of the trajectory
                p = self.__env.index_to_pos(i)
                s = self.__env.index_to_speed(k)
                x = State(p, s)
                # Adding everything to the list
                list_sample.append(x)
        # Start of the Q fitted Iteration algo
        for n in range(CST.NB_EPISODES):
            # Fist iteration, the learning set is different
            if n == 0:
                # Using predefined (in Util.py) function to get the LS
                learning_set = getLSFirstIter(list_sample)
                # Predicting first Q
                last_Q = sml.getQ(learning_set)
            else:
                # Using predefined (in Util.py) function to get the LS
                learning_set = getLS(list_sample, last_Q)
                # Predicting next Q
                last_Q = sml.getQ(learning_set)
