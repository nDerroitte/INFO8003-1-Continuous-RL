#! /usr/bin/env python
# -*- coding: utf-8 -*-

import random

# Import packages
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor
from sklearn import neighbors
import matplotlib.pyplot as plt
import glob
import imageio

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# DOWNLOAD OF A PACKAGE FOR THE VIDEO
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# imageio.plugins.ffmpeg.download()
from moviepy.editor import *
from displaycaronthehill import save_caronthehill_image

# ===========================================
#          GLOBAL VARIABLES
# ===========================================

p_BOUND = 1  # Maximal Position (in absolute value)
s_BOUND = 3  # Maximal Speed (in absolute value)

ACTION_BOUND = 4

MASS = 1
GRAVITY = 9.81

T = 0.1  # (in seconds)
DURATION = 50  # (in seconds)
INTEGRATION_STEP = 0.001
N = 40  # Number of steps to evaluate Q (ideally infinite!)
NB_DEC = 3  # Can't be lower, otherwise we may "loop in the same tuple"

# For discretization
POSITION_STEP = 0.05
SPEED_STEP = 0.1

GAMMA = 0.95  # Discount factor

# Representation of the state through a tuple
INDEX_p = 0
INDEX_s = 1

INDEX_STATE = 0
INDEX_ACTION = 1
INDEX_REWARD = 2
INDEX_NEXTSTATE = 3

START_p = 0
START_s = 0

NB_EPISODES = 40
SIZE_EPISODE = 5000

IMAGE_DURATION = 0.3
fps = 10  # Number of Frames per seconds

DEFAULT_ALGORITHM = "ExtraTreesRegressor"
NB_ADD_SAMPLES = 30  # For the Q-fitted algorithm

ALPHA = 0.05  # Learning algorithm  (Q6)


# ===========================================
#          Creation of the domain
# ===========================================
class Domain:
    def __init__(self):
        self.start_state = (START_p, START_s)
        self.possible_actions = [-ACTION_BOUND, ACTION_BOUND]

    # --------------------- is_terminal_state --------------------------------
    # This function returns true if the given state in a terminal state
    # ------------------------------------------------------------------------
    def is_terminal_state(self, state):
        return p_BOUND <= abs(state[INDEX_p]) or s_BOUND <= abs(state[INDEX_s])

    # --------------------- derivation_hill --------------------------------
    # This function returns the first derivation of the hill function
    # ----------------------------------------------------------------------
    def derivation_hill(self, p):
        if p < 0:
            return 2 * p + 1
        else:
            return 1 / pow(1 + 5 * p ** 2, 3 / 2)

    # ----------------- second_derivation_hill -----------------------------
    # This function returns the second derivation of the hill function
    # ----------------------------------------------------------------------
    def second_derivation_hill(self, p):
        if p < 0:
            return 2
        else:
            return -15 * p / pow(1 + 5 * p ** 2, 5 / 2)

    # --------------------------- s_prime ------------------------------
    # This function returns the current acceleration corresponding to
    # the speed of the next time
    # -----------------------------------------------------------------
    def s_prime(self, p, s, u):

        hill_prime = self.derivation_hill(p)
        hill_prime_prime = self.second_derivation_hill(p)

        first_term = u / (MASS * (1 + hill_prime ** 2))
        second_term = -GRAVITY * hill_prime / (1 + hill_prime ** 2)
        third_term = - (s ** 2) * hill_prime * hill_prime_prime / (1 + hill_prime ** 2)

        return first_term + second_term + third_term

    # --------------------- get_next_state --------------------------------
    # IN: state: current state
    #     u: action to perform from this current state
    #     round_values = True if positions and speed must be rounded
    # OUT: resulting next state (next_pos, next_speed)
    # ----------------------------------------------------------------------
    def get_next_state(self, state, u, round_values=False):

        next_speed = self.s_prime(state[INDEX_p], state[INDEX_s], u)

        # Application of the Euler Equation to discretize
        next_pos = state[INDEX_p] + INTEGRATION_STEP * state[INDEX_s]
        next_speed = state[INDEX_s] + INTEGRATION_STEP * next_speed

        if round_values:
            next_pos = round(next_pos, NB_DEC)
            next_speed = round(next_speed, NB_DEC)

        return next_pos, next_speed

    # ---------------------- get_reward --------------------------------
    # IN: state: (p,s): representing the current state
    #     u: action to perform from the given state
    #        If None: the state directly corresponds to the next state
    # OUT: the associated reward
    # ------------------------------------------------------------------
    def get_reward(self, state, u=None):

        if u is None:
            next_state = state
        else:
            # Get the resulting next state
            next_state = self.get_next_state(state, u)

        # Find associate reward
        if next_state[INDEX_p] < - 1 or 3 < abs(next_state[INDEX_s]):
            return -1
        elif 1 < next_state[INDEX_p] and abs(next_state[INDEX_s]) <= 3:
            return 1
        else:
            return 0

    # ---------------------------- build_episodes -------------------------------------
    # This method returns episodes where each episode is a set of 4-tuples
    # like (x,u,r,x_next)
    # IN: is_trajectory: if True: each episode corresponds to a trajectory
    #                    Otherwise: all the tuples are totally independent from each other
    # ------------------------------------------------------------------------------------

    def build_episodes(self, is_trajectory=False, nb_episodes=NB_EPISODES):
        i_episode = 0
        episodes = []

        while i_episode < nb_episodes:
            episodes.append([])
            i_4tuple = 0

            while i_4tuple < SIZE_EPISODE and (len(episodes[i_episode]) == 0 or
                        (is_trajectory and not self.is_terminal_state(next_state)) or not is_trajectory):
                episodes[i_episode].append([])

                # Be sure to have the start state (optional)
                if i_4tuple == 0:
                    state = self.start_state

                # Ensure to have a trajectory
                elif is_trajectory:
                    state = next_state

                # Pick randomly a state
                else:
                    bound_p = p_BOUND * pow(10, NB_DEC)
                    bound_s = s_BOUND * pow(10, NB_DEC)
                    pos = random.randint(-bound_p, bound_p) / pow(10, NB_DEC)
                    speed = random.randint(-bound_s, bound_s) / pow(10, NB_DEC)
                    state = (pos, speed)

                    # The current state can't be a terminal one (because no next corresponding state)
                    while domain.is_terminal_state(state):
                        pos = random.randint(-bound_p, bound_p) / pow(10, NB_DEC)
                        speed = random.randint(-bound_s, bound_s) / pow(10, NB_DEC)
                        state = (pos, speed)

                episodes[i_episode][i_4tuple].append(state)

                # Pick randomly an action
                act = random.choice(self.possible_actions)
                episodes[i_episode][i_4tuple].append(act)

                # Compute the corresponding reward
                reward = self.get_reward(state, act)
                episodes[i_episode][i_4tuple].append(reward)

                # Compute the corresponding next state
                next_state = self.get_next_state(state, act, round_values=True)
                episodes[i_episode][i_4tuple].append(next_state)

                i_4tuple += 1
            i_episode += 1

        return episodes

    # ---------------------- build_data_sets -------------------------------
    # This method builds a dataset having the form:
    # [[[(x1,u1),Q1], [(x2,u2),Q2] ...][ [(x1,u1),Q1], [(x2,u2),Q2] ...]...]
    # OUT: dataset (from one-step transitions)
    #      episode: containing the remaining tuple which have not been used
    # REM: the episode will be used to have information about transitions
    # ---------------------------------------------------------------------

    def build_data_sets(self, nb_steps=N):

        # Build episodes and take only the first
        episode = self.build_episodes(nb_episodes=1)[0]

        # Initialization of the dataset
        tmp_ds, state_target = get_success(episode)
        datasets = [tmp_ds]
        all_considered_states = state_target
        datasets[-1].extend(get_poor_states(all_considered_states))

        j = 1

        while j <= nb_steps:
            # Get the direct possible preceding states
            tmp_ds, state_target = get_success(episode, step=j, state_list=state_target)
            all_considered_states.extend(
                x for x in state_target if x[INDEX_STATE] not in all_considered_states[INDEX_STATE])
            tmp_ds = [x for x in tmp_ds if x[INDEX_STATE] not in all_considered_states[INDEX_STATE]]

            # ------ Create current dataset ------------
            # Add the preceding relevant data
            prec_dataset = datasets[-1]
            datasets.append([x for x in prec_dataset if x[1] != 0])

            # Add the new data
            datasets[-1].extend(tmp_ds)

            # Provide information about the non-visited states
            datasets[-1].extend(get_poor_states(all_considered_states))

            j += 1

        return datasets, episode

# =====================================================
#           Creation of the POLICY
# =====================================================

class Policy:
    def __init__(self, domain):
        self.domain = domain
        self.nb_interval_positions = int((2 * p_BOUND) / POSITION_STEP)
        self.nb_interval_speed = int((2 * s_BOUND) / SPEED_STEP)
        self.policy = np.ones((self.nb_interval_positions, self.nb_interval_speed)) * ACTION_BOUND

    # ---------------------- get_policy -------------------------------
    # This method defines a policy
    # IN: action: action which have to be performed whatever the state
    #             If None: a random action is picked for each state
    # -----------------------------------------------------------------

    def get_policy(self, action=None):

        for i in range(self.nb_interval_positions):
            for j in range(self.nb_interval_speed):
                if action is None:
                    self.policy[i][j] = random.choice(domain.possible_actions)
                else:
                    self.policy[i][j] = action

    # ---------------------- get_action -------------------------------
    # This method returns the action to perform for a given state
    # IN: state: (p,s)
    # -----------------------------------------------------------------
    def get_action(self, state):

        index_pos, index_speed = self.get_indexes(state)
        return self.policy[index_pos][index_speed]

    # ---------------------- get_indexes ------------------------------
    # This method returns the indexes for a given state
    # IN: state: (p,s)
    # -----------------------------------------------------------------
    def get_indexes(self, state):

        # If terminal state, no meaning, default return
        if self.domain.is_terminal_state(state):
            return self.nb_interval_positions-1, self.nb_interval_speed-1

        # Find the index of the line from the position
        up_pos = -p_BOUND + POSITION_STEP
        index_pos = 0
        while up_pos < state[INDEX_p]:  # round(up_pos,1)
            up_pos += POSITION_STEP
            index_pos += 1

        # Find the index of the column from the speed
        up_speed = -s_BOUND + SPEED_STEP
        index_speed = 0
        while up_speed < state[INDEX_s]:
            up_speed += SPEED_STEP
            index_speed += 1

        return index_pos, index_speed

    # ---------------------- get_state -------------------------------
    # This method returns the state corresponding to the given indexes
    # -----------------------------------------------------------------
    def get_state(self, index_pos, index_speed):

        # Find the index of the line from the position
        pos = -p_BOUND + POSITION_STEP / 2
        curr_pos_index = 0

        while curr_pos_index < index_pos:
            pos += POSITION_STEP
            curr_pos_index += 1

        # Find the index of the column from the speed
        speed = -s_BOUND + SPEED_STEP / 2
        curr_speed_index = 0

        while curr_speed_index < index_speed:
            speed += SPEED_STEP
            curr_speed_index += 1

        return (pos, speed)

    # ------------- get_learnt_policy -------------------------
    # This function turns the evaluated Q function
    # into a policy.
    # IN: 2 possible forms: estimator_qn: tuple of estimators
    #                       qn: tuple of "hashlist"
    #
    # REM: the first element of the tuple is for negative action
    #      and the second one for positive action
    # ----------------------------------------------------------
    def get_learnt_policy(self, estimator_qn=None, qn=None):

        for i in range(self.nb_interval_positions):
            for j in range(self.nb_interval_speed):
                state = self.get_state(i, j)

                if qn is None:
                    Q_neg = estimator_qn[0].predict([np.array(state)])[0]
                    Q_pos = estimator_qn[1].predict([np.array(state)])[0]
                else:
                    index_pos, index_speed = self.get_indexes(state)
                    Q_neg = qn[0][index_pos][index_speed]
                    Q_pos = qn[1][index_pos][index_speed]

                if Q_neg < Q_pos:
                    self.policy[i][j] = ACTION_BOUND
                else:
                    self.policy[i][j] = -ACTION_BOUND

    # ---------------------- evaluate_policy ------------------------------
    # This method evaluates the policy, using the Monte Carlo Method
    # IN: episodes: list of episodes, each of them being a list of 4-tuples
    # -------------------------------------------)-------------------------
    def evaluate_policy(self, episodes, nb_steps=N):
        evaluated_return = 0
        for i in range(NB_EPISODES):
            reward = 0
            state = self.domain.start_state
            for j in range(nb_steps):
                action = self.get_action(state)
                closest_tuple = find_closest_4tuple(episodes[i], state, action)
                reward += closest_tuple[INDEX_REWARD]
                state = closest_tuple[INDEX_NEXTSTATE]
            evaluated_return += reward

        return evaluated_return / NB_EPISODES

    # ----------------- get_states_sequence -----------------------
    # IN: nb_steps: the number of states transitions to represent
    # OUT: list of the consecutive states, following (with respect
    #      to the policy
    # ------------------------------------------------------------

    def get_states_sequence(self, nb_steps=DURATION / T):

        states = []
        curr_state = self.domain.start_state
        i = 0

        while i < nb_steps and not self.domain.is_terminal_state(curr_state):
            states.append(curr_state)

            # Representation of the direction of the car
            curr_action = self.get_action(curr_state)

            # Evaluation of the new state, according to the policy
            curr_state = self.domain.get_next_state(curr_state, curr_action, round_values=True)

            i += 1

        return states

    # ---------------------------- get_Q_learning --------------------------------------
    # IN: episodes: list of trajectories: [[x0, u0, r0, x1],[x1...]...], [[x0...]...]]
    # where:  - xi is the current position
    #         - ui si the action to perform
    #         - ri is the reward which has been got
    #         - x_next_i (= x_i+1) is the next pos which has been reached
    #           [pos1][speed1] = 0
    #
    # OUT: Tuple of the estimated Q for each of the 2 actions
    # ----------------------------------------------------------------------------------
    def get_Q_learning(self, episodes):

        Q_neg = []  # Q-function for action -4
        Q_pos = []  # Q-function for action 4

        # Initialization of the Q
        for i in range(self.nb_interval_positions):
            Q_neg.append([])
            Q_pos.append([])
            for j in range(self.nb_interval_speed):
                Q_neg[i].append(0)
                Q_pos[i].append(0)

        for i, episode in enumerate(episodes):
            final_reward = episode[-1][INDEX_REWARD]  # To give the sign of GAMMA

            # Trial by using an episode
            for j, tuple in enumerate(episode):
                index_pos, index_speed = self.get_indexes(tuple[INDEX_STATE])

                # Assignment of the evaluated Q
                Q_est = pow(final_reward * GAMMA, len(episode) - 1 - j)

                if tuple[INDEX_ACTION] == -ACTION_BOUND:
                    Q_neg[index_pos][index_speed] = (1 - ALPHA) * Q_neg[index_pos][index_speed] + ALPHA * Q_est
                else:
                    Q_pos[index_pos][index_speed] = (1 - ALPHA) * Q_pos[index_pos][index_speed] + ALPHA * Q_est

        return Q_neg, Q_pos

    # ----------------- get_representation -----------------------
    # IN: nb_steps: the number of states transitions to represent
    # ------------------------------------------------------------

    def get_representation(self, nb_steps=DURATION / T):
        print("======================= START ==============================")

        curr_state = self.domain.start_state
        i = 0

        while i < nb_steps and not self.domain.is_terminal_state(curr_state):

            # Representation of the direction of the car
            position = str(round(curr_state[INDEX_p], NB_DEC))
            curr_action = self.get_action(curr_state)

            if divmod(i, 15)[1] == 0:  # To avoid really printing All the positions
                print("Position " + position, end='')
                for k in range(7 - len(position)):
                    print(" ", end='')
                print(":   ", end='')
                if curr_state[INDEX_s] == 0:
                    print(".", end='')
                else:
                    if curr_state[INDEX_s] < 0:
                        print("<", end='')

                    # Representation of the speed
                    for j in range(abs(int(10 * curr_state[INDEX_s]))):
                        print("-", end='')

                    # Representation of the direction of the car
                    if 0 < curr_state[INDEX_s]:
                        print(">", end='')

                # Print the current action which is going to be performed
                print("   NEXT ACTION: " + str(curr_action))

            # Evaluation of the new state, according to the policy
            curr_state = self.domain.get_next_state(curr_state, curr_action)
            i += 1

        reward = self.domain.get_reward(curr_state)
        print("==================== END with reward " + str(reward) + " =============================")


# ================================================================
#         IMPLEMENTATION OF THE MACHINE LEARNING PART
# ================================================================


# --------------------------- split_data_set ----------------------------------
# IN: dataset: List of data ((state, action), Q)
#     action: the action the data must be related to
# OUT: dataset_action: List lists of data ((state, action), Q) for the given
#                      specific action
# REM: each list in dataset_action is for a specific N
# -----------------------------------------------------------------------------
def split_data_set(data_set, action):
    dataset_action = []

    for i in range(len(data_set)):
        dataset_action.append([])

        for j in range(len(data_set[i])):
            if data_set[i][j][0][1] == action:  # "0" to take the input; "1" to take the action
                dataset_action[i].append(data_set[i][j])

    return dataset_action


# --------------------- formatting_data --------------------------------
# IN: dataset: (((s1,p1),u1),Q1),((s2,p2),u2),Q2), ...,((sN,pN),uN),Q_N))
#     p = proportion which is used for learning
#
# OUT: Formatted learning and testing sets
# ----------------------------------------------------------------------

def formatting_data(dataset, p=0.75):
    testing_set = []
    learning_set = []

    for i in range(len(dataset)):

        if i / len(dataset) < p:
            learning_set.append(dataset[i])
        else:
            testing_set.append(dataset[i])

    # ------- Formatting the learning and the testing sets ------------

    x_ls = np.ones((len(learning_set), 2), dtype=np.float)
    y_ls = np.zeros((x_ls.shape[0],), dtype=np.float)

    x_ts = np.ones((len(testing_set), 2), dtype=np.float)
    y_ts = np.zeros((x_ts.shape[0],), dtype=np.float)

    for k, data in enumerate(learning_set):
        x_ls[k, INDEX_p] = learning_set[k][0][INDEX_STATE][INDEX_p]
        x_ls[k, INDEX_s] = learning_set[k][0][INDEX_STATE][INDEX_s]
        y_ls[k] = learning_set[k][1]

    for k, data in enumerate(testing_set):
        x_ts[k, INDEX_p] = testing_set[k][0][INDEX_STATE][INDEX_p]
        x_ts[k, INDEX_s] = testing_set[k][0][INDEX_STATE][INDEX_s]
        y_ts[k] = testing_set[k][1]

    return [x_ls, y_ls, x_ts, y_ts]


# --------------------- build_predict --------------------------------
# IN: dataset: formatted dataset [x_ls, y_ls, x_ts, y_ts]
# OUT: The average of the relative error on the prediction
# --------------------------------------------------------------------

def build_predict(dataset, algorithm):
    index_xls = 0
    index_yls = 1

    # ------------ Management of the learning algorithm ----------------------
    if algorithm == "BaggingRegressor":
        return BaggingRegressor(n_estimators=10, max_samples=1.0, bootstrap=True,
                                n_jobs=1, random_state=7).fit(dataset[index_xls], dataset[index_yls])

    elif algorithm == "ExtraTreesRegressor":
        return ExtraTreesRegressor(n_estimators=10, n_jobs=1, random_state=7).fit(dataset[index_xls],
                                                                                  dataset[index_yls])

    elif algorithm == "KNeighborsRegressor":
        if len(dataset[index_yls]) < 10:
            n_neighbors = len(dataset[index_yls])
        else:
            n_neighbors = 10
        return neighbors.KNeighborsRegressor(n_neighbors).fit(dataset[index_xls], dataset[index_yls])

    else:
        return LinearRegression(fit_intercept=True, normalize=False, n_jobs=1).fit(dataset[index_xls],
                                                                                   dataset[index_yls])


# --------------------- additional_sample --------------------------------
# IN: estimator: to allow to access the evaluated Q for a given state
#     action: action corresponding to the given estimator
#     episode: list of tuples providing the transition information:
#              (x_t,u_t) => x_t+1
#     nb_samples: nb of additional samples to produce
# --------------------------------------------------------------------

def additional_sample(estimator, action, episode, nb_samples=NB_ADD_SAMPLES):
    samples = []

    if len(episode) < nb_samples:
        nb_samples = len(episode)

    for i in range(nb_samples):
        # Compute a preceding possible state (by taking the given action)
        state = episode[i][INDEX_NEXTSTATE]
        prec_state = episode[i][INDEX_STATE]

        # Compute the evaluated Q
        Q = estimator.predict([np.array(state)])

        samples.append([])
        samples[-1].append((prec_state, action))  # INPUT
        samples[-1].append(GAMMA * Q[0])  # OUTPUT

    return samples


# ===============================================
#          FUNCTIONS for evaluating Q
# ===============================================


# -------------------- Q_fitted_algorithm ----------------------------------
# This function applies the Q fitted algorithm by evaluation each Q_i
# and using them to add information in the training set.
# --------------------------------------------------------------------------
def Q_fitted_algorithm(algorithm=DEFAULT_ALGORITHM, action=ACTION_BOUND):
    global estimatorQ

    # Generation of 2 datasets which are not empty
    datasets, episode = domain.build_data_sets()
    datasets_action = split_data_set(datasets, action)

    # At each iteration, we evaluate Q_i and use it to improve the training set
    for i in range(len(datasets)):
        # Formatting of the data and estimation of Q
        ds_action = formatting_data(datasets_action[i])

        # Build estimator
        estimatorQ = build_predict(ds_action, algorithm)

        # Use the predicted Q_i to add new data in the training sample
        if i < len(datasets) - 1:
            datasets_action[i + 1] += additional_sample(estimatorQ, -ACTION_BOUND, episode)

    return estimatorQ


# ===============================================
#               TOOLS FUNCTIONS
# ===============================================

# --------------------- find_closest_4tuple ----------------------------
# IN: episode: list of 4-tuples (x,u,r,x_next), where x = (p,s)
#     state: tuple made up of a position x' and a speed s'
#     action: an action (-4 or 4)
#
# OUT: The closest tuple in episode which is such that:
#       - the actions u and "action" must be the same if possible
#       - |p-p'| + |s-s'| must be minimum
# ----------------------------------------------------------------------
def find_closest_4tuple(episode, state, action):
    best_tuple = episode[0]
    min_dif = float("inf")
    filtered_episode = [tuple for tuple in episode if tuple[INDEX_ACTION] == action]

    for i in range(len(filtered_episode)):
        x = filtered_episode[i][INDEX_STATE][INDEX_p]
        s = filtered_episode[i][INDEX_STATE][INDEX_s]
        dif = abs(state[INDEX_p] - x) + abs(state[INDEX_s] - s)

        # Update the closest tuple if the difference lower than the current one
        if dif < min_dif:
            best_tuple = episode[i]

    return best_tuple


# --------------------------------- get_success -------------------------------------
# IN: episode: list of 4-tuples (x,u,r,x_next), where x = (p,s)
#     state_list: list of states [x0,x1,x2...]
#                 If none: the condition is "having a reward of 1"
#                 Otherwise: the condition is "leading to a state in the list"
#
# OUT: - List of data ((state, action), Q)
#      - List of the states and the corresponding FINAL reward (1 or -1)
# ----------------------------------------------------------------------------------
def get_success(episode, step=None, state_list=None):
    success_list = []
    target_state = []
    i = 0

    while i < len(episode):
        # START CASE
        if state_list is None and abs(episode[i][INDEX_REWARD]) == 1:
            origin_reward = episode[i][INDEX_REWARD]
            Q = origin_reward

        # ITERATIVE CASE
        elif state_list is not None and is_closed_from(episode[i][INDEX_NEXTSTATE], state_list):
            origin_reward = is_closed_from(episode[i][INDEX_NEXTSTATE], state_list)
            Q = pow(origin_reward * GAMMA, step)

        # THE CURRENT TUPLE CAN'T BE USED
        else:
            i += 1
            continue

        # BUILD THE TUPLE
        target_state.append(((episode[i][INDEX_STATE], origin_reward)))
        success_list.append([])
        success_list[-1].append((episode[i][INDEX_STATE], episode[i][INDEX_ACTION]))

        # Assign a value for Q_step
        success_list[-1].append(Q)
        del episode[i]

    return success_list, target_state


# ------------------------------------- get_poor_states ------------------------------------
# IN: list of states whose corresponding Q function is not 0 (with the corresponding reward)
# OUT: - List like [[(x0,u0), 0], [(x1,u1), 0], ..., [(xN,uN), 0]]
# -------------------------------------------------------------------------------------------
def get_poor_states(all_considered_states):
    dataset0 = []

    for pos in np.arange(-p_BOUND, p_BOUND, POSITION_STEP * 5):
        for speed in np.arange(-s_BOUND, s_BOUND, SPEED_STEP * 5):
            if is_closed_from((pos, speed), all_considered_states) is None:
                dataset0.append([((pos, speed), -ACTION_BOUND), 0])
                dataset0.append([((pos, speed), ACTION_BOUND), 0])

    return dataset0


# --------------------- is_closed_from (to discretize) ----------------------------------
# IN: state: (p, s)
#     state_list: list of states and corresponding final [(x0,1),(x1, -1),(x2, -1) ...]
#
# OUT: True if it's possible close enough to one of the states belong to the list
#      from the single input state
# ----------------------------------------------------------------------------------------
def is_closed_from(state, states):
    origin_reward = None
    k = 0
    while k < len(states):
        if (POSITION_STEP < abs(state[INDEX_p] - states[k][INDEX_STATE][INDEX_p])) or (SPEED_STEP < abs(
                    state[INDEX_s] - states[k][INDEX_STATE][INDEX_s])):
            k += 1
        else:
            origin_reward = states[k][1]  # Collect the corresponding reward
            k += 1

    return origin_reward


# -------------------- display_QN ----------------------------------
# This function displays a graph allowing to visualize Q_N estimated
# By different learning algorithm, for specific states
# ------------------------------------------------------------------
def display_QN(action=ACTION_BOUND):
    # Create the list of algorithm to use

    print("COMPARISON OF THE LEARNING ALGORITHMS ... ")
    algos = ["LinearRegression", "BaggingRegressor", "ExtraTreesRegressor"]

    # Create the list of state we want to evaluate QN (4*4 states)
    states = []
    positions = [-0.9, -0.5, 0, 0.7, 0.9]
    speeds = [-2, 0, 2]

    for i, pos in enumerate(positions):
        for j, speed in enumerate(speeds):
            states.append((pos, speed))

    QN = []

    # Each iteration uses one specific algorithm to evaluate Q for the picked states
    for i, algo in enumerate(algos):
        print("Evaluation using algorithm " + algo + "...")
        QN.append({key: [] for key in states})
        estim = Q_fitted_algorithm(algorithm=algo, action=action)

        for j, state in enumerate(states):
            QN[i][state] = estim.predict([np.array(state)])[0]

    # Visualisation through a bar chart
    bar_chart(QN[0], QN[1], QN[2], "Q_n approximations", first_title=algos[0], second_title=algos[1],
              third_title=algos[2], y_label="Q_N")


# ===========================================
#          VISUALIZATION TOOL
# ===========================================
# -------------- bar_chart --------------------
# IN: dictionary1 and dictionary2 have the same
#     keys and 1 value in a list
# ---------------------------------------------

def bar_chart(dict1, dict2, dict3, title, first_title='', second_title='', third_title='', y_label=''):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(len(dict1))  # 2 bars to consider
    width = 0.2

    first_vals = []
    second_vals = []
    third_vals = []

    for key in dict1.keys():
        first_vals.append(dict1[key])
        second_vals.append(dict2[key])
        third_vals.append(dict3[key])

    rects1 = ax.bar(ind, first_vals, width, color='#FF8C00')
    rects2 = ax.bar(ind + width, second_vals, width, color='#255D79')
    rects3 = ax.bar(ind + 2 * width, third_vals, width, color='#009999')

    ax.set_ylabel(y_label)
    ax.set_xticks(range(len(dict1)))
    ax.set_xticklabels(dict1.keys())
    ax.legend((rects1[0], rects2[0], rects3[0]), (first_title, second_title, third_title))

    plt.title(title)
    print("DISPLAY OF THE FIGURE (delete it to carry on)")
    plt.show()



# ---------------------- visualize ------------------------
# IN: states: Sequence of states which has to be visualized
# OUT: mp4 file representing the displacement of the car
# ---------------------------------------------------------
def visualize(filename, states=None):
    # Default sequence just for testing
    if states is None:
        print("lol")
        states = [(-1, 2.4), (-0.9, -0.4), (-0.8, -0.4), (-0.7, -0.4), (-0.6, -0.4), (-0.5, 0), (-0.4, 0.4), (-0.3, 0.4),
                  (-0.2, 0.4), (-0.1, 0.4)]

    # Creation of the images to visualize the change in states
    for i, state in enumerate(states):
        save_caronthehill_image(state[0], state[1], str(i) + ".jpeg")

    file_list = glob.glob('*.jpeg')  # Get all the pngs in the current directory

    clips = [ImageClip(m).set_duration(IMAGE_DURATION)
             for m in file_list]

    concat_clip = concatenate_videoclips(clips, method="compose")
    concat_clip.write_videofile(filename + "car_displacement.mp4", fps=30)

    # Suppression of the tmp images which have been generated
    #for i in range(len(states)):
        #os.remove(str(i) + ".jpeg")


# ===============================================
#         MAIN FUNCTION TO TEST
# ===============================================
if __name__ == "__main__":
    # Initialization of the domain
    domain = Domain()
    # Build the set of episodes the evaluation of the policy will be based on
    episodes = domain.build_episodes()

    num_question = 4

    if num_question == 2 or num_question is None:
        print("================= QUESTION 2 - Domain ============================= ")
        policy_q2 = Policy(domain)
        policy_q2.get_policy()
        policy_q2.get_representation()

    if num_question == 3 or num_question is None:
        print("================= QUESTION 3 - Expected return ===================== ")
        policy_q3 = Policy(domain)
        policy_q3.get_policy(action=ACTION_BOUND)
        print("The expected return is: " + str(policy_q3.evaluate_policy(episodes)))

    if num_question == 4 or num_question is None:
        print("================= QUESTION 4 - Visualization ======================= ")
        visualize("question4_")  # Applied on a default sequence of states

    if num_question == 5 or num_question is None:
        print("================= QUESTION 5 - Fitted-Q-Iteration ================== ")
        display_QN()  # Comparison of the algorithms

        # Evaluation of an optimal policy
        print("Estimation of the optimal policy ...")
        est_neg = Q_fitted_algorithm(action=-ACTION_BOUND)
        est_pos = Q_fitted_algorithm(action=ACTION_BOUND)

        policy_q5 = Policy(domain)
        policy_q5.get_learnt_policy(estimator_qn=(est_neg, est_pos))
        print("The expected return is: " + str(policy_q5.evaluate_policy(episodes)))

        print("Building the video to visualize the corresponding behaviour...")
        states = policy_q5.get_states_sequence()
        visualize(filename="question5_", states=states)

    if num_question == 6 or num_question is None:
        print("================= QUESTION 6 - Parametric Q-Learning================ ")
        # Build trajectories
        trajectories = domain.build_episodes(is_trajectory=True)

        policy_q6 = Policy(domain)
        qn = policy_q6.get_Q_learning(episodes)
        policy_q6.get_learnt_policy(qn=qn)
        print("The expected return is: " + str(policy_q6.evaluate_policy(episodes)))

        print("Building the video to visualize the corresponding behaviour...")
        states = policy_q6.get_states_sequence()
        visualize(filename="question6_", states=states)

    if num_question == 7:
        pass
