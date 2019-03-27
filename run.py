from argparse import ArgumentParser, ArgumentTypeError
from constants import CST
from agent import *
import time
from util import *

if __name__ == "__main__":
    start = time.time()
    usage = """
    USAGE:      python3 run.py <options>
    EXAMPLES:   (1) python run.py
                    - Launch the Q4 of the project : Visualisation of a policy
    """

    # Using argparse to select the different setting for the run
    parser = ArgumentParser(usage)

    # Q : Allow to select the question the user want to run.
    parser.add_argument(
        '--Q',
        help='Selection the question of the project one wants to run',
        type=int,
        default=5
    )

    # t: Size of the trajectory
    parser.add_argument(
        '--t',
        help='Size of the history one wants to consider',
        type=int,
        default=10000
    )

    # nb_episodes
    parser.add_argument(
        '--nb_episodes',
        help='Number of episodes used during the Q-learing algorithm',
        type=int,
        default=100
    )

    # discount_factor : gamma parameter
    parser.add_argument(
        '--discount_factor',
        help='Discount factor (gamma)',
        type=float,
        default=0.99
    )

    # time
    parser.add_argument(
        '--time',
        help="""Number of seconds one wants the video.
                This number is for 25 fps! (See report)""",
        type=int,
        default=30
    )

    # Algo
    parser.add_argument(
        '--algo',
        help=""" The supervised machine learning algorithm used fo the Q fitted
                 iteration. Possible value : "LinearRegression" (or "LR"),
                 "ExtremelyRandomizedTrees" (or "ERT"), NeuralNetworks
                 (or "NN") or "all" for all the algorithms previously
                 quoted.""",
        type=str,
        default="ERT"
    )
    # Getting the arguments and setting the constants.
    args = parser.parse_args()
    if args.algo == "LR" or args.algo == "LinearRegression":
        CST.ALGORITHM = "LR"
    elif args.algo == "ERT" or args.algo == "ExtremelyRandomizedTrees":
        CST.ALGORITHM = "ERT"
    elif args.algo == "NN" or args.algo == "NeuralNetworks":
        CST.ALGORITHM = "NN"
    elif args.algo == "all":
        CST.ALGORITHM = "all"
    else:
        print("Unknown supervised learning algorithm. To see the available "
              "ones, please use python3 run.py --help.\nUsing \"ERT\" as "
              "default")
        CST.ALGORITHM = "ERT"

    question_number = args.Q
    CST.TIME_VISUALISATION = args.time
    CST.NB_EPISODES = args.nb_episodes
    CST.LENGTH_EPISODE = args.t
    CST.DISCOUT_FACTOR = args.discount_factor

    if question_number == 2 or question_number == 4:
        # Create Agent
        a = Agent(-0.99, 0.0)
        # Create Policy
        a.createSimplePolicy(CST.DIR_LEFT)
        # Visualise it
        a.visualisePolicy("Q{}".format(question_number))

    if question_number == 3:
        # Create Agent
        a = Agent()
        # Create simple POLICY
        a.createSimplePolicy(direction=CST.DIR_LEFT)
        # Computing the expected return
        J_est, J_mean = a.evaluatePolicy(MonteCarlo=True)
        # Ploting the result. Uncomment to get the plot
        # plot3D(J_est)
        print("""The mean of the expected return is : {}.""".format(J_mean))
    if question_number == 5:
        a = Agent(start_p=-0.5)
        # We want to check all algo
        if CST.ALGORITHM == "all":
            # 1 : Linear Regression
            print("1) Using RL algoritm:")
            # Getting Q
            estimated_q = a.fittedQ("RL")
            # Updating the policy
            policy = a.updatePolicy(estimated_q)
            # Assessing the results
            J_est, J_mean = a.evaluatePolicy(MonteCarlo=True)
            print("The mean of the expected return is : {}.".format(J_mean))
            # 2 : Extremely Randomized Trees
            print("2) Using ERT algoritm:")
            # Getting Q
            estimated_q = a.fittedQ("ERT")
            # Updating the policy
            policy = a.updatePolicy(estimated_q)
            # Assessing the results
            J_est, J_mean = a.evaluatePolicy(MonteCarlo=True)
            # 3 : NeuralNetworks
            print("3) Using Neural Networks:")
            # Getting Q
            estimated_q = a.fittedQ("NN")
            # Updating the policy
            policy = a.updatePolicy(estimated_q)
            # Assessing the results
            J_est, J_mean = a.evaluatePolicy(MonteCarlo=True)
        else:
            # Algo specific
            # Getting Q
            estimated_q = a.fittedQ(CST.ALGORITHM)
            # Updating the policy
            policy = a.updatePolicy(estimated_q)
            # Assessing the results
            J_est, J_mean = a.evaluatePolicy(MonteCarlo=True)
            print("The mean of the expected return is : {}.".format(J_mean))
            # Creating video
            a.visualisePolicy("Q{}_{}".format(question_number, CST.ALGORITHM))
    if question_number == 6:
        a = Agent(start_p=-0.5)
        # Using Parametric Q
        Q = a.ParamQLearning()
        # Updating policy
        a.updatePolicy(Q, skitlearnModel=False)
        # Assessing the results
        J_est, J_mean = a.evaluatePolicy(MonteCarlo=True)
        print("The mean of the expected return is : {}.".format(J_mean))
        a.visualisePolicy("Q{}".format(question_number))

    print("--------- Comp. time : {} ---------".format(time.time() - start))
