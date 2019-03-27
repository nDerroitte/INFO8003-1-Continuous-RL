# INFO8003-1 - Optimal decision making for complex problems
## Assignment 2 - Section 1 to 6
This file comments on how to use the code done in the second assignment. The first step is to install the packages needed for the code.
### Intsall
To create the video, `opencv` has been used. In the second part, `skitlearn` was used.
To install the related packages, one can just run the install bash script : 
```sh
$ ./install.sh
```
### Run the code
In order to run the code, one should simply use the following :
```sh
$ python3 run.py <options>
```
When called without option, the code runs the question 5 with the default value of the arguments. In order to change that, one can use the following parameters:
* `--Q` **{2, 3, 4}** : Select the question to run.
* `--t`  **int** : Size of the trajectory (>0).
* `--nb_episodes` **int** : Number of episodes used during the Q-learing algorithm (>0).
* `--discount_factor` **float** : Discount factor (gamma).
* `--time` **float** : Time of the simulation in seconds (>0).
* `--algo` **str** : The supervised machine learning algorithm used fo the Q fittediteration. Possible value : "LinearRegression" (or "LR"), "ExtremelyRandomizedTrees" (or "ERT"), NeuralNetworks, (or "NN") or "all" for all the algorithms previously quoted.
