import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import torch


#This code use reinforcement learnin. the environment is that of a pirate robots that is in search for lost treasuere.
# The agent can only see the square around it, and the agent can only see the square around it.
# The environment represents by a grid,where each cell of this grid represents a square whrere, by being
# finds the agent or elements of the environment, such as holes or treasuere.

# Assume, that robot always starts at the corner upper left and the trasure in the lower right corner.
# Furthemore, to generate different situations, the holes are randomly generated in the environment.
# that is, the random position of the holes is generated and this position is maintained throughout the
# entire period episode. If another episode is started, the holes are randomly generated again.

# The robot cannot go outside the grid or into a space occupied by a hole. If any of these situations occurs,
#, end the episode. The robot agent can perform four categorical actions: up, down, left, right. 
# When performing an action, the agent moves from according to its direction, only one position that is.

# The enviroment can be represented as a 7x7 matrix, Table 1, where the numerical value 1 represents an
# off-grid state, the value 2 represents a hole, the value 3 represents the treasure. If the agent is in
# a state with the values 1,2 and 3, the episode ends. The value 0 represents a free state, that is, the

#The REINFORCE algorithm must be implemented using neural networks to learn the policy for the pirate robot, executing
# actions to find the treasure with the minimum number of steps.

# - The results must be presented in terms of graph for average return
#of episodes vs. training epochs.
# - The results must be presented in terms of graph for the quantity
#of average steps executed by the robot in the episodes vs training epochs.
# - The execution of an episode must be presented at the end with the best
#obtained policy.


# Create the environment
# Create the agent
# Create the neural network
# Create the REINFORCE algorithm
# Train the agent
# Test the agent



# Simple pygame program

import torch

probs = torch.tensor([0.5, 0.5])

#Inicializa uma fun¸c~ao de distribui¸c~ao categorica
pd = torch.distributions.Categorical(logits=probs)

action = pd.sample()
#amostra uma categoria
print(action.item())

        








