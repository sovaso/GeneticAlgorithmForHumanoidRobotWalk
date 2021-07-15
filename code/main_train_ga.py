import gym
import numpy as np
import torch
import time
from gym.wrappers import Monitor
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torch.optim as optim
import math
import copy
import warnings
import pybulletgym
from genetic_algorithm import GeneticAlgorithm


if __name__ == "__main__":
    """ Main funtion for training the GA """
    
    env = gym.make('HumanoidPyBulletEnv-v0')
    ga = GeneticAlgorithm(number_of_agents = 4, 
                          number_of_generations = 3, 
                          number_of_iterations = 3, 
                          top_limit_agents = 2,  
                          mutation_chance = 0.1,
                          environment=env,
                          path_to_save_results = "../results")
    ga.run_genetic_algorithm()
