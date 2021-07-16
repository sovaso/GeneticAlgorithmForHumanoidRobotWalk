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
    ga = GeneticAlgorithm(number_of_agents = 15, 
                          number_of_generations = 20, 
                          number_of_iterations = 40, 
                          top_limit_agents = 1,  
                          mutation_chance = 0.1,
                          environment=env,
                          path_to_save_results = "../results")
    ga.run_genetic_algorithm()
