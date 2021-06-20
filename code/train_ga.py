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
    # main funtion for training
    environment = gym.make('HumanoidPyBulletEnv-v0')
    #hyper-parameter, set from https://arxiv.org/pdf/1712.06567.pdf
    ga = GeneticAlgorithm(number_of_agents = 20, 
                          number_of_generations = 1000, 
                          number_of_episodes = 3, 
                          top_limit_agents = 3, 
                          environment = environment, 
                          mutation_power = 0.00224)
    ga.run_genetic_algorithm()
