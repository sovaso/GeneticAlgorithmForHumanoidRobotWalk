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
    ga = GeneticAlgorithm(number_of_agents = 100, 
                          number_of_generations = 50, 
                          number_of_episodes = 3, 
                          top_limit_agents = 6, 
                          environment = environment, 
                          mutation_power = 0.02,
                          mutation_chance = 0.25,
                          path_to_save_results = "../results")
    ga.run_genetic_algorithm()
