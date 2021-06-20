import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import torch.optim as optim
import math
import warnings


class AgentModel(nn.Module):
    """ Agent model used for genetic algorithm """

    def __init__(self, input_dim, output_dim):
        """ Initialize the agent model

        Args:
            input_dim (integer): dimension of input layer of nn (observation space number)
            output_dim (integer): dimension of output layer of nn (action space number)
        """

        super(AgentModel, self).__init__()

        self.first_layer = nn.Linear(input_dim, 128)
        self.second_layer = nn.Linear(128, 64)
        self.third_layer = nn.Linear(64, output_dim)

    def forward(self, x):
        """ Forward pass through agent neural network

        Args:
            x (tensor): current state

        Returns:
            (tensor): sample mean of the next action to perform
        """

        warnings.filterwarnings("ignore")
        x = torch.tensor(x, dtype=torch.float32)
        x = F.relu(self.first_layer(x))
        x = F.relu(self.second_layer(x))
        x = self.third_layer(x)
        return x


class Agent():
    """ Main class that is used as an agent in GA """

    def __init__(self, input_dim, output_dim):
        """ Initialize the agent class

        Args:
            input_dim (integer): dimension of input layer of nn (observation space number)
            output_dim (integer): dimension of output layer of nn (action space number)
        """

        self.model = AgentModel(input_dim, output_dim)
        self.covariance_matrix = torch.diag(input=torch.full(size=(output_dim,), fill_value=0.5), diagonal=0)

    def get_action(self, state):
        """ Getting action from current state for continuous space

        Args:
            state (tensor): current state

        Returns:
            (Tensor): next action to perform
            (Tensor): logaritmic probability of that action
        """

        mu = self.model.forward(state)
        multivariate_gaussian_distribution = MultivariateNormal(loc=mu, covariance_matrix=self.covariance_matrix)
        action = multivariate_gaussian_distribution.sample()
        log_probability = multivariate_gaussian_distribution.log_prob(value=action)
        return action, log_probability
