import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import warnings


class CriticModel(nn.Module):
    """ Critic model used for state value function approximation """

    def __init__(self, input_dim, output_dim):
        """ Initialize the critic model

        Args:
            input_dim (int): dimension of input layer of nn (observation space number)
            output_dim (int): dimension of output layer of nn (one -> value of the state)
        """

        super(CriticModel, self).__init__()

        self.first_layer = nn.Linear(input_dim, 128)
        self.second_layer = nn.Linear(128, 64)
        self.third_layer = nn.Linear(64, output_dim)

    def forward(self, x):
        """ Forward pass through critic neural network
        
        Args:
            x (tensor): state
        
        Returns:
            (float): value of the state
        """

        warnings.filterwarnings("ignore")
        x = torch.tensor(x, dtype=torch.float32)
        x = F.relu(self.first_layer(x))
        x = F.relu(self.second_layer(x))
        x = self.third_layer(x)
        return x


class Critic():
    """ Main class that is used as Critic in TRPO algorithm """

    def __init__(self, input_dim, output_dim, learning_rate):
        """ Initialize the critic class

        Args:
            input_dim (int): dimension of input layer of nn (observation space number)
            output_dim (int): dimension of output layer of nn (one -> value of the state)
            learning_rate (float): learning rate used for critic model neural netwoek
        """

        self.model = CriticModel(input_dim, output_dim)
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)


    def update_critic(self, advantages):
        """ Gradient descent on estimation of advantages
        
        Args:
            advantages (array): estimation of advantages
        """

        loss = (advantages ** 2).mean()  # MSE
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
