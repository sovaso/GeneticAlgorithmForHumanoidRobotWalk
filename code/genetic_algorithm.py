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
from agent import Agent


class GeneticAlgorithm():
    """ Main class for the whole Genetic Algorithm process """

    def __init__(self, number_of_agents, number_of_generations, number_of_episodes, top_limit_agents, 
                 environment, mutation_power):
        """ Constructor for the Genetic Algorithm class 
        
        Args:
            number_of_agents (integer): number of population per each generation
            number_of_generations (integer): number of generations to run ga
            number_of_episodes (integer): number of episodes to run each agent for fitness function
            top_limit_agents (integer): number of how many agents to pick as possible parents for next
                                        generation
            environment (object): environment that agent will perform action and observe the space of it
            mutation_power (float): coefficient used for mutation of agent's neural network's parameters
        """

        self.number_of_agents = number_of_agents
        self.number_of_generations = number_of_generations
        self.number_of_episodes = number_of_episodes
        self.top_limit_agents = top_limit_agents
        self.environment = environment
        self.mutation_power = mutation_power

    def initialize_weights(self, agent_model):
        """ Initialize weight for agents neural network using He initialization (because nn's uses relu
        activation function) while biases initialize with zeros

        Args:
            agent_model (object): neural network of the agent
        """

        if ((type(agent_model) == nn.Linear) | (type(agent_model) == nn.Conv2d)):
            #he instead of xavier weight initialization is used because of relu activation
            torch.nn.init.kaiming_uniform(agent_model.weight, nonlinearity="relu")
            #put biases to zeros
            agent_model.bias.data.fill_(0.00)

    def create_initial_population(self):
        """ Creating the first generation of the population by instancing new agents

        Returns:
            (array): population of agents
        """

        agents = []
        #for each expected agent number create one agent
        for _ in range(self.number_of_agents):
            #input dimension is the size of expected state tensor, and output dimension is
            #the size of expected action tensor
            agent = Agent(44,17)
            #we do not need to calculate gradients for this algorithm
            for param in agent.model.parameters():
                param.requires_grad = False
            self.initialize_weights(agent.model)
            agents.append(agent)
        return agents

    def run_population(self, agents):
        """ Run current generation of the population to calculate their fitness function

        Args:
            agents (array): current generation of the population

        Returns:
            (array): rewards of fitness function for each agent
        """

        reward_agents = []
        for agent in agents:
            agent.model.eval()
            #reset your environment
            current_state = self.environment.reset()
            cumulative_reward = 0
            #1600 is maximal number of steps per one episode
            for _ in range(1600):
                #take action based on current state
                action, _ = agent.get_action(current_state)
                current_action = action.numpy()
                # running current action in the environment
                next_state, reward, done, _ = self.environment.step(current_action)
                cumulative_reward = cumulative_reward + reward
                current_state = next_state
                if done:
                    break
            reward_agents.append(cumulative_reward)
        return reward_agents

    def calculate_fitness_for_one_agent(self, agent):
        """ Calculating fitness function for one specific agent by running him into the environment
        n times and taking mean value from all runnings

        Args:
            agent (object): one agent from the population

        Returns:
            (float): fitness score for that agent
        """

        fitness_score = 0.
        for _ in range(self.number_of_episodes):
            fitness_score += self.run_population([agent])[0]
        return fitness_score/self.number_of_episodes

    def calculate_fitness_for_all_agents(self, agents):
        """ Calculating fitness function for every agent from the current generation of the
        population

        Args:
            agents (array): current generation of the population

        Returns:
            (array): fitness score for every agent
        """

        fitness_scores = []
        for agent in agents:
            fitness_scores.append(self.calculate_fitness_for_one_agent(agent))
        return fitness_scores

    def mutate_agent(self, agent):
        """ Mutating agent using equation θ = θ + σ * e, where θ are the agent's nn's parameters,
        σ is mutation coefficient (hyperparameter) and e is a number from normal distribution

        Args:
            agent (object): one agent from the population

        Returns:
            (object): mutated agent for a new generation of the population
        """

        child_agent = copy.deepcopy(agent)
        for param in child_agent.model.parameters():
            #weights of Conv2D
            if (len(param.shape) == 4):  
                for i0 in range(param.shape[0]):
                    for i1 in range(param.shape[1]):
                        for i2 in range(param.shape[2]):
                            for i3 in range(param.shape[3]):
                                param[i0][i1][i2][i3] += self.mutation_power * np.random.randn()
            #weights of linear layer
            elif (len(param.shape) == 2):  
                for i0 in range(param.shape[0]):
                    for i1 in range(param.shape[1]):
                        param[i0][i1] += self.mutation_power * np.random.randn()
            #biases of linear layer or conv layer
            elif (len(param.shape) == 1):  
                for i0 in range(param.shape[0]):
                    param[i0] += self.mutation_power * np.random.randn()
        return child_agent

    def create_next_generation(self, agents, sorted_parent_indexes, elite_index):
        """ Based on fitness score of current generation, create new generation by mutating
        N-1 agents and adding one elite agent in order to form the next generation

        Args:
            agents (array): current generation of the population
            sorted_parent_indexes (array): indices of the agents with the best fitness score
            elite_index (integer): index where elite agent can be found in current generation

        Returns:
            (array): the next generation of the population
            (integer): index of elite agent in new generation
        """

        children_agents = []
        #first take selected parents from sorted_parent_indexes and generate N-1 children
        for i in range(len(agents) - 1):
            selected_agent_index = sorted_parent_indexes[np.random.randint(len(sorted_parent_indexes))]
            children_agents.append(self.mutate_agent(agents[selected_agent_index]))
        #now add one elite
        elite_child = self.add_elite(agents, sorted_parent_indexes, elite_index)
        children_agents.append(elite_child)
        #it will be stored as the last one
        elite_index = len(children_agents) - 1
        return children_agents, elite_index

    def add_elite(self, agents, sorted_parent_indexes, elite_index=None, only_consider_top_n=10):
        """ Creating one elite agent by considering only 10 n agents that performed with their
        fitness score, by taking another run to calculate fitness score and choosing the best
        among them to select the elite agent for the next generation.

        Args:
            agents (array): current generation of the population
            sorted_parent_indexes (array): indices of the agents with the best fitness score
            elite_index (integer, optional): index of the current elite agent. Defaults to None.
            only_consider_top_n (int, optional): number to define how many agents will be 
                                                 considered for the next elite agent. Defaults 
                                                 to 10.

        Returns:
            (object): new elite agent
        """

        #taking only top n agents
        candidate_elite_index = sorted_parent_indexes[:only_consider_top_n]
        if (elite_index is not None):
            candidate_elite_index = np.append(candidate_elite_index, [elite_index])
        top_score = None
        top_elite_index = None
        #calculating fitness score for each elite candidate agent once again
        for i in candidate_elite_index:
            score = self.calculate_fitness_for_one_agent(agents[i])
            print("Score for elite i ", i, " is ", score)
            if (top_score is None):
                top_score = score
                top_elite_index = i
            elif (score > top_score):
                top_score = score
                top_elite_index = i
        print("Elite selected with index ", top_elite_index, " and score", top_score)
        child_agent = copy.deepcopy(agents[top_elite_index])
        return child_agent

    def run_genetic_algorithm(self):
        """ Main funtion to run the genetic algorithm """

        torch.set_grad_enabled(False)
        agents = self.create_initial_population()
        elite_index = None
        for generation in range(self.number_of_generations):
            # return rewards of agents
            rewards = self.calculate_fitness_for_all_agents(agents)
            # sort by rewards
            sorted_parent_indexes = np.argsort(rewards)[::-1][:self.top_limit_agents]  
            print("")
            print("")
            top_rewards = []
            for best_parent in sorted_parent_indexes:
                top_rewards.append(rewards[best_parent])
            print("Generation ", generation, " | Mean rewards: ", np.mean(rewards), " | Mean of top 5: ",
                np.mean(top_rewards[:5]))
            print("Top ", self.top_limit_agents, " scores", sorted_parent_indexes)
            print("Rewards for top: ", top_rewards)
            # setup an empty list for containing children agents
            children_agents, elite_index = self.create_next_generation(agents, sorted_parent_indexes, 
                                                                       elite_index)
            # kill all agents, and replace them with their children
            agents = children_agents
