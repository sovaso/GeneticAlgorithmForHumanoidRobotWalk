import gym
import numpy as np
import torch
import time
import random
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
import os
import json


class GeneticAlgorithm():
    """ Main class for the whole Genetic Algorithm process """

    def __init__(self, number_of_agents, number_of_generations, number_of_episodes, top_limit_agents, 
                 environment, mutation_power, mutation_chance, path_to_save_results):
        """ Constructor for the Genetic Algorithm class 
        
        Args:
            number_of_agents (integer): number of population per each generation
            number_of_generations (integer): number of generations to run ga
            number_of_episodes (integer): number of episodes to run each agent for fitness function
            top_limit_agents (integer): number of how many agents to pick as possible parents for next
                                        generation
            environment (object): environment that agent will perform action and observe the space of it
            mutation_power (float): coefficient used for mutation of agent's neural network's parameters
            mutation_chance (float): number between 0 and 1 to present the chance of agent's parameters 
                                     being mutated
            path_to_save_results (string): path where the results will be saved
        """

        self.number_of_agents = number_of_agents
        self.number_of_generations = number_of_generations
        self.number_of_episodes = number_of_episodes
        self.top_limit_agents = top_limit_agents
        self.environment = environment
        self.mutation_power = mutation_power
        self.mutation_chance = mutation_chance
        self.path_to_save_results = path_to_save_results

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

        c = random.uniform(0, 1)
        if c > self.mutation_chance:
            return agent
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

    def crossover_agents(self, first_agent, second_agent):
        """ Creating two children by taking a random uniform number C between 0 and 1 and summing
        both parent nns parameters by the formula: param1*C+param2(1-C) and other replacing param1
        and param2 in the formula.

        Args:
            first_agent (object): first parent
            second_agent (object): second parent

        Returns:
            (object, object): 2 children for next generation
        """

        c = random.uniform(0, 1)
        first_child = copy.deepcopy(first_agent)
        second_child = copy.deepcopy(second_agent)
        for (param_p1, param_p2, param_c1, param_c2) in zip(first_agent.model.parameters(),
                           second_agent.model.parameters(), first_child.model.parameters(),
                           second_child.model.parameters()):
            #weights of Conv2D
            if (len(param_p1.shape) == 4):  
                for i0 in range(param_p1.shape[0]):
                    for i1 in range(param_p1.shape[1]):
                        for i2 in range(param_p1.shape[2]):
                            for i3 in range(param_p1.shape[3]):
                                param_c1[i0][i1][i2][i3] += c*param_p1[i0][i1][i2][i3]+ \
                                                            (1-c)*param_p2[i0][i1][i2][i3]
                                param_c2[i0][i1][i2][i3] += c*param_p2[i0][i1][i2][i3]+ \
                                                            (1-c)*param_p1[i0][i1][i2][i3]
            #weights of linear layer
            elif (len(param_p1.shape) == 2):  
                for i0 in range(param_p1.shape[0]):
                    for i1 in range(param_p1.shape[1]):
                        param_c1[i0][i1] += c*param_p1[i0][i1]+(1-c)*param_p2[i0][i1]
                        param_c2[i0][i1] += c*param_p2[i0][i1]+(1-c)*param_p1[i0][i1]
            #biases of linear layer or conv layer
            elif (len(param_p1.shape) == 1):  
                for i0 in range(param_p1.shape[0]):
                    param_c1[i0] += c*param_p1[i0]+(1-c)*param_p2[i0]
                    param_c2[i0] += c*param_p2[i0]+(1-c)*param_p1[i0]
        return first_child, second_child

    def create_next_generation(self, agents, sorted_parent_indexes):
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
        for i in range((len(agents) - self.top_limit_agents) // 2):
            selected_agent_index_1, selected_agent_index_2 = self.selection_of_parents(
                                                                  sorted_parent_indexes)
            child_agent_1, child_agent_2 = self.crossover_agents(agents[selected_agent_index_1],
                                                                 agents[selected_agent_index_2])
            children_agents.append(self.mutate_agent(child_agent_1))
            children_agents.append(self.mutate_agent(child_agent_2))
        #TODO
        for index in sorted_parent_indexes[:self.top_limit_agents]:
            children_agents.append(self.mutate_agent(agents[index]))
        return children_agents

    def selection_of_parents(self, sorted_parent_indexes):
        """ Selecting possible parents for the crossover to create new generation and selecting them
        based on fitness function (reward from episodes) using softmax to give everyone a chance

        Args:
            sorted_parent_indexes (array): indexes of parents sorted by fitness function

        Returns:
            (int,int): indexes of two selected parents
        """

        selection_util_array = np.zeros(self.number_of_agents)
        for iter in range(self.number_of_agents):
            selection_util_array[iter] = np.exp(-1 * (iter+1) * random.uniform(0, 1))
        selection_util_array /= np.sum(selection_util_array)
        parent_indexes = np.random.choice(sorted_parent_indexes, 2, p=selection_util_array)
        return parent_indexes[0], parent_indexes[1]

    def save_generation_results(self, agents, all_mean_rewards, all_mean_top_rewards, 
                                all_top_indexes, all_top_rewards):
        result_json_string = "{\"all_mean_rewards\":\"" + str(all_mean_rewards) + "\"," + \
                             "\"all_mean_top_rewards\":\"" + str(all_mean_top_rewards) + "\"," + \
                             "\"all_top_indexes\":\"" + str(all_top_indexes) + "\"," + \
                             "\"all_top_rewards\":\"" + str(all_top_rewards) + "\"," + \
                             "\"mutation_power\":\"" + str(mutation_power) + "\"," + \
                             "\"mutation_chance\":\"" + str(mutation_chance) + "\"}"

        index = 1
        while os.path.isfile(self.path_to_save_results + "/experiment_" + str(index) + ".json"):
            index += 1
        with open(self.path_to_save_results + "/experiment_" + str(index) + ".json", 'w') as f:
            json.dump(result_json_string, f)


    def run_genetic_algorithm(self):
        """ Main funtion to run the genetic algorithm """

        #parameters to save about the generation
        all_mean_rewards = []
        all_mean_top_rewards = []
        all_top_rewards = []
        all_top_indexes = []
        torch.set_grad_enabled(False)
        agents = self.create_initial_population()
        elite_index = None
        for generation in range(self.number_of_generations):
            # return rewards of agents
            rewards = self.calculate_fitness_for_all_agents(agents)
            # sort by rewards
            sorted_parent_indexes = np.argsort(rewards)[::-1] 
            print("")
            print("")
            top_rewards = []
            for best_parent in sorted_parent_indexes[:self.top_limit_agents]:
                top_rewards.append(rewards[best_parent])
            print("Generation ", generation, " | Mean rewards: ", np.mean(rewards), " | Mean of top 5: ",
                np.mean(top_rewards[:5]))
            print("Top ", self.top_limit_agents, " scores", sorted_parent_indexes[:self.top_limit_agents])
            print("Rewards for top: ", top_rewards)
            # setup an empty list for containing children agents
            children_agents = self.create_next_generation(agents, sorted_parent_indexes)
            # kill all agents, and replace them with their children
            agents = children_agents
            # save important results in lists
            all_mean_rewards.append(np.mean(rewards))
            all_mean_top_rewards.append(np.mean(top_rewards[:self.top_limit_agents]))
            all_top_rewards.append(sorted_parent_indexes)
            all_top_indexes.append(top_rewards)
        self.save_generation_results(agents, all_mean_rewards, all_mean_top_rewards, all_top_indexes, 
                                     all_top_rewards)
