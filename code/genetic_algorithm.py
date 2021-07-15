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
from trpo_agent import TRPOAgent
from actor import Actor
from critic import Critic
import os
import json


class UnitResultUtil:
    """ Util class to help us keed information about one unit in the current generation """

    def __init__(self, delta_a2, delta_a1, delta_a0, reward):
        """ Constructor for the unit result util class

        Args:
            delta_a2 (float): part of a delta function
            delta_a1 (float): part of a delta function
            delta_a0 (float): part of a delta function
            reward (float): fitness score the agent with these parameters achieved
        """
        self.delta_a2 = delta_a2
        self.delta_a1 = delta_a1
        self.delta_a0 = delta_a0
        self.reward = reward


class GenerationResultUtil:
    """ Util class to help us keep all information about units in current generation """

    def __init__(self, units):
        """ Constructor for the generation result util class

        Args:
            units (array): all units in current generation
        """
        self.units = units

    def to_json(self):
        """ Converting the object to a string that is in json format and can be saved as json file

        Returns:
            (string): string as json format of the object
        """

        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)


class GeneticAlgorithm():
    """ Main class for the whole Genetic Algorithm process """

    def __init__(self, number_of_agents, number_of_generations, number_of_iterations, top_limit_agents, 
                 mutation_chance, environment, path_to_save_results = "../results"):
        """ Constructor for the Genetic Algorithm class 
        
        Args:
            number_of_agents (integer): number of population per each generation
            number_of_generations (integer): number of generations to run ga
            number_of_iterations (integer): number of epochs to train each agent for fitness function
            top_limit_agents (integer): number of how many agents from current generation to pick to
                                        keep in next generation
            mutation_chance (float): chance that crossovered agent will be mutated
            environment (object): environment where the game will be played
            path_to_save_results (string): path where the results will be saved
        """

        self.number_of_agents = number_of_agents
        self.number_of_generations = number_of_generations
        self.number_of_iterations = number_of_iterations
        self.top_limit_agents = top_limit_agents
        self.mutation_chance = mutation_chance
        self.path_to_save_results = path_to_save_results
        self.environment = environment

    def create_initial_population(self):
        """ Creating the first generation of the population by instancing new agents

        Returns:
            (array): population of agents
        """

        #possible starter values for agents to pick for parameters we want to optimize
        a2_options = [1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4,5e-5,1e-5,5e-6]
        a1_options = [1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4,5e-5,1e-5,5e-6]
        a0_options = [1.0 ,2.0 ,3.0 ,4.0 ,5.0 ,6.0 ,7.0 ,8.0 ,9.0 ,10.0]

        agents = []
        #for each expected agent number we create one agent
        for iter in range(self.number_of_agents):
            agent = TRPOAgent(actor=Actor(44, 17),
                              critic=Critic(44, 1, 2.5e-4),
                              delta_a2=a2_options[iter],
                              delta_a1=a1_options[iter],
                              delta_a0=a0_options[iter],
                              gamma=0.99,
                              cg_delta=1e-2,
                              cg_iterations = 10,
                              alpha=0.99,
                              backtrack_steps_num=100,
                              critic_epoch_num=20,
                              epochs=self.number_of_iterations,
                              num_of_timesteps=4800,
                              max_timesteps_per_episode=1600)
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
            cumulative_reward = agent.train(env=self.environment, return_only_rewards=True)
            reward_agents.append(cumulative_reward)
        return reward_agents

    def calculate_fitness_for_one_agent(self, agent):
        """ Calculating fitness function for one specific agent by training it for N iterations and
            then calculating cumulative reward it gets during that training

        Args:
            agent (object): one agent from the population

        Returns:
            (float): fitness score for that agent
        """

        return self.run_population([agent])[0]

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
        """ Mutating agent in a range +/-10percent for its parameters value, with some chance

        Args:
            agent (object): one agent from the population

        Returns:
            (object): mutated agent for a new generation of the population
        """

        c = random.uniform(0, 1)
        if c > self.mutation_chance:
            return agent
        child_agent = copy.deepcopy(agent)
        c2 = random.uniform(0, 1)
        child_agent.delta_a2 += 0.1 * (2*c2 - 1.0) * child_agent.delta_a2
        child_agent.delta_a1 += 0.1 * (2*c2 - 1.0) * child_agent.delta_a1
        child_agent.delta_a0 += 0.1 * (2*c2 - 1.0) * child_agent.delta_a0
        return child_agent

    def crossover_agents(self, first_agent, second_agent):
        """ Creating two children by taking a random uniform number C between 0 and 1 and summing
            both parent parameters by the formula: param1*C+param2(1-C) and other replacing param1
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
        first_child.delta_a2 = first_agent.delta_a2*c + (1-c)*second_agent.delta_a2
        first_child.delta_a1 = first_agent.delta_a1*c + (1-c)*second_agent.delta_a1
        first_child.delta_a0 = first_agent.delta_a0*c + (1-c)*second_agent.delta_a0
        second_child.delta_a2 = second_agent.delta_a2*c + (1-c)*first_agent.delta_a2
        second_child.delta_a1 = second_agent.delta_a1*c + (1-c)*first_agent.delta_a1
        second_child.delta_a0 = second_agent.delta_a0*c + (1-c)*first_agent.delta_a0
        return first_child, second_child

    def create_next_generation(self, agents, sorted_parent_indexes):
        """ Based on fitness score of current generation, take best agents from current generation
            and the rest of needed population create using crossover with the mutation

        Args:
            agents (array): current generation of the population
            sorted_parent_indexes (array): indices of the agents with the best fitness score

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
        #take best agents into the next generation
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

    def save_generation_results(self, agents, rewards, number_of_generation):
        """ Save information about the current generation by saving parameters and rewards of each unit
            current generation

        Args:
            agents (array): current generation
            rewards (array): fitness score of current generation
            number_of_generation (int): index of current generation
        """
        agent_results = []
        for iter in range(len(agents)):
            agent_results.append(UnitResultUtil(agents[iter].delta_a2,
                                                agents[iter].delta_a1,
                                                agents[iter].delta_a0,
                                                rewards[iter]))
        with open(self.path_to_save_results + "/generation_" + str(number_of_generation) + ".json", 'w') as f:
            json.dump(GenerationResultUtil(agent_results).to_json(), f)

    def run_genetic_algorithm(self):
        """ Main funtion to run the genetic algorithm """

        agents = self.create_initial_population()
        for generation in range(self.number_of_generations):
            #return rewards of agents
            rewards = self.calculate_fitness_for_all_agents(agents)
            #sort by rewards
            sorted_parent_indexes = np.argsort(rewards)[::-1] 
            top_rewards = []
            for best_parent in sorted_parent_indexes[:self.top_limit_agents]:
                top_rewards.append(rewards[best_parent])
            print("Generation ", generation, " | Mean rewards: ", np.mean(rewards), " | Mean of top 5: ",
                np.mean(top_rewards[:5]))
            print("Top ", self.top_limit_agents, " scores", sorted_parent_indexes[:self.top_limit_agents])
            print("Rewards for top: ", top_rewards)
            print("")
            print("")
            #save informations about the generation
            self.save_generation_results(agents, rewards, generation)
            #setup an empty list for containing children agents
            children_agents = self.create_next_generation(agents, sorted_parent_indexes)
            #kill all agents, and replace them with their children
            agents = children_agents
            #reset each actor and critic of TRPO agent
            for agent in agents:
                agent.actor=Actor(44, 17)
                agent.critic=Critic(44, 1, 2.5e-4)
