import gym
import pybulletgym
import time
import torch
from actor import Actor
from critic import Critic
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torch.distributions import MultivariateNormal

Episode = namedtuple('Episode', ['states', 'actions', 'rewards', 'next_states', 'probabilities'])


class TRPOAgent():
    """ Main class that implements TRPO algorithm to improve actor and critic neural networks """

    def __init__(self, actor, critic, delta_a2, delta_a1, delta_a0, gamma, cg_delta, cg_iterations, 
                 alpha, backtrack_steps_num, critic_epoch_num, epochs, num_of_timesteps,
                 max_timesteps_per_episode, starting_with=0, elementary_path=""):
        """
        Initialize the parameters of TRPO class

        Args:
            actor (object): actor model for this problem that is used as a policy function
            critic (object): critic model for this problem that is used as a value state function
            delta_a2 (float): part of a number used as a KL divergence constraint between two distributions
            delta_a1 (float): part of a number used as a KL divergence constraint between two distributions
            delta_a0 (float): part of a number used as a KL divergence constraint between two distributions
            gamma (float): discount factor
            cg_delta (float): conjugate gradient constraint to tell us when to stop with the process
            cg_iterations (int): maximal number of iterations for conjugate gradient algortihm to perform
            alpha (float): factor to compute max step to update actor parameters in order to satisfy the KL 
                           divergence constraint (delta)
            backtrack_steps_num (int): number of steps to compute max step to update actor parameters
            critic_epoch_num (int): number of epoch to train critic neural network
            epochs (int): number of epochs for TRPO algorithm to train
            num_of_timesteps (int): number how many actions to take in one epoch
            max_timesteps_per_episode (int): maximal number of actions in one episode
            starting_with (int, optional): number that represents from which epoch we are starting the 
                                           training. Defaults to 0. 
            elementary_path(string, optional): path where actor and critic model with rewards will be saved.
                                               Defaults to "".
        """

        self.actor = actor
        self.critic = critic
        self.delta_a2 = delta_a2
        self.delta_a1 = delta_a1
        self.delta_a0 = delta_a0
        self.gamma = gamma
        self.cg_delta = cg_delta
        self.cg_iterations = cg_iterations
        self.alpha = alpha
        self.backtrack_steps_num = backtrack_steps_num
        self.critic_epoch_num = critic_epoch_num
        self.epochs = epochs
        self.num_of_timesteps = num_of_timesteps
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.starting_with = starting_with
        self.delta = 0
        self.elementary_path = elementary_path 

    def estimate_advantages(self, states, rewards):
        """ Estimating the advantage based on trajectories for one episode

        Args:
            states (tensor): states we visited during the episode
            rewards (tensor): collected rewards in that concrete episode
        
        Returns:
            (array): estimated advantage
        """

        # using critic nn to get state values
        values = self.critic.model.forward(states)
        # defining a variable to store rewards-to-go values
        rtg = torch.zeros_like(rewards)
        # setting last value on zero
        last_value = 0
        # calculating rewards-to-go
        for i in reversed(range(rewards.shape[0])):
            last_value = rtg[i] = rewards[i] + self.gamma * last_value
        # advantage = rewards-to-go - values
        return rtg - values

    def surrogate_loss(self, new_probs, old_probs, advantages):
        """ Calculating surrogate loss that is used for calculating policy network gradients.
            Formula: mean(e^(p1-p2)*adv), where p1 and p2 are log probabilities
        
        Args:
            new_probs (array): log probabilities of the new policy
            old_probs (array): log probabilities of the old policy
            advantages (array): estimated advantage of all episodes of current epoch
        
        Returns:
            (float): surrogate loss
        """
        return (torch.exp(new_probs - old_probs) * advantages).mean()

    def kl_divergence(self, mean1, std1, mean2, std2):
        """ Calculating the KL divergence between two distributions (old and new policy)
            Formula: log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / 
                     (2.0 * std1.pow(2)) - 0.5
        
        Args:
            mean1 (array): new means
            std1 (array): new standard deviations
            mean2 (array): old means
            std2 (array): old standard deviations
        
        Returns:
            (float): KL divergence between two distributions
        """

        mean2 = mean2.detach()
        std2 = std2.detach()
        kl_div = torch.log(std1)-torch.log(std2)+(std2.pow(2)+(mean1-mean2).pow(2))/(2.0*std1.pow(2))-0.5
        return kl_div.sum(1, keepdim=True).mean()

    def compute_grad(self, y, x, retain_graph=False, create_graph=False):
        """ Calculating the derivative of y with respect to x -> dx/dy
        
        Args:
            y (function): function
            x (array): parameter
            retain_graph (boolean): boolean value should we retain a graph
            create_graph (boolean): boolean value to define should we create a graph

        Returns:
            (function): derivation dy/dx
        """

        if create_graph:
            retain_graph = True
        grad = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
        grad = torch.cat([t.view(-1) for t in grad])
        return grad

    def conjugate_gradient(self, hvp_function, b):
        """ Calculate the H^1 * g using the conjugate gradient algorithm
        
        Args:
            hvp_function (function): hessian vector product function
            b (vector): vector that will be multiplied with inverse hessian matrix
        
        Returns:
            (function): multiplication of vector g and inverse hessian matrix H^-1
        """

        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        i = 0
        while i < self.cg_iterations:
            AVP = hvp_function(p)
            dot_old = r @ r
            alpha = dot_old / (p @ AVP)
            x = x + alpha * p
            r = r - alpha * AVP
            if r.norm() <= self.cg_delta:
                return x
            beta = (r @ r) / dot_old
            p = r + beta * p
            i += 1
        return x

    def get_advantage_estimation(self, episodes):
        """ Function to gather all estimated advantages of each episode of the epoch in one variable 
            and normalize it

        Args:
            episodes (array): episodes of the epoch
        
        Returns:
            (array): estimated normalized advantages
        """

        #collect all advantages
        advantages = [self.estimate_advantages(states, rewards) for states, _, rewards, _, _ in episodes]
        advantages = torch.cat(advantages, dim=0).flatten()
        #normalizing the advantages
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def get_probability(self, actions, states):
        """ Calculating logaritmic probability of actions based on states
        
        Args:
            actions (array): actions of the trajectories
            states (array): states of the trajectories
        
        Returns:
            (array): logarithmic probability
        """

        #mean of the distribution
        mu = self.actor.model.forward(states)
        multivariate_gaussian_distribution = MultivariateNormal(loc=mu, covariance_matrix=self.actor.covariance_matrix)
        logarithmic_probability = multivariate_gaussian_distribution.log_prob(value=actions)
        return logarithmic_probability

    def calculate_general_reward_for_agent(self, rewards):
        """ Calculate reward for the agent that will represent fitness score for GA

        Args:
            rewards (array): array of rewards per each epoch that the agent was trained

        Returns:
            (float): cumulative reward as a fitness function for GA
        """

        sum = 0
        gamma_coef = 1
        for i in reversed(range(len(rewards))):
            if i == 0:
                rewards[i] = 0
            if rewards[i] > 0:
                sum += rewards[i] * gamma_coef
            else:
                sum += rewards[i] * (2.0 - gamma_coef)
            gamma_coef *= self.gamma
        return sum

    def update_agent(self, episodes):
        """ Function that update both critic and actor neural network's parameters
        
        Args:
            episodes (array): episodes in the epoch
        """

        # PART 1: get states and actions provided through parameter episodes
        states = torch.cat([r.states for r in episodes], dim=0)
        actions = torch.cat([r.actions for r in episodes], dim=0)

        # PART 2: calculate advantages based on trajectories and normalize it
        advantages = self.get_advantage_estimation(episodes).detach()

        # PART 3: update critic parameters based on advantage estimation
        for iter in range(self.critic_epoch_num):
            self.critic.update_critic(self.get_advantage_estimation(episodes))

        # PART 4: get distribution of the policy and define surrogate loss and kl divergence
        probability = self.get_probability(actions, states)
        mean, std = self.actor.get_mean_std(states)
        L = self.surrogate_loss(probability, probability.detach(), advantages)
        KL = self.kl_divergence(mean, std, mean, std)

        # PART 5: compute gradient for surrogate loss and kl divergence
        parameters = list(self.actor.model.parameters())
        g = self.compute_grad(L, parameters, retain_graph=True)
        d_kl = self.compute_grad(KL, parameters, create_graph=True)
        #print('Gradient -> ', g)

        # PART 6: define hessian vector product function, compute search direction and max_length to get max step
        def HVP(v):
            return self.compute_grad(d_kl @ v, parameters, retain_graph=True)

        search_dir = self.conjugate_gradient(HVP, g)
        max_length = torch.sqrt(2 * self.delta / (search_dir @ HVP(search_dir)))
        max_step = max_length * search_dir

        # PART 7: check if max step satisfy the constraint, if not make it smaller
        def criterion(step):
            #print('Step ->', step)
            self.actor.update_parameters(step)
            with torch.no_grad():
                mean_new, std_new = self.actor.get_mean_std(states)
                probability_new = self.get_probability(actions, states)
                L_new = self.surrogate_loss(probability_new, probability, advantages)
                KL_new = self.kl_divergence(mean_new, std_new, mean, std)
            L_improvement = L_new - L
            #print('Distribution difference ->', KL_new)
            #print('Loss improvement ->', L_new)
            if L_improvement > 0 and KL_new <= self.delta:
                return True
            self.actor.update_parameters(-step)
            return False

        i = 0
        while not criterion((self.alpha ** i) * max_step) and i < self.backtrack_steps_num:
            i += 1

    def train(self, env, render_frequency=None, return_only_rewards=False):
        """ Main function used for training the TRPO agent

        Args:
            env (object): environment to run your agent on
            render_frequency (int, optional): speed of rendering the environment, expressed in ms, 
                                              if None then there is no rendering. Defaults to None.
            return_only_rewards (bool, optional): if true, function will return the cumulative reward 
                                                  for the whole training process. Defaults to False.

        Returns:
            (float): cumulative reward
        """

        mean_total_rewards = []
        global_episode = 0

        if self.starting_with > 0:
            #if we want to continue  our training, we need to load previous state and results
            self.actor.model.load_state_dict(torch.load(self.elementary_path + '/actor' + str(self.starting_with) + '.pt'))
            self.critic.model.load_state_dict(torch.load(self.elementary_path + '/critic' + str(self.starting_with) + '.pt'))
            rewards_txt = open(self.elementary_path + '/rewards' + str(self.starting_with) + '.txt', "r")
            rewards_str = rewards_txt.read()
            rewards_str = rewards_str[1:][:-1]
            mean_total_rewards = [eval(rwd) for rwd in rewards_str.split(",")]

        for epoch in range(self.epochs):

            #calculate delta based on formula 1/(a2*epoch^2+a1*epoch+a0)
            iteration = epoch+1+self.starting_with
            self.delta = 1.0 / (self.delta_a2 * (iteration+1)**2 + self.delta_a1 * (iteration+1) + self.delta_a0)

            episodes, episode_total_rewards = [], []
            curr_number_of_timesteps, num_of_episodes = 0, 0
            while curr_number_of_timesteps < self.num_of_timesteps:

                num_of_steps = 0
                state = env.reset()
                done = False
                samples = []
                curr_episode_steps = 0
                while not done and curr_episode_steps < self.max_timesteps_per_episode:

                    #rendering the environment
                    if render_frequency is not None and global_episode % render_frequency == 0:
                        env.render()
                    #getting action and probability of it based on current state
                    action, probability = self.actor.get_action(state)
                    curr_action = action.numpy()
                    #running current action in the environment
                    next_state, reward, done, _ = env.step(curr_action)
                    num_of_steps += 1
                    samples.append((state, action, reward, next_state, probability))
                    state = next_state
                    curr_episode_steps += 1

                num_of_episodes += 1
                curr_number_of_timesteps += curr_episode_steps
                states, actions, rewards, next_states, probabilities = zip(*samples)
                states = torch.stack([torch.from_numpy(state) for state in states], dim=0).float()
                next_states = torch.stack([torch.from_numpy(state) for state in next_states], dim=0).float()
                actions = torch.stack([torch.tensor(action) for action in actions], dim=0).float()
                rewards = torch.as_tensor(rewards).unsqueeze(1)
                probabilities = torch.tensor(probabilities, requires_grad=True).unsqueeze(1)
                episodes.append(Episode(states, actions, rewards, next_states, probabilities))
                episode_total_rewards.append(rewards.sum().item())
                global_episode += 1

            #updating the agent
            self.update_agent(episodes)
            mtr = np.mean(episode_total_rewards)
            mean_total_rewards.append(mtr)
            #printing the statistics of current epoch
            print(f'E: {epoch+1+self.starting_with}.\tMean total reward across {num_of_episodes} episodes and {curr_number_of_timesteps} timesteps: {mtr}')

            #every 50 epoch, save all mean rewards and model for actor & critic
            if epoch % 100 == 4:
                torch.save(self.actor.model.state_dict(),
                           self.elementary_path + '/actor' + str(epoch + self.starting_with + 1) + '.pt')
                torch.save(self.critic.model.state_dict(),
                           self.elementary_path + '/critic' + str(epoch + self.starting_with + 1) + '.pt')
                with open(self.elementary_path + '/rewards' + str(epoch + self.starting_with + 1) + '.txt', 'w+') as fp:
                    fp.write(str(mean_total_rewards))
        print("")
        if return_only_rewards:
            return self.calculate_general_reward_for_agent(mean_total_rewards)
