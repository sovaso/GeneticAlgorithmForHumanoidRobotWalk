import gym
import pybulletgym
import time
import torch
from actor import Actor
from critic import Critic
from trpo_agent import TRPOAgent


if __name__ == "__main__":
    """ Main function for training the TRPO Agent """

    env = gym.make('HumanoidPyBulletEnv-v0')
    #env.render()
    env.reset()
    actor = Actor(44, 17)
    critic = Critic(44, 1, 2.5e-4)
    trpo = TRPOAgent(actor=actor,
                     critic=critic,
                     delta_a2=1e-2,
                     delta_a1=1e-2,
                     delta_a0=3,
                     gamma=0.99,
                     cg_delta=1e-2,
                     cg_iterations = 10,
                     alpha=0.99,
                     backtrack_steps_num=100,
                     critic_epoch_num=20,
                     epochs=5000,
                     num_of_timesteps=4800,
                     max_timesteps_per_episode=1600,
                     starting_with=1700,
                     elementary_path="../results/eksperiment2")
    trpo.train(env=env)
