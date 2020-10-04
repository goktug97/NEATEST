#!/usr/bin/env python3
from typing import List, Union, cast
import random

import gym #type: ignore
import numpy as np

import neatest

Array = Union[np.ndarray, np.generic]

class Agent(neatest.Agent):
    def __init__(self):
        self.env = gym.make('CartPole-v1')

    def rollout(self, genome: neatest.Genome, render=False) -> float:
        total_reward: float = 0.0
        for i in range(10):
            observation: Array = self.env.reset()
            done: bool = False
            while not done:
                output: np.int  = np.argmax(genome(cast(List[float], observation)))
                observation, reward, done, info = self.env.step(output)
                if render:
                    self.env.render()
                total_reward += reward
        return total_reward / 10

agent = Agent()

SEED = 123
agent.env.seed(SEED)
agent.env.action_space.seed(SEED)

a = neatest.NEATEST(
    agent,
    neatest.Adam,
    n_networks = 128,
    es_population = 256,
    input_size = agent.env.observation_space.shape[0],
    output_size = agent.env.action_space.n,
    bias = True,
    dominant_gene_rate = 0.5,
    dominant_gene_delta = 0.05,
    elite_rate = 0.05,
    sigma = 0.01,
    save_checkpoint_n = 50,
    optimizer_kwargs = {'lr': 0.01},
    disable_connection_mutation_rate = 0.3,
    seed = SEED,
    logdir = './logs/test',
    hidden_activation=neatest.relu,
    node_mutation_rate = 0.3,
    connection_mutation_rate = 0.3)

a.train(3)

print(a.best_genome)
a.best_genome.save('LunarLander.genome')
print(agent.rollout(a.best_genome, render=True))

agent.env.close()
