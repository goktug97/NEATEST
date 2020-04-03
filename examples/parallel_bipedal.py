import os
import subprocess
import sys
import time
import math
import pickle

from mpi4py import MPI
import gym
gym.logger.set_level(40)

import neat

# Working prototype for parallel NEAT.

def fork(n_proc):
    if os.getenv('MPI_PARENT') is None:
        env = os.environ.copy()
        env['MPI_PARENT'] = '1'
        subprocess.call(['mpirun', '-use-hwthread-cpus', '-np',
            str(n_proc), sys.executable, '-u', __file__], env=env)
        return True
    return False


class ParallelNEAT():
    def __init__(self):
        self.env = gym.make('BipedalWalker-v3')
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.n_workers = self.comm.Get_size() 

        self.job_per_worker = 32
        if self.rank == 0:
            self.ai = neat.NEAT(n_networks = self.n_workers * self.job_per_worker,
                                input_size = self.env.observation_space.shape[0],
                                output_size = self.env.action_space.shape[0],
                                bias = True,
                                hidden_activation = math.tanh,
                                output_activation = math.tanh,
                                c1 = 1.0, c2 = 1.0, c3 = 5.0,
                                distance_threshold = 3.0,
                                weight_mutation_rate = 0.8,
                                node_mutation_rate = 0.03,
                                connection_mutation_rate = 0.3,
                                interspecies_mating_rate = 0.001,
                                disable_rate = 0.75,
                                noise_magnitude = 0.01,
                                stegnant_threshold = 30)

    def eval_genomes(self, genomes):
        rewards = []
        for genome in genomes:
            total_reward = 0.0
            observation = self.env.reset()
            done = False
            while not done:
                output = genome(observation)
                observation, reward, done, info = self.env.step(output)
                total_reward += reward
            rewards.append(total_reward)
        return rewards

    def master(self):
        while True:
            prev_time = time.time()
            print(f'Step: {self.ai.generation}')
            population = self.ai.population
            for i in range(1, self.n_workers):
                self.comm.send(population[
                    i*self.job_per_worker:i*self.job_per_worker+self.job_per_worker],
                               dest=i)
            rewards = [0] * self.ai.n_networks
            rewards[:self.job_per_worker] = self.eval_genomes(
                population[:self.job_per_worker])
            for i in range(1, self.n_workers):
                rewards[i*self.job_per_worker:i*self.job_per_worker+self.job_per_worker
                ] = self.comm.recv(source=i)
            self.ai.next_generation(rewards)
            print(f'Step Took: {time.time() - prev_time} seconds')
            print(f'Max Reward Session: {self.ai.best_fitness}')
            print(f'Max Reward Generation: {max(rewards)}')
            if not (self.ai.generation % 100):
                self.ai.save_checkpoint()
                self.ai.best_genome.save('best_genome.pickle')

    def slave(self):
        while True:
            genomes = self.comm.recv(source=0)
            rewards = self.eval_genomes(genomes)
            self.comm.send(rewards, dest=0)

    def main(self):
        if fork(n_proc = 4): sys.exit(0)
        self.master() if self.rank == 0 else self.slave()
    

if __name__ == '__main__':
    ai = ParallelNEAT()
    ai.main()
