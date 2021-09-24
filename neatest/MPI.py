import numpy as np

class MPI():
    @property
    def FLOAT(self):
        return np.float32

    @property
    def DOUBLE(self):
        return np.float64

    @property
    def COMM_WORLD(self):
        return self

    @staticmethod
    def Allgatherv(reward, reward_array):
        reward, dtype = reward
        reward_array, dtype = reward_array
        reward_array[...] = reward

    @staticmethod
    def Allgather(reward, reward_array):
        reward, dtype = reward
        reward_array, dtype = reward_array
        reward_array[...] = reward

    @staticmethod
    def allgather(reward):
        return [reward]

    @staticmethod
    def allgatherv(reward):
        return [reward]

    @staticmethod
    def Get_rank():
        return 0

    @staticmethod
    def Get_size():
        return 1

    @staticmethod
    def bcast(a, **kwargs):
        return a

    @property
    def rank(self):
        return self.Get_rank()
