import neat
from utils.SubstrateCNN import *

class PythonNeatCPPN(BaseCPPN):


    def __init__(self, genome, config:neat.Config):
        self._net = neat.nn.FeedForwardNetwork.create(genome, config)


    def get_weights(self, pos):
        weights = torch.empty(pos.shape[0])
        for i in range(pos.shape[0]):
            weights[i] = self._net.activate(list(pos[i]))[0]
        return weights

    def get_bias(self, pos):
        return torch.full((pos.shape[0], ), 0)
        pos = torch.cat([pos, torch.full((pos.shape[0],3), 0)], dim=1)
        bias = torch.empty(pos.shape[0])
        for i in range(pos.shape[0]):
            bias[i] = self._net.activate(list(pos[i]))[1]
        return bias
