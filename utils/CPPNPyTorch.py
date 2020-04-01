from utils.SubstrateCNN import *
from pytorch_neat.recurrent_net import RecurrentNet

class PyTorchCPPN(BaseCPPN):

    def __init__(self, genome, config, batch_size=64):
        self._batch_size = batch_size
        self._net = RecurrentNet.create(genome, config, batch_size)

    def _activate_dynamic_batch(self, pos):
        weights = torch.empty((pos.shape[0], self._net.n_outputs))
        for i in range(0, pos.shape[0], self._batch_size):
            batch = pos[i:i+self._batch_size]
            if len(batch) == self._batch_size:
                weights[i:i+self._batch_size] = self._net.activate(batch)
            else:
                padded = torch.empty((self._batch_size, batch.shape[1]))
                padded[:len(batch),:] = batch
                weights[i:i+len(batch)] = self._net.activate(padded)[:len(batch)]
        return weights

    def get_weights(self, pos):
        return self._activate_dynamic_batch(pos)[:, 0]


    def get_bias(self, pos):
        pos = torch.cat([pos, torch.full((pos.shape[0],3), 0)], dim=1)
        return self._activate_dynamic_batch(pos)[:, 1]


