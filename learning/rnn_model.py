import torch.nn as nn
from rl_games.algos_torch.models import ModelA2CContinuousLogStd
import torch

class RNN(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)
        return

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network):
            super().__init__(a2c_network)
            return

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)

            result = super().forward(input_dict)
            if is_train:
                return result

            return result
        

