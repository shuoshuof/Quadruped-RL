import torch.nn as nn
from rl_games.algos_torch.models import ModelA2CContinuousLogStd
import torch

class RNN(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)
        return
    
    def build(self, config):
        net = self.network_builder.build("rnn", **config)
        for name, _ in net.named_parameters():
            print(name)
        return RNN.Network(net, **config)

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network, **config):
            for key in list(config.keys()):
                if key not in ["obs_shape", "normalize_value", "normalize_input", "value_size"]:
                    config.pop(key)
            super().__init__(a2c_network, **config)
            return

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)

            result = super().forward(input_dict)
            if is_train:
                return result

            return result
        

