from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from rl_games.algos_torch.network_builder import NetworkBuilder
import torch
import torch.nn as nn
import numpy as np

class RNNBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)
        return

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actor_rnn_len = kwargs.pop('actor_rnn_len')
            actor_rnn_input_size = kwargs.pop('actor_rnn_input_size')
            actions_num = kwargs.pop('actions_num')

            critic_input_size = kwargs.pop('critic_input_size')
            
            command_size = kwargs.pop('command_size')
            world_model_size = kwargs.pop('world_model_size')

            self.actor_rnn_len = actor_rnn_len
            self.actor_rnn_input_size = actor_rnn_input_size
            self.critic_input_size = critic_input_size
            self.command_size = command_size
            self.world_model_size = world_model_size
            self.totoal_input_dim = self.actor_rnn_len * self.actor_rnn_input_size + self.critic_input_size + self.command_size + self.world_model_size 

            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            self.rnn_name = "gru"
            self.actor_rnn = self._build_rnn(self.rnn_name, actor_rnn_input_size, self.rnn_unit, self.rnn_layers)
            actor_mlp_args = {"input_size": self.rnn_unit*self.rnn_layers + command_size, "units": self.actor_units, "activation": self.activation, 'dense_func' : torch.nn.Linear, "norm_func_name": self.normalization}
            self.actor_mlp = self._build_mlp(**actor_mlp_args)
            self.actor_mu = torch.nn.Linear(self.rnn_unit, actions_num)

            critic_mlp_args = {"input_size": critic_input_size + command_size, "units": self.critic_units, "activation": self.activation, 'dense_func' : torch.nn.Linear, "norm_func_name": self.normalization}
            self.critic_mlp = self._build_mlp(**critic_mlp_args)
            self.critic_value = torch.nn.Linear(self.rnn_unit, 1)

            world_model_mlp_args = {"input_size": self.rnn_unit*self.rnn_layers, "units": self.world_model_units, "activation": self.activation, 'dense_func' : torch.nn.Linear, "norm_func_name": self.normalization}
            self.world_model_mlp = self._build_mlp(**world_model_mlp_args)
            self.world_model_out = torch.nn.Linear(self.world_model_units[-1], world_model_size)

            for m in self.modules():
                if isinstance(m, nn.Linear):
                    self.init_factory.create(**self.initializer)(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            return 
            

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            B, D = obs.size()
            assert D == self.totoal_input_dim

            actor_rnn_input = obs[:, :self.actor_rnn_len * self.actor_rnn_input_size].view(B, self.actor_rnn_len, self.actor_rnn_input_size)
            critic_input = obs[:, self.actor_rnn_len * self.actor_rnn_input_size:self.actor_rnn_len * self.actor_rnn_input_size + self.critic_input_size]
            command_input = obs[:, -self.command_size-self.world_model_size:-self.world_model_size]
            world_model_input = obs[:, -self.world_model_size:]

            # TODO: check rnn is time first or batch first
            actor_actor_rnn_out, _ = self.actor_rnn(actor_rnn_input.permute(1, 0, 2))

            mu = self.actor_mu(self.actor_mlp(torch.cat([actor_actor_rnn_out[-1], command_input], dim=1)))
            value = self.critic_value(self.critic_mlp(torch.cat([critic_input, command_input], dim=1)))
            world_model_out = self.world_model_out(self.world_model_mlp(actor_actor_rnn_out[-1]))
            world_model_loss = torch.nn.functional.mse_loss(world_model_out, world_model_input)

            sigma = torch.ones_like(mu) * 0.1
            return mu, sigma, value, {"world_model_loss": world_model_loss}
        
        
                    
        def is_separate_critic(self):
            raise NotImplementedError

        def is_rnn(self):
            return False

        def load(self, params):

            self.rnn_layers = params['rnn']['layers']
            self.rnn_unit = params['rnn']['unit']
            
            self.actor_units = params['actor']['units']
            self.critic_units = params['critic']['units']
            self.world_model_units = params['world_model']['units']

            self.activation = params['activation']
            self.initializer = params['initializer']
            self.normalization = params.get('normalization', None)


    def build(self, name, **kwargs):
        net = RNNBuilder.Network(self.params, **kwargs)
        return net