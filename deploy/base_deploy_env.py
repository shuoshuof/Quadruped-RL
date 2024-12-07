# -*- coding: utf-8 -*-
"""
@Time ： 2024/12/6 16:12
@Auth ： shuoshuof
@File ：base_deploy_env.py
@Project ：Quadruped-RL
"""
from abc import ABC, abstractmethod
import threading
import copy

import numpy as np

import gym
from gym import spaces
import torch

class BaseDeployEnv(ABC):
    def __init__(self,num_envs,num_obs, num_actions,device="cuda:0"):
        self.device = device
        self.num_envs = num_envs
        self.num_obs = num_obs
        self.num_actions = num_actions

        self.act_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)
        self.obs_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)

        self._allocate_buffers()
        self._init_thread_flags()

    def _init_thread_flags(self):
        self.state_lock = threading.Lock()
        self.obs_lock = threading.Lock()
        self.action_lock = threading.Lock()
        self.has_started = threading.Event()
    def _allocate_buffers(self):
        self.obs_buf = torch.zeros((self.num_envs,self.num_obs),device=self.device,dtype=torch.float32)
        self.action_buf = torch.zeros((self.num_envs,self.num_actions),device=self.device,dtype=torch.float32)
        self.state_dict = {}
    @abstractmethod
    def get_obs(self):
        raise NotImplementedError

    def update_state(self,state_dict):
        with self.state_lock:
            self.state_dict = state_dict
    def get_state(self):
        with self.state_lock:
            return copy.deepcopy(self.state_dict)
    def update_action(self,action):
        with self.action_lock:
            self.action_buf[:] = action
    def get_action(self):
        with self.action_lock:
            return self.action_buf.clone()
    def step(self, action):
        self.update_action(action)
        obs = self.get_obs()
        obs_dict = {}
        obs_dict['obs'] = obs

        return (obs_dict,
                torch.tensor([0.],dtype=torch.float32,device=self.device),
                torch.tensor([False],dtype=torch.bool,device=self.device),
                {})
    @abstractmethod
    def _reset_robot(self):
        raise NotImplementedError
    def reset(self):
        self._reset_robot()
        self.has_started.set()

        obs = self.get_obs()
        obs_dict = {}
        obs_dict['obs'] = obs

        return obs_dict

    @property
    def observation_space(self) -> gym.Space:
        """Get the environment's observation space."""
        return self.obs_space

    @property
    def action_space(self) -> gym.Space:
        """Get the environment's action space."""
        return self.act_space
