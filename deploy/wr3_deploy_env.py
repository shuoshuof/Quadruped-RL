# -*- coding: utf-8 -*-
"""
@Time ： 2024/12/4 20:09
@Auth ： shuoshuof
@File ：wr3_deploy_env.py
@Project ：Quadruped-RL
"""

class BaseDeployEnv:
    def __init__(self):
        self.env = None
    def step(self):
        raise NotImplementedError()
    def reset(self):
        pass

class Wr3DeployEnv:
    def __init__(self, env_name='Wr3BulletEnv', **kwargs):
        pass

    def step(self):
        pass

    def reset(self):
        pass

