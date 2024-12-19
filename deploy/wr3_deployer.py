# -*- coding: utf-8 -*-
"""
@Time ： 2024/12/4 18:02
@Auth ： shuoshuof
@File ：wr3_deployer.py
@Project ：Quadruped-RL
"""
import time

import torch

from rl_games.algos_torch.players import PpoPlayerContinuous
from loop_rate_limiters import RateLimiter


class Wr3Deployer(PpoPlayerContinuous):
    def __init__(self, params):
        super().__init__(params)

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_deterministic = self.is_deterministic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None


        self.wait_for_checkpoint()

        need_init_rnn = self.is_rnn
        from deploy.wr3_deploy_env import Wr3DeployEnv
        from deploy.wr3_sim2sim_env import Wr3MujocoEnv
        # from typing import Union
        # self.env: Wr3DeployEnv

        for _ in range(n_games):
            if games_played >= n_games:
                break
            # if isinstance(self.env,Wr3MujocoEnv):
            #     self.env.start_control_thread()
            obses = self.env_reset(self.env)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            rate = RateLimiter(frequency=50.0, warn=False)
            import time
            while True:
                start = time.time()
                obses = self.env.get_obs(is_policy_input=True)
                action = self.get_action(obses, is_deterministic)

                self.env_step(self.env, action)
                rate.sleep()
                end = time.time()
                # print(end - start)




