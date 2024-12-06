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

    # def run(self):
    #     n_games = self.games_num
    #     render = self.render_env
    #     n_game_life = self.n_game_life
    #     is_deterministic = self.is_deterministic
    #     sum_rewards = 0
    #     sum_steps = 0
    #     sum_game_res = 0
    #     n_games = n_games * n_game_life
    #     games_played = 0
    #     has_masks = False
    #     has_masks_func = getattr(self.env, "has_action_mask", None) is not None
    #
    #
    #     if has_masks_func:
    #         has_masks = self.env.has_action_mask()
    #
    #     self.wait_for_checkpoint()
    #
    #     need_init_rnn = self.is_rnn
    #
    #     from deploy.wr3_deploy_env import Wr3MujocoEnv
    #
    #     mujoco_env = Wr3MujocoEnv()
    #
    #     for _ in range(n_games):
    #         if games_played >= n_games:
    #             break
    #
    #         obses = self.env_reset(self.env)
    #         batch_size = 1
    #         batch_size = self.get_batch_size(obses, batch_size)
    #
    #         if need_init_rnn:
    #             self.init_rnn()
    #             need_init_rnn = False
    #
    #         cr = torch.zeros(batch_size, dtype=torch.float32)
    #         steps = torch.zeros(batch_size, dtype=torch.float32)
    #
    #
    #         for n in range(self.max_steps):
    #             if self.evaluation and n % self.update_checkpoint_freq == 0:
    #                 self.maybe_load_new_checkpoint()
    #
    #             if has_masks:
    #                 masks = self.env.get_action_mask()
    #                 action = self.get_masked_action(
    #                     obses, masks, is_deterministic)
    #             else:
    #                 action = self.get_action(obses, is_deterministic)
    #
    #             obses, r, done, info = self.env_step(self.env, action)
    #             cr += r
    #             steps += 1
    #
    #             if render:
    #                 self.env.render(mode='human')
    #                 time.sleep(self.render_sleep)
    #
    #             all_done_indices = done.nonzero(as_tuple=False)
    #             done_indices = all_done_indices[::self.num_agents]
    #             done_count = len(done_indices)
    #             games_played += done_count

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

        from deploy.wr3_sim2sim_env import Wr3MujocoEnv

        mujoco_env = Wr3MujocoEnv()

        for _ in range(n_games):
            if games_played >= n_games:
                break

            obses = self.env_reset(mujoco_env)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            rate = RateLimiter(frequency=60.0, warn=False)
            import time
            while True:
                start = time.time()
                action = self.get_action(obses, is_deterministic)

                obses, r, done, info = self.env_step(mujoco_env, action)
                rate.sleep()
                end = time.time()
                print(end - start)




