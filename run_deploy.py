# -*- coding: utf-8 -*-
"""
@Time ： 2024/12/4 19:10
@Auth ： shuoshuof
@File ：run_deploy.py
@Project ：Quadruped-RL
"""
import hydra
from omegaconf import DictConfig, OmegaConf



def preprocess_train_config(cfg, config_dict):
    """
    Adding common configuration parameters to the rl_games train config.
    An alternative to this is inferring them in task-specific .yaml files, but that requires repeating the same
    variable interpolations in each config.
    """

    train_cfg = config_dict['params']['config']

    train_cfg['device'] = cfg.rl_device

    train_cfg['population_based_training'] = cfg.pbt.enabled
    train_cfg['pbt_idx'] = cfg.pbt.policy_idx if cfg.pbt.enabled else None

    train_cfg['full_experiment_name'] = cfg.get('full_experiment_name')

    print(f'Using rl_device: {cfg.rl_device}')
    print(f'Using sim_device: {cfg.sim_device}')
    print(train_cfg)
    # the number of units in each hidden layer will be enlarged or reduced proportionally
    # if model_size_multiplier is not 1
    try:
        model_size_multiplier = config_dict['params']['network']['mlp']['model_size_multiplier']
        if model_size_multiplier != 1:
            units = config_dict['params']['network']['mlp']['units']
            for i, u in enumerate(units):
                units[i] = u * model_size_multiplier
            print(
                f'Modified MLP units by x{model_size_multiplier} to {config_dict["params"]["network"]["mlp"]["units"]}')
    except KeyError:
        pass

    return config_dict


@hydra.main(version_base="1.1", config_name="config", config_path="cfgs/cfg")
def launch_rlg_hydra(cfg: DictConfig):
    import logging
    import os
    from datetime import datetime

    # noinspection PyUnresolvedReferences
    import isaacgym
    from isaacgymenvs.pbt.pbt import PbtAlgoObserver, initial_pbt_check
    from isaacgymenvs.utils.rlgames_utils import multi_gpu_get_rank
    from hydra.utils import to_absolute_path
    from isaacgymenvs.tasks import isaacgym_task_map
    import gym
    from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed

    if cfg.pbt.enabled:
        initial_pbt_check(cfg)

    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, MultiObserver, ComplexObsRLGPUEnv
    from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.wandb_name}_{time_str}"

    # Population-Based Training
    if cfg.pbt.enabled:
        initial_pbt_check(cfg)

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    # convert to dict
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)

    def create_deploy_env(**kwargs):
        from deploy.wr3_sim2sim_env import Wr3MujocoEnv
        from deploy.wr3_deploy_env import Wr3DeployEnv
        if cfg.task.name=="Wr3Mujoco":
            env = Wr3MujocoEnv(DictConfig(cfg.task))
        if cfg.task.name=="Wr3Deploy":
            env = Wr3DeployEnv(DictConfig(cfg.task))
        return env

    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': lambda **kwargs: create_deploy_env(**kwargs),
    })


    rlg_config_dict = omegaconf_to_dict(cfg.train)
    rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict)

    # log stats from the env along with the algorithm running stats
    observers = [RLGPUAlgoObserver()]

    if cfg.pbt.enabled:
        pbt_observer = PbtAlgoObserver(cfg)
        observers.append(pbt_observer)

    # multi-gpu
    if cfg.wandb_activate:
        cfg.seed += global_rank
        if global_rank == 0:
            # initialize wandb only once per multi-gpu run
            wandb_observer = WandbAlgoObserver(cfg)
            observers.append(wandb_observer)

    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        from deploy.wr3_deployer import Wr3Deployer
        runner.player_factory.register_builder('deployer', lambda **kwargs: Wr3Deployer(**kwargs))
        return runner

    # convert CLI arguments into dictionary
    # create runner and set the settings
    runner = build_runner(MultiObserver(observers))
    runner.load(rlg_config_dict)
    runner.reset()

    # dump config dict
    if not cfg.test:
        experiment_dir = os.path.join('runs', cfg.train.params.config.name +
        '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))

        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))

    runner.run({
        'train': not cfg.test,
        'play': cfg.test,
        'checkpoint': cfg.checkpoint,
        'sigma': cfg.sigma if cfg.sigma != '' else None
    })

if __name__ == "__main__":
    launch_rlg_hydra()