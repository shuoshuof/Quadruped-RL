# -*- coding: utf-8 -*-
"""
@Time ： 2024/12/4 20:09
@Auth ： shuoshuof
@File ：wr3_deploy_env.py
@Project ：Quadruped-RL
"""
from abc import ABC, abstractmethod
import copy
import threading
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from collections import OrderedDict
import time

from loop_rate_limiters import RateLimiter
import gym
from isaacgymenvs.utils.torch_jit_utils import *
import torch

from deploy.base_deploy_env import BaseDeployEnv
from deploy.robot_communication import DataReceiver, MotorCmdDataHandler


class Wr3DeployEnv(BaseDeployEnv):

    def __init__(self, cfg) -> None:
        self.device = 'cuda:0'
        super().__init__(cfg)

        self.num_dofs = self.num_actions

        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]

        self.gravity_vec = torch.tensor([0., 0., -1.], dtype=torch.float32, device=self.device)
        self.forward_vec = torch.tensor([1., 0., 0.], dtype=torch.float32, device=self.device).repeat(
            (self.num_envs, 1))

        # x vel, y vel, yaw vel, heading

        self.commands_scale = torch.tensor([self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale],
                                           device=self.device)
        self.use_default_commands = self.cfg["env"]["useDefaultCommands"]
        # control
        # self.default_dof_pos = np.array(self.cfg["env"]["defaultJointAngles"],dtype=np.float32)
        self.default_dof_pos = np.array([angle for angle in dict(self.cfg["env"]["defaultJointAngles"]).values()],
                                        dtype=np.float32)
        self.default_Kp = self.cfg["env"]["control"]["stiffness"]
        self.default_Kd = self.cfg["env"]["control"]["damping"]
        self.Kp = copy.deepcopy(self.default_Kp)
        self.Kd = copy.deepcopy(self.default_Kd)
        self.max_torque = self.cfg["env"]["control"]["maxTorque"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        self.num_height_points = 140

        self._init_SDK()
        self.state_thread = threading.Thread(target=self._update_state_thread)
        self.state_thread.daemon = True
        self.state_thread.start()

    def _init_SDK(self):
        self.receiver = DataReceiver()
        self.motor_cmd = MotorCmdDataHandler(num_motors=12, header1=0x57, header2=0x4C, sequence=0, data_type=0x01)
        # h-thigh a-hip k-calf
        self.real2sim_dof_map = [
            3, 4, 5,
            0, 1, 2,
            9, 10, 11,
            6, 7, 8,
        ]

        self.sim2real_dof_map = [
            3, 4, 5,
            0, 1, 2,
            9, 10, 11,
            6, 7, 8,
        ]

        # TODO: motor order may be different with the sim
        for _ in range(100):
            imu_data = self.receiver.get_imu_data()
            self.start_quat = np.array(imu_data.quaternion, dtype=np.float32)[[1, 2, 3, 0]]
            self.start_quat = normalize(to_torch(self.start_quat, device=self.device))
            state_dict = self._acquire_robot_state()
            time.sleep(0.01)
        dof_pos = state_dict['dof_pos'][self.sim2real_dof_map]

        self._set_motor_pd(kp=self.Kp, kd=self.Kd)
        for i in range(self.num_dofs):
            self.motor_cmd.cmd[i].pos = dof_pos[i]

        self.motor_cmd.send_data()
        print("_init_SDK:", state_dict)

    def _set_motor_pd(self, kp=0., kd=0., send_data=True):
        self.Kp = kp
        self.Kd = kd
        state_dict = self._acquire_robot_state()
        dof_pos = state_dict['dof_pos'][self.sim2real_dof_map]
        for i in range(self.num_dofs):
            self.motor_cmd.cmd[i].kp = self.Kp
            self.motor_cmd.cmd[i].kd = self.Kd
            self.motor_cmd.cmd[i].mode = 10
            self.motor_cmd.cmd[i].pos = dof_pos[i]
        
        if send_data:
            self.motor_cmd.send_data()

    def _reset_robot(self):
        self._set_motor_pd(kp=20., kd=0.2)
        while True:
            action = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=self.device)
            self.update_action(action)
            self._apply_action_to_robot(action, torque_threshold=0.5)
            time.sleep(0.01)
            state_dict = self.get_state()
            if np.all(np.abs(state_dict["dof_pos"] - self.default_dof_pos) < 1e-1):
                self._set_motor_pd(kp=self.default_Kp, kd=self.default_Kd, send_data=False)
                # raise EOFError
                break

    def _update_state_thread(self):
        rate = RateLimiter(frequency=1000, warn=True)
        while True:
            start = time.time()
            state_dict = self._acquire_robot_state()
            self.update_state(state_dict)

            rate.sleep()
            end = time.time()
            # print("Control Thread Time: ", end - start)

    def _acquire_robot_state(self):
        odometer_data = self.receiver.get_odometer_data()
        imu_data = self.receiver.get_imu_data()

        # TODO: need to change the order of the quaternion?????
        base_quat = np.array(imu_data.quaternion, dtype=np.float32)[[1, 2, 3, 0]]
        base_quat = normalize(to_torch(base_quat, device=self.device))
        base_quat = quat_mul(calc_heading_quat_inv(self.start_quat[None, ...]), base_quat[None, ...])[0].cpu().numpy()

        base_lin_vel = np.array([
            odometer_data.linear_x,
            odometer_data.linear_y,
            odometer_data.linear_z,
        ], dtype=np.float32)

        base_ang_vel = np.array(imu_data.gyroscope, dtype=np.float32)

        heights = np.zeros((self.num_height_points,), dtype=np.float32)

        motor_data = self.receiver.get_motor_state_data()

        dof_pos = []
        dof_vel = []
        dof_mode = []
        for i in range(self.num_dofs):
            dof_pos.append(motor_data.state[i].pos)
            dof_vel.append(motor_data.state[i].w)
            dof_mode.append(motor_data.state[i].mode)
        dof_pos = np.array(dof_pos, dtype=np.float32)[self.real2sim_dof_map]
        dof_vel = np.array(dof_vel, dtype=np.float32)[self.real2sim_dof_map]

        state_dict = OrderedDict(
            base_quat=base_quat.copy(),
            base_lin_vel=base_lin_vel.copy(),
            base_ang_vel=base_ang_vel.copy(),
            heights=heights.copy(),
            dof_pos=dof_pos.copy(),
            dof_vel=dof_vel.copy()
        )
        return state_dict

    def _apply_action_to_robot(self, actions, torque_threshold=None):
        state_dict = self.get_state()
        dof_vel = state_dict['dof_vel']
        dof_pos = state_dict['dof_pos']
        actions = actions.squeeze(0).cpu().numpy()
        # target_dof_pos = self.action_scale * actions + self.default_dof_pos

        torque = self.Kp*(self.action_scale * actions + self.default_dof_pos - dof_pos) - self.Kd * dof_vel
        max_torque = torque_threshold if torque_threshold is not None else self.max_torque
        clipped_torque = np.clip(torque, -max_torque, max_torque)
        clipped_target_dof_pos = (clipped_torque + self.Kd * dof_vel) / self.Kp + dof_pos

        # # TODO: add protection for out of bound
        # if clip_action:
        #     target_dof_pos = np.clip(target_dof_pos, dof_pos - clip_action_threshold, dof_pos + clip_action_threshold)
        # # TODO: motor order may be different with the sim
        # target_dof_pos = target_dof_pos[self.sim2real_dof_map]
        # # print(target_dof_pos)
        # for i in range(self.num_dofs):
        #     self.motor_cmd.cmd[i].pos = target_dof_pos[i]

        clipped_target_dof_pos = clipped_target_dof_pos[self.sim2real_dof_map]
        for i in range(self.num_dofs):
            self.motor_cmd.cmd[i].pos = clipped_target_dof_pos[i]
        self.motor_cmd.send_data()

    def get_obs(self):
        state_dict = self.get_state()
        action = self.get_action()
        base_quat = torch.from_numpy(state_dict['base_quat']).to(self.device)

        # TODO: quat rotate inverse may be not correct
        base_lin_vel = torch.from_numpy(state_dict['base_lin_vel']).to(self.device)
        base_lin_vel = quat_rotate_inverse(base_quat.unsqueeze(0), base_lin_vel.unsqueeze(0))

        # TODO: confirm the ang vel is local or global
        base_ang_vel = torch.from_numpy(state_dict['base_ang_vel']).to(self.device)
        base_ang_vel = quat_rotate_inverse(base_quat.unsqueeze(0), base_ang_vel.unsqueeze(0))

        projected_gravity = quat_rotate_inverse(base_quat.unsqueeze(0), self.gravity_vec.clone().unsqueeze(0))

        forward = quat_apply(base_quat.unsqueeze(0), self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        dof_pos = torch.from_numpy(state_dict['dof_pos']).to(self.device)
        dof_vel = torch.from_numpy(state_dict['dof_vel']).to(self.device)

        heights = torch.from_numpy(state_dict['heights']).to(self.device)


        if self.use_default_commands:
            self.commands *= 0
            self.commands[:, 0] = 0

        obs_tensor = torch.concatenate([
            # base_lin_vel * self.lin_vel_scale,
            base_ang_vel * self.ang_vel_scale,
            projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            dof_pos.unsqueeze(0) * self.dof_pos_scale,
            dof_vel.unsqueeze(0) * self.dof_vel_scale,
            # heights.unsqueeze(0),
            action,
        ], dim=-1)

        assert obs_tensor.shape == (self.num_envs, self.num_obs)
        return obs_tensor
    def step(self, action):
        self._apply_action_to_robot(action)
        return super().step(action)

@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


if __name__ == '__main__':
    from hydra import core
    from deploy.wr3_sim2sim_env import Wr3MujocoEnv

    # ip 192.168.93.107 255.255.255.0 192.168.93.1

    core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path='../cfgs/cfg/task/')
    deploy_cfg = hydra.compose(config_name='Wr3Deploy.yaml')

    deploy_env = Wr3DeployEnv(deploy_cfg)

    # core.global_hydra.GlobalHydra.instance().clear()
    # hydra.initialize(config_path='../cfgs/cfg/task/')
    # mujoco_cfg = hydra.compose(config_name='Wr3Mujoco.yaml')
    #
    # mujoco_env = Wr3MujocoEnv(mujoco_cfg)
    # # mujoco_env.start_control_thread()
    # rate = RateLimiter(frequency=100, warn=True)
    # while True:
    #     state_dict = deploy_env._acquire_robot_state()
    #     if "base_pos" is not state_dict.keys():
    #         # x,y,z,w
    #         state_dict["base_pos"] = np.array([0., 0., 0.5], dtype=np.float32)
    #     # state_dict["base_quat"] = state_dict["base_quat"][[3,0,1,2]]
    #     print(np.round(state_dict["base_quat"], 4))
    #     mujoco_env._apply_state_in_mujoco(state_dict)
    #
    #     rate.sleep()

    deploy_env.start_control_thread()
