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
import mujoco
import mujoco.viewer
from isaacgymenvs.utils.torch_jit_utils import quat_rotate_inverse,quat_apply
import torch




class BaseDeployEnv(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def _allocate_buffers(self):
        pass
    @abstractmethod
    def step(self,action):
        raise NotImplementedError
    @abstractmethod
    def reset(self):
        raise NotImplementedError

class Wr3MujocoEnv(BaseDeployEnv):
    def __init__(self):
        super().__init__()
        from hydra import core
        core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(config_path='../cfgs/cfg_deploy/task/')
        cfg = hydra.compose(config_name='Wr3Mujoco.yaml')
        self.cfg = DictConfig(cfg)

        self.device = 'cuda:0'

        self.num_envs = self.cfg["env"]['numEnvs']
        self.num_obs = self.cfg["env"]['numObservations']
        self.num_actions = self.cfg["env"]['numActions']
        self.num_dofs = self.num_actions

        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]

        self.gravity_vec = torch.tensor([0., 0., -1.],dtype=torch.float32,device=self.device)
        self.forward_vec = torch.tensor([1., 0., 0.], dtype=torch.float32,device=self.device).repeat((self.num_envs, 1))

        # x vel, y vel, yaw vel, heading
        self.commands = torch.zeros((self.num_envs,4),dtype=torch.float32,device=self.device)
        self.commands[:,0] = 1.
        self.commands_scale = torch.tensor([self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale],device=self.device)
        self.use_default_commands = self.cfg["env"]["useDefaultCommands"]
        # control
        self.default_dof_pos = np.zeros(self.num_dofs, dtype=np.float32)
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        self.num_height_points = 140

        self.scene_path = 'assets/wr3/scene.xml'
        self._init_sim()
        self._launch_viewer()

        self._allocate_buffers()

        self._init_thread_values()
        sim_thread = threading.Thread(target=self._run_sim_thread)
        sim_thread.start()

    def _init_sim(self):
        self.scene = mujoco.MjModel.from_xml_path(self.scene_path)
        self.mj_data = mujoco.MjData(self.scene)

        self.opt = mujoco.MjvOption()
    def _launch_viewer(self):
        self.viewer = mujoco.viewer.launch_passive(self.scene, self.mj_data)
        self.viewer.cam.lookat = [0.5, 0., 0.5]
        self.viewer.cam.elevation = -90
        self.viewer.cam.azimuth = 0

    def _allocate_buffers(self):
        self.obs_buf = torch.zeros((self.num_envs,self.num_obs),device=self.device,dtype=torch.float32)
        self.action_buf = torch.zeros((self.num_envs,self.num_actions),device=self.device,dtype=torch.float32)
        self.state_dict = {}
    def _init_thread_values(self):
        self.state_lock = threading.Lock()
        self.obs_lock = threading.Lock()
        self.action_lock = threading.Lock()
        self.has_started = threading.Event()
    def _reset_robot(self):
        # initialize the robot with start pose
        joint_start_poses = self.cfg["env"]["defaultJointAngles"]

        for idx,(joint_name, joint_angle) in enumerate(joint_start_poses.items()):
            joint_idx = mujoco.mj_name2id(self.scene, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            self.mj_data.qpos[6+joint_idx] = joint_angle
            self.default_dof_pos[idx] = joint_angle

        robot_base_state = self.cfg["env"]['baseInitState']

        self.mj_data.qpos[:3] = np.array(robot_base_state['pos'],dtype=np.float32)
        self.mj_data.qvel[3:7] = np.array(robot_base_state['rot'],dtype=np.float32)[[1, 2, 3, 0]]
        self.mj_data.qvel[:3] = np.array(robot_base_state['vLinear'],dtype=np.float32)
        self.mj_data.qvel[3:6] = np.array(robot_base_state['vAngular'],dtype=np.float32)

        self.mj_data.qvel = 0.
        self.mj_data.ctrl = 0.

        mujoco.mj_step(self.scene, self.mj_data)
        self.viewer.sync()

        state_dict = self._acquire_mujoco_state()
        self.update_state(state_dict)

    def _run_sim_thread(self):
        self._reset_robot()
        self.has_started.set()

        rate = RateLimiter(frequency=200.0, warn=False)
        while self.viewer.is_running():
            start = time.time()
            state_dict = self._acquire_mujoco_state()
            self.update_state(state_dict)
            actions = self.get_action()
            self._apply_action_in_mujoco(actions,state_dict)

            mujoco.mj_step(self.scene,self.mj_data)
            self.viewer.sync()
            rate.sleep()
            end = time.time()
            print('Simulation step took {} seconds'.format(end - start))


    def _acquire_mujoco_state(self):
        base_quat = self.mj_data.qpos[3:7].copy().astype(np.float32)[[1,2,3,0]]

        base_lin_vel = self.mj_data.qvel[:3].copy().astype(np.float32)

        base_ang_vel = self.mj_data.qvel[3:6].copy().astype(np.float32)

        # TODO: add heights
        heights = np.zeros((self.num_height_points,),dtype=np.float32)
        # TODO: dof may be not correct
        dof_pos = self.mj_data.qpos[7:7+12].copy().astype(np.float32)
        dof_vel = self.mj_data.qvel[6:6+12].copy().astype(np.float32)

        state_dict = OrderedDict(
            base_quat = base_quat,
            base_lin_vel = base_lin_vel,
            base_ang_vel = base_ang_vel,
            heights = heights,
            dof_pos = dof_pos,
            dof_vel = dof_vel
        )

        return state_dict

    def get_obs(self):
        state_dict = self.get_state()
        base_quat = torch.from_numpy(state_dict['base_quat']).to(self.device)

        # TODO: quat rotate inverse may be not correct
        base_lin_vel = torch.from_numpy(state_dict['base_lin_vel']).to(self.device)
        base_lin_vel = quat_rotate_inverse(base_quat.unsqueeze(0),base_lin_vel.unsqueeze(0))

        base_ang_vel = torch.from_numpy(state_dict['base_ang_vel']).to(self.device)
        base_ang_vel = quat_rotate_inverse(base_quat.unsqueeze(0),base_ang_vel.unsqueeze(0))

        projected_gravity = quat_rotate_inverse(base_quat.unsqueeze(0),self.gravity_vec.clone().unsqueeze(0))

        forward = quat_apply(base_quat.unsqueeze(0), self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.commands[:,2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        dof_pos = torch.from_numpy(state_dict['dof_pos']).to(self.device)
        dof_vel = torch.from_numpy(state_dict['dof_vel']).to(self.device)

        heights = torch.from_numpy(state_dict['heights']).to(self.device)

        action = self.get_action()

        if self.use_default_commands:
            self.commands*=0
            self.commands[:, 0] = 1

        obs_tensor = torch.concatenate([
            base_lin_vel*self.lin_vel_scale,
            base_ang_vel*self.ang_vel_scale,
            projected_gravity,
            self.commands[:,:3]*self.commands_scale,
            dof_pos.unsqueeze(0)*self.dof_pos_scale,
            dof_vel.unsqueeze(0)*self.dof_vel_scale,
            heights.unsqueeze(0),
            action,
        ],dim=-1)

        assert obs_tensor.shape == (self.num_envs,self.num_obs)
        return obs_tensor

    def _apply_action_in_mujoco(self, actions, state_dict):
        dof_vel = state_dict['dof_vel']
        dof_pos = state_dict['dof_pos']
        actions = actions.squeeze(0).cpu().numpy()
        torques = self.Kp * (self.action_scale * actions + self.default_dof_pos - dof_pos) - self.Kd * dof_vel
        torques_clipped = np.clip(torques,-80.,80.)
        assert self.mj_data.ctrl.shape == (12,)
        assert torques_clipped.shape == (12,)
        self.mj_data.ctrl[:12] = torques_clipped

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

        return (obs,
                torch.tensor([0.],dtype=torch.float32,device=self.device),
                torch.tensor([False],dtype=torch.bool,device=self.device),
                {})

    def reset(self):
        self._reset_robot()
        self.has_started.set()

        obs = self.get_obs()
        obs_dict = {}
        obs_dict['obs'] = obs

        return obs_dict


@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles

if __name__ == "__main__":
    env = Wr3MujocoEnv()
    i=0
    while True:
        i+=1
        # if i%1000==0:
        #     env.reset()
        # env.step(action=torch.ones((1,12),dtype=torch.float32,device='cuda:0'))
        time.sleep(1/1000)











