import os

import copy
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from gym.envs.robotics import utils
from gym.envs.robotics import rotations

import mujoco_py

PATH = os.getcwd()

MODEL_XML_PATH = os.path.join(PATH,'SpaceRobotEnv','assets', 'spacerobot', 'spacerobot_dualarm.xml')
DEFAULT_SIZE = 500


class RobotEnv(gym.GoalEnv):
    def __init__(self, model_path, initial_qpos, n_substeps):

        # load model and simulator
        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model, nsubsteps=n_substeps)

        # render setting
        self.viewer = None
        self._viewers = {}
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }

        # seed
        self.seed()

        # initalization
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())
        self.goal = self._sample_goal()

        # set action_space and observation_space
        obs = self._get_obs()
        self._set_action_space()
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["desired_goal"].shape, dtype="float32"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float32"
                ),
            )
        )

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy()
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    def _detecte_collision(self):
        self.collision = self.sim.data.ncon
        return self.collision

    def _sensor_torque(self):
        self.sensor_data = self.sim.data.sensordata
        return self.sensor_data

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert action.shape == (12,)
        old_action = self.sim.data.ctrl.copy() * (1 / 0.8)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)  # do one step simulation here
        self._step_callback()
        obs = self._get_obs()
        done = False
        info = { 
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
            "act": action,
            "old_act": old_action,
            }
        reward = self.compute_reward(
            obs["achieved_goal"], self.goal.copy(), action, old_action, info
        )
        return obs, reward, done, info

    def reset(self):
        """Attempt to reset the simulator. Since we randomize initial conditions, it
        is possible to get into a state with numerical issues (e.g. due to penetration or
        Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        In this case, we just keep randomizing until we eventually achieve a valid initial
        configuration.
        """
        super(RobotEnv, self).reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()

        self.goal = self._sample_goal()
        obs = self._get_obs()

        # TODO: set the position of cube

        # body_id = self.sim.model.geom_name2id("cube")
        # self.sim.model.geom_pos[body_id] = np.array([0, 0, 6])
        return obs

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def render(self, mode="human", width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        # self._render_callback()
        if mode == "rgb_array":
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == "human":
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)

        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)
                self._viewer_setup()

            elif mode == "rgb_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
                self._viewer_setup()
                # self.viewer.cam.trackbodyid = 0
                # latest modification
                cam_pos = np.array([0.5, 0, 5, 0.3, -30, 0])
                for i in range(3):
                    self.viewer.cam.lookat[i] = cam_pos[i]
                self.viewer.cam.distance = cam_pos[3]
                self.viewer.cam.elevation = cam_pos[4]
                self.viewer.cam.azimuth = cam_pos[5]
                # self.viewer.cam.trackbodyid = -1

            self._viewers[mode] = self.viewer
        return self.viewer

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it is successful.
        If a reset is unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    def _get_obs(self):
        """Returns the observation."""
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation."""
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal."""
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it."""
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback【自定义回调】 that is called before rendering. Can be used
        to implement custom visualizations.【可实现自定义可视化】
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.【对模拟状态强制附加约束】
        """
        pass


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class SpacerobotEnv(RobotEnv):
    """Superclass for all SpaceRobot environments."""

    def __init__(
        self,
        model_path,
        n_substeps,
        distance_threshold,
        rotdis_threshold,
        initial_qpos,
        reward_type,
        c_coeff,
        dr_ratio,
    ):
        """Initializes a new Fetch environment.
        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            pro_type ('MDP' or 'CMDP'):  the problem setting whether contains cost or not
            c_coeff: cost coefficient
        """
        self.n_substeps = n_substeps
        #        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.rotdis_threshold = rotdis_threshold
        self.reward_type = reward_type
        self.c_coeff = c_coeff
        self.dr_ratio = dr_ratio

        super(SpacerobotEnv, self).__init__(
            model_path=model_path,
            n_substeps=n_substeps,
            initial_qpos=initial_qpos,
        )

    def compute_reward(self, achieved_goal, desired_goal, action, old_action, info):
        a = achieved_goal.copy()
        d = desired_goal.copy()
        # Compute distance between goal and the achieved goal.
        rd1 = goal_distance(a[:3], d[:3])
        rr1 = self.dr_ratio * goal_distance(a[3:6], d[3:6])
        rd2 = goal_distance(a[6:9], d[6:9])
        rr2 = self.dr_ratio * goal_distance(a[9:12], d[9:12])
        d = rd1 + rd2 + self.dr_ratio*(rr1 + rr2)

        l0 = np.linalg.norm(action[0:3] - old_action[0:3])**2
        l1 = np.linalg.norm(action[3:6] - old_action[3:6])**2
        l2 = np.linalg.norm(action[6:9] - old_action[6:9])**2
        l3 = np.linalg.norm(action[9:12] - old_action[9:12])**2

        tt0 = np.linalg.norm(action[0:3])**2
        tt1 = np.linalg.norm(action[3:6])**2
        tt2 = np.linalg.norm(action[6:9])**2
        tt3 = np.linalg.norm(action[9:12])**2

        reward = {
            "sparse": -(d > self.distance_threshold).astype(np.float32),
            # "r0": -(0.001 * rd1 ** 2 + np.log10(rd1 ** 2 + 1e-6)) - (0.001 * rr1 ** 2 + np.log10(rr1 ** 2 + 1e-6)),
            # "r1": -(0.001 * rd1 ** 2 + np.log10(rd1 ** 2 + 1e-6)) - (0.001 * rr1 ** 2 + np.log10(rr1 ** 2 + 1e-6)),
            # "r2": -(0.001 * rd2 ** 2 + np.log10(rd2 ** 2 + 1e-6)) - (0.001 * rr2 ** 2 + np.log10(rr2 ** 2 + 1e-6)),
            # "r3": -(0.001 * rd2 ** 2 + np.log10(rd2 ** 2 + 1e-6)) - (0.001 * rr2 ** 2 + np.log10(rr2 ** 2 + 1e-6)),
            # "r0": - (rd1 > self.distance_threshold).astype(np.float32),
            # "r0": - (0.001 * rd1 ** 2 + np.log10(rd1 ** 2 + 1e-6) + 0.01 * tt0 + 0.01 * l0),
            "r0": - (0.001 * rd1 ** 2 + np.log10(rd1 ** 2 + 1e-6) + 0.01 * l0),
            "r1": - (0.001 * rr1 ** 2 + np.log10(rr1 ** 2 + 1e-6) + 0.05 * tt1),
            "r2": - (0.001 * rd2 ** 2 + np.log10(rd2 ** 2 + 1e-6) + 0.01 * l2),
            "r3": - (0.001 * rr2 ** 2 + np.log10(rr2 ** 2 + 1e-6) + 0.05 * tt3),
            # "r1": 0 ,
            # "r3": 0 ,
            # "r1": -self.dr_ratio*(0.001 * rr1 ** 2 + np.log10(rr1 ** 2 + 1e-6)),
            # "r3": -self.dr_ratio*(0.001 * rr2 ** 2 + np.log10(rr2 ** 2 + 1e-6)),
            # "r1": -(0.001 * rr1 ** 2 + np.log10(rr1 ** 2 + 1e-6)),
            # "r3": -(0.001 * rr2 ** 2 + np.log10(rr2 ** 2 + 1e-6)),
            # "r0": -rd1,
            # "r1": -rr1,
            # "r2": -rd2,
            # "r3": -rr2,
            "dense": -(0.001 * d ** 2 + np.log10(d ** 2 + 1e-6)),
        }
        # print("r0: ",reward["r0"],"r1: ",reward["r1"],"r2: ",reward["r2"],"r3: ",reward["r3"])
        # print("r0=", 0.001 * rd1 ** 2 , np.log10(rd1 ** 2 + 1e-6) , 0.01 * l0)

        return reward

    def _set_action(self, action):
        """
        output action (velocity)
        :param action: angle velocity of joints
        :return: angle velocity of joints
        """
        act = action.copy()  # ensure that we don't change the action outside of this scope
        self.sim.data.ctrl[:] = act * 0.8
        for _ in range(self.n_substeps):
            self.sim.step()

    def _get_obs(self):
        # positions
        grip_pos1 = self.sim.data.get_body_xpos("tip_frame").copy()
        grip_pos2 = self.sim.data.get_body_xpos("tip_frame1").copy()

        # get the rotation angle of the target
        grip_rot1 = self.sim.data.get_body_xquat('tip_frame').copy()
        grip_rot1 = rotations.quat2euler(grip_rot1)
        grip_rot2 = self.sim.data.get_body_xquat('tip_frame1').copy()
        grip_rot2 = rotations.quat2euler(grip_rot2)     

        grip_velp1 = self.sim.data.get_body_xvelp("tip_frame").copy() * self.dt
        grip_velp2 = self.sim.data.get_body_xvelp("tip_frame1").copy() * self.dt
        grip_velr1 = self.sim.data.get_body_xvelr("tip_frame").copy() * self.dt
        grip_velr2 = self.sim.data.get_body_xvelr("tip_frame1").copy() * self.dt

        achieved_goal = np.concatenate([grip_pos1.copy(),grip_rot1.copy(),grip_pos2.copy(),grip_rot2.copy()])

        qpos_tem = self.sim.data.qpos[:19].copy()
        qvel_tem = self.sim.data.qvel[:18].copy()

        obs = np.concatenate(
            [
                qpos_tem,
                qvel_tem,
                grip_pos1,
                grip_pos2,
                grip_velp1,
                grip_velp2,
                grip_rot1,
                grip_rot2,
                grip_velr1,
                grip_velr2,
                self.goal.copy(),
            ]
        )

        ob0 = np.concatenate(
            [
                qpos_tem[:10].copy(),
                qvel_tem[:9].copy(),
                grip_pos1,
                grip_velp1,
                self.goal[0:6].copy()
            ]
        )

        ob1 = np.concatenate(
            [
                qpos_tem[:7].copy(),
                qpos_tem[10:13].copy(),
                qvel_tem[:6].copy(),
                qvel_tem[9:12].copy(),
                grip_rot1,
                grip_velr1,
                self.goal[0:6].copy()
            ]
        )

        ob2 = np.concatenate(
            [
                qpos_tem[:7].copy(),
                qpos_tem[13:16].copy(),
                qvel_tem[:6].copy(),
                qvel_tem[12:15].copy(),
                grip_pos2,
                grip_velp2,
                self.goal[6:12].copy()
            ]
        )

        ob3 = np.concatenate(
            [
                qpos_tem[:7].copy(),
                qpos_tem[16:19].copy(),
                qvel_tem[:6].copy(),
                qvel_tem[15:18].copy(),
                grip_rot2,
                grip_velr2,
                self.goal[6:12].copy()
            ]
        )

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
            "observation_0":ob0.copy(),
            "observation_1":ob1.copy(),
            "observation_2":ob2.copy(),
            "observation_3":ob3.copy()
        }

    def _viewer_setup(self):
        # body_id = self.sim.model.body_name2id('forearm_link')
        body_id = self.sim.model.body_name2id("wrist_3_link")
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.0
        self.viewer.cam.elevation = -14.0

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    def _sample_goal(self):
        goal_pos1 = np.array([0,0,0],dtype=np.float32)
        goal_pos2 = np.array([0,0,0],dtype=np.float32)
        goal_rot1 = np.array([0,0,0],dtype=np.float32)
        goal_rot2 = np.array([0,0,0],dtype=np.float32)

        goal_pos1[0] = self.initial_gripper1_pos[0] - np.random.uniform(-0.10, 0.15)
        goal_pos1[1] = self.initial_gripper1_pos[1] - np.random.uniform(0.15, 0.35)
        goal_pos1[2] = self.initial_gripper1_pos[2] + np.random.uniform(-0.10, 0.10)

        goal_rot1[0] = self.initial_gripper1_rot[0] + np.random.uniform(-0.20, 0.30)
        goal_rot1[1] = self.initial_gripper1_rot[1] + np.random.uniform(-0.15, 0.35)
        goal_rot1[2] = self.initial_gripper1_rot[2] + np.random.uniform(-0.30, 0.20)

        goal_pos2[0] = self.initial_gripper2_pos[0] - np.random.uniform(-0.10, 0.15) #two end-effectors have the same x
        goal_pos2[1] = self.initial_gripper2_pos[1] + np.random.uniform(0.15, 0.35) #twp end-effector have opposite y (minus)
        goal_pos2[2] = self.initial_gripper2_pos[2] - np.random.uniform(-0.10, 0.10) #two end-effector have different z 

        goal_rot2[0] = self.initial_gripper2_rot[0] + np.random.uniform(-0.20, 0.30) #the difference between two arms:rotate along x axis with pi deg
        goal_rot2[1] = self.initial_gripper2_rot[1] - np.random.uniform(-0.15, 0.35) #So the target has opposite y and z.
        goal_rot2[2] = self.initial_gripper2_rot[2] - np.random.uniform(-0.30, 0.20)

        # goal_pos1[0] = self.initial_gripper1_pos[0] - 0.2
        # goal_pos1[1] = self.initial_gripper1_pos[1] - 0.3
        # goal_pos1[2] = self.initial_gripper1_pos[2] + 0.05

        # goal_rot1[0] = self.initial_gripper1_rot[0] + 0.5
        # goal_rot1[1] = self.initial_gripper1_rot[1] + 0.1
        # goal_rot1[2] = self.initial_gripper1_rot[2] + 0.4

        # goal_pos2[0] = self.initial_gripper2_pos[0] - 0.2
        # goal_pos2[1] = self.initial_gripper2_pos[1] + 0.3
        # goal_pos2[2] = self.initial_gripper2_pos[2] - 0.05

        # goal_rot2[0] = self.initial_gripper2_rot[0] + 0.5
        # goal_rot2[1] = self.initial_gripper2_rot[1] - 0.1
        # goal_rot2[2] = self.initial_gripper2_rot[2] - 0.2

        """
        goal = np.concatenate((goal_pos, goal_rot)) #一维度的数据不影响
        goal1 = np.concatenate((goal_pos1, goal_rot1))
        """
        site_id = self.sim.model.site_name2id("target0")
        self.sim.model.site_pos[site_id] = goal_pos1.copy()
        self.sim.model.site_quat[site_id] = rotations.euler2quat(goal_rot1.copy())
        site_id1 = self.sim.model.site_name2id("target1")
        self.sim.model.site_pos[site_id1] = goal_pos2.copy()
        self.sim.model.site_quat[site_id1] = rotations.euler2quat(goal_rot2.copy())
        self.sim.forward()
        goal = np.concatenate([goal_pos1.copy(), goal_rot1.copy(),goal_pos2.copy(),goal_rot2.copy()])
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d1 = goal_distance(achieved_goal[:3], desired_goal[:3])
        d2 = goal_distance(achieved_goal[6:9], desired_goal[6:9])
        r1 = goal_distance(achieved_goal[3:6], desired_goal[3:6])
        r2 = goal_distance(achieved_goal[9:12], desired_goal[9:12])
        return (d1 < self.distance_threshold) and (d2 < self.distance_threshold) and (r1 < self.rotdis_threshold) and (r2 < self.rotdis_threshold)
        # return d

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)

        # Extract information for sampling goals.
        self.initial_gripper1_pos = self.sim.data.get_body_xpos("tip_frame").copy()
        self.initial_gripper2_pos = self.sim.data.get_body_xpos("tip_frame1").copy()
        gripper1_rot = self.sim.data.get_body_xquat('tip_frame').copy()
        gripper2_rot = self.sim.data.get_body_xquat('tip_frame1').copy()
        self.initial_gripper1_rot = rotations.quat2euler(gripper1_rot)
        self.initial_gripper2_rot = rotations.quat2euler(gripper2_rot)

        # get the initial base attitude
        self.initial_base_att = self.sim.data.get_body_xquat("chasersat").copy()

        # get the initial base position
        self.initial_base_pos = self.sim.data.get_body_xpos("chasersat").copy()

    def render(self, mode="human", width=500, height=500):
        return super(SpacerobotEnv, self).render(mode, width, height)


class SpaceRobotDualArmWithRot(SpacerobotEnv, gym.utils.EzPickle):
    def __init__(self, reward_type="sparse"):
        initial_qpos = {
            "arm:shoulder_pan_joint": 1.57*2,
            "arm:shoulder_lift_joint": -1.57,
            "arm:elbow_joint": 0.0,
            "arm:wrist_1_joint": -1.57,
            "arm:wrist_2_joint": 0.0,
            "arm:wrist_3_joint": 1.57*2,
            "arm:shoulder_pan_joint1": 0.0,
            "arm:shoulder_lift_joint1": -1.57,
            "arm:elbow_joint1": 0.0,
            "arm:wrist_1_joint1": -1.57,
            "arm:wrist_2_joint1": 0.0,
            "arm:wrist_3_joint1": 0.0,
        }
        SpacerobotEnv.__init__(
            self,
            MODEL_XML_PATH,
            n_substeps=20,
            distance_threshold=0.05,
            rotdis_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            c_coeff=0.1,
            dr_ratio=0.2
        )
        gym.utils.EzPickle.__init__(self)
