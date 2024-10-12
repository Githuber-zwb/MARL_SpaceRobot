import numpy as np
import reachTest
from gym.envs.robotics import rotations

np.random.seed(1234)
env = reachTest.SpaceReachEnv()
env.seed(1234)
max_action = env.action_space.high

obs_dict = env.reset()  # 获取初始状态,每次循环之前先reset()
o = obs_dict['observation']
ag = obs_dict['achieved_goal']
g = obs_dict['desired_goal']

dims = {
    'o': o.shape[0],
    'u': env.action_space.shape[0],
    'g': g.shape[0],
}


# agent = DDPG(input_dims=dims, max_u=1., scope='ddpg', clip_obs=200., norm_eps=0.01, norm_clip=5)

for epoch in range(20):
    obs_dict = env.reset()  # 获取初始状态,每次循环之前先reset()
    o = obs_dict['observation']
    g = np.concatenate([obs_dict['desired_goal'], obs_dict['desired_goal1']])

    for t in range(1000):
        env.render()
        # 使targetsat旋转
        curr_rot = env.sim.data.get_body_xquat('targetsat').copy()
        curr_rot = rotations.quat2euler(curr_rot)  # 3维欧拉角
        print('curr_rot', curr_rot)
        curr_rot1 = curr_rot + [0.0, 0.00, 0.01]  # 绕x,y,z轴旋转
        target_id = env.sim.model.body_name2id('targetsat')  # 设置target的位置
        env.sim.model.body_quat[target_id] = rotations.euler2quat(curr_rot1)
        # targetsat的线速度
        # curr_pos = self.sim.data.get_body_xpos('targetsat').copy()
        # self.sim.model.body_pos[target_id] = curr_pos + [0.001,0,0]

        env.sim.forward()


env.close()
