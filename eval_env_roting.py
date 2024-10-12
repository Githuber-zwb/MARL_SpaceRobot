import gym
import imageio
from PIL import Image, ImageDraw
import SpaceRobotEnv
import numpy as np
import torch
import sys
import os
from gym import spaces
from gym.envs.robotics import rotations
from PIL import Image 

# os.environ["MUJOCO_GL"] = "egl"

parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(parent_dir)
sys.path.append(parent_dir+"/RL_algorithms/Torch/MAPPO/onpolicy")
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from onpolicy.envs.spacerobot.SpaceRobotBaseRot_env import BaseRot
from onpolicy.envs.spacerobot.SpaceRobotFourArm_env import FourArm
from onpolicy.config import get_config

def _t2n(x):
    return x.detach().cpu().numpy()

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str,
                        default="SpaceRobotDualArmWithRot", 
                        help="which scenario to run on.")
    parser.add_argument("--num_agents", type=int, default=4,
                        help="number of agents.")
    parser.add_argument("--share_reward", action='store_false', 
                        default=True, 
                        help="by default true. If false, use different reward for each agent.")
                        
    all_args = parser.parse_known_args(args)[0]

    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False. Note that GRF is a fully observed game, so ippo is rmappo.")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    env = BaseRot(all_args)
    env2 = FourArm(all_args)

    eval_episode_rewards = []
    actors = []
    obs = env.reset()

    # set reach goals
    # compute current goal
    goal_pos1 = np.array(env.env.sim.data.get_body_xpos("targethold_h3").copy())
    goal_rot1 = np.array(rotations.quat2euler(env.env.sim.data.get_body_xquat("targethold_h3").copy()))
    goal_pos2 = np.array(env.env.sim.data.get_body_xpos("targethold_h4").copy())
    goal_rot2 = np.array(rotations.quat2euler(env.env.sim.data.get_body_xquat("targethold_h4").copy()))

    site_id0 = env.env.sim.model.site_name2id("target0")
    env.env.sim.model.site_pos[site_id0] = goal_pos1.copy()
    env.env.sim.model.site_quat[site_id0] = rotations.euler2quat(goal_rot1.copy())
    site_id1 = env.env.sim.model.site_name2id("target1")
    env.env.sim.model.site_pos[site_id1] = goal_pos2.copy()
    env.env.sim.model.site_quat[site_id1] = rotations.euler2quat(goal_rot2.copy())

    # set base goal
    # site_id = env.env.sim.model.site_name2id("targetbase")
    # env.env.sim.model.site_pos[site_id] = np.array([0, 0, 4], dtype=np.float32)
    # env.env.sim.model.site_quat[site_id] = rotations.euler2quat(np.array([0, 0, 0.1]))

    # get state
    grip_pos1 = env.env.sim.data.get_body_xpos("tip_frame").copy()
    grip_rot1 = env.env.sim.data.get_body_xquat('tip_frame').copy()
    grip_rot1 = rotations.quat2euler(grip_rot1)  
    grip_velp1 = env.env.sim.data.get_body_xvelp("tip_frame").copy() * 0.1
    grip_velr1 = env.env.sim.data.get_body_xvelr("tip_frame").copy() * 0.1

    grip_pos2 = env.env.sim.data.get_body_xpos("tip_frame1").copy()
    grip_rot2 = env.env.sim.data.get_body_xquat('tip_frame1').copy()
    grip_rot2 = rotations.quat2euler(grip_rot2)  
    grip_velp2 = env.env.sim.data.get_body_xvelp("tip_frame1").copy() * 0.1
    grip_velr2 = env.env.sim.data.get_body_xvelr("tip_frame1").copy() * 0.1

    qpos_tem = env.env.sim.data.qpos[:31].copy()
    qvel_tem = env.env.sim.data.qvel[:30].copy()

    ob0 = np.concatenate(
        [
            qpos_tem[:7].copy(),
            qpos_tem[7:10].copy(),
            qvel_tem[:6].copy(),
            qvel_tem[6:9].copy(),
            grip_pos1,
            grip_velp1,
            goal_pos1, 
            goal_rot1
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
            goal_pos1,
            goal_rot1
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
            goal_pos2,
            goal_rot2
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
            goal_pos2,
            goal_rot2
        ]
    )

    obs_new = [ob0, ob1, ob2, ob3]

    eval_rnn_states = np.zeros((1),dtype=np.float32)
    eval_masks = np.ones((1), dtype=np.float32)

    for i in range(4):
        act = R_Actor(all_args,env2.observation_space[i],env.action_space[i])
        act.load_state_dict(torch.load("/home/zhaowenbo/Documents/MARL_SpaceRobot/RL_algorithms/Torch/MAPPO/onpolicy/scripts/results/SpaceRobotEnv/SpaceRobotFourArm/mappo/EightAgents/run1/models/actor_agent"+str(i)+".pt"))
        actors.append(act)
    for i in range(4,8):
        act = R_Actor(all_args,env.observation_space[i],env2.action_space[i])
        act.load_state_dict(torch.load("/home/zhaowenbo/Documents/MARL_SpaceRobot/RL_algorithms/Torch/MAPPO/onpolicy/scripts/results/SpaceRobotEnv/SpaceRobotBaseRot/mappo/EightAgents/run1/models/actor_agent"+str(i)+".pt"))
        actors.append(act)

    frames = []
    print("init goal:", env.env.goal)

    with torch.no_grad():
        # print(env.env.initial_gripper1_pos,env.env.initial_gripper1_rot,env.env.initial_gripper2_pos,env.env.initial_gripper2_rot)
        for eval_step in range(all_args.episode_length):
            print("step: ",eval_step)

            env.env.render()
            # img = env.env.render("rgb_array")

            curr_rot = env.env.sim.data.get_body_xquat('targetsat').copy()
            curr_rot = rotations.quat2euler(curr_rot)  # 3维欧拉角
            curr_rot1 = curr_rot + [0.0, 0.005, 0.00]  # 绕x,y,z轴旋转
            target_id = env.env.sim.model.body_name2id('targetsat')  # 设置target的位置
            env.env.sim.model.body_quat[target_id] = rotations.euler2quat(curr_rot1)

            # compute current goal
            goal_pos1 = np.array(env.env.sim.data.get_body_xpos("targethold_h3").copy())
            goal_rot1 = np.array(rotations.quat2euler(env.env.sim.data.get_body_xquat("targethold_h3").copy()))
            goal_pos2 = np.array(env.env.sim.data.get_body_xpos("targethold_h4").copy())
            goal_rot2 = np.array(rotations.quat2euler(env.env.sim.data.get_body_xquat("targethold_h4").copy()))

            site_id0 = env.env.sim.model.site_name2id("target0")
            env.env.sim.model.site_pos[site_id0] = goal_pos1.copy()
            env.env.sim.model.site_quat[site_id0] = rotations.euler2quat(goal_rot1.copy())
            site_id1 = env.env.sim.model.site_name2id("target1")
            env.env.sim.model.site_pos[site_id1] = goal_pos2.copy()
            env.env.sim.model.site_quat[site_id1] = rotations.euler2quat(goal_rot2.copy())


            # frames.append(img)
            action = []
            # if eval_step == 0 or eval_step == 10 or eval_step == 100:
            #     adv = Image.fromarray(np.uint8(img))
            #     adv.save(str(parent_dir) + "/render/mixed_goal/fig%d_mixed.jpg" %eval_step, quality = 100)
            # print("observation: ",np.array(list(obs[0,:])).reshape(1,25))
            for agent_id in range(4):
                actor = actors[agent_id]
                actor.eval()
                eval_action,_,rnn_states_actor = actor(
                    np.array(obs_new[agent_id]).reshape(1,31),
                    eval_rnn_states,
                    eval_masks,
                    deterministic=True,
                )
                eval_action = eval_action.detach().cpu().numpy()
                # print("step: ",eval_step,"action: ",eval_action)
                action.append(eval_action)
            for agent_id in range(4,8):
                actor = actors[agent_id]
                actor.eval()
                eval_action,_,rnn_states_actor = actor(
                    np.array(list(obs[agent_id,:])).reshape(1,25),
                    eval_rnn_states,
                    eval_masks,
                    deterministic=True,
                )
                eval_action = eval_action.detach().cpu().numpy()
                # print("step: ",eval_step,"action: ",eval_action)
                action.append(eval_action)

            obs, eval_rewards, done, infos = env.step(np.stack(action).squeeze().reshape(all_args.num_agents,3))

            # get state
            grip_pos1 = env.env.sim.data.get_body_xpos("tip_frame").copy()
            grip_rot1 = env.env.sim.data.get_body_xquat('tip_frame').copy()
            grip_rot1 = rotations.quat2euler(grip_rot1)  
            grip_velp1 = env.env.sim.data.get_body_xvelp("tip_frame").copy() * 0.1
            grip_velr1 = env.env.sim.data.get_body_xvelr("tip_frame").copy() * 0.1

            grip_pos2 = env.env.sim.data.get_body_xpos("tip_frame1").copy()
            grip_rot2 = env.env.sim.data.get_body_xquat('tip_frame1').copy()
            grip_rot2 = rotations.quat2euler(grip_rot2)  
            grip_velp2 = env.env.sim.data.get_body_xvelp("tip_frame1").copy() * 0.1
            grip_velr2 = env.env.sim.data.get_body_xvelr("tip_frame1").copy() * 0.1

            qpos_tem = env.env.sim.data.qpos[:31].copy()
            qvel_tem = env.env.sim.data.qvel[:30].copy()

            ob0 = np.concatenate(
                [
                    qpos_tem[:7].copy(),
                    qpos_tem[7:10].copy(),
                    qvel_tem[:6].copy(),
                    qvel_tem[6:9].copy(),
                    grip_pos1,
                    grip_velp1,
                    goal_pos1, 
                    goal_rot1
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
                    goal_pos1,
                    goal_rot1
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
                    goal_pos2,
                    goal_rot2
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
                    goal_pos2,
                    goal_rot2
                ]
            )

            obs_new = [ob0, ob1, ob2, ob3]

            # print("reward: ",eval_rewards)
            # print("action: ",np.stack(action).squeeze().reshape(all_args.num_agents,3))
            eval_episode_rewards.append(eval_rewards)
        print("episode reward: ",np.array(eval_episode_rewards).sum())
        
    print("goal:, ", env.env.goal)
    # np.save("fig_plot/error_mixed_goal1.npy",error)
    # print(error)

    # imageio.mimsave(
    #     str(parent_dir) + "/render/mixed_goal/render_mix4_tmp2.mp4",
    #     frames,
    # #     # duration=0.01,
    # )
        
        # writer = imageio.get_writer(parent_dir + "/render.gif")
        # # print('reward is {}'.format(self.reward_lst))
        # for frame, reward in zip(frames, eval_episode_rewards):
        #     print(eval_step)
        #     frame = Image.fromarray(frame)
        #     draw = ImageDraw.Draw(frame)
        #     draw.text((70, 70), '{}'.format(reward), fill=(255, 255, 255))
        #     frame = np.array(frame)
        #     writer.append_data(frame)
        # writer.close()
        # env.close()

if __name__ == "__main__":
    main(sys.argv[1:])