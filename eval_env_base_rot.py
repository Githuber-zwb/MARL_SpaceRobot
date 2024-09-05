import gym
import imageio
from PIL import Image, ImageDraw
import SpaceRobotEnv
import numpy as np
import torch
import sys
import os
from gym import spaces
from PIL import Image 
import matplotlib.pyplot as plt
from gym.envs.robotics import rotations

# os.environ["MUJOCO_GL"] = "egl"

parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(parent_dir)
sys.path.append(parent_dir+"/RL_algorithms/Torch/MAPPO/onpolicy")
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from onpolicy.envs.spacerobot.SpaceRobotBaseRot_env import BaseRot
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

    # env.seed(all_args.seed)
    # seed
    # torch.manual_seed(all_args.seed)
    # torch.cuda.manual_seed_all(all_args.seed)
    # np.random.seed(all_args.seed)

    eval_episode_rewards = []
    actors = []
    obs = env.reset()

    eval_rnn_states = np.zeros((1),dtype=np.float32)
    eval_masks = np.ones((1), dtype=np.float32)

    for i in range(all_args.num_agents):
        act = R_Actor(all_args,env.observation_space[i],env.action_space[i])
        act.load_state_dict(torch.load("/home/zhaowenbo/Documents/MARL_SpaceRobot/RL_algorithms/Torch/MAPPO/onpolicy/scripts/results/SpaceRobotEnv/SpaceRobotBaseRot/mappo/EightAgents/run1/models/actor_agent"+str(i)+".pt"))
        actors.append(act)
        # print(act.act.action_out.logstd._bias)

    frames = []
    x = np.linspace(0, 19.9, 200)
    y = np.zeros(200)
    dist_time = [i for i in range(50,55)]
    print("init goal:", env.env.goal)

    with torch.no_grad():
        # print(env.env.initial_gripper1_pos,env.env.initial_gripper1_rot,env.env.initial_gripper2_pos,env.env.initial_gripper2_rot)
        for eval_step in range(all_args.episode_length):
            env_goal = env.env.goal
            print("step: ",eval_step)
            # env.env.render()
            img = env.env.render("rgb_array")
            frames.append(img)
            action = []
            # if eval_step == 0 or eval_step == 5 or eval_step == 10 or eval_step == 20:
            #     adv = Image.fromarray(np.uint8(img))
            #     adv.save(str(parent_dir) + "/render/base_reorien/fig%d.jpg" %eval_step, quality = 100)
            # print("observation: ",np.array(list(obs[0,:])).reshape(1,25))
            for agent_id in range(all_args.num_agents):
                actor = actors[agent_id]
                actor.eval()
                # print(actor.act.action_out.logstd._bias)
                # print("observation: ",np.array(list(obs[agent_id,:])).reshape(1,25))
                eval_action,_,rnn_states_actor = actor(
                    np.array(list(obs[agent_id,:])).reshape(1,25),
                    eval_rnn_states,
                    eval_masks,
                    deterministic=True,
                )
                eval_action = eval_action.detach().cpu().numpy()
                # print("step: ",eval_step,"action: ",eval_action)
                action.append(eval_action)

            ob_i = np.array(list(obs[0,:])).reshape(25,)
            error_i = goal_distance(ob_i[19:22], env_goal)
            y[eval_step] = error_i

            exert_action = np.stack(action).squeeze().reshape(all_args.num_agents,3)
            if eval_step in dist_time:  # disturbance
                exert_action = np.ones([8,3])
            # exert_action[6,:] = np.zeros(3)
            # exert_action[7,:] = np.zeros(3)
            obs, eval_rewards, done, infos = env.step(exert_action)

            # reset attitude of base 
            # if eval_step == 80:
            #     goal = np.array([ 0, 0, 0], dtype=np.float32)
            #     site_id = env.env.sim.model.site_name2id("targetbase")
            #     env.env.sim.model.site_pos[site_id] = np.array([0, 0, 4], dtype=np.float32)
            #     env.env.sim.model.site_quat[site_id] = rotations.euler2quat(goal.copy())
            #     env.env.goal = goal
            #     env.env.sim.forward()
            print("reward: ",eval_rewards)
            # print("action: ",np.stack(action).squeeze().reshape(all_args.num_agents,3))
            eval_episode_rewards.append(eval_rewards)
        print("episode reward: ",np.array(eval_episode_rewards).sum())
        
        # np.save('fig_plot/error_base_rot.npy', y)
        # plt.plot(x, y, label = "error" )
        # plt.legend()
        # plt.show()
        # plt.savefig('sim1.png')
        
    print("goal:, ", env.env.goal)
    imageio.mimsave(
        str(parent_dir) + "/render/base_reorien/render_base_rot_dist1.mp4",
        frames,
        # duration=0.01,
    )


if __name__ == "__main__":
    main(sys.argv[1:])