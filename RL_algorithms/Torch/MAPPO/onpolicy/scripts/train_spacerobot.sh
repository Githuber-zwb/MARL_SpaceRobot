#!/bin/sh
# exp param
env="SpaceRobotEnv"
scenario="SpaceRobotDualArmWithRot"
algo="mappo" # "rmappo" "ippo"
exp="OneArm"

# SpaceRobot param
# num_agents=4
num_agents=1

# train param
num_env_steps=20000000 #20M
episode_length=50

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, num agents is ${num_agents}"

CUDA_VISIBLE_DEVICES=0 python RL_algorithms/Torch/MAPPO/onpolicy/scripts/train/train_spacerobot.py \
--env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--n_rollout_threads 128  --use_wandb --share_policy --share_reward --hidden_size 256 --layer_N 2 --ppo_epoch 5 \
--save_interval 200 --log_interval 5 --use_valuenorm --use_popart --gamma 0.96 --use_policy_active_masks --use_value_active_masks \
--entropy_coef 0.0 --lr 1e-3 --critic_lr 1e-3 \
# --use_valuenorm \
# --use_popart
# --lr 1e-3 --critic_lr 1e-3 \
# --use_feature_normalization --use_orthogonal \
