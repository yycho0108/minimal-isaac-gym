from dqn import DQN
from ppo import PPO
from ppo_discrete import PPO_Discrete

import numpy as np
import torch as th
import cv2
from torch.distributions import Categorical
import random
import argparse
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default=None, type=str)
parser.add_argument('--port', default=None, type=int)

parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
parser.add_argument('--compute_device_id', default=0, type=int)
parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics Device ID')
parser.add_argument('--num_envs', default=4, type=int)
parser.add_argument('--headless', action='store_true', default=True)
parser.add_argument('--method', default='ppo', type=str)

args = parser.parse_args()
args.headless = True

th.manual_seed(0)
random.seed(0)

if args.method == 'ppo':
    policy = PPO(args)
elif args.method == 'ppo_d':
    policy = PPO_Discrete(args)
elif args.method == 'dqn':
    policy = DQN(args)

policy.load('/tmp/cartpole-policy.pt')
policy.net.eval()

# Evaluation...? With render()
env=policy.env
env.reset()
for step in range(512):
    # Get observation.
    # TODO(ycho): Figure out why clone?
    # Rather than e.g. detach
    obs = env.obs_buf.clone()

    # Get action.
    with th.no_grad():
        action_probs = policy.net.pi(obs)
        dist = Categorical(action_probs)
        action = dist.sample()
        # map action to -1 to 1
        action = 2 * (action / (policy.act_space - 1) - 0.5)  

    # Step.
    env.step(action)
    # next_obs, reward, done = env.obs_buf.clone(), env.reward_buf.clone(), env.reset_buf.clone()
    env.reset()

    imgs = env.render(mode='rgb_array')
    #print(imgs[0].min(axis=(0,1)), imgs[0].max(axis=(0,1)),
    #        imgs[0].dtype, imgs[0].shape)
    for env_id, img in enumerate(imgs):
        cv2.imwrite(F'/tmp/imgs/{env_id:01d}-{step:03d}.png', img[..., :3])
