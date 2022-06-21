#!/usr/bin/env python3

from dqn import DQN
from ppo import PPO
from ppo_discrete import PPO_Discrete

import numpy as np
import torch as th
import random
import argparse
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default=None, type=str)
parser.add_argument('--port', default=None, type=int)

parser.add_argument('--sim_device', type=str, default="cuda:0", help='Physics Device in PyTorch-like syntax')
parser.add_argument('--compute_device_id', default=0, type=int)
parser.add_argument('--graphics_device_id', type=int, default=0, help='Graphics Device ID')
parser.add_argument('--num_envs', default=512, type=int)
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

try:
    with tqdm(range(8192)) as pbar:
        for step in pbar:
            policy.run()
            #episode_rewards = policy.run()
            #if len(episode_rewards) > 0:
            #    avg_rew = th.mean(episode_rewards).detach().cpu().numpy()
            #    pbar.set_description(F'avg.ep.rew = {avg_rew}')
finally:
    if isinstance(policy, PPO_Discrete):
        policy.save('/tmp/cartpole-policy.pt')
