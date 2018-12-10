#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/RoboschoolHumanoid-v1.py --render \
            --num_rollouts 20
"""

import os
import argparse
import pickle
import tensorflow as tf
import numpy as np
import gym
import importlib


import gym, roboschool
# envname = params['env']
import numpy as np
class Environment():
    def __init__(self, envname):
        self.env = gym.make(envname)
        self.reset()

    def set_action(self, action):
        self.action = action

    def reset(self):
        self.action = None
        self.steps = 0
        self.observations = []
        self.actions = []
        self.rewards = []

    def generator(self):
        obs = self.env.reset()
        yield obs, 0
        done = False
        while not done:
            obs, reward, done, _ = self.env.step(self.action)
            self.observation = obs
            self.reward = reward
            self.actions.append(self.action)
            self.observations.append(self.observation)
            self.rewards.append(self.reward)
            self.action = None # reset action
            self.steps += 1

            yield obs, reward

def get_roboschool_policy(expert_policy_file):
    print('loading expert policy')
    module_name = expert_policy_file.replace('/', '.')
    if module_name.endswith('.py'):
        module_name = module_name[:-3]
    policy_module = importlib.import_module(module_name)
    print('loaded')
    _, policy = policy_module.get_env_and_policy()
    return policy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')

    args = parser.parse_args()
    policy = get_roboschool_policy(args.expert_policy_file)
    env = Environment(module_name.split('.')[-1])
    max_steps = args.max_timesteps or env.env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for i in range(args.num_rollouts):
        print('iter', i)
        for (obs, reward) in env.generator():
            act = policy.act(obs)
            env.set_action(act)
            observations.append(act)
            actions.append(act)
            if args.render:
                env.env.render()
            if env.steps % 100 == 0: print("%i/%i"%(env.steps, max_steps))
        returns.append(np.sum(env.rewards))

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    expert_data = {'observations': np.array(observations),
                    'actions': np.array(actions),
                    'returns': returns}

    with open(os.path.join('expert_data', args.envname + '.pkl'), 'wb') as f:
        pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
