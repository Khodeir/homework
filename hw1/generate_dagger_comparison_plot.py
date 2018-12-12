import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

import pickle

import numpy as np

data = []
rollout_data = sorted([(int(path.replace('dagger_runs/5/rollouts ', '')), path) for path in glob('dagger_runs/5/rollouts*')])
for (step, rollout_path) in rollout_data:
    with open(rollout_path, 'rb') as open_file:
        rollout = pickle.load(open_file)
    for reward in rollout['rewards']:
        data.append(dict(step=step, reward=np.sum(reward)))

max_step = rollout_data[-1][0]

with open('expert_data/RoboschoolHumanoid-v1.py-100.pkl', 'rb') as expert_file:
    expert_rollouts = pickle.load(expert_file)
expert_data = []
for i in range (max_step):
    for reward in expert_rollouts['returns']:
        expert_data.append(dict(step=i, reward=reward))
expert_df = pd.DataFrame(expert_data)

with open('models/humanoid/test_rollouts.pkl', 'rb') as clone_file:
    bc_agent_rollouts = pickle.load(clone_file)
bc_agent_data = []
for i in range (max_step):
    
    for reward in bc_agent_rollouts['rewards']:
        bc_agent_data.append(dict(step=i, reward=np.sum(reward)))
bc_agent_df = pd.DataFrame(bc_agent_data)

df = pd.DataFrame(data)

plt.figure(figsize=(10,6))
sns.lineplot(data=df, x='step', y='reward', ci='sd', label='dagger')
sns.lineplot(data=expert_df, x='step', y='reward', ci='sd', label='expert')
sns.lineplot(data=bc_agent_df, x='step', y='reward', ci='sd', label='bc agent')
plt.savefig('3.2 - Dagger Comparison.png')