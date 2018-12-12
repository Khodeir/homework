import seaborn as sns
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

expert = pickle.load(open('expert_data/RoboschoolHalfCheetah-v1.py-100.pkl', 'rb'))
rollouts = [pickle.load(open('models/half_cheetah-0.%d/test_rollouts.pkl' % d, 'rb')) for d in range(1, 10, 2)]
rollout_returns  = [[np.sum(rollout) for rollout in rollout['rewards']] for rollout in rollouts]


# imgs = ['models/half_cheetah-0.%d/expert_rewards_comparsion.png' % d for d in range(1, 10, 2)]
# imgs = [plt.imread(img) for img in imgs]
# plt.figure(figsize=(12,6))
# hist = sns.distplot(expert['returns'], label='expert')
# for clone, d in zip(rollout_returns, range(1, 10, 2)):
#     hist = sns.distplot(clone, label='0.%d' % d)
# figure = hist.get_figure()
# figure.legend()
# figure.savefig('out.png', dpi=400)

df = pd.concat([
    pd.DataFrame(dict(rewards=clone, model='0.%d' % d)) for clone, d in zip(rollout_returns, range(1, 10, 2))
] + [
    pd.DataFrame(dict(rewards=expert['returns'], model='expert')),
]
)

plt.figure()
sns.boxplot(x='model', y='rewards', data=df)
plt.show()