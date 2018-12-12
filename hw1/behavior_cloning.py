import numpy as np
import tensorflow as tf
import sacred
import pickle
from core import *
ex = sacred.Experiment('behavior_cloning')
from sacred.observers import FileStorageObserver
ex.observers.append(FileStorageObserver.create('my_runs'))

@ex.config
def get_params():
    use_data_prop = 1.
    relu1_size = 128
    relu2_size = 64
    learning_rate = 0.1
    batch_size = 10
    env = 'RoboschoolHalfCheetah-v1.py'
    num_epochs = 25
    num_steps = 80000
    actions_dim = 6
    model_dir = 'models/half-cheetah'
    data_path = 'expert_data/RoboschoolHalfCheetah-v1.py-100.pkl'
    lambda_l2 = 0
    num_test_rollouts = 200
    test_rollout_filename = '%s/test_rollouts.pkl' % model_dir
    final_comparison_histpath = '%s/expert_rewards_comparsion.png' % model_dir
    exported_model_base = '%s/exports/' % model_dir
    should_render = False

@ex.named_config
def half_cheetah():
    env = 'RoboschoolHalfCheetah-v1.py'
    num_epochs = 25
    actions_dim = 6
    model_dir = 'models/half-cheetah'
    data_path = 'expert_data/RoboschoolHalfCheetah-v1.py-100.pkl'

@ex.named_config
def ant():
    env = 'RoboschoolAnt-v1.py'
    num_epochs = 25
    actions_dim = 8
    model_dir = 'models/ant'
    data_path = 'expert_data/RoboschoolAnt-v1.py-100.pkl'

@ex.named_config
def hopper():
    env = 'RoboschoolHopper-v1.py'
    num_epochs = 25
    actions_dim = 3
    model_dir = 'models/hopper'
    data_path = 'expert_data/RoboschoolHopper-v1.py-100.pkl'

@ex.named_config
def humanoid():
    relu1_size = 256
    relu2_size = 128
    env = 'RoboschoolHumanoid-v1.py'
    num_epochs = 25
    actions_dim = 17
    model_dir = 'models/humanoid'
    data_path = 'expert_data/RoboschoolHumanoid-v1.py-100.pkl'

@ex.named_config
def reacher():
    env = 'RoboschoolReacher-v1.py'
    num_epochs = 25
    actions_dim = 2
    model_dir = 'models/reacher'
    data_path = 'expert_data/RoboschoolReacher-v1.py-100.pkl'

@ex.named_config
def walker():
    env = 'RoboschoolWalker2d-v1.py'
    num_epochs = 25
    actions_dim = 6
    model_dir = 'models/walker'
    data_path = 'expert_data/RoboschoolWalker2d-v1.py-100.pkl'

@ex.command
def test_policy_command(_config):
    params = _config
    test_policy(params)
    ex.add_artifact(_config['test_rollout_filename'])

@ex.command
def test_exported_policy_command(_config):
    params = _config
    test_policy_fn(params)
    ex.add_artifact(_config['test_rollout_filename'])

@ex.command
def train_command(_config):
    params = _config
    train(params)

@ex.command(unobserved=True)
def export_model(_config):
    params = _config
    estimator = get_estimator(params)
    def serving_input_receiver_fn():
        feat = tf.placeholder(dtype=tf.float32, shape=(1, 44), name='observation')
        return tf.estimator.export.TensorServingInputReceiver(features=feat, receiver_tensors=feat)

    estimator.export_savedmodel(
        params['exported_model_base'],
        serving_input_receiver_fn
    )

@ex.command(unobserved=True)
def analyze(_config):
    params = _config
    with open(params['test_rollout_filename'], 'rb') as open_file:
        clone_rollouts = pickle.load(open_file)
    clone_total_rewards = [np.sum(rollout) for rollout in clone_rollouts['rewards']]
    with open(params['data_path'], 'rb') as open_file:
        expert_rollouts = pickle.load(open_file)
    expert_total_rewards = expert_rollouts['returns']

    print(clone_total_rewards)
    print('Expert Rewards: MEAN=%.2f, STD=%.2f' % (np.mean(expert_total_rewards), np.std(expert_total_rewards)))
    print('Clone Rewards: MEAN=%.2f, STD=%.2f' % (np.mean(clone_total_rewards), np.std(clone_total_rewards)))
    import seaborn as sns
    hist = sns.distplot(expert_total_rewards, label='expert')
    hist = sns.distplot(clone_total_rewards, label='clone')
    figure = hist.get_figure()
    figure.legend()
    figure.savefig(params['final_comparison_histpath'], dpi=400)
    # plt.hist(clone_total_rewards, name='clone')
    # plt.savefig('example.png')

@ex.automain
def main(_config):
    params = _config
    train(params)
    test_policy_estimator(params)
    ex.add_artifact(_config['test_rollout_filename'])
    analyze(_config)
    ex.add_artifact(_config['final_comparison_histpath'])

