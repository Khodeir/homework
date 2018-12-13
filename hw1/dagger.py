import numpy as np
import tensorflow as tf
import sacred
import pickle
from core import *
ex = sacred.Experiment('dagger')
from sacred.observers import FileStorageObserver
ex.observers.append(FileStorageObserver.create('dagger_runs'))
from shutil import copyfile, move, copytree, rmtree


@ex.config
def humanoid():
    use_data_prop = 1.
    learning_rate = 0.1
    batch_size = 10
    lambda_l2 = 0
    model_dir = 'models/humanoid-dagger'

    test_rollout_filename = '%s/test_rollouts.pkl' % model_dir
    final_comparison_histpath = '%s/expert_rewards_comparsion.png' % model_dir
    exported_model_base = '%s/exports/' % model_dir
    should_render = False
    relu1_size = 256
    relu2_size = 128
    env = 'RoboschoolHumanoid-v1.py'
    actions_dim = 17
    initial_rollout = 'expert_data/RoboschoolHumanoid-v1.py-10.pkl'
    data_path = 'dagger_data/RoboschoolHumanoid-v1.py-10.pkl'
    num_test_rollouts = 10
    num_episodes = 100
    num_steps = 80000 # for training
    start_from_scratch = True

def aggregate_datasets(params):
    actions, observations = read_data(params['data_path'])
    with open(params['test_rollout_filename'], 'rb') as open_file:
        rollouts_new = pickle.load(open_file)

    for actions_new, observations_new in zip(rollouts_new['actions'], rollouts_new['observations']):
        actions = np.concatenate([actions, actions_new], axis=0)
        observations = np.concatenate([observations, observations_new], axis=0)
    assert len(actions) == len(observations), 'Actions has length %d, observations has length %d' % (len(actions), len(observations))
    with open(params['data_path'], 'wb') as f:
        pickle.dump(dict(actions=actions, observations=observations), f)

def analyze(params, episode_num):
    with open(params['test_rollout_filename'], 'rb') as open_file:
        clone_rollouts = pickle.load(open_file)
    clone_total_rewards = [np.sum(rollout) for rollout in clone_rollouts['rewards']]
    return dict(mean_reward=np.mean(clone_total_rewards), std_reward=np.std(clone_total_rewards))


def test_policy_estimator_dagger(params):
    estimator = get_estimator(params)
    env = Environment(params['env'][:-3], should_render=params['should_render'])
    expert_policy = get_roboschool_policy('experts/'+params['env'])
    input_fn  = lambda: tf.data.Dataset.from_generator(
        env.generator, tf.float32, tf.TensorShape([1, env.dim_observations])
    ).make_one_shot_iterator().get_next()
    # with tf.Session() as sess:
    #     sess.run(iter.initializer)
    #     x = sess.run(input_fn())
    #     print(x.shape)
    rewards, observations, actions = [], [], []
    expert_actions = []
    for rollout in range(params['num_test_rollouts']):
        expert_actions_current = []
        for action in estimator.predict(input_fn):
            expert_actions_current.append(expert_policy.act(action['observations']))
            env.set_action(action['actions'])
        expert_actions.append(expert_actions_current)
        rewards.append(env.rewards.copy())
        observations.append(env.observations.copy())
        actions.append(env.actions.copy())

    rollouts = dict(rewards=rewards, actions=expert_actions, observations=observations)

    with open(params['test_rollout_filename'], 'wb') as out_file:
        pickle.dump(rollouts, out_file)

    return rollouts

@ex.automain
def train_dagger(_config, _run):
    params = _config
    copyfile(params['initial_rollout'], params['data_path'])
    for episode in range(params['num_episodes']):
        train(params)
        test_policy_estimator_dagger(params)
        metrics = analyze(params, episode)
        for key in metrics:
            _run.log_scalar(key, metrics[key], episode)
        aggregate_datasets(params)
        ex.add_artifact(params['test_rollout_filename'], 'rollouts %d.pkl' % episode)
        if params['start_from_scratch']:
            rmtree(params['model_dir'])
        # copyfile(params['test_rollout_filename'], '%s/%s.%d' % (params['model_dir'], params['test_rollout_filename'], episode))
        # copytree(params['model_dir'],  params['model_dir'] + '.%s' % episode)

