import numpy as np
import tensorflow as tf
import sacred
import pickle
ex = sacred.Experiment('behavior_cloning')
from sacred.observers import FileStorageObserver
ex.observers.append(FileStorageObserver.create('my_runs'))


class Environment():
    def __init__(self, envname):
        import gym, roboschool
        self.env = gym.make(envname)
        self.reset()
        self.dim_observations = len(self.env.observation_space.high)
        self.dim_actions = len(self.env.action_space.high)

    def set_action(self, action):
        self.action = action

    def reset(self):
        self.action = None
        self.steps = 0
        self.observations = []
        self.actions = []
        self.rewards = []

    def sleep_until_action_ready(self):
        while self.action is None:
            time.sleep(0.5)

    def generator(self):
        self.reset()
        obs = self.env.reset()
        yield [obs]
        done = False
        while not done:
            # self.sleep_until_action_ready()
            obs, reward, done, info = self.env.step(self.action)

            self.observation = obs
            self.reward = reward
            self.actions.append(self.action)
            self.observations.append(self.observation)
            self.rewards.append(self.reward)
            self.action = None # reset action
            self.steps += 1
            # print('OBS', obs, done, info)
            yield [obs]



def test_policy(params):
    estimator = get_estimator(params)
    env = Environment(params['env'][:-3])
    # policy = get_roboschool_policy('experts/'+params['env'])
    input_fn  = lambda: tf.data.Dataset.from_generator(
        env.generator, tf.float32, tf.TensorShape([1, env.dim_observations])
    ).make_one_shot_iterator().get_next()
    # with tf.Session() as sess:
    #     sess.run(iter.initializer)
    #     x = sess.run(input_fn())
    #     print(x.shape)
    rewards, observations, actions = [], [], []
    for rollout in range(params['num_test_rollouts']):
        for action in estimator.predict(input_fn):
            # env.set_action(policy.act(action['observations']))
            env.set_action(action['actions'])
        rewards.append(env.rewards.copy())
        observations.append(env.observations.copy())
        actions.append(env.actions.copy())

    rollouts = dict(rewards=rewards, actions=actions, observations=observations)

    with open(params['test_rollout_filename'], 'wb') as out_file:
        pickle.dump(rollouts, out_file)

    return rollouts

def get_dataset(params):
    actions, observations = read_data(params['data_path'])
    num_datapoints, dim_observations = np.shape(observations)
    dataset = tf.data.Dataset.from_tensor_slices((observations, actions))
    # dataset = dataset
    return dataset, num_datapoints

def get_estimator(params):
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params=params,
        model_dir=params['model_dir']
    )
    return estimator

def train(params):
    dataset, num_datapoints = get_dataset(params)
    train_size = int(0.9*num_datapoints)
    train, eval_set = dataset.take(train_size), dataset.skip(train_size)
    train = train.shuffle(num_datapoints).batch(params['batch_size'])
    train_input_fn = lambda: train.make_one_shot_iterator().get_next()
    eval_input_fn = lambda: eval_set.batch(1).repeat(1).make_one_shot_iterator().get_next()
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, throttle_secs=10, steps=None)
    estimator = get_estimator(params)

    for epoch in range(params['num_epochs']):
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def read_data(data_path):
    with open(data_path, 'rb') as raw_file:
        rollouts = pickle.load(raw_file)
    actions = rollouts['actions'] # targets
    observations = rollouts['observations'] # inputs/sensors
    return actions.astype('float32'), observations.astype('float32')

def model_fn(features, labels, mode, params):
    relu1 = tf.layers.Dense(params['relu1_size'], activation=tf.nn.relu, name='relu1')
    relu2 = tf.layers.Dense(params['relu2_size'], activation=tf.nn.relu, name='relu2')
    lin1 = tf.layers.Dense(params['actions_dim'], name='output')

    X = features
    r1 = relu1(X)
    r2 = relu2(r1)
    z = lin1(r2)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode,
            predictions={
                'actions': z,
                'observations': X
            }
        )
    l2 = tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables() ]) * params['lambda_l2']
    mse = tf.losses.mean_squared_error(labels, z)
    loss = l2 + mse

    metrics = {"mse": tf.metrics.mean_squared_error(labels, z)}

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics
        )


    assert mode == tf.estimator.ModeKeys.TRAIN
    global_step = tf.train.get_global_step()

    learning_rate = params['learning_rate']

    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('mse', mse)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    train_op = optimizer.minimize(
        loss,
        global_step=global_step
    )
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


@ex.config
def get_params():
    relu1_size = 128
    relu2_size = 64
    learning_rate = 0.1
    batch_size = 10
    env = 'RoboschoolHalfCheetah-v1.py'
    num_epochs = 25
    actions_dim = 6
    model_dir = 'models/half-cheetah'
    data_path = 'expert_data/RoboschoolHalfCheetah-v1.py-100.pkl'
    lambda_l2 = 0
    num_test_rollouts = 100
    test_rollout_filename = '%s/test_rollouts.pkl' % model_dir
    final_comparison_histpath = '%s/expert_rewards_comparsion.png' % model_dir

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
    relu1_size = 128
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
def train_command(_config):
    params = _config
    train(params)

@ex.command(unobserved=True)
def analyze(_config):
    params = _config
    with open(params['test_rollout_filename'], 'rb') as open_file:
        clone_rollouts = pickle.load(open_file)
    clone_total_rewards = [np.sum(rollout) for rollout in clone_rollouts['rewards']]
    with open(params['data_path'], 'rb') as open_file:
        expert_rollouts = pickle.load(open_file)
    expert_total_rewards = expert_rollouts['returns']


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
    test_policy(params)
    ex.add_artifact(_config['test_rollout_filename'])
    analyze(_config)
    ex.add_artifact(_config['final_comparison_histpath'])

