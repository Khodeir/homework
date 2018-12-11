import numpy as np
import tensorflow as tf
import sacred
import pickle
ex = sacred.Experiment('behavior_cloning')
from sacred.observers import FileStorageObserver
ex.observers.append(FileStorageObserver.create('my_runs'))

class Environment():
    def __init__(self, envname, should_render=False):
        import gym, roboschool
        self.env = gym.make(envname)
        self.should_render = should_render
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
            if self.should_render:
                self.env.render()
            # self.sleep_until_action_ready()
            obs, reward, done, info = self.env.step(self.action)

            self.observation = obs
            self.reward = reward
            self.actions.append(self.action)
            self.observations.append(self.observation)
            self.rewards.append(self.reward)
            self.action = None # reset action
            self.steps += 1
            yield [obs]


class ExportedModelPolicy():
    def __init__(self, exported_model_base):
        self.sess = tf.Session()
        tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], exported_model_base)
    def act(self, observation):
        action = self.sess.graph.get_tensor_by_name('final_output:0')
        return self.sess.run(action, feed_dict={"observation:0": observation})


def test_exported_policy(params):
    policy = ExportedModelPolicy(exported_model_base=params['exported_model_base'])
    env = Environment(params['env'][:-3], should_render=params['should_render'])

    rewards, observations, actions = [], [], []
    for rollout in range(params['num_test_rollouts']):
        for observation in env.generator():
            # env.set_action(policy.act(action['observations']))
            action = policy.act(observation)
            env.set_action(action[0])

        rewards.append(env.rewards.copy())
        observations.append(env.observations.copy())
        actions.append(env.actions.copy())

    rollouts = dict(rewards=rewards, actions=actions, observations=observations)

    with open(params['test_rollout_filename'], 'wb') as out_file:
        pickle.dump(rollouts, out_file)

    return rollouts


def test_policy(params):
    estimator = get_estimator(params)
    env = Environment(params['env'][:-3], should_render=params['should_render'])
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
    datapoints_to_use = int(params['use_data_prop'] * num_datapoints)
    dataset = dataset.take(datapoints_to_use)
    # dataset = dataset
    return dataset, datapoints_to_use

def get_estimator(params):
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params=params,
        model_dir=params['model_dir']
    )
    return estimator

def train_and_evaluate(params):
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

def train(params):
    dataset, num_datapoints = get_dataset(params)
    dataset = dataset.shuffle(num_datapoints).batch(params['batch_size'])
    train_input_fn = lambda: dataset.make_one_shot_iterator().get_next()
    estimator = get_estimator(params)
    num_steps = num_datapoints * params['num_epochs']
    estimator.train(train_input_fn, steps=num_steps)

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

    Y = tf.identity(z, name='final_output')

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode,
            predictions={
                'actions': Y,
                'observations': X
            }
        )
    l2 = tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables() ]) * params['lambda_l2']
    mse = tf.losses.mean_squared_error(labels, Y)
    loss = l2 + mse

    metrics = {"mse": tf.metrics.mean_squared_error(labels, Y)}

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
    use_data_prop = 1.
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
def test_exported_policy_command(_config):
    params = _config
    test_exported_policy(params)
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
    test_policy(params)
    ex.add_artifact(_config['test_rollout_filename'])
    analyze(_config)
    ex.add_artifact(_config['final_comparison_histpath'])

