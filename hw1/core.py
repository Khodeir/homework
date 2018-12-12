import numpy as np
import tensorflow as tf
import sacred
import pickle
import importlib
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
        self.observations.append(obs)
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
        return self.sess.run(action, feed_dict={"observation:0": observation})[0]

def get_roboschool_policy(expert_policy_file):
    print('loading expert policy')
    module_name = expert_policy_file.replace('/', '.')
    if module_name.endswith('.py'):
        module_name = module_name[:-3]
    policy_module = importlib.import_module(module_name)
    print('loaded')
    _, policy = policy_module.get_env_and_policy()
    return policy

def test_policy_fn(params, policy=None):
    if policy is None:
        policy = ExportedModelPolicy(exported_model_base=params['exported_model_base'])
    env = Environment(params['env'][:-3], should_render=params['should_render'])

    rewards, observations, actions = [], [], []
    for rollout in range(params['num_test_rollouts']):
        for observation in env.generator():
            # env.set_action(policy.act(action['observations']))
            action = policy.act(observation)
            env.set_action(action)

        rewards.append(env.rewards.copy())
        observations.append(env.observations.copy())
        actions.append(env.actions.copy())

    rollouts = dict(rewards=rewards, actions=actions, observations=observations)

    with open(params['test_rollout_filename'], 'wb') as out_file:
        pickle.dump(rollouts, out_file)

    return rollouts


def test_policy_estimator(params):
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
    dataset = dataset.shuffle(num_datapoints).batch(params['batch_size']).repeat()
    train_input_fn = lambda: dataset.make_one_shot_iterator().get_next()
    estimator = get_estimator(params)
    num_steps = int(num_datapoints * params['num_epochs'] / params['batch_size']) if params['num_steps'] is None else params['num_steps']
    estimator.train(train_input_fn, steps=num_steps)


def read_data(data_path):
    with open(data_path, 'rb') as raw_file:
        rollouts = pickle.load(raw_file)
    actions = rollouts['actions'] # targets
    observations = rollouts['observations'] # inputs/sensors
    actions = np.array(actions).astype('float32')
    observations = np.array(observations).astype('float32')
    return actions, observations

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
