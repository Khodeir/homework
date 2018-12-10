import numpy as np
import tensorflow as tf
import sacred
import pickle
from sacred.stflow import LogFileWriter
ex = sacred.Experiment('behavior_cloning')

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

@LogFileWriter(ex)
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
    relu2_size = 128
    learning_rate = 0.1
    batch_size = 10
    max_epochs = 10
    env = 'RoboschoolHalfCheetah-v1.py'
    num_epochs = 50
    actions_dim = 6
    model_dir = 'models/half-cheetah'
    lambda_l2 = 0
    data_path = 'expert_data/RoboschoolHalfCheetah-v1.py.pkl'


@ex.named_config
def half_cheetah():
    relu1_size = 128
    relu2_size = 128
    learning_rate = 0.1
    batch_size = 10
    max_epochs = 10
    env = 'RoboschoolHalfCheetah-v1.py'
    num_epochs = 50
    actions_dim = 6
    model_dir = 'models/half-cheetah'
    lambda_l2 = 0
    data_path = 'expert_data/RoboschoolHalfCheetah-v1.py.pkl'

@ex.named_config
def ant():
    relu1_size = 128
    relu2_size = 128
    learning_rate = 0.1
    batch_size = 10
    max_epochs = 10
    env = 'RoboschoolAnt-v1.py'
    num_epochs = 100
    actions_dim = 8
    model_dir = 'models/ant'
    lambda_l2 = 0
    data_path = 'expert_data/RoboschoolAnt-v1.py.pkl'

@ex.automain
def main(_config):
    params = _config
    train(params)

