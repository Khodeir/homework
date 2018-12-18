from train_pg_f18 import *
from scipy.stats import norm

import numpy as np

def get_agent_from_params(params):
  # if not(os.path.exists('data')):
  #     os.makedirs('data')
  return get_agent(
    exp_name=params.get('exp_name', None),
    env_name=params.get('env_name', None),
    n_iter=params.get('n_iter', None),
    gamma=params.get('discount', None),
    min_timesteps_per_batch=params.get('batch_size', None),
    max_path_length=params.get('ep_len', None),
    learning_rate=params.get('learning_rate', None),
    reward_to_go=params.get('reward_to_go', None),
    animate=params.get('render', None),
    logdir=params.get('logdir', None),
    normalize_advantages=not(params.get('dont_normalize_advantages', None)),
    nn_baseline=params.get('nn_baseline,', None),
    seed=params.get('seed', None),
    n_layers=params.get('n_layers', None),
    size=params.get('size', None)
  )
def get_params(batch_size=1000, learning_rate=5e-3):
  params = {}
  params['exp_name'] = 'hc_b%d_r%.2f' % (batch_size, learning_rate)
  params['env_name'] = 'InvertedPendulum-v2'

  logdir = params.get('exp_name') + '_' + params.get('env_name') + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
  logdir = os.path.join('exp', logdir)
  params['logdir'] = logdir
  # params['exp_name'] = None
  params['render'] = False
  params['discount'] = 0.9
  params['n_iter'] = 100

  params['batch_size'] = int(batch_size)

  params['ep_len'] = 1000
  params['learning_rate'] = float(learning_rate)
  params['reward_to_go'] = True
  params['dont_normalize_advantages'] = False
  params['nn_baseline'] = False
  params['seed'] = 1
  params['n_layers'] = 2
  params['size'] = 64
  return params

def get_agent(
        exp_name,
        env_name,
        n_iter, 
        gamma, 
        min_timesteps_per_batch, 
        max_path_length,
        learning_rate, 
        reward_to_go, 
        animate, 
        logdir, 
        normalize_advantages,
        nn_baseline, 
        seed,
        n_layers,
        size):

    start = time.time()

    #========================================================================================#
    # Set Up Logger
    #========================================================================================#
    setup_logger(logdir, locals())

    #========================================================================================#
    # Set Up Env
    #========================================================================================#

    # Make the gym environment
    print('making env')
    env = gym.make(env_name)
    print('hello')

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    # Is this env continuous, or self.discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    #========================================================================================#
    # Initialize Agent
    #========================================================================================#
    computation_graph_args = {
        'n_layers': n_layers,
        'ob_dim': ob_dim,
        'ac_dim': ac_dim,
        'discrete': discrete,
        'size': size,
        'learning_rate': learning_rate,
        }

    sample_trajectory_args = {
        'animate': animate,
        'max_path_length': max_path_length,
        'min_timesteps_per_batch': min_timesteps_per_batch,
    }

    estimate_return_args = {
        'gamma': gamma,
        'reward_to_go': reward_to_go,
        'nn_baseline': nn_baseline,
        'normalize_advantages': normalize_advantages,
    }

    agent = Agent(computation_graph_args, sample_trajectory_args, estimate_return_args)

    # build computation graph
    agent.build_computation_graph()
    # tensorflow: config, session, variable initialization
    agent.init_tf_sess()
    return agent


def test_logprob():
	params = get_params()
	agent = get_agent_from_params(params)

	sample_observations = [[0.1, 0.2, 0.1, 0.2]]*100000

	mu, sigma = agent.sess.run(agent.policy_parameters, feed_dict={agent.sy_ob_no: sample_observations[:1]})
	sigma = np.exp(sigma)

	sample_actions = agent.sess.run(agent.sy_sampled_ac, feed_dict={agent.sy_ob_no: sample_observations})
	sample_logprobs = agent.sess.run(agent.sy_logprob_n, feed_dict={agent.sy_ob_no: sample_observations, agent.sy_ac_na: sample_actions})

	print(mu, sigma)

	comparison_logprobs = norm.logpdf(sample_actions, loc=mu[0][0], scale=sigma[0])

	print(np.mean(np.square((comparison_logprobs - sample_logprobs))))

test_logprob()

