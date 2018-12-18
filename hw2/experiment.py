from skopt import Optimizer
from skopt.learning import GaussianProcessRegressor
from skopt.space import Real, Integer
from multiprocessing import Pool
from plot import get_datasets
from train_pg_f18 import train_PG
import time
import os
import mujoco_py
import IPython
def train_func(args):
  # if not(os.path.exists('data')):
  #     os.makedirs('data')

  # print(args.get('ep_len', None))
  train_PG(
    exp_name=args.get('exp_name', None),
    env_name=args.get('env_name', None),
    n_iter=args.get('n_iter', None),
    gamma=args.get('discount', None),
    min_timesteps_per_batch=args.get('batch_size', None),
    max_path_length=args.get('ep_len', None),
    learning_rate=args.get('learning_rate', None),
    reward_to_go=args.get('reward_to_go', None),
    animate=args.get('render', None),
    logdir=args.get('logdir', None),
    normalize_advantages=not(args.get('dont_normalize_advantages', None)),
    nn_baseline=args.get('nn_baseline,', None),
    seed=args.get('seed', None),
    n_layers=args.get('n_layers', None),
    size=args.get('size', None)
  )
  results = get_datasets(args.get('logdir'))
  return results[0]['MaxReturn'][-10:].mean()


def get_params(batch_size=250, learning_rate=5e-3):
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





import sys
import os
def mute():
  sys.stdout = open(os.devnull, 'w') 

def f(x):
  (batch_size, learning_rate) = x
  params = get_params(batch_size, learning_rate)
  try:
    return train_func(params)
  except:
    return 0

optimizer = Optimizer(
  dimensions=[Integer(15, 1000), Real(0.00001, 1.0)],
  random_state=1
)
n_jobs = 4
for i in range(10): 
  pool = Pool(n_jobs, initializer=mute)
  x = optimizer.ask(n_points=n_jobs)  # x is a list of n_points points
  # y = Parallel()(delayed(branin)(v) for v in x)  # evaluate points in parallel
  y = pool.map(f, x)
  pool.close()
  optimizer.tell(x, y)
# print(x)
# print(y)
IPython.embed()
  # p = Process(target=f, args=x)
  # p.start()
  # processes.append(p)

  # for p in processes:
  #     p.join()



