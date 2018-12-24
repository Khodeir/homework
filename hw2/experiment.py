
from multiprocessing import Pool
from plot import get_datasets

from subprocess import check_output
import time
import os
import IPython
from glob import glob
import numpy as np
import pickle
def train_func(batch_size, learning_rate):
  exp_name = 'hc_b%d_r%.2f' % (batch_size, learning_rate)
  command  = 'python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b %d -lr %.2f -rtg --exp_name %s' % (batch_size, learning_rate, exp_name)
  print(command)
  check_output(command, shell=True)
  fpath = glob('data/%s*' % exp_name)[-1]
  results = get_datasets(fpath)
  return -np.mean([result['AverageReturn'][-5:].mean() for result in results])




import sys
import os
def mute():
  sys.stdout = open(os.devnull, 'w')

def f(x):
  (batch_size, learning_rate) = x
  return train_func(batch_size, learning_rate)

def skopt_main():
  from skopt import Optimizer, dump, load, Space
  from skopt.learning import GaussianProcessRegressor
  from skopt.space import Real, Integer
  fname = 'optimizer-exp-pendulum-4.pkl'
  dims = [Integer(15, 500), Real(0.025, 0.1, prior="log-uniform")]
  try:
    optimizer = load(fname)
    optimizer.space = Space(dims)
  except:
    optimizer = Optimizer(
      dimensions=dims,
      random_state=1
    )
  n_jobs = 2
  for i in range(3): 
    pool = Pool(n_jobs, initializer=mute)
    x = optimizer.ask(n_points=n_jobs)  # x is a list of n_points points
    print(x)
    y = pool.map(f, x)
    pool.close()
    optimizer.tell(x, y)
    print('Iteration %d. Best yi %.2f' % (i, min(optimizer.yi)))

  dump(optimizer, fname)

def read_results(fname):
  with open(fname, 'rb') as open_file:
    results = pickle.load(open_file)
  return results

def manual_main():
  fname = 'manual-exp-pend.pkl'
  try:
    results = read_results(fname)
  except:
    results = []

  params = [
    # (10000, 0.1),
    # (10000, 0.01),
    # (1000, 0.01),
    # (500, 0.01),
    # (500, 0.001),
    # (100, 0.001),
    # (500, 0.02),
    # (500, 0.04),
    # (100, 0.02),
    # (100, 0.04)
    # (750, 0.02),
    # (750, 0.04),
    # (250, 0.02),
    # (250, 0.04),
    # (750, 0.025),
    # (500, 0.025),
    # (750, 0.03),
    # (500, 0.03),
    # (750, 0.01),
    # (750, 0.015),
    # (1200, 0.01),
    # (1400, 0.01),
    # (2000, 0.01),
    # (4000, 0.01),
    (10000, 0.02),
    (10000, 0.04),
  ]
  n_jobs = 2
  pool = Pool(n_jobs)
  for i in range(0, len(params) - n_jobs + 1, n_jobs):
    jobs = params[i:i+n_jobs]

    y = pool.map(f, jobs)
    for xi, yi in zip(jobs, y):
      results.append((yi, xi))

    print('Iteration %d. Best yi %s' % (i, str(min(results))))

  pool.close()
  with open(fname, 'wb') as open_file:
    results = pickle.dump(results, open_file)

def show_results():
  results = read_results('manual-exp-pend.pkl')
  for p in sorted(results):
    print(p)
  # return
  # print(sorted(results))

if __name__ == '__main__':
  manual_main()
  # show_results()