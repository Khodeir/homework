from run_dqn_atari import *

def main():
    # Get Atari games.
    task = gym.make('PongNoFrameskip-v4')

    # Run training
    seed = 1500
    print('random seed = %d' % seed)
    env = get_env(task, seed)
    session = get_session()
  	
    atari_learn(env, session, num_timesteps=4 * 5e6, double_q=False, rew_file='p1q1.pkl')


if __name__ == '__main__':
	main()