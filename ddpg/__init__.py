import logging


def initialize_ddpg(options):
    # Heavy imports here to make `main.py --help` fast somewhere upstream
    import gym
    import tensorflow as tf
    from ddpg.setup import Setup

    # Hard-coded choice of environment, for now
    GYM_ENV = 'Pendulum-v0'
    env = gym.make(GYM_ENV)

    logging.debug("Using environment '{}'".format(GYM_ENV))
    logging.debug("Action space: {}".format(env.action_space))
    logging.debug("Observation space: {}".format(env.observation_space))

    sess = tf.InteractiveSession()
    setup = Setup(env, sess, options)

    return env, sess, setup


def run_ddpg(options):
    env, sess, setup = initialize_ddpg(options)

    try:
        for episode in range(options.num_episodes):
            setup.reset()

            for time in range(options.num_timesteps):
                if options.render_training:
                    env.render()

                done = setup.make_an_exploratory_move()

                if len(setup.buffer) >= options.batch_size:
                    setup.run_one_step_of_training()

                if done:
                    logging.info("Done. Finishing episode after {} timesteps.".format(time+1))
                    setup.log_results()
                    break
    except KeyboardInterrupt:
        # FIXME: Better output
        print(setup.total_reward_per_training_episode[:-1])  # Last one might be corrupted