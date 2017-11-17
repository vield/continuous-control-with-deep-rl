import numpy as np
import tensorflow as tf

from ddpg import replay


class Setup:
    """Container for an experiment."""
    def __init__(self, env, sess, options):
        self.sess = sess
        self.env = env

        self.action_dims = env.action_space.shape[0]
        self.state_dims = env.observation_space.shape[0]
        self.action_range = np.stack((env.action_space.low, env.action_space.high), axis=0)

        # TODO: Initialize actor
        # TODO: initialize critic

        self.buffer = replay.ReplayBuffer(
            buffer_size=options.buffer_size,
            action_dimensions=env.action_space.shape[0],
            state_dimensions=env.observation_space.shape[0]
        )

        self.generate_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_dims))

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def reset(self):
        self.env.reset()

    def make_a_move(self):
        raise NotImplementedError("Stepping forward not implemented yet!")

    def run_one_step_of_training(self):
        raise NotImplementedError("Training not implemented yet!")

    def choose_action(self, state):
        raise NotImplementedError("Action choosing not implemented yet!")

    def choose_noisy_action(self, state):
        return self.choose_action(state) + self.generate_noise()


# Copied from from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class OrnsteinUhlenbeckActionNoise:
    """Process to generate temporally correlated noise.

    The previous noise value is stored and used as a starting point
    for the next value."""
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = (
            self.x_prev +
            self.theta * (self.mu - self.x_prev) * self.dt +
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)