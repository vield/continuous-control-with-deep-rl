import numpy as np
import tensorflow as tf

from ddpg import actor, replay


class Setup:
    """Container for an experiment."""
    def __init__(self, env, sess, options):
        self.sess = sess
        self.env = env

        self.action_dims = env.action_space.shape[0]
        self.state_dims = env.observation_space.shape[0]
        self.action_range = np.stack((env.action_space.low, env.action_space.high), axis=0)

        self.actor = actor.Actor(
            sess=self.sess,
            action_range=self.action_range,
            action_dimensions=self.action_dims,
            state_dimensions=self.state_dims
        )
        # TODO: initialize critic

        self.buffer = replay.ReplayBuffer(
            buffer_size=options.buffer_size,
            action_dimensions=env.action_space.shape[0],
            state_dimensions=env.observation_space.shape[0]
        )

        self.generate_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_dims))

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.total_reward_per_training_episode = []
        self.state = None

    def reset(self):
        self.state = self.env.reset()
        self.total_reward_per_training_episode.append(0.0)

    def make_a_move(self):
        action = self.choose_noisy_action(self.state)
        new_state, reward, done, info = self.env.step(action)
        self.buffer.store(self.state, action, reward, new_state, done)

        self.total_reward_per_training_episode[-1] += reward

        raise NotImplementedError("Stepping forward not implemented yet!")

    def run_one_step_of_training(self):
        raise NotImplementedError("Training not implemented yet!")

    def choose_action(self, state):
        """Use the actor to choose the best possible action given the current parameters."""
        return self.actor.predict(np.reshape(state, (1, self.state_dims)))

    def choose_noisy_action(self, state):
        """Choose the best possible action and add some 'exploratory' noise."""
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