import logging

import numpy as np
import tensorflow as tf

from ddpg import actor, critic, replay


class Setup:
    """Container for an experiment."""
    def __init__(self, env, sess, options):
        self.sess = sess
        self.env = env

        self.batch_size = options.batch_size
        self.gamma = options.gamma

        self.action_dims = env.action_space.shape[0]
        self.state_dims = env.observation_space.shape[0]
        self.action_range = np.stack((env.action_space.low, env.action_space.high), axis=0)

        self.actor = actor.Actor(
            sess=self.sess,
            action_range=self.action_range,
            action_dimensions=self.action_dims,
            state_dimensions=self.state_dims
        )
        self.critic = critic.Critic(
            sess=self.sess,
            actor=self.actor,
            action_dimensions=self.action_dims,
            state_dimensions=self.state_dims,
            gamma=self.gamma
        )

        self.actor.set_train_step(self.critic, batch_size=options.batch_size)

        self.buffer = replay.ReplayBuffer(
            buffer_size=options.buffer_size,
            action_dimensions=env.action_space.shape[0],
            state_dimensions=env.observation_space.shape[0]
        )

        self.generate_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_dims), sigma=options.noise_sigma, theta=0.15)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # TODO: Do something with this
        # TODO: Especially if KeyboardInterrupted... would like it to be written down somewhere for plotting
        self.current_episode = 0
        self.total_reward_per_training_episode = []
        self.state = None
        self.evaluation_reward = []

    def reset(self, advance_episode=True):
        self.state = self.env.reset()
        if advance_episode:
            self.total_reward_per_training_episode.append(0.0)
            self.current_episode += 1

    def make_an_exploratory_move(self):
        """Choose action with exploratory noise; observe and store outcome.

        :return done:
            True if the latest move took us to a terminal state
            in the environment. This ends a training episode.
        """
        action = self.choose_noisy_action(self.state)
        new_state, reward, done, info = self.env.step(action)
        self.buffer.store(self.state, action, reward, new_state, done)

        try:
            self.total_reward_per_training_episode[-1] += reward[0]
        except TypeError:
            self.total_reward_per_training_episode[-1] += reward

        self.state = new_state

        return done

    def make_a_move(self):
        action = self.choose_action(self.state)
        new_state, reward, done, info = self.env.step(action)

        try:
            self.evaluation_reward[-1] += reward[0]
        except TypeError:
            self.evaluation_reward[-1] += reward

        self.state = new_state
        return done

    def run_one_step_of_training(self):
        """Train the critic and the actor from a sampled minibatch.

        Also update the target networks.
        """
        # Sample uncorrelated transitions from the replay buffer
        old_states, actions, rewards, new_states, is_terminal = self.buffer.next_batch(self.batch_size)

        # Train critic
        self.critic.run_one_step_of_training(
            old_states=old_states,
            actions=actions,
            rewards=rewards,
            new_states=new_states
        )
        # Train actor
        self.actor.run_one_step_of_training(
            states=old_states
        )

        # Update the target networks
        self.actor.update_target_network()
        self.critic.update_target_network()

    def choose_action(self, state):
        """Use the actor to choose the best possible action given the current parameters."""
        return self.actor.predict(np.reshape(state, (1, self.state_dims)))

    def choose_noisy_action(self, state):
        """Choose the best possible action and add some 'exploratory' noise."""
        return self.choose_action(state) + self.generate_noise()

    def evaluate(self, max_timesteps, render_testing=False):
        self.evaluation_reward.append(0.0)
        self.reset(advance_episode=False)

        for time in range(max_timesteps):
            if render_testing:
                self.env.render()

            done = self.make_a_move()
            if done:
                break

        logging.debug('Testing. Episode: {} Testing reward: {}'.format(self.current_episode, self.evaluation_reward[-1]))

        return self.evaluation_reward[-1]

    def log_results(self):
        logging.debug("Episode: {} Total reward: {}".format(self.current_episode, self.total_reward_per_training_episode[-1]))


# Copied from from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class OrnsteinUhlenbeckActionNoise:
    """Process to generate temporally correlated noise.

    The previous noise value is stored and used as a starting point
    for the next value."""
    def __init__(self, mu, sigma=0.3, theta=0.15, dt=1e-2, x0=None):
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