import logging

import numpy as np


class ReplayBuffer:
    """Fixed-size container for (old_state, action, reward, new_state, is_terminal) tuples.

    Older entries are overwritten by new entries after the buffer becomes full.

    Supports sampling for training purposes."""

    def __init__(self, buffer_size, action_dimensions=1, state_dimensions=1):
        """Initialize new, empty replay buffer.

        :param buffer_size:
            Maximum number of entries that may be in the buffer.
        :param action_dimensions:
            Dimensions of the action space.
            Affects actions.
        :param state_dimensions:
            Dimensions of the observation space.
            Affects new_states and old_states.
        """
        self.current_index = 0
        self.size = 0
        assert buffer_size > 0
        self.buffer_size = buffer_size

        self.action_dimensions = action_dimensions
        self.state_dimensions = state_dimensions

        self.old_states = np.zeros([buffer_size, state_dimensions])
        self.new_states = np.zeros([buffer_size, state_dimensions])
        self.actions = np.zeros([buffer_size, action_dimensions])
        self.rewards = np.zeros([buffer_size, 1])
        self.is_terminal = np.zeros([buffer_size, 1], dtype=bool)

        logging.debug("Initialized ReplayBuffer with size {}.".format(buffer_size))

    def store(self, old_state, action, reward, new_state, is_terminal):
        """Add new entry to the buffer, possibly overwriting the oldest entry.

        :param old_state:
            Starting state. Must be convertible to shape [1, state_dimensions].
        :param action:
            Action. Must be convertible to shape [1, state_dimensions].
        :param reward:
            Reward. Must be convertible to shape [1, 1] (effectively, this is a
            real-value scalar).
        :param new_state:
            Next state. Must be convertible to shape [1, state_dimensions].
        :param is_terminal:
            Whether this is a terminal state. Must be convertible to a 1 * 1
            boolean array.
        """
        self.old_states[self.current_index, :] = np.reshape(old_state, [1, self.state_dimensions])
        self.new_states[self.current_index, :] = np.reshape(new_state, [1, self.state_dimensions])
        self.actions[self.current_index, :] = np.reshape(action, [1, self.action_dimensions])
        self.rewards[self.current_index, :] = np.reshape(reward, [1, 1])
        self.is_terminal[self.current_index, :] = np.reshape(is_terminal, [1, 1])

        if self.size < self.buffer_size:
            # We have actually added something to the buffer without
            # overwriting
            self.size += 1

        self.current_index += 1
        self.current_index %= self.buffer_size

    def next_batch(self, batch_size):
        if batch_size > self.size:
            raise Exception("Not enough elements in replay buffer!")

        random_indices = np.random.choice(self.size, batch_size)

        return (
            self.old_states[random_indices],
            self.actions[random_indices],
            self.rewards[random_indices],
            self.new_states[random_indices],
            self.is_terminal[random_indices]
        )

    def __len__(self):
        return self.size