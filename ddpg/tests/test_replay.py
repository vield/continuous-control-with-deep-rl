import unittest

import numpy as np
import numpy.testing

from ddpg.replay import ReplayBuffer


class ReplayBufferTestCase(unittest.TestCase):
    def test_replay_buffer_overwrites_old_entries(self):
        BUFFER_SIZE = 3

        buffer = ReplayBuffer(BUFFER_SIZE)

        new_states = np.array([1, 2, 3, 4, 5, 6, 7])
        old_states = np.array([10, 20, 30, 40, 50, 60, 70])
        rewards = np.array([11, 22, 33, 44, 55, 66, 77])
        is_terminal = np.array([True, False, False, False, False, False, False])
        actions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

        for i in range(4):
            buffer.store(
                old_state=old_states[i],
                new_state=new_states[i],
                action=actions[i],
                is_terminal=is_terminal[i],
                reward=rewards[i]
            )
        # Buffer should be full, oldest entry should have been overwritten
        numpy.testing.assert_array_equal(
            buffer.new_states,
            np.reshape(new_states[[3,1,2]], [BUFFER_SIZE, 1])
        )
        numpy.testing.assert_array_equal(
            buffer.old_states,
            np.reshape(old_states[[3,1,2]], [BUFFER_SIZE, 1])
        )
        numpy.testing.assert_array_equal(
            buffer.is_terminal,
            np.reshape(is_terminal[[3,1,2]], [BUFFER_SIZE, 1])
        )

        for i in range(4, 7):
            buffer.store(
                old_state=old_states[i],
                new_state=new_states[i],
                action=actions[i],
                is_terminal=is_terminal[i],
                reward=rewards[i]
            )
        # More entries should have been overwritten
        numpy.testing.assert_array_equal(
            buffer.rewards,
            np.reshape(rewards[[6,4,5]], [BUFFER_SIZE, 1])
        )
        numpy.testing.assert_array_equal(
            buffer.actions,
            np.reshape(actions[[6,4,5]], [BUFFER_SIZE, 1])
        )

    def test_len_function(self):
        buffer = ReplayBuffer(3)
        self.assertEqual(len(buffer), 0)

        buffer.store(1, 2, 3, False, 4)
        self.assertEqual(len(buffer), 1)

        buffer.store(2, 3, 4, False, 5)
        self.assertEqual(len(buffer), 2)

        buffer.store(3, 4, 5, False, 6)
        self.assertEqual(len(buffer), 3)

        buffer.store(4, 5, 6, False, 7)
        self.assertEqual(len(buffer), 3)