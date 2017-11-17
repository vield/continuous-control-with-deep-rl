import numpy as np
import tensorflow as tf

from ddpg import network


class Actor:
    """Container for the actor network and its target network."""
    def __init__(self, sess, action_range,
                 action_dimensions=1, state_dimensions=1,
                 learning_rate=0.0001, tau=0.001):
        """Initialize actor and actor target networks and ops for training.

        :param sess:
            TensorFlow session.
        :param action_range:
            [low, high] for the action range.
        :param action_dimensions:
            Dimensions of the (continuous) action space.
        :param state_dimensions:
            Dimensions of the (continuous) state space.
        :param learning_rate:
            Learning rate to initialize the optimizer with.
        :param tau:
            Weight to control how fast the target network is
            shifted towards the current actor network.
        """
        self.sess = sess
        self.action_dimensions = action_dimensions
        self.state_dimensions = state_dimensions

        self.learning_rate = learning_rate
        self.tau = tau

        self.network = ActorNetwork(state_dimensions, action_dimensions, action_range)
        self.target_network = ActorNetwork(state_dimensions, action_dimensions, action_range)

        # TensorFlow op for shifting the critic target network params
        # slightly towards the critic network params (controlled by
        # tau).
        self._update_target_network_op = []
        network_params = self.network.get_trainable_params()
        target_params = self.target_network.get_trainable_params()

        for i in range(len(network_params)):
            self._update_target_network_op.append(
                tf.assign(
                    target_params[i],
                    tf.add(
                        tf.multiply(network_params[i], tau),
                        tf.multiply(target_params[i], 1.0 - tau)
                    )
                )
            )

        # TODO: Add training ops

    def update_target_network(self):
        """Shift target network weights towards critic network weights."""
        self.sess.run(self._update_target_network_op)

    def predict(self, states):
        """Predict best actions using the actor network."""
        return self.network.predict(self.sess, states)


class ActorNetwork:
    def __init__(self, state_dimensions, action_dimensions, action_range):
        # Inputs for prediction
        self.states = tf.placeholder(tf.float32, [None, state_dimensions])

        # Create fully-connected architecture
        nodes_per_layer = [state_dimensions, 400, 300, action_dimensions]

        self._biases = network.create_bias_shaped_variables(nodes_per_layer, stddev=0.1)
        self._weights = network.create_weight_shaped_variables(nodes_per_layer, stddev=0.1)
        self._var_list = self._biases + self._weights

        self._raw_outputs = network.create_fully_connected_architecture(self.states, self._biases, self._weights)

        # The raw outputs have arbitrary magnitudes.
        # We use tanh for an activation function, it casts them to be in (-1, 1).
        # Then we scale them so the actual outputs are within the action range
        # (so that we will always only predict "valid" actions, but will also
        # be able to predict practically any valid actions).
        self._outputs = tf.tanh(self._raw_outputs)

        # XXX What if the exploration noise takes us outside the range?

        # Naive scaling, assert symmetric action space
        # Could just shift it up or down as appropriate but this'll work for the Pendulum.
        assert np.all(np.equal(action_range[0], -action_range[1]))
        self._action_range = action_range

        self.scaled_outputs = tf.multiply(
            tf.cast(action_range[1], tf.float32),
            self._outputs)

    def predict(self, sess, states):
        """Return predicted actions for each given state."""
        return sess.run(self.scaled_outputs, feed_dict={self.states: states})

    def get_trainable_params(self):
        return self._var_list