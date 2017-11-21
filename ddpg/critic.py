import numpy as np
import tensorflow as tf

from ddpg import network


class Critic:
    """Container for the critic network and its target network."""
    def __init__(self, sess, actor,
                 action_dimensions=1, state_dimensions=1,
                 learning_rate=0.001, tau=0.001):
        """Initialize actor and actor target networks and ops for training.

        :param sess:
            TensorFlow session.
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

        self.network = CriticNetwork(state_dimensions, action_dimensions)
        # TODO: Actually in DDPG, they should be initialized to start from the same set of weights
        self.target_network = CriticNetwork(state_dimensions, action_dimensions, actor=actor)

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

        # Predicted value for Q(s, a) -- "y_i" in the paper
        # Used to compute the cost
        self._predicted_outputs = tf.placeholder(tf.float32, [None, 1])

        network_outputs = self.network.get_unscaled_outputs()

        self._mse = tf.reduce_mean(
            tf.square(self._predicted_outputs - network_outputs)
        )
        self._train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self._mse)

        # Gradients to feed into the actor update
        self._action_grads = tf.gradients(network_outputs, self.network.actions)

    def update_target_network(self):
        """Shift target network weights towards learned network weights."""
        self.sess.run(self._update_target_network_op)

    def target_predict(self, states, actions):
        """Predict quality values using the target network."""
        return self.target_network.predict(self.sess, states, actions)

    def run_one_step_of_training(self, states, actions, predicted_outputs):
        """What the function name says.

        :param states:
            States.
        :param actions:
            Actions.
        :param predicted_outputs:
            y_i, or predicted Q(s, a) values based on the current reward and
            discounted future rewards computed using the target actor and critic.
            The training attempts to minimize the MSE between the actual output
            and the target output, see __init__ for setup.
        """
        self.sess.run(
            self._train_step,
            feed_dict={
                self.network.states: states,
                self.network.actions: actions,
                self._predicted_outputs: predicted_outputs
            }
        )

    def compute_action_gradients(self, states, actions):
        """Compute action gradients to feed into the actor update."""
        return self.sess.run(
            self._action_grads,
            feed_dict={
                self.network.states: states,
                self.network.actions: actions
            }
        )


class CriticNetwork:
    def __init__(self, state_dimensions, action_dimensions, actor=None):
        # Inputs for the quality function
        if actor is None:
            self.states = tf.placeholder(tf.float32, [None, state_dimensions])
            self.actions = tf.placeholder(tf.float32, [None, action_dimensions])
        else:
            self.states = actor.target_network.states
            self.actions = actor.target_network.scaled_outputs

        # Create network architecture
        # The actions are merged in for the second layer
        input_weights = network.create_weight_shaped_variables([state_dimensions, 400, 300, 1], stddev=0.01)
        input_biases = network.create_bias_shaped_variables([state_dimensions, 400, 300, 1], stddev=0.01)
        action_weights = network.create_weight_shaped_variables([action_dimensions, 300], stddev=0.01)

        self._var_list = input_weights + action_weights + input_biases

        # First layer output, from states only
        outputs1 = tf.nn.relu(tf.add(tf.matmul(self.states, input_weights[0]), input_biases[0]))

        # Second layer output, combination of states and actions
        inputs2_1 = tf.matmul(outputs1, input_weights[1])
        inputs2_2 = tf.matmul(self.actions, action_weights[0])
        outputs2 = tf.nn.relu(
            tf.add(
                tf.add(inputs2_1, inputs2_2),
                input_biases[1]
            )
        )

        self._raw_outputs = tf.add(tf.matmul(outputs2, input_weights[2]), input_biases[2])

    def get_unscaled_outputs(self):
        return self._raw_outputs

    def get_trainable_params(self):
        return self._var_list

    def predict(self, sess, states, actions):
        """Predict quality values for (state, action) pairs.

        :param sess:
            TensorFlow session where we should do the computation.
        :param states:
            N states.
        :param actions:
            N actions.
        :return:
            Floating point value representing the goodness of the
            state-action combination.
        """
        return sess.run(
            self._raw_outputs,
            feed_dict={
                self.states: states,
                self.actions: actions
            }
        )