import tensorflow as tf

from ddpg import network


class Critic:
    """Container for the critic network and its target network."""
    def __init__(self, sess, actor, gamma=0.99,
                 action_dimensions=1, state_dimensions=1,
                 learning_rate=0.001, tau=0.001):
        """Initialize actor and actor target networks and ops for training.

        :param sess:
            TensorFlow session.
        :param actor:
            Actor class. Outputs from the policy and the target policy
            feed into some Critic operations.
        :param gamma:
            Discount factor for future rewards.
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

        self.network = CriticNetwork(state_dimensions, action_dimensions, actor=actor, add_gradients=True)
        self.target_network = CriticNetwork(state_dimensions, action_dimensions, actor=actor, use_actions_from_actor=True)

        # TensorFlow op for shifting the critic target network params
        # slightly towards the critic network params (controlled by
        # tau).
        self._update_target_network_op = []
        self._initialize_target_network_op(tau)

        # Training params
        self._rewards_input = tf.placeholder(tf.float32, [None, 1])
        self._train_step = None
        self._set_train_step(gamma, learning_rate)

        # Gradients to feed into the actor update
        self._action_grads = tf.gradients(self.network._raw_grad_outputs, actor.network.scaled_outputs)

    def get_action_gradients_op(self):
        """Return gradients of the Q function w.r.t. mu (the current policy from the actor)."""
        return self._action_grads[0]

    def update_target_network(self):
        """Shift target network weights towards learned network weights."""
        self.sess.run(self._update_target_network_op)

    def run_one_step_of_training(self, old_states, actions, rewards, new_states):
        self.sess.run(
            self._train_step,
            feed_dict={
                self.target_network.states: new_states,
                self.network.states: old_states,
                self.network.actions: actions,
                self._rewards_input: rewards
            }
        )

    #
    #
    # Helper functions
    def _initialize_target_network_op(self, tau):
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

    def _set_train_step(self, gamma, learning_rate):
        # Predicted value for Q(s, a) from "current reward + discounted future
        # reward as given by the target network".
        # FIXME: Consider terminal states (no future expected reward from there)
        target_outputs = tf.stop_gradient(
            tf.add(
                self._rewards_input,
                tf.multiply(gamma, self.target_network.get_unscaled_outputs())
            )
        )
        current_outputs = self.network.get_unscaled_outputs()
        mse = tf.reduce_mean(
            tf.square(target_outputs - current_outputs)
        )

        self._train_step = tf.train.AdamOptimizer(learning_rate).minimize(mse)


class CriticNetwork:
    def __init__(self, state_dimensions, action_dimensions, actor=None,
                 use_actions_from_actor=False, add_gradients=False):
        # Inputs for the quality function
        if not use_actions_from_actor:
            self.states = tf.placeholder(tf.float32, [None, state_dimensions])
            self.actions = tf.placeholder(tf.float32, [None, action_dimensions])
        else:
            assert actor is not None, "Actor must be given if using it for predictions"
            self.states = actor.target_network.states
            self.actions = actor.target_network.scaled_outputs

        input_weights = network.create_weight_shaped_variables([state_dimensions, 400, 300, 1], stddev=0.01)
        input_biases = network.create_bias_shaped_variables([state_dimensions, 400, 300, 1], stddev=0.01)
        action_weights = network.create_weight_shaped_variables([action_dimensions, 300], stddev=0.01)

        self._var_list = input_weights + action_weights + input_biases

        self._raw_outputs = self._create_network_architecture(self.states, self.actions,
                                                              input_weights, input_biases, action_weights)

        if add_gradients:
            assert actor is not None, "Actor must be given if adding gradient output"
            self._raw_grad_outputs = self._create_network_architecture(actor.network.states, actor.network.scaled_outputs,
                                                                       input_weights, input_biases, action_weights)
        else:
            self._raw_grad_outputs = None

    def get_unscaled_outputs(self):
        return self._raw_outputs

    def get_trainable_params(self):
        return self._var_list

    def predict(self, sess, states, actions):
        """Predict quality values for (state, action) pairs."""
        return sess.run(
            self._raw_outputs,
            feed_dict={
                self.states: states,
                self.actions: actions
            }
        )

    #
    #
    # Helper functions
    def _create_network_architecture(self, states, actions, input_weights, input_biases, action_weights):
        # First layer output, from states only
        outputs1 = tf.nn.relu(tf.add(tf.matmul(states, input_weights[0]), input_biases[0]))

        # Second layer output, combination of states and actions
        inputs2_1 = tf.matmul(outputs1, input_weights[1])
        inputs2_2 = tf.matmul(actions, action_weights[0])
        outputs2 = tf.nn.relu(
            tf.add(
                tf.add(inputs2_1, inputs2_2),
                input_biases[1]
            )
        )

        outputs = tf.add(tf.matmul(outputs2, input_weights[2]), input_biases[2])
        return outputs