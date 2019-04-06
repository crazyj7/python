import numpy as np
import tensorflow as tf


class DQN:
    def __init__(self, session, input_size, output_size, name="main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        # self._build_network()
        self._build_network(32, 0.1)
        # self._build_network(10, 0.1)

    def _build_network(self, h_size=10, l_rate=0.1):
        print('hidden=', h_size, 'learning rate=', l_rate)
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")

            # First layer of weights
            W1 = tf.get_variable("W1", shape=[self.input_size, h_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            # layer1 = tf.nn.tanh(tf.matmul(self._X, W1))
            # layer1 = tf.nn.sigmoid(tf.matmul(self._X, W1))
            layer1 = tf.nn.relu(tf.matmul(self._X, W1))

            # W15 = tf.get_variable('W15', shape=[h_size, 64],
            #                       initializer=tf.contrib.layers.xavier_initializer())
            # layer2 = tf.nn.relu(tf.matmul(layer1, W15))


            # Second layer of Weights
            W2 = tf.get_variable("W2", shape=[h_size, self.output_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            # W2 = tf.get_variable("W2", shape=[64, self.output_size],
            #                      initializer=tf.contrib.layers.xavier_initializer())
            # Q prediction
            self._Qpred = tf.matmul(layer1, W2)
            # self._Qpred = tf.matmul(layer2, W2)

            self.saver = tf.train.Saver(max_to_keep=None)

        # We need to define the parts of the network needed for learning a policy
        self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)

        # Loss function
        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
        # Learning
        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})

    def save(self, path):
        self.saver.save(self.session, path)


    def load(self, path):
        self.saver.restore(self.session, path)

