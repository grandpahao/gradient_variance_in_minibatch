import tensorflow as tf
from IPython import embed

from utils import jacobian


class Net:
    def __init__(self, lr0, epsilon=0.0001):
        self.lr = lr0
        self.epsilon = epsilon
        self.iter = 0
        self.sess = tf.Session()

        self.writer = tf.summary.FileWriter(
            'log/batch_adjust_g_{}/'.format(epsilon))
        self._net_build()
        self.sess.run(tf.global_variables_initializer())
        # self.saver = tf.train.Saver(self.sess.trainable_variables, max_to_keep=5)

    def _net_build(self):
        self.input = tf.placeholder(shape=[None, 784], dtype=tf.float32)
        self.target = tf.placeholder(shape=[None, 10], dtype=tf.float32)
        self.fc1 = tf.contrib.layers.fully_connected(
            self.input, 500, activation_fn=tf.nn.relu)
        self.fc2 = tf.contrib.layers.fully_connected(
            self.fc1, 10, activation_fn=None)

        self.loss_vector = tf.reshape(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.target, logits=self.fc2), [-1])
        self.real_loss = tf.reduce_mean(self.loss_vector)

        self.var = tf.trainable_variables()
        self.grad = tf.gradients(self.real_loss, self.var)

        # embed()

        self.jacobian = jacobian(self.loss_vector, self.var, False)

        flat_j = tf.concat(
            [
                tf.reshape(self.jacobian[0], shape=[-1, 784 * 500]),
                self.jacobian[1],
                tf.reshape(self.jacobian[2], shape=[-1, 500 * 10]),
                self.jacobian[3]
            ],
            axis=1
        )
        root_j = tf.sqrt(tf.reduce_mean(
            tf.square(flat_j), axis=0)) + self.epsilon

        cur_g = tf.concat([tf.reshape(g, [-1]) for g in self.grad], axis=0)

        self.batch_var = tf.reduce_mean(
            tf.reduce_sum(tf.square(flat_j - cur_g), axis=1))

        [g1, g2, g3, g4] = tf.split(tf.divide(cur_g, root_j), [
                                    784 * 500, 500, 500 * 10, 10])
        adjust_g = [tf.reshape(g1, [784, 500]), g2,
                    tf.reshape(g3, [500, 10]), g4]

        # embed()

        self.minimizer = []
        for v, g in zip(self.var, adjust_g):
            self.minimizer.append(tf.assign_sub(v, g * self.lr))
        self.minimizer = tf.group(self.minimizer)

        tf.identity(self.real_loss, name='loss')
        tf.identity(self.batch_var, name='batch_variance')
        tf.summary.scalar('loss', self.real_loss)
        tf.summary.scalar('batch_variance', self.batch_var)
        self.merge_summary = tf.summary.merge_all()

    def train(self, data, target):
        _, cur_summary = self.sess.run(
            [
                self.minimizer,
                self.merge_summary
            ],
            feed_dict={
                self.input: data,
                self.target: target
            }
        )
        self.writer.add_summary(cur_summary, self.iter)
        # self.saver.save(self.sess, 'model/model.ckpt', global_step=self.iter)
        self.iter += 1
