import tensorflow as tf
import numpy as np


class MultiGRUCell:
    def __init__(self, cellsize, layer_count, alphasize):
        self.cellsize = cellsize
        self.layer_count = layer_count
        self.alphasize = alphasize
        self.state_size = cellsize * layer_count

        cell = tf.contrib.rnn.GRUCell(cellsize)
        self.mcell = tf.contrib.rnn.MultiRNNCell(
            [cell] * layer_count,
            state_is_tuple=False
        )

        self.Wi = tf.Variable(tf.truncated_normal(
            [alphasize, cellsize],
            stddev=0.1))
        self.Bi = tf.Variable(tf.zeros([cellsize]))

        self.Wo = tf.Variable(tf.truncated_normal(
            [cellsize, alphasize],
            stddev=0.1))
        self.Bo = tf.Variable(tf.zeros([alphasize]))

        # for run this cell
        self.gruX = tf.placeholder(tf.uint8, [1])
        X_one_hot = tf.one_hot(self.gruX, alphasize, 1.0, 0.0)
        self.Hin = tf.placeholder(tf.float32, [1, self.state_size])
        Yout, self.Hout = self.mcell(X_one_hot, self.Hin)
        self.Y = tf.argmax(
            tf.matmul(Yout, self.Wo) + self.Bo, 0
        )

    def tfget(self):
        return self.mcell

    def save(self, sess):
        saver = tf.train.Saver()
        saver.save(sess, "./MultiGRUCell.ckpt")

    def restore(self, sess):
        saver = tf.train.Saver()
        saver.restore(sess, "./MultiGRUCell.ckpt")

    def zero_state(self):
        return np.zeros(np.float32, [1, self.state_size])

    def run(self, sess, X, Hin):
        Y, Hout = sess.run([self.Y, self.Hout],
                        feed_dict={
                            self.gruX: [X],
                            self.Hin: Hin
                        })
        return Y[0], Hout
