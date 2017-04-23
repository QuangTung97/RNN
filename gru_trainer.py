import tensorflow as tf
import numpy as np


class GRUTrainer:
    def __init__(self, unroll_gru):
        self.unroll_gru = unroll_gru
        gru = unroll_gru

        Y_ = tf.placeholder(
            tf.uint8,
            [gru.batchsize, gru.seqlen],
            name="Y_"
        )
        Y_one_hot = tf.one_hot(Y_, gru.mcell.alphasize, 1.0, 0.0)

        self.cross_entropy = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=gru.logits,
                labels=Y_one_hot)
        )

        self.train_step = tf.train.AdamOptimizer(1e-3).minimize(self.cross_entropy)

    def train(self, sess, X, Y, loop):
        gru = self.unroll_gru
        Hzero = Hin = np.zeros([gru.batchsize,
                                gru.mcell.state_size],
                       dtype=np.float32)

        batch_count = len(X)
        for i in range(loop):
            pos = i % batch_count
            if (pos == 0):
                Hin = Hzero

            input_data = {
                "X:0": X[pos],
                "Hin:0": Hin,
                "Y_:0": Y[pos]
            }

            _, entropy, Hin = sess.run(
                [self.train_step, self.cross_entropy, gru.Hout],
                feed_dict=input_data
            )
            print(i, entropy)
