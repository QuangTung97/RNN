import tensorflow as tf


class UnrollGRU:
    def __init__(self, mcell, batchsize, seqlen):
        self.batchsize = batchsize
        self.seqlen = seqlen
        self.mcell = mcell

        Hin = tf.placeholder(
            tf.float32,
            [batchsize, mcell.state_size],
            name="Hin"
        )

        X = tf.placeholder(tf.uint8,
                           [batchsize, seqlen], name="X")
        self.X = X

        X_one_hot = tf.one_hot(X, mcell.alphasize, 1.0, 0.0)
        Hr, H = tf.nn.dynamic_rnn(mcell.tfget(), X_one_hot,
                                  initial_state=Hin)

        Hr_reshaped = tf.reshape(
            Hr, [batchsize * seqlen, mcell.cellsize]
        )
        self.Hout = H
        self.logits = tf.matmul(Hr_reshaped, mcell.Wo) + mcell.Bo
        self.logits = tf.reshape(
            self.logits, [batchsize, seqlen, mcell.alphasize]
        )
        self.Y = tf.nn.softmax(self.logits)
