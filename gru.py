import tensorflow as tf


# Constants
CELLSIZE = 512
ALPHASIZE = 98
NLAYERS = 3
SEQLEN = 30
batch_size = 10

cell = tf.contrib.rnn.GRUCell(CELLSIZE)
mcell = tf.contrib.rnn.MultiRNNCell([cell] * NLAYERS,
                                    state_is_tuple=False)

# Input of [BATCHSIZE, SEQLEN]
Xd = tf.placeholder(tf.uint8, [None, None])
# [BATCHSIZE, SEQLEN, ALPHASIZE]
X = tf.one_hot(Xd, ALPHASIZE, 1.0, 0.0)

# Train data with shape [BATCHSIZE, SEQLEN]
Yd_ = tf.placeholder(tf.uint8, [None, None])
# Train data after one-hot transform
# shape [BATCHSIZE, SEQLEN, ALPHASIZE]
Y_ = tf.one_hot(Yd_, ALPHASIZE, 1.0, 0.0)
# Reshape to [BATCHSIZE * SEQLEN, ALPHASIZE]
Y_ = tf.reshape(Y_, (batch_size * SEQLEN, ALPHASIZE))

Hin = tf.placeholder(tf.float32, [None, CELLSIZE * NLAYERS])

# Hr output shaped [BATCHSIZE, SEQLEN, CELLSIZE]
# H output state
Hr, H = tf.nn.dynamic_rnn(mcell, X, initial_state=Hin)

W = tf.Variable(tf.truncated_normal((CELLSIZE, ALPHASIZE),
                                    stddev=0.1))
b = tf.Variable(tf.zeros([ALPHASIZE]))

Hr_reshaped = tf.reshape(Hr, [batch_size * SEQLEN, CELLSIZE])
logits = tf.matmul(Hr_reshaped, W) + b
Y = tf.softmax(logits)

cross_entropy = tf.reduce_sum(
    tf.nn.softmax_cross_entropy_with_logits(
        logits=logits,
        labels=Y_)
)

train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

# for test result
# Shape [batch_size * SEQLEN]
predictions = tf.argmax(Y, 1)
predictions = tf.reshape(predictions, [batch_size, SEQLEN])

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

step_count = 100
for step in range(step_count):
    _, outH = sess.run([train_step, H])
