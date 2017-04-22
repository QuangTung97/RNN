import tensorflow as tf

CELLSIZE = 128
batch_size = 10
num_steps = 8
input_size = 8  # bit

lstm = tf.contrib.rnn.BasicLSTMCell(CELLSIZE)
W = tf.Variable(tf.truncated_normal((input_size, CELLSIZE), stddev=0.1))
b = tf.Variable(tf.zeros([CELLSIZE]))

# weigth_words = tf.placeholder(tf.float32, [num_steps, batch_size, CELLSIZE])

init_state = state = lstm.zero_state(batch_size, dtype=tf.float32)

word = tf.placeholder(tf.float32, [batch_size, input_size])
weigth_word = tf.matmul(word, W) + b

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
