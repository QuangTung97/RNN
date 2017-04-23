import tensorflow as tf
from multi_gru_cell import MultiGRUCell


mcell = MultiGRUCell(
    cellsize=512,
    layer_count=3, alphasize=256
)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

mcell.restore(sess)


Hin = mcell.zero_state()
X = 15
result = []
for i in range(1000):
    X, Hin = mcell.run(sess, X, Hin)
    result.append(X)

print(str(bytearray(result)))
