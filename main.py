import tensorflow as tf
from multi_gru_cell import MultiGRUCell
from unroll_gru import UnrollGRU
from gru_trainer import GRUTrainer
from batch_splitter import BatchSplitter


mcell = MultiGRUCell(
    cellsize=512, layer_count=3,
    alphasize=256
)

unroll = UnrollGRU(
    mcell, batchsize=100, seqlen=30
)

trainer = GRUTrainer(unroll)
X, Y = BatchSplitter.split(
    filename="input.txt",
    batchsize=100, seqlen=30
)
print(len(X))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

trainer.train(
    sess=sess, X=X,
    Y=Y, loop=100
)

mcell.save(sess)
print("Saved")
