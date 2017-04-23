import codecs
import numpy as np


class BatchSplitter:
    @staticmethod
    def get_bytes_from_file(filename):
        f = codecs.open(filename, 'r', 'utf-8')
        return map(ord, f.read().encode('utf-8'))

    @staticmethod
    def split(filename, seqlen, batchsize):
        A = BatchSplitter.get_bytes_from_file(filename)
        length = len(A)
        print("length: ", length)
        batch_step = length / batchsize
        batch_count = batch_step / seqlen
        print("batch_step: ", batch_step)
        print("batch_count: ", batch_count)
        X = []
        Y = []

        for batch in range(batch_count):
            X_data = []
            Y_data = []
            for j in range(batchsize):
                for i in range(seqlen):
                    pos = j * batch_step + batch * seqlen + i
                    x = A[pos]
                    y = A[(pos + 1) % length]
                    X_data.append(x)
                    Y_data.append(y)

            X.append(
                np.array(X_data, dtype=np.uint8)
                .reshape((batchsize, seqlen))
            )

            Y.append(
                np.array(Y_data, dtype=np.uint8)
                .reshape((batchsize, seqlen))
            )

        return X, Y
