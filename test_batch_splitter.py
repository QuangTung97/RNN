from batch_splitter import BatchSplitter
from mock import Mock


BatchSplitter.get_bytes_from_file = Mock(
    return_value=[1, 2, 3, 4, 5, 6, 7, 8,
                  9, 10, 11, 12, 13, 14, 15, 16, 17]
)

X, Y = BatchSplitter.split("abc.txt", 2, 4)

for i in range(len(X)):
    print(X[i].shape)
    print(X[i])
    print(Y[i])
    print("------------------------------------")
