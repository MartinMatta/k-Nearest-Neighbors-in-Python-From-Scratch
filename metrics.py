import numpy as np


class Euclidean(object):

    def __call__(sejf, train_data, test_data):
        square = (train_data - test_data) ** 2
        return np.sqrt(np.sum(square, axis=-1))


class Manhattan(object):

    def __call__(self, train_data, test_data):
        return np.sum(np.abs(train_data - test_data), axis=-1)


class Accuracy(object):
    
    def __call__(self, Y, Y_pred):
        acc = (Y == Y_pred).astype("int")
        acc = sum(acc)/len(Y)
        return acc