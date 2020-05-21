import numpy as np


class KNNClassifier:

    def __init__(self, k, metric):
        self.metric = metric
        self.k = k
       
    def fit(self, X, y):
        self.X = np.copy(X)
        self.y = np.copy(y)

    def predict(self, x_test):
        preds = []
        for index in range(x_test.shape[0]):

            distances = self.metric(self.X, x_test[index])

            sorted_labels = self.y[np.argsort(distances)]

            k_sorted_labels = sorted_labels[:self.k]

            unique, counts = np.unique(k_sorted_labels, return_counts=True)

            predict = unique[np.argmax(counts)]
            preds.append(predict)

        return np.array(preds)