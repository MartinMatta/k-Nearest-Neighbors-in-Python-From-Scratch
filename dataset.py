import pandas as pd
import numpy as np 
import os

class  Data(object):

    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.data = self.data.reindex(
            np.random.permutation(self.data.index)
        )
        self.data.drop("Id", axis=1, inplace=True)

    def get_data(self):
        return self.data[
            [col for col in self.data.columns if col!="Species"]
            ]

    def get_labels(self):
        return self.data["Species"]


class Dataset(Data):

    def __init__(self, path, test_split=0.3):
        super().__init__(path)
        self.test_size = int(len(self.get_labels()) * test_split)

    def train(self):
        return (self.get_data()[self.test_size:].values,
         self.get_labels()[self.test_size:].values)

    def test(self):
        return (self.get_data()[:self.test_size].values,
         self.get_labels()[:self.test_size].values)


