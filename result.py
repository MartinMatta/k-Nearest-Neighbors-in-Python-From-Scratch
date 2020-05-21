from knn_classification import KNNClassifier
from metrics import Manhattan
from metrics import Euclidean
from metrics import Accuracy
from dataset import Dataset

import matplotlib.pyplot as plt
import numpy as np
import argparse 
import sys
import os


ap = argparse.ArgumentParser()

ap.add_argument("-s", "--test_split",
                required=True,
	            help="data split")

ap.add_argument("-m", "--metric",
                required=True,
	            help="Knn metrics")

args = vars(ap.parse_args())


metric = Euclidean()

if str(args["metric"])=="eucludean":
    metric = Euclidean()

elif str(args["metric"])=="manhattan":
    metric = Manhattan()

else:
    sys.exit(1)



data = Dataset(os.getcwd() + "/dataset/Iris.csv",
               test_split=float(args["test_split"])
               )

train_x, train_y = data.train()
test_x, test_y = data.test()


mse = {}

for k in range(1, 21):
    knn = KNNClassifier(k=k, metric=metric)
    knn.fit(train_x, train_y)

    pred = knn.predict(test_x)

    mse[k] = Accuracy()(test_y, pred)
        
print("Max ACC:", max(mse.values()))
print("best k", np.argmax([*mse.values()])+1)



plt.figure(figsize=(8, 4))
plt.plot(list(mse.keys()), list(mse.values()))
plt.xticks(range(1, 21))
plt.xlabel("k")
plt.ylabel("ACC")
plt.title("metrics: " + str(args["metric"]), fontsize=15)
plt.show()