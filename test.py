import load_csv
import numpy as np

train=load_csv.csv_reader("./pima/train.csv")
print(np.shape(train))
print(train[0])
print(train[1])
"""
for row in train:
  print(row)
"""
