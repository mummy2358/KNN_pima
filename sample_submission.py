import load_csv
import numpy as np

test_groundtruth=load_csv.csv_reader("./pima/test_answer.csv")
test_groundtruth=test_groundtruth[1:]
load_csv.csv_writer(test_groundtruth,"./pima/groundtruth_submission.csv")
