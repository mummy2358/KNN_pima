import numpy as np
import load_csv
import os

def divide(dataset,root="./",train_val_test=[0.6,0.2,0.2]):
  title=dataset[0]
  if np.sum(train_val_test)!=1:
    train_val_test=train_val_test/np.sum(train_val_test)
  total_num=len(dataset)-1
  train=[title]
  val=[title]
  test_inputs=[title[:-1]]
  test_ans=[[title[0],title[-1]]]
  for i in range(total_num):
    prob=np.random.rand()
    if prob<=train_val_test[0]:
      train.append(dataset[i+1])
    elif prob<=train_val_test[0]+train_val_test[1]:
      val.append(dataset[i+1])
    else:
      test_inputs.append(dataset[i+1][:-1])
      test_ans.append([dataset[i+1][0],dataset[i+1][-1]])
  if not os.path.exists(root):
    os.system("mkdir "+root)
  load_csv.csv_writer(train,root+"/train.csv")
  load_csv.csv_writer(val,root+"/val.csv")
  load_csv.csv_writer(test_inputs,root+"/test.csv")
  load_csv.csv_writer(test_ans,root+"/test_answer.csv")
  print("division complete")
  

train,val,test=0.6,0.2,0.2
# train and val is given to competitors to validate their models and adjust hyper-parameters or methods
# the test answers are not given and is later divided by kaggle into "public" and "private"
dataset=load_csv.csv_reader("./Pima.csv")

divide(dataset,root="./pima",train_val_test=[train,val,test])
