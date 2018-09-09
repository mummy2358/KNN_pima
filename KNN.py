import numpy as np
import load_csv
import matplotlib.pyplot as plt
def distance_L1(x1,x2):
  # x1 and x2 can be in different shapes on the sample dim
  # if one of them is the test point and the other is the whole training set, then the minimum distance is in the returned distance array
  x1=np.array(x1)
  x2=np.array(x2)
  return np.sum(np.abs(x1-x2),axis=-1)

def distance_L2(x1,x2):
  x1=np.array(x1)
  x2=np.array(x2)
  return np.sqrt(np.sum(np.power(x1-x2,2),axis=-1))

def std_normalization(inputs,mean_std=None,axis=0):
  if mean_std==None:
    mean=np.mean(inputs,axis=axis)
    std=np.std(inputs,axis=axis)
    return [mean,std,(inputs-mean)/std]
  else:
    mean=mean_std[0]
    std=mean_std[1]
    return [mean,std,(inputs-mean)/std]

def scaling(inputs,axis=0):
  # scale inputs to 0-1 
  return inputs/np.amax(inputs,axis=axis)

def KNN(x,train,K=3):
  # train should have one more column than x as the label column
  train=np.array(train)
  x=np.array(x)
  train_inputs=train[:,:-1]
  
  [mean,std,train_inputs]=std_normalization(train_inputs)
  [_,_,x]=std_normalization(x,mean_std=[mean,std])
  dist=distance_L2(x,train_inputs)
  sorted_indices=np.argsort(dist,axis=0)
  bin_results=np.bincount(np.int32(train[sorted_indices[:K]][-1]))
  return np.float32(np.argmax(bin_results))

def accuracy(pred):
  answer=load_csv.csv_reader("./pima/test_answer.csv")
  correct=0
  for pred_row,answer_row in zip(pred[1:],answer[1:]):
    if answer_row[1]==pred_row[1]:
      correct+=1
  return correct/(len(answer)-1)
  
def predict(test_inputs,K):  
  pred=[["ID","Outcome"]]  
  for i in range(len(test_inputs)):
    pred.append([test_inputs[i][0],KNN(test_inputs[i][1:],train,K=K)])
  return pred

if __name__=="__main__":
  train_load=load_csv.csv_reader("./pima/train.csv")
  val_load=load_csv.csv_reader("./pima/val.csv")
  test_load=load_csv.csv_reader("./pima/test.csv")
  
  train=np.array(train_load[1:])
  train=train[:,1:]
  test_inputs=test_load[1:]

  acc=[]
  pred=[]
  for k in range(1,201,2):
    pred.append(predict(test_inputs,k))
    acc.append(accuracy(pred[-1]))
  print(acc)
  plt.plot(list(range(1,201,2)),acc)
  plt.show()
  load_csv.csv_writer(pred[0],"./pima/KNN_pred.csv")


