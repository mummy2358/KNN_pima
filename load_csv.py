# This tool is not dependent on dataset
# Dataset should be in the format:
# [ ['id', 'age',...,'category_n'],
#   [   1,    20,...,     2019.90],
#   ...
#   ,
#   [ 100,    30,...,     2100.90]]

import tensorflow as tf
import csv

def csv_reader(filename):
  # read csv data as a list and convert the second and subsequent lines to float
  ans=[]
  with open(filename,"r",newline="") as f:
    data=csv.reader(f,delimiter=",")
    i=0
    for row in data:
      if i>0:
        for j in range(len(row)):
          row[j]=float(row[j])
      ans.append(row)
      i+=1
  return ans

def csv_writer(inputs,filename):
  # write inputs into a given csv file name (eg:"mydata.csv")
  # take care of the type of inputs. It's elements should be transformed back to "str"s
  with open(filename,"w") as f:
    writer=csv.writer(f,delimiter=",")
    ans=inputs
    for i in range(len(inputs)):
      for e in range(len(inputs[i])):
        if not isinstance(inputs[i][e],str):
          ans[i][e]=str(inputs[i][e])
      writer.writerow(ans[i])
    print("successfully written to "+filename)


"""
#[Run this part for debuging the reader and writer]#

pima=csv_reader("./Pima.csv")
print(pima)
csv_writer(pima,"./Pima_written.csv")
pima_read=csv_reader("./Pima_written.csv")
print(pima_read)

#[End]#
"""
