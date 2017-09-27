import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import csv
import numpy as np

json_train=open('./train.json').read()
traindata=json.loads(json_train)

json_test=open('./test.json').read()
testdata=json.loads(json_test)


x_train=[]
y_train=[]

for d in traindata:
    x=''
    for i in d["X"]:
        x+=(str(i)+' ')
    x_train.append(x)
    y_train.append(d["Y"])

x_train=np.array(x_train)
y_train=np.array(y_train)


x_test=[]

for d in testdata:
    x=''
    for i in d["X"]:
        x+=(str(i)+' ')
    x_test.append(x)

x_test=np.array(x_test)

vctzr=TfidfVectorizer(ngram_range=(1,3),min_df=0)
x_train=vctzr.fit_transform(x_train)
x_test=vctzr.transform(x_test)


model=svm.LinearSVC(C=0.34)
model.fit(x_train,y_train)



pred=model.predict(x_test)

with open('./output.csv','wb') as f:
    wr=csv.writer(f,delimiter=',')
    wr.writerow(("Id","Expected"))
    for val in range(len(pred)):
        wr.writerow((val+1,pred[val]))
