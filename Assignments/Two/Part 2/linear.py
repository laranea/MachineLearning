import numpy as np
from sklearn import svm
import h5py
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['x'][:]
		Y = hf['y'][:]
	return X, Y

################################################################################

#x,y=load_h5py('./data_1.h5')
#x,y=load_h5py('./data_2.h5')
#x,y=load_h5py('./data_3.h5')
#x,y=load_h5py('./data_4.h5')
x,y=load_h5py('./data_5.h5')


number_of_samples=len(y)

random_indices=np.random.permutation(number_of_samples)

num_train=int(number_of_samples*0.7)
num_test=int(number_of_samples*0.30)

x_train=x[random_indices[:num_train]]
y_train=y[random_indices[:num_train]]

x_test=x[random_indices[num_train:]]
y_test=y[random_indices[num_train:]]


C_val=[1,2,3,4,5,6,7,8,9,10]
gamma_val=[0.0001,0.005,0.01,0.02,0.03,0.04,0.05,0.10,0.2]
accuracy=[]
cmatrix=[]

for gamma in gamma_val:
    for c in C_val:
        model = svm.SVC(kernel='linear',C=c)
        model.fit(x_train, y_train)
        y_pred=model.predict(x_test)
        acc=accuracy_score(y_test,y_pred)*100
        accuracy.append(acc)
        confusion=confusion_matrix(y_test,y_pred)
        cmatrix.append(confusion)

plt.plot(accuracy)
plt.show()
print accuracy
print cmatrix
