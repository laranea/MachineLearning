import numpy as np
from sklearn import svm
import h5py
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

def kernel(params,sv,X):
    k = [math.exp(-params['gamma'] * np.linalg.norm(vi - X) ** 2) for vi in sv]
    return k

def predict(params, sv, nv, a, b, cs, X):
    #k=kernel(params,sv,X)
    k = [math.exp(-params['gamma'] * np.linalg.norm(vi - X) ** 2) for vi in sv]
    start = [sum(nv[:i]) for i in range(len(nv))]
    end = [start[i] + nv[i] for i in range(len(nv))]
    c = [ sum(a[ i ][p] * k[p] for p in range(start[j], end[j])) + sum(a[j-1][p] * k[p] for p in range(start[i], end[i])) for i in range(len(nv)) for j in range(i+1,len(nv))]
    decision= [sum(x) for x in zip(c, b)]
    votes = [(i if decision[p] > 0 else j) for p,(i,j) in enumerate((i,j) for i in range(len(cs)) for j in range(i+1,len(cs)))]
    return cs[max(set(votes), key=votes.count)]

def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['x'][:]
		Y = hf['y'][:]
	return X, Y


x,y=load_h5py('../data_5.h5')



number_of_samples=len(y)

random_indices=np.random.permutation(number_of_samples)
num_train=int(number_of_samples*0.7)
num_test=int(number_of_samples*0.3)

x_train=x[random_indices[:num_train]]
y_train=y[random_indices[:num_train]]

x_test=x[random_indices[num_train:]]
y_test=y[random_indices[num_train:]]


clf = svm.SVC(gamma=0.001, C=100)
clf.fit(x_train, y_train)
params = clf.get_params()
sv = clf.support_vectors_
nv = clf.n_support_
a  = clf.dual_coef_
b  = clf._intercept_
cs = clf.classes_

plt.scatter(x[:,0], x[:,1], c=y, s=30, cmap=plt.cm.Paired)


ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,linestyles=['--', '-', '--'])
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,linewidth=1, facecolors='none')
plt.show()


pred=[]
for x in x_test:
    pred.append(predict(params, sv, nv, a, b, cs, x))

correct_classification=0
for i in range(len(pred)):
    if pred[i]==y_test[i]:
        correct_classification+=1
accuracy=(correct_classification/len(y_test))*100
print accuracy






confusion=confusion_matrix(y_test,pred)
print confusion
fpr,tpr,threshold=roc_curve(y_test,pred)
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
