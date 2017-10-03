from sklearn.manifold import TSNE
import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['x'][:]
		Y = hf['y'][:]
	return X, Y


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


################################################################################


x,y=load_h5py('./data_1.h5')

C=1.0
fig, sub = plt.subplots(2, 2)

models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(x, y) for clf in models)


titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')


x_coordinates=x[:,0]
y_coordinates=x[:,1]

xmean=np.mean(x_coordinates)
ymean=np.mean(y_coordinates)

xsd=np.std(x_coordinates)
ysd=np.std(y_coordinates)

newx=[]
newy=[]

for index,val in enumerate(x_coordinates):
	if (val > (xmean-2*xsd)) or (val < (xmean+2*xsd)):
		 	newx.append(val)
			newy.append(y_coordinates[index])

newxx=[]
newyy=[]

for index,val in enumerate(newy):
	if (val > (ymean-2*ysd)) or (val < (ymean+2*ysd)):
		 	newyy.append(val)
			newxx.append(newx[index])

newxx=np.array(newxx)
newyy=np.array(newyy)

xx, yy = make_meshgrid(newxx, newyy)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(x_coordinates, y_coordinates, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title('SVM decision Boundary')

plt.show()


################################################################################

x,y=load_h5py('./data_2.h5')

C=1.0
fig, sub = plt.subplots(2, 2)

models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(x, y) for clf in models)


titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')


x_coordinates=x[:,0]
y_coordinates=x[:,1]

xmean=np.mean(x_coordinates)
ymean=np.mean(y_coordinates)

xsd=np.std(x_coordinates)
ysd=np.std(y_coordinates)

newx=[]
newy=[]

for index,val in enumerate(x_coordinates):
	if (val > (xmean-2*xsd)) or (val < (xmean+2*xsd)):
		 	newx.append(val)
			newy.append(y_coordinates[index])

newxx=[]
newyy=[]

for index,val in enumerate(newy):
	if (val > (ymean-2*ysd)) or (val < (ymean+2*ysd)):
		 	newyy.append(val)
			newxx.append(newx[index])

newxx=np.array(newxx)
newyy=np.array(newyy)

xx, yy = make_meshgrid(newxx, newyy)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(x_coordinates, y_coordinates, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title('SVM decision Boundary')

plt.show()


################################################################################


x,y=load_h5py('./data_3.h5')


C=1.0
fig, sub = plt.subplots(2, 2)

models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(x, y) for clf in models)


titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')


x_coordinates=x[:,0]
y_coordinates=x[:,1]

xmean=np.mean(x_coordinates)
ymean=np.mean(y_coordinates)

xsd=np.std(x_coordinates)
ysd=np.std(y_coordinates)

newx=[]
newy=[]

for index,val in enumerate(x_coordinates):
	if (val > (xmean-2*xsd)) or (val < (xmean+2*xsd)):
		 	newx.append(val)
			newy.append(y_coordinates[index])

newxx=[]
newyy=[]

for index,val in enumerate(newy):
	if (val > (ymean-2*ysd)) or (val < (ymean+2*ysd)):
		 	newyy.append(val)
			newxx.append(newx[index])

newxx=np.array(newxx)
newyy=np.array(newyy)

xx, yy = make_meshgrid(newxx, newyy)


for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(x_coordinates, y_coordinates, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title('SVM decision Boundary')

plt.show()


################################################################################


x,y=load_h5py('./data_4.h5')


C=1.0
fig, sub = plt.subplots(2, 2)

models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(x, y) for clf in models)


titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')


x_coordinates=x[:,0]
y_coordinates=x[:,1]

xmean=np.mean(x_coordinates)
ymean=np.mean(y_coordinates)

xsd=np.std(x_coordinates)
ysd=np.std(y_coordinates)

newx=[]
newy=[]

for index,val in enumerate(x_coordinates):
	if (val > (xmean-2*xsd)) or (val < (xmean+2*xsd)):
		 	newx.append(val)
			newy.append(y_coordinates[index])

newxx=[]
newyy=[]

for index,val in enumerate(newy):
	if (val > (ymean-2*ysd)) or (val < (ymean+2*ysd)):
		 	newyy.append(val)
			newxx.append(newx[index])

newxx=np.array(newxx)
newyy=np.array(newyy)

xx, yy = make_meshgrid(newxx, newyy)


for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(x_coordinates, y_coordinates, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title('SVM decision Boundary')

plt.show()


################################################################################


x,y=load_h5py('./data_5.h5')

C=1.0
fig, sub = plt.subplots(2, 2)

models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(x, y) for clf in models)

titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')


x_coordinates=x[:,0]
y_coordinates=x[:,1]

xmean=np.mean(x_coordinates)
ymean=np.mean(y_coordinates)

xsd=np.std(x_coordinates)
ysd=np.std(y_coordinates)

newx=[]
newy=[]

for index,val in enumerate(x_coordinates):
	if (val > (xmean-2*xsd)) or (val < (xmean+2*xsd)):
		 	newx.append(val)
			newy.append(y_coordinates[index])

newxx=[]
newyy=[]

for index,val in enumerate(newy):
	if (val > (ymean-2*ysd)) or (val < (ymean+2*ysd)):
		 	newyy.append(val)
			newxx.append(newx[index])

newxx=np.array(newxx)
newyy=np.array(newyy)

xx, yy = make_meshgrid(newxx, newyy)


for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(x_coordinates, y_coordinates, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title('SVM decision Boundary')

plt.show()


################################################################################
