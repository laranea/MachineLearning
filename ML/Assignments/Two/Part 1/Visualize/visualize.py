from sklearn.manifold import TSNE
import h5py
import matplotlib.pyplot as plt

def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['x'][:]
		Y = hf['y'][:]
	return X, Y


################################################################################

x,y=load_h5py('./data_1.h5')

x_coordinates=x[:,0]
y_coordinates=x[:,1]

labels=['Class 0', 'Class 1']
colors=['r','g']

plt.scatter(x_coordinates,y_coordinates,color=colors,label=labels)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()



################################################################################


#x,y=load_h5py('./data_2.h5')


#labels=['Class 0', 'Class 1']
#colors=['r','g']


#tsne=TSNE(n_components=2,random_state=4).fit_transform(x)

#x_coordinates=x[:,0]
#y_coordinates=x[:,1]

#plt.scatter(x_coordinates,y_coordinates,color=colors,label=labels)
#plt.xlabel('Feature 1')
#plt.ylabel('Feature 2')
#plt.legend()
#plt.show()



################################################################################


#x,y=load_h5py('./data_3.h5')

#labels=['Class 0', 'Class 1']
#colors=['r','g']


#tsne=TSNE(n_components=2,random_state=5).fit_transform(x)

#x_coordinates=x[:,0]
#y_coordinates=x[:,1]

#plt.scatter(x_coordinates,y_coordinates,color=colors,label=labels)
#plt.xlabel('Feature 1')
#plt.ylabel('Feature 2')
#plt.legend()
#plt.show()



################################################################################


#x,y=load_h5py('./data_4.h5')

#labels=['Class 0', 'Class 1', 'Class 2']
#colors=['r','g','b']

#tsne=TSNE(n_components=2,random_state=6).fit_transform(x)

#x_coordinates=x[:,0]
#y_coordinates=x[:,1]

#plt.scatter(x_coordinates,y_coordinates,color=colors,label=labels)
#plt.xlabel('Feature 1')
#plt.ylabel('Feature 2')
#plt.legend()
#plt.show()



################################################################################


#x,y=load_h5py('./data_5.h5')


#labels=['Class 0', 'Class 1', 'Class 2']
#colors=['r','g','b']

#tsne=TSNE(n_components=2,random_state=34).fit_transform(x)

#x_coordinates=x[:,0]
#y_coordinates=x[:,1]

#plt.scatter(x_coordinates,y_coordinates,color=colors,label=labels)
#plt.xlabel('Feature 1')
#plt.ylabel('Feature 2')
#plt.legend()
#plt.show()
