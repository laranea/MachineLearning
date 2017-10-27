import os
import os.path
import argparse
from sklearn.manifold import TSNE
import h5py
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--data", type = str  )
parser.add_argument("--plots_save_dir", type = str  )

args = parser.parse_args()
datapath=args.data
plot_dir=args.plots_save_dir

def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y

x,y=load_h5py(datapath)

classlabel=[]

for k in y:
	for i in range(0,y.shape[1]):
		if k[i]==1:
			classlabel.append(i)


tsne=TSNE(n_components=2).fit_transform(x)

x_coordinates=tsne[:,0]
y_coordinates=tsne[:,1]

plt.scatter(x_coordinates,y_coordinates,label=classlabel)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


#print tsne
