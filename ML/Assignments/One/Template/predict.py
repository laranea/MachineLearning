import os
import os.path
import argparse
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str  )
parser.add_argument("--weights_path", type = str)
parser.add_argument("--test_data", type = str  )
parser.add_argument("--output_preds_file", type = str  )

args = parser.parse_args()
args = parser.parse_args()
mdl=args.model_name
datapath=args.train_data
outputfile=args.output_preds_file

# load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y


x,y=load_h5py(datapath)

Y=[]
for k in y:
	for i in range(0,y.shape[1]):
		if k[i]==1:
			Y.append(i)



number_of_samples=len(Y)


num_train=int(number_of_samples*0.7)
num_test=int(number_of_samples*0.30)

random_indices = np.random.permutation(number_of_samples)

x_train = x[random_indices[:num_train]]
x_test = x[random_indices[num_train]]


y_train=[]
y_test=[]

r1=random_indices[:num_train]
r2=random_indices[num_train:]
for i in r1:
	y_train.append(Y[i])
for i in r2:
	y_test.append(Y[i])



if args.model_name == 'GaussianNB':
	model= GaussianNB()
	model.fit(x_train,y_train)

elif args.model_name == 'LogisticRegression':
	model=linear_model.LogisticRegression(C=1e5)
	model.fit(x_train,y_train)

elif args.model_name == 'DecisionTreeClassifier':
	model=tree.DecisionTreeClassifier()
	model.fit(x_train,y_train)

else:
	raise Exception("Invald Model name")


# Save to File
fl = open(outputfile, 'w')
predictions=model.predict(x_test)
print(predictions)
for item in predictions:
  fl.write("%s\n" % item)
