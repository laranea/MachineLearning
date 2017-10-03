import os
import os.path
import argparse
import h5py
from sklearn import linear_model,tree
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str  )
parser.add_argument("--weights_path", type = str)
parser.add_argument("--train_data", type = str  )
parser.add_argument("--plots_save_dir", type = str  )

args = parser.parse_args()
mdl=args.model_name
datapath=args.train_data
plot_dir=args.plots_save_dir
weights_path=args.weights_path

# Load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y


# Preprocess data and split it
x,y=load_h5py(datapath)

Y=[]
for k in y:
	for i in range(0,y.shape[1]):
		if k[i]==1:
			Y.append(i)



number_of_samples=len(Y)



# k=1
num_train=int(number_of_samples*0.7)
num_test=int(number_of_samples*0.3)

random_indices = np.random.permutation(number_of_samples)

x_train = x[random_indices[:num_train]]
x_test = x[random_indices[num_train:]]

y_train=[]
y_test=[]

r1=random_indices[:num_train]
r2=random_indices[num_train:]
for i in r1:
	y_train.append(Y[i])
for i in r2:
	y_test.append(Y[i])


# Train the models

if args.model_name == 'GaussianNB':


#	model= GaussianNB()
#	model.fit(x_train,y_train)
#	y_pred=model.predict(x_test)
#	classification=0
#	for i in range(len(y_pred)):
#		if y_pred[i]==y_test[i]:
#			classification+=1

#	accuracy = (float(classification)/float(len(y_pred)))*100
#	print('Accuracy:', accuracy)


#	output = open('./Weights/GaussianNB_C.pkl', 'wb')
#	pickle.dump(x_train, output)
#	pickle.dump(x_test, output)
#	pickle.dump(y_train, output)
#	pickle.dump(y_test, output)
#	output.close()

	acc=[]
	for k in range(0,3):
		model=GaussianNB()
		if k==0:
			model.fit(x_train_1,y_train_1)
			y_pred=model.predict(x_test_1)
			classification=0
			for i in range(len(y_pred)):
				if y_pred[i]==y_test_1[i]:
					classification+=1

					accuracy = (float(classification)/float(len(y_pred)))*100
					acc.append(accuracy)
					print('Accuracy:', accuracy)

		if k==1:
			model.fit(x_train_2,y_train_2)
			y_pred=model.predict(x_test_2)
			classification=0
			for i in range(len(y_pred)):
				if y_pred[i]==y_test_2[i]:
					classification+=1

					accuracy = (float(classification)/float(len(y_pred)))*100
					acc.append(accuracy)
					print('Accuracy:', accuracy)

		if k==2:
			model.fit(x_train_3,y_train_3)
			y_pred=model.predict(x_test_3)
			classification=0
			for i in range(len(y_pred)):
				if y_pred[i]==y_test_3[i]:
					classification+=1

					accuracy = (float(classification)/float(len(y_pred)))*100
					acc.append(accuracy)
					print('Accuracy:', accuracy)

	plt.plot(acc)
	plt.title('GaussianNB K Cross Validation')
	plt.show()

	if args.weights_path:
		output = open(weights_path+'/GaussianNB.pkl', 'wb')
		pickle.dump(x_train, output)
		pickle.dump(x_test, output)
		pickle.dump(y_train, output)
		pickle.dump(y_test, output)
		output.close()






elif args.model_name == 'LogisticRegression':


#	model= linear_model.LogisticRegression()
#	model.fit(x_train,y_train)
#	y_pred=model.predict(x_test)
#	classification=0
#	for i in range(len(y_pred)):
#		if y_pred[i]==y_test[i]:
#			classification+=1

#	accuracy = (float(classification)/float(len(y_pred)))*100
#	print('Accuracy:', accuracy)


#	output = open('./Weights/LogisticRegression_C.pkl', 'wb')
#	pickle.dump(x_train, output)
#	pickle.dump(x_test, output)
#	pickle.dump(y_train, output)
#	pickle.dump(y_test, output)
#	output.close()

	for k in range(0,3):
		model=linear_model.LogisticRegression()
		if k==0:
			model.fit(x_train_1,y_train_1)
			y_pred=model.predict(x_test_1)
			classification=0
			for i in range(len(y_pred)):
				if y_pred[i]==y_test_1[i]:
					classification+=1

					accuracy = (float(classification)/float(len(y_pred)))*100
					print('Accuracy:', accuracy)

			penalty_val=['l1','l2']
			max_iter_val=[100,150,200]
			C_val=[1,2,3]

			acc=[]
			param=[]

			for pen in penalty_val:
				for iterv in max_iter_val:
					for c in C_val:
							model=linear_model.LogisticRegression(penalty=pen,max_iter=iterv,C=c)
							model.fit(x_train_1,y_train_1)
							y_pred=model.predict(x_test_1)
							classification=0
							for i in range(len(y_pred)):
								if y_pred[i]==y_test_1[i]:
									classification+=1
							accuracy = (float(classification)/float(len(y_pred)))*100
							tup=(pen,iterv,c)
							acc.append(accuracy)
							param.append(tup)
			print acc
			plt.plot(acc)
			plt.show()


		if k==1:
			model.fit(x_train_2,y_train_2)
			y_pred=model.predict(x_test_2)
			classification=0
			for i in range(len(y_pred)):
				if y_pred[i]==y_test_2[i]:
					classification+=1

					accuracy = (float(classification)/float(len(y_pred)))*100
					print('Accuracy:', accuracy)

			penalty_val=['l1','l2']
			max_iter_val=[100,150,200]
			C_val=[1,2,3]

			acc=[]
			param=[]

			for pen in penalty_val:
				for iterv in max_iter_val:
					for c in C_val:
							model=linear_model.LogisticRegression(penalty=pen,max_iter=iterv,C=c)
							model.fit(x_train_2,y_train_2)
							y_pred=model.predict(x_test_2)
							classification=0
							for i in range(len(y_pred)):
								if y_pred[i]==y_test_2[i]:
									classification+=1
							accuracy = (float(classification)/float(len(y_pred)))*100
							tup=(pen,iterv,c)
							acc.append(accuracy)
							param.append(tup)
			print acc
			plt.plot(acc)
			plt.show()

		if k==2:
			model.fit(x_train_3,y_train_3)
			y_pred=model.predict(x_test_3)
			classification=0
			for i in range(len(y_pred)):
				if y_pred[i]==y_test_3[i]:
					classification+=1

					accuracy = (float(classification)/float(len(y_pred)))*100
					print('Accuracy:', accuracy)

			penalty_val=['l1','l2']
			max_iter_val=[100,150,200]
			C_val=[1,2,3]

			acc=[]
			param=[]

			for pen in penalty_val:
				for iterv in max_iter_val:
					for c in C_val:
							model=linear_model.LogisticRegression(penalty=pen,max_iter=iterv,C=c)
							model.fit(x_train_3,y_train_3)
							y_pred=model.predict(x_test_3)
							classification=0
							for i in range(len(y_pred)):
								if y_pred[i]==y_test_3[i]:
									classification+=1
							accuracy = (float(classification)/float(len(y_pred)))*100
							tup=(pen,iterv,c)
							acc.append(accuracy)
							param.append(tup)
			print acc
			plt.plot(acc)
			plt.show()



	if args.plots_save_dir:
		plt.savefig(plot_dir)

		if args.weights_path:
			output = open(weights_path+'/LogisticRegression.pkl', 'wb')
			pickle.dump(x_train, output)
			pickle.dump(x_test, output)
			pickle.dump(y_train, output)
			pickle.dump(y_test, output)
			output.close()


elif args.model_name == 'DecisionTreeClassifier':

#	model=tree.DecisionTreeClassifier()
#	model.fit(x_train,y_train)
#	y_pred=model.predict(x_test)
#	classification=0
#	for i in range(len(y_pred)):
#		if y_pred[i]==y_test[i]:
#			classification+=1

#	accuracy = (float(classification)/float(len(y_pred)))*100
#	print('Accuracy:', accuracy)


#	output = open('./Weights/DecisionTreeClassifier_A.pkl', 'wb')
#	pickle.dump(x_train, output)
#	pickle.dump(x_test, output)
#	pickle.dump(y_train, output)
#	pickle.dump(y_test, output)
#	output.close()

	model=tree.DecisionTreeClassifier()


	for k in range(0,3):
		model=tree.DecisionTreeClassifier()
		if k==0:
			model.fit(x_train_1,y_train_1)
			y_pred=model.predict(x_test_1)
			classification=0
			for i in range(len(y_pred)):
				if y_pred[i]==y_test_1[i]:
					classification+=1

					accuracy = (float(classification)/float(len(y_pred)))*100
					print('Accuracy:', accuracy)


			min_samples_split_val=[2,3,4]
			max_depth_val=[5,7,10]
			min_samples_leaf_val=[1,2,3]

			acc=[]
			param=[]

			for depth in max_depth_val:
				for sleaf in min_samples_leaf_val:
					for ssplit in min_samples_split_val:
						model=tree.DecisionTreeClassifier(max_depth=depth,min_samples_split=ssplit,min_samples_leaf=sleaf)
						model.fit(x_train_1,y_train_1)
						y_pred=model.predict(x_test_1)
						classification=0
						for i in range(len(y_pred)):
							if y_pred[i]==y_test_1[i]:
								classification+=1
						accuracy = (float(classification)/float(len(y_pred)))*100
						tup=(depth,sleaf,ssplit)
						acc.append(accuracy)
						param.append(tup)
			print acc,param
			plt.plot(acc)
			plt.show()


		if k==1:
			model.fit(x_train_2,y_train_2)
			y_pred=model.predict(x_test_2)
			classification=0
			for i in range(len(y_pred)):
				if y_pred[i]==y_test_2[i]:
					classification+=1

					accuracy = (float(classification)/float(len(y_pred)))*100
					print('Accuracy:', accuracy)

			penalty_val=['l1','l2']
			max_iter_val=[100,150,200]
			C_val=[1,2,3]

			acc=[]
			param=[]

			min_samples_split_val=[2,3,4]
			max_depth_val=[5,7,10]
			min_samples_leaf_val=[1,2,3]

			acc=[]
			param=[]

			for depth in max_depth_val:
				for sleaf in min_samples_leaf_val:
					for ssplit in min_samples_split_val:
						model=tree.DecisionTreeClassifier(max_depth=depth,min_samples_split=ssplit,min_samples_leaf=sleaf)
						model.fit(x_train_2,y_train_2)
						y_pred=model.predict(x_test_2)
						classification=0
						for i in range(len(y_pred)):
							if y_pred[i]==y_test_2[i]:
								classification+=1
						accuracy = (float(classification)/float(len(y_pred)))*100
						tup=(depth,sleaf,ssplit)
						acc.append(accuracy)
						param.append(tup)
			print acc,param
			plt.plot(acc)
			plt.show()

		if k==2:
			model.fit(x_train_3,y_train_3)
			y_pred=model.predict(x_test_3)
			classification=0
			for i in range(len(y_pred)):
				if y_pred[i]==y_test_3[i]:
					classification+=1

					accuracy = (float(classification)/float(len(y_pred)))*100
					print('Accuracy:', accuracy)

			min_samples_split_val=[2,3,4]
			max_depth_val=[5,7,10]
			min_samples_leaf_val=[1,2,3]

			acc=[]
			param=[]

			for depth in max_depth_val:
				for sleaf in min_samples_leaf_val:
					for ssplit in min_samples_split_val:
						model=tree.DecisionTreeClassifier(max_depth=depth,min_samples_split=ssplit,min_samples_leaf=sleaf)
						model.fit(x_train_3,y_train_3)
						y_pred=model.predict(x_test_3)
						classification=0
						for i in range(len(y_pred)):
							if y_pred[i]==y_test_3[i]:
								classification+=1
						accuracy = (float(classification)/float(len(y_pred)))*100
						tup=(depth,sleaf,ssplit)
						acc.append(accuracy)
						param.append(tup)
			print acc,param
			plt.plot(acc)
			plt.show()


	if args.plots_save_dir:
		plt.savefig(plot_dir) # save image

	if args.weights_path:
		output = open(weights_path+'/DecisionTreeClassifier.pkl', 'wb')
		pickle.dump(x_train, output)
		pickle.dump(x_test, output)
		pickle.dump(y_train, output)
		pickle.dump(y_test, output)
		output.close()

else:
	raise Exception("Invalid Model name")
