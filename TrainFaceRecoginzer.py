# develop a classifier for the 5 Celebrity Faces Dataset
from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import os

import pickle
import numpy as np

# load dataset
data = load('data/Employe-faces-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
if not os.path.exists("models"):
	os.mkdir("models")
def TrainSkLearnModel(trainX,testX,trainy,testy):
	# normalize input vectors
	in_encoder = Normalizer(norm='l2')
	trainX = in_encoder.transform(trainX)
	testX = in_encoder.transform(testX)
	# label encode targets
	out_encoder = LabelEncoder()
	out_encoder.fit(trainy)
	classes=[c+"\n" for c in list(out_encoder.classes_)]
	with open("models/classes_name.txt",'w') as f:
		f.writelines(classes)
	trainy = out_encoder.transform(trainy)
	testy = out_encoder.transform(testy)
	# fit model
	model = SVC(kernel='linear', probability=True)
	model.fit(trainX, trainy)
	pickle.dump(model, open("models/detect.p", 'wb'))
	# predict
	yhat_train = model.predict(trainX)
	yhat_test = model.predict(testX)
	# score
	score_train = accuracy_score(trainy, yhat_train)
	score_test = accuracy_score(testy, yhat_test)
	# summarize
	print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

def getAvereageFeatureOfClasses(trainX,trainy):
	unique_Label=np.unique(trainy)
	train_data=np.concatenate((trainX,np.expand_dims(trainy,axis=1)),axis=1)
	train_data=[train_data[train_data[:,-1]==ul][:,:-1] for ul in unique_Label]
	train_mean=np.array([np.mean(data.astype(np.float),axis=0) for data in train_data])
	np.savez_compressed('data/classes_mean.npz', train_mean)
	print(("Seaved Class mean"))


def removePerson(name,trainX, testX, trainy, testy):
	trainX = trainX[trainy != name]
	trainy = trainy[trainy != name]
	testX = testX[testy != name]
	testy = testy[testy != name]
	return trainX, testX, trainy, testy


if __name__=="__main__":
	print(trainX.shape,trainy.shape)
	# trainX, testX, trainy, testy=removePerson("Parmpal", trainX, testX, trainy, testy)
	print(trainX.shape,trainy.shape)
	getAvereageFeatureOfClasses(trainX,  trainy)
	TrainSkLearnModel(trainX, testX, trainy, testy)
