import os 
import caffe 
import datetime
import random
import numpy as np 
from sklearn.externals import joblib

FEATURES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Features')

def save_features(app, label):
	# set category 
	if app.classifier == app.SVMClf:
		category = 'SIFT'
	elif app.classifier == app.AlexClf:
		category = 'CNNALEX'
	else:
		category = 'CNNLENET'

	# get features
	if app.classifier == app.AlexClf:
		features = app.classifier.net.blobs['fc7'].data[0]
	elif app.classifier == app.SVMClf:
		features = app.classifier.features
	else:
		features = app.classifier.net.blobs['loss3/classifier'].data[0]

	fileName = str(datetime.datetime.now()).replace(' ','_') + '_featrure.txt'
	joblib.dump((features, category, app.imName, label), os.path.join(FEATURES_FOLDER, fileName), compress=3)

def get_history():
	listFile = [os.path.join(FEATURES_FOLDER, f) for f in os.listdir(FEATURES_FOLDER)]
	history = []
	if len(listFile) > 10:
		random.shuffle(listFile)
		for f in listFile[0:9]:
			features, category, imName, label = joblib.load(f)
			history.append([imName,label])
	elif len(listFile) > 0:
		for f in listFile:
			features, category, imName, label = joblib.load(f)
			history.append([imName,label])
	return history 