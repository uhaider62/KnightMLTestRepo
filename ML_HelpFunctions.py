# import the necessary packages
import numpy as np
import cv2
import os
import time
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.preprocessing.image import img_to_array
from PIL import Image

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

class ML_hFunctions:
	
	def kfolds_distrb(kfolds=5, X_train=None, Y_train=None):
		
		data_length = len(Y_train)
		kfold_dataset_len = np.int_(data_length/kfolds)
		
		starting_point = 0
		cross_val_train_data=[]
		cross_val_train_labels=[]
		
		for i in range (0,kfolds):
			
			_data=X_train[starting_point:(starting_point + kfold_dataset_len)]
			_labels=Y_train[starting_point:(starting_point + kfold_dataset_len)]    
			cross_val_train_data.append(_data)
			cross_val_train_labels.append(_labels) 
			
			if i < (kfolds-1):
				starting_point = starting_point + kfold_dataset_len
		
		return cross_val_train_data, cross_val_train_labels
	
	def getDataAndLabels(path='dataset', with_FaceDetect=0):

		imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
		faceSamples=[]
		ids = []
		id_new=0
		
		for imagePath in imagePaths:
			
			PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
			img_numpy = np.array(PIL_img,'uint8')
			id = int(os.path.split(imagePath)[-1].split(".")[1])
			if id_new !=0:
				if id_new != id:
					id_new = id
					classes = classes + 1    
				else:
					id_new = id
			else:
				id_new = id
				classes = 1
				
			if with_FaceDetect != 0:   	 
				
				faces = detector.detectMultiScale(img_numpy,
		                                          scaleFactor = 1.2,#1.2
		                                          minNeighbors = 5)#5
				for (x,y,w,h) in faces:
				    image = img_numpy[y:y+h,x:x+w]
				#face_image = cv2.resize(image, (100, 100))
			else: 
				image = img_numpy
			
			faceSamples.append(image)
			id = classes-1
			ids.append(id)
		return faceSamples,ids,classes
	
	def getDataAndLabels_Resize(path='dataset',max_h=120,max_w=120):

	    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
	    faceSamples=[]
	    ids = []
	    id_new = 0
	    classes=0
	
	    for imagePath in imagePaths:
	
	        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
	        img_numpy = np.array(PIL_img,'uint8')
	
	        id = int(os.path.split(imagePath)[-1].split(".")[1])
	        if id_new !=0:
	            if id_new != id:
	                id_new = id
	                classes = classes + 1    
	            else:
	                id_new = id
	        else:
	            id_new = id
	            classes = 1        
	                
	        if len(img_numpy) < max_h or len(img_numpy[0]) < max_w:
	                 
	            face_image = cv2.resize(img_numpy, (max_h, max_w), interpolation = cv2.INTER_CUBIC)
	        else:
	            face_image = cv2.resize(img_numpy, (max_h, max_w), interpolation = cv2.INTER_AREA)
	            
	        face_image = img_to_array(face_image)
	        faceSamples.append(face_image)
	        id = classes-1
	        ids.append(id)
	       
	    return faceSamples,ids,classes
	   
	def validate_model_predict_func(model_type, model, data_set, labels, classes=None):
		
	    count=0
	    indx=0
	    acc=0
	    p=[]
	    w = len(data_set[0])
	    h= 1
	    data_set = np.array(data_set)
	    test_sets = len(data_set)
	    ro = [[0 for x in range(w)] for y in range(h)] 
	    avg_time = 0	    
	    
	    for i in range(0,test_sets):
	        
	        
	        start=time.time()
	        if model_type == 'svc':
	        	ro[0][:] = (data_set[i][:])
	        	p = np.array(model.predict(np.array(ro)))
	        	if (labels[i] == p[1]):
	        		count= count + 1
	        		
	        elif model_type == 'cnn':
	        	ro = []
	        	ro = data_set[i]
	        	a = len(ro)
	        	b= len(ro[0])
	        	ro = ro.reshape(1,a,b,1)
	        	class_prob = model.predict(ro)[0]
	        	for j in range(0,classes):
	        		if (class_prob[j] > 0.51):
	        			if labels[i] == j:
	        				print(labels[i],j,count)
	        				count = count + 1 
	        			break
	        end=time.time()      
	        avg_time = avg_time + (end-start)    
	    accu = (count/(len(labels)))*100
	    avg_time=avg_time/i
	    
	    return accu,avg_time
	
	def create_SURF_BoW(dataset=None, labels=None, vocabulary_size=50, hessianThreshold=100, nOctaves=2, nOctaveLayers=3, extended=0, upright=0):
		
		imgs2Keypoints = {}
		kmeansTrainer = cv2.BOWKMeansTrainer(vocabulary_size)  
		
		hessianThreshold = 100
		nOctaves = 2
		nOctaveLayers = 3
		extended = 0
		upright = 0 
		
		surf = cv2.xfeatures2d.SURF_create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright)
		bow_extr = cv2.BOWImgDescriptorExtractor(surf, cv2.BFMatcher(cv2.NORM_L2))
		
		avg_time_preprocessing  = 0
		avg_time0 = 0
		avg_time1 = 0
		
		for face in dataset:
		    start= time.time()
		    kk, des = surf.detectAndCompute(face, None)
		    des = np.float32(des)
		    kmeansTrainer.add(des)
		    end = time.time()
		    avg_time0 = avg_time0 + (end-start)
		    
		vocab = kmeansTrainer.cluster()
		
		bow_extr.setVocabulary(vocab) 
		idlist = np.array(labels)
		sampledata = []
		samplelabels = []
		count=0
		
		for face in dataset:
		    start= time.time()
		    kp = surf.detect(face)
		    bowsig = bow_extr.compute(face, kp)
		    end = time.time()
		    sampledata.extend( bowsig )
		    samplelabels.append(idlist[count])
		    count= count+1
		    avg_time1 = avg_time1 + (end-start)
		
		avg_time0 = avg_time0/len(dataset)
		avg_time1 = avg_time1/len(dataset)
		avg_time_preprocessing = avg_time0 + avg_time1
		
		return vocab,sampledata,samplelabels,avg_time_preprocessing 

