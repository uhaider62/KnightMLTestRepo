# -*- coding: utf-8 -*-
# MLP for Pima Indians Dataset Serialize to JSON and HDF5
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.utils import to_categorical
from keras.models import load_model
from keras import backend as K
from keras.preprocessing.image import img_to_array
from keras.layers import Dropout
from PIL import Image
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from ML_HelpFunctions import ML_hFunctions
import matplotlib.pyplot as plt
import argparse
import sklearn.model_selection as skl
import numpy as np
import os
import cv2

# Path for face image database
path = 'dataset'

#np.random.seed(42)

#face detector vaiola jones algorithm, to detect and extract face from picture
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# fix random seed for reproducibility


use_data_generator = np.array(input("should I use data generator to increase the datasize (rotate,flip,height-width shifts of images \n - 0 for no \n - 1 for yes \n"), 'uint8')

# some global defines for trainning CNN
if use_data_generator == 1:
    
    EPOCHS = 75
else:
    EPOCHS = 75
use_data_generator = 0    
INIT_LR = 1e-3
#np.random.seed(7)
max_w = 120
max_h = 120
dataset_func = ML_hFunctions
# --------------------------------------------------------------------------
""" Extract faces and labels from dataset """
#        
faces,ids,classes = dataset_func.getDataAndLabels_Resize(path, max_h, max_w)

# --------------------------------------------------------------------------
""" Prepare dataset for training and testing """
# --------------------------------------------------------------------------
            
faces = np.array(faces, dtype="float") / 255.0
ids = np.array(ids)
a = len(faces)
faces = faces.reshape(a,max_h,max_w,1)
# create training, test data
X_train, X_test, Y_train, Y_test = skl.train_test_split(faces, ids, test_size=0.1)#, random_state=42)

# --------------------------------------------------------------------------
""" Create CNN model architecture """
# --------------------------------------------------------------------------
# create model sequentially. Add layers on top of previous one
model = Sequential()
""" Input layer with convolution """
inputShape = (max_h,max_w,1)
#conv layer with 34 5x5 filters. 
model.add(Conv2D(21, (3, 3), padding="same", input_shape=inputShape, data_format="channels_last", use_bias=True, bias_initializer="zeros"))
#using non-linear function for neuron activation. relu -> output x if x>0, otherwise output 0.
model.add(Activation("relu"))
# 2,2 pool/stride means reducing the size of input to the next layer by 2. 
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#regularization. Use drop out to prevent (to an extent) the model from overfitting. It will turn off some neurons while training.
model.add(Dropout(0.01))
""" Convolutional layer """
# second set of CONV => RELU => POOL layers
# 55  5x5 filters
model.add(Conv2D(34, (2, 2), padding="same", use_bias=True, bias_initializer="zeros"))
#relu -> output x if x>0, otherwise output 0.
model.add(Activation("relu"))
#regularization. Use drop out to prevent (to an extent) the model from overfitting. It will turn off some neurons while training.
model.add(Dropout(0.01))
""" first hidden layer """
# flatten inputs to the hidden layer
model.add(Flatten())
# Dense is -> fully connected layer of 100 neurons
model.add(Dense(50))
#relu -> output x if x>0, otherwise output 0.
model.add(Activation("relu"))
#regularization. Use drop out to prevent (to an extent) the model from overfitting. It will turn off some neurons while training.
model.add(Dropout(0.01))
""" Second hidden layer """
# Dense is -> fully connected layer of 100 neurons
model.add(Dense(50, use_bias=True, bias_initializer="zeros"))
#relu -> output x if x>0, otherwise output 0.
model.add(Activation("relu"))
#regularization. Use drop out to prevent (to an extent) the model from overfitting. It will turn off some neurons while training.
model.add(Dropout(0.01))
""" Third hidden layer """
# Dense is -> fully connected layer of 100 neurons
model.add(Dense(50, use_bias=True, bias_initializer="zeros"))
#relu -> output x if x>0, otherwise output 0.
model.add(Activation("relu"))
#regularization. Use drop out to prevent (to an extent) the model from overfitting. It will turn off some neurons while training.
model.add(Dropout(0.01))
""" Fourth hidden layer """
# Dense is -> fully connected layer of 100 neurons
model.add(Dense(100, use_bias=True, bias_initializer="zeros"))
#relu -> output x if x>0, otherwise output 0.
model.add(Activation("relu"))
#regularization. Use drop out to prevent (to an extent) the model from overfitting. It will turn off some neurons while training.
model.add(Dropout(0.01))
""" Output layer """
# softmax classifier
model.add(Dense(classes, use_bias=True, bias_initializer="zeros"))
#use softmax for activation for multi-class problem.
model.add(Activation("softmax"))

filepath="weights.best.hdf5"
EarlyStoppingLoss = EarlyStopping(monitor='val_loss', verbose=1, patience=30, min_delta=np.float64(0.0000001), mode='min', baseline=np.float64(0.0000001))
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [EarlyStoppingLoss, checkpoint]
# --------------------------------------------------------------------------
""" Compile and train CNN model 
    - using ADAM optimization algorithm
    - using crossentropy cost function for learning
    - cross-entropy cost function is C=−(1/n)∑x[ylna+(1−y)ln(1−a)]
    - where a=σ(z)
    - and z=∑jwjxj+b
    - w is weights of inputs to the neuron
    - b is bias
    """
# --------------------------------------------------------------------------
# generate extra data
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")
#compile model
#
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy", "crossentropy"])
print("[INFO] training network...")
# Fit the model
# train the network
batch_size = int(len(X_train)/10)
# convert the labels from integers to vectors
Y_train_gen = Y_train
Y_train = to_categorical(Y_train, num_classes=classes)
test_labels = Y_test
Y_test = to_categorical(Y_test, num_classes=classes)

if use_data_generator == 1:
    
    print("[INFO] using generator to increase datasize...")
    X_train1, X_test1, Y_train1, Y_test1 = skl.train_test_split(X_train, Y_train_gen, test_size=0.3)#, random_state=42)
    batch_size = int(len(X_train1)/5)
    # convert the labels from integers to vectors
    Y_train1 = to_categorical(Y_train1, num_classes=classes)
    Y_test1 = to_categorical(Y_test1, num_classes=classes)
    
    history = model.fit_generator(aug.flow(X_train1, Y_train1, batch_size=batch_size),
    validation_data=(X_test1, Y_test1), steps_per_epoch=len(X_train1) // batch_size,
    epochs=EPOCHS, callbacks=callbacks_list, verbose=1)

else:
    
    history = model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=batch_size, callbacks=callbacks_list, verbose=1, validation_split=0.1)

# --------------------------------------------------------------------------
""" Evaluate new CNN model """
# --------------------------------------------------------------------------
# evaluate the model
print("[INFO] validating network...")
[test_loss, test_accuracy, test_crossentropy] = model.evaluate(X_test, Y_test, verbose=1)
print("%s: %.2f%%" % ("validation accuracy: ", test_accuracy*100))
model.summary()
print("returned metrics for train-validation: "+str(history.history.keys()))
# --------------------------------------------------------------------------
""" Check prediction time of trained model and memory usage"""
# --------------------------------------------------------------------------
print("--------------------------------------------------------------------------------------------")
print("[INFO] network estimated prediction time and memory usage is...")
accu,avg_time = dataset_func.validate_model_predict_func('cnn',model,X_test,test_labels,classes)
print("prediction accuracy: "+str(accu) + " %")
print("average prediction time: "+str(avg_time) + " sec")
model.save("cnnmodel_temp")  
model_info = os.stat('cnnmodel_temp')
model_mem_size = model_info.st_size
print("estimate total memory usage for this model and BoW: "+str((model_mem_size)/1024)+" KB")
print("--------------------------------------------------------------------------------------------")
os.remove('cnnmodel_temp')
# --------------------------------------------------------------------------
""" model loss and accuracy plots"""
# --------------------------------------------------------------------------
#Plot the Loss Curves
plt.figure(num=1,figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
plt.show()
#Plot the Accuracy Curves
plt.figure(num=2,figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
plt.show()
# --------------------------------------------------------------------------
""" Save model in a file for later use """
# --------------------------------------------------------------------------
save_model = np.array(input("Do you want to save the model?  \n - 0 for no \n - 1 for yes \n"), 'uint8')


if save_model !=0:
    model_name = input("Give model name to save as - no spaces.")
    model.save(str(model_name))


#########################################################################################################
# --------------------------------------------------------------------------
""" Load model file once to check save was successfull """
# --------------------------------------------------------------------------
# CNNModel = load_model("CNNModel")
# face = faces[0]
# face = face.reshape(1,100,100,1)
# a = model.predict(face)
# print("load model and use test image to verify load successfull "+str(np.amax(a)))


# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cv2.destroyAllWindows()
# To save model as json 

# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")
#  
# # later...
#  
# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
#  
# # evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# score = loaded_model.evaluate(X_test, Y_test, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


#########################################################################################################
