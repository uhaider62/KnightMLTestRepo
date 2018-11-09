import numpy as np
import cv2
from PIL import Image
import os
import time
import datetime
import sklearn.model_selection as skl
from ML_HelpFunctions import ML_hFunctions

# Path for face image database
path = 'dataset'

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
Vocab_size = 80
dataset_func = ML_hFunctions
# --------------------------------------------------------------------------
""" load and prepare data """
# --------------------------------------------------------------------------
faces,ids,classes = dataset_func.getDataAndLabels(path, with_FaceDetect=1)
# --------------------------------------------------------------------------
""" Prepare BoW (using SURF) for training and testing """
# --------------------------------------------------------------------------
hessianThreshold = 100
nOctaves = 2
nOctaveLayers = 3
extended = 0
upright = 0 
vocab,sampledata,samplelabels,avg_time_preprocessing = dataset_func.create_SURF_BoW(dataset=faces, labels=ids, vocabulary_size=Vocab_size, hessianThreshold=100, nOctaves=2, nOctaveLayers=3,extended=0,upright=0)
fs = cv2.FileStorage("vocab_BoW_size.yml", cv2.FILE_STORAGE_WRITE) 
fs.write(name='matrix', val=vocab)
vocab_info = os.stat('vocab_BoW_size.yml')
vocab_mem_size = vocab_info.st_size
# --------------------------------------------------------------------------
""" split data """
# --------------------------------------------------------------------------
X_train, X_test, Y_train, Y_test = skl.train_test_split(sampledata, samplelabels, test_size=0.1, random_state=42)
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
""" Prepare data for kFolds cross validation """
# -------------------------------------------------------------------------- 
kfolds=5
cross_val_train_data, cross_val_train_labels = dataset_func.kfolds_distrb(kfolds, X_train, Y_train)
# --------------------------------------------------------------------------

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
# --------------------------------------------------------------------------
""" Train SVM models """
# -------------------------------------------------------------------------- 
gamma_range = np.logspace(-9, 3, 13)
print(gamma_range)
C_range = np.logspace(-2, 10, 13)
print(C_range)
kernel_type = [cv2.ml.SVM_LINEAR, cv2.ml.SVM_RBF, cv2.ml.SVM_POLY]
kernel_str = ['linear', 'RBF', 'POLY']
Y_test = np.array(Y_test)

# --------------------------------------------------------------------------
""" C parameter and gamma parameter grid training"""
# -------------------------------------------------------------------------- 


for types in  [0, 1, 2]:
    
    # Train the SVM:
    model = cv2.ml.SVM_create()
    model.setType(cv2.ml.SVM_C_SVC)
    model.setKernel(kernel_type[types])
    print("****************************")    
    print(kernel_str[types])
    print("****************************") 
    # make a log grid for C and gamma, select the diognals 
    for C in C_range:
        
        for gamma in gamma_range:
            # --------------------------------------------------------------------------
            """ Model settings for this iteration """
            # --------------------------------------------------------------------------  
            model.setC(C)
            if types > 0:
                model.setGamma(gamma)

            if types > 1:
                model.setDegree(5)
            
            model.setTermCriteria((cv2.TERM_CRITERIA_EPS, 1000, 1e-5))
            
            max_valid_score = 0
            crossvalidated_models = []
            crossvalid_models_accur = []
            # --------------------------------------------------------------------------
            """ K-fold cross-validation training """
            # --------------------------------------------------------------------------
            
            for kfold_models in range(0,kfolds):
                
                crossvalidated_models.append(model)
                 
            for kfolds_valid in range(0,kfolds):
                
                if kfolds_valid == (kfolds-1):
                    
                    valid_dat = cross_val_train_data[kfolds_valid]
                    valid_lab = cross_val_train_labels[kfolds_valid]
                else:
                    valid_dat = cross_val_train_data[kfolds_valid+1]
                    valid_lab = cross_val_train_labels[kfolds_valid+1]
                    
                for j in range(0,kfolds):
                    if j != kfolds_valid:
                        crossvalidated_models[kfolds_valid].train(np.array(cross_val_train_data[j]), cv2.ml.ROW_SAMPLE, np.array(cross_val_train_labels[j]))
                        
                val_acc,avg_time = dataset_func.validate_model_predict_func('svc',crossvalidated_models[kfolds_valid],valid_dat,valid_lab,classes)
                crossvalid_models_accur.append(val_acc)
            
            # --------------------------------------------------------------------------
            """ Select the model with best validation accuracy from K-folds """
            # --------------------------------------------------------------------------    
            for j in range(0,kfolds):
                
                if j > 0:
                    if crossvalid_models_accur[j] > crossvalid_models_accur[j-1]:
                        max_valid_score = crossvalid_models_accur[j]
                        selected_model = crossvalidated_models[j]
                else:
                    max_valid_score = crossvalid_models_accur[j]
                    selected_model = crossvalidated_models[j]
            
            # --------------------------------------------------------------------------
            """ Retrain the model on full train data and then validate with test data """
            # --------------------------------------------------------------------------  
            # train the selected model on the whole train data set 
            selected_model.train(np.array(X_train), cv2.ml.ROW_SAMPLE, np.array(Y_train))      
            accu,avg_time = dataset_func.validate_model_predict_func('svc',selected_model,X_test,Y_test,classes)
            vecs = np.array(model.getSupportVectors())
            
            # --------------------------------------------------------------------------
            """ If true accuracy of model is more than 80% then it is in short list of selectable models """
            # --------------------------------------------------------------------------  
            if accu > 80:
                
                print("--------------------------------------------------------------------------------------------")
                print(kernel_str[types]+", value C "+str(C)+", value gamma "+str(gamma))
                for i in range(0,kfolds):
                    print ("model valid score with kfold "+str(i)+" is: "+str(crossvalid_models_accur[i]))
                print("Best k-fold cross validation Accuracy: "+str(max_valid_score))
                print("Test set Accuracy: "+str(accu))
                print("number of SVs: "+str(len(vecs)))
                print("Average prediction time is: "+str(avg_time+avg_time_preprocessing) + " sec")
                selected_model.save("model_temp.xml")  
                model_info = os.stat('model_temp.xml')
                model_mem_size = model_info.st_size
                print("estimate total memory usage for this model and BoW: "+str((model_mem_size+vocab_mem_size)/1024)+" KB")
                print("--------------------------------------------------------------------------------------------")
                os.remove('model_temp.xml')
            # --------------------------------------------------------------------------
            """ Save the final selected model """
            # --------------------------------------------------------------------------  
            if types == 1 and C == 100 and gamma == 0.1:
                    now = datetime.datetime.now()
                    print(now)
                    selected_model.save("SVMmodel_"+str(len(vecs))+"_BoWVocab_"+str(Vocab_size)+str(now)+".xml")    
                    #store the vocabulary
                    fs = cv2.FileStorage("vocab_BoW_SVM"+str(Vocab_size)+str(now)+".yml", cv2.FILE_STORAGE_WRITE) 
                    fs.write(name='matrix', val=vocab)
                    fs.release()


#model.trainAuto(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))
os.remove('vocab_BoW_size.yml')
# Now create a new SVM & load the model:
#model2=cv2.ml.SVM_load("SVMModel.xml")
#model2.load()

# Predict with model2:
#model2.predict(np.array([[1.0, 2.1]], dtype=np.float32))


