'''
Created on 9 nov. 2018

@author: prshe
'''
import os
import cv2
from cv2 import imread, imwrite
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, exposure, img_as_float
import skimage.io as io 

path = 'dataset'
files = os.listdir(path)

i=0
flag=0
for file in files:
    # full path of the image
    imagePath = os.path.join(path,file)
    #convert to float
    image = img_as_float(io.imread(imagePath))
    #adaptive histogram exposure augmentation 
    image_moded = exposure.equalize_adapthist(image,clip_limit=0.03)
    # saving as a new jpg file
    id = os.path.split(imagePath)[-1].split(".")[1]
    image_number = os.path.split(imagePath)[-1].split(".")[2]
    new_num = 200 + int(image_number)
    io.imsave("dataset/User." + str(id)+ "." +str(new_num)+".jpg", image_moded,cmap=plt.cm.gray)
    i += 1
    
    