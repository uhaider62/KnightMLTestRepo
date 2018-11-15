import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import os
import operator
import socket 
import urllib.request
import time
from keras.utils import plot_model


# --------------------------------------------------------------------------
""" Load face recognition model"""
# --------------------------------------------------------------------------
model="CNN_4Cls_120x120_2Conv_4HidLay_v2"
recognizer = load_model(model)
recognizer.summary()

classes=3
max_w = 120
max_h = 120
# input to check if to use wifi based recognition or local
use_network=0 #int(input('\input to check if to use wifi based recognition or local 0=local, 1=Network  '))
# --------------------------------------------------------------------------
""" Load face detection model"""
# --------------------------------------------------------------------------
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# --------------------------------------------------------------------------
""" Setup socket for wifi network based recognition"""
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
""" Setup camera"""
# --------------------------------------------------------------------------
# cap = cv2.VideoCapture(0)
# cap.set(3,640) # set Width
# cap.set(4,480) # set Height

font = cv2.FONT_HERSHEY_SIMPLEX

#indiciate id counter
id = 0
#names against ids 
names = ['Robin', 'Usman', 'Gabriel','Prithvi','unknown']


videoFrameWidth = 640
videoFrameHeight = 480
videoFrameWidthMiddle = round(videoFrameWidth/2)
videoFrameHeightMiddle = round(videoFrameHeight/2)
faceHyst = 80

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, videoFrameWidth) # set video widht
cam.set(4, videoFrameHeight) # set video height
#CV_CAP_PROP_CONVERT_RGB = 15
CV_CAP_PROP_WHITE_BALANCE_U = 16
cam.set(CV_CAP_PROP_WHITE_BALANCE_U, 100)

# Define min window size to be recognized as a face
minW = 0.05*cam.get(3)
minH = 0.05*cam.get(4)
POS_MSEC = 0
POS_FRAMES = 1
POS_AVI_RATIO = 2
FRAME_WIDTH = 3
FRAME_HEIGHT = 4
FPS = 5
FOURCC = 6
FRAME_COUNT = 7
FORMAT = 8
MODE = 9
BRIGHTNESS = 10
CONTRAST = 11 
SATURATION =  12
HUE = 13 
GAIN = 14
EXPOSURE = 15
CONVERT_RGB = 16
WHITE_BALANCE = 17
RECTIFICATION = 18
props = [['Height', cam.get(FRAME_HEIGHT)],
         ['Brightness', cam.get(BRIGHTNESS)],
         ['Exposure', cam.get(EXPOSURE)]]



faces_average = np.zeros((3,classes), dtype=np.float)
old_id = np.zeros((len(faces_average),1), dtype=np.int)

for i in range(0,len(faces_average)):
    old_id[i] = len(names)-1

average_10frames = 0
old_no_faces = 0
# --------------------------------------------------------------------------
""" Infinite loop """
# --------------------------------------------------------------------------
while True:
    
    if average_10frames < 10:
        # --------------------------------------------------------------------------
        """ Read Frame """
        # --------------------------------------------------------------------------   
        # read camera frame
        ret, img =cam.read()
        # turn to gray scale
        frame = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # --------------------------------------------------------------------------
        """ Detect faces """
        # --------------------------------------------------------------------------   
        start= time.time()
        # detect faces in frame
        faces = faceCascade.detectMultiScale( 
            frame,
            scaleFactor = 1.2,#1.2
            minNeighbors = 6,#5
            minSize = (int(minW), int(minH)),
           )
        end = time.time()
        print("Face detection time: "+str(end-start))
        add_face = 0
        # --------------------------------------------------------------------------
        """ If new faces detected in camera, then restart the recognition for all faces. """
        # --------------------------------------------------------------------------  
        if len(faces) != old_no_faces:
            
            for j in range (0, len(faces_average)):
                old_id[j] = len(names)-1
                for i in range (0,classes):
                    faces_average[j,i] = 0
            
            average_10frames = 0
        
        old_no_faces = len(faces)       
        # --------------------------------------------------------------------------
        """ Loop through faces and take average of 10 frames for recognition """
        # --------------------------------------------------------------------------   
        # loop through detected faces for recognition
        for(x,y,w,h) in faces:
            id = "unknown"
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            
            face = frame[y:y+h,x:x+w]
            
            start= time.time()
            if len(face) < max_h or len(face[0]) < max_w:
                 
                face_image = cv2.resize(face, (max_h, max_w), interpolation = cv2.INTER_CUBIC)
            else:
                face_image = cv2.resize(face, (max_h, max_w), interpolation = cv2.INTER_AREA)
            
            
            #face_image = cv2.resize(face, (max_h, max_w))
            face_image = img_to_array(face_image)
            face_image = np.array(face_image, dtype="float") / 255.0
            face_image = face_image.reshape(1,max_h,max_w,1)
            
            if w > max_w and h > max_h:
                
                # recognize face
                start= time.time()
                class_prob = recognizer.predict(face_image)[0]
                end = time.time()
                print("prediction time: "+str(end-start))
                print("face "+str(add_face)+": "+str(class_prob))
                # take average of 10 samples to get the id. maximum for 5 faces.
                if add_face < 3 and add_face < len(faces):
                    
                    if average_10frames < 9:
                        
                        for i in range (0,classes):
                            
                            faces_average[add_face,i] = faces_average[add_face,i] + class_prob[i]*100
                        
                        id = names[old_id[add_face, 0]]    
                        
                    else:
                        
                        for i in range(0,classes):

                            if (faces_average[add_face,i]/10 > 85): # or (old_id[add_face,0] == 0 and faces_average[add_face,0]/10 > 70):
                                id = names[i]
                                old_id[add_face,0] = i
                                break

                            else:
                                id = names[len(names)-1]
                                old_id[add_face,0] = len(names)-1
    
                        print("face "+str(add_face)+": "+str(faces_average[0,:]))
                        
                        for i in range (0,classes):
                            
                            faces_average[add_face,i] = class_prob[i]*100
                else:
                    bb=0
           
                add_face = add_face + 1
            else:
                
                id = "unknown"
                
            # --------------------------------------------------------------------------
            """ Display name and rectangle around face """
            # --------------------------------------------------------------------------   
            #print("x=",x,"y=",y,"w=",w,"h=",h)
            xFaceMiddle = (x + (round(w/2)))
            yFaceMiddle = (y + (round(h/2)))
            
            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    
        #display square around face and 
        cv2.imshow('camera',img)
         
        average_10frames = average_10frames + 1
        
    else:
        average_10frames  = 0
    
    
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
