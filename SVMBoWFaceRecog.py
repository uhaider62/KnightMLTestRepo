import numpy as np
import time
import cv2
import os

# --------------------------------------------------------------------------
""" Load SVM recognition model and find total classes in the model """
# --------------------------------------------------------------------------
model_file = "SVMmodel_56_BoWVocab_80.xml"
recognizer = cv2.ml.SVM_load("SVMmodel_56_BoWVocab_80.xml")

classes=0
model_info = open(model_file,"r")
model_lines = model_info.readlines()
for line in model_lines: 
    a = line.find('class_count')
    if a !=-1:
        str1 = line.split('>')[1]
        str2 = str1.split('<')[0]
        classes = int(str2)
        print("number of classes in the model are: "+str2)
        break
for line in model_lines: 
    a = line.find('var_count')
    if a !=-1:
        str1 = line.split('>')[1]
        str2 = str1.split('<')[0]
        input_size = int(str2)
        print("number of inputs in the model are: "+str2)
        break
model_info.close
# --------------------------------------------------------------------------
""" Load face detection model """
# --------------------------------------------------------------------------
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# --------------------------------------------------------------------------
""" Bag of words vocabulary file """
# --------------------------------------------------------------------------
fs = cv2.FileStorage("vocab_BoW_SVM80.yml", cv2.FILE_STORAGE_READ) 
vocab = fs.getNode("matrix")
vocabulary = vocab.mat()
# --------------------------------------------------------------------------
""" Create SURF feature and BoW detection/extraction setup """
# --------------------------------------------------------------------------
total_vocabulary = input_size
hessianThreshold = 100
nOctaves = 4
nOctaveLayers = 3
extended = 0
upright = 0 
surf = cv2.xfeatures2d.SURF_create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright )
bow_extr = cv2.BOWImgDescriptorExtractor(surf, cv2.BFMatcher(cv2.NORM_L2))
# --------------------------------------------------------------------------
""" Camera setup settings """
# --------------------------------------------------------------------------
font = cv2.FONT_HERSHEY_SIMPLEX
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
# --------------------------------------------------------------------------
""" Local functions """
# --------------------------------------------------------------------------
def empty(value):
    s = 0
    try:
        value = len(value)
    except ValueError:
        pass
        s=1
    return bool(s)
# --------------------------------------------------------------------------
""" Setup of variables used """
# --------------------------------------------------------------------------
id = 0
classes=classes+1
data_not_enough = 0
names = ['unknown', 'Robin', 'usman', 'Gabriel']
tot_faces = 3
avg_ids = np.zeros((3,classes), dtype=np.float)
old_id = np.zeros((tot_faces,), dtype=np.int)
print(old_id)
confirm_threshold=7
tot_samples=0
bow_extr.setVocabulary(vocabulary)
# --------------------------------------------------------------------------
""" Application run """
# --------------------------------------------------------------------------
while True:
    
    # --------------------------------------------------------------------------
    """ Get camera frame and grayscale """
    # --------------------------------------------------------------------------
    ret, img =cam.read()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # --------------------------------------------------------------------------
    """ Detect face """
    # --------------------------------------------------------------------------
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,#1.2
        minNeighbors = 5,#5
        minSize = (int(minW), int(minH)),
       )
    
    face_count = 0
    
    for(x,y,w,h) in faces:
        
        # --------------------------------------------------------------------------
        """ Scaling of face """
        # --------------------------------------------------------------------------
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        
        face = gray[y:y+h,x:x+w]
        # --------------------------------------------------------------------------
        """ Get BoW features from frame """
        # --------------------------------------------------------------------------
        start= time.time()
        #face_image = cv2.resize(face, (100, 100))
        des=[]
        kk, des = surf.detectAndCompute(face, None)
        if not(des == []):
            
            if len(des) >= total_vocabulary:
                data_not_enough = 0    
                bowsig = bow_extr.compute(face, kk)
                # --------------------------------------------------------------------------
                """ recognise face """
                # --------------------------------------------------------------------------
                id = recognizer.predict(bowsig)
                end = time.time()
                print("prediction time: "+str(end-start))
                id = np.array(id, 'uint8')
                a = id[1]
                # --------------------------------------------------------------------------
                """ Run through faces if more than 1 """
                # --------------------------------------------------------------------------
                if face_count < 3:
                    # --------------------------------------------------------------------------
                    """ Average of 10 predictions to verify the face """
                    # --------------------------------------------------------------------------
                    if tot_samples < 10:
                        
                        avg_ids[face_count][a+1] = avg_ids[face_count][a+1] + 1

                        id = names[old_id[face_count]]
                        tot_samples = tot_samples + 1
                        
                    else:
                        # --------------------------------------------------------------------------
                        """ Verify the id after 10 frames """
                        # --------------------------------------------------------------------------   
                        id_flag = False
                        for i in range(classes-1,-1,-1):
                            
                            if avg_ids[face_count][i] > confirm_threshold:
                                id = names[i]
                                id_flag = True
                                old_id[face_count] = i

                        if id_flag == False:
                            id = names[0]
                            old_id[face_count] = 0
                            
                        print(avg_ids[face_count][:])   
                        tot_samples = 0    
                        for i in range(0,classes):
                            avg_ids[face_count][i] = 0 
    
            else:
                # --------------------------------------------------------------------------
                """ Reset to unkown if features not enough for 10 consecutive frames """
                # --------------------------------------------------------------------------   
                id = names[old_id[face_count]]
                if data_not_enough > 10:
                    id = names[0]
                    
                    for i in range(0,tot_faces):
                        old_id[i] = 0
                else:
                    data_not_enough = data_not_enough + 1
                    
        else:
            id = names[old_id[face_count]]
        # --------------------------------------------------------------------------
        """ Settings for camera view """
        # --------------------------------------------------------------------------   
        xFaceMiddle = (x + (round(w/2)))
        yFaceMiddle = (y + (round(h/2)))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
    
    cv2.imshow('camera',img) 
    
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
fs.release()
cam.release()
cv2.destroyAllWindows()
