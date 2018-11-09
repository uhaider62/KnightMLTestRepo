''''
Capture multiple Faces from multiple users to be stored on a DataBase (dataset directory)
	==> Faces will be stored on a directory: dataset/ (if does not exist, pls create one)
	==> Each face will have a unique numeric integer ID as 1, 2, 3, etc                       

Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18    

'''

import cv2
from PIL import Image
import numpy as np
import os

# Path for face image database
path = 'other'

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize individual sampling face count
count = 0

imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     

for imagePath in imagePaths:

    PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale  
    img_numpy = np.array(PIL_img,'uint8')
    faces = face_detector.detectMultiScale(img_numpy,
                                      scaleFactor = 1.3,#1.2
                                      minNeighbors = 5)#5
                                      #minSize = (int(minW), int(minH)),)

    
    for (x,y,w,h) in faces:
         # Save the captured image into the datasets folder
         cv2.imwrite(imagePath, img_numpy[y:y+h,x:x+w]) #gray[y:y+h,x:x+w]

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cv2.destroyAllWindows()


