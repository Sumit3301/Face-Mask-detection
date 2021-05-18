#Detecting mask
import numpy as np
import cv2
 
haar_cascade=cv2.CascadeClassifier('haar_face.xml')
people=["without_mask","with_mask"]



face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')
img=cv2.imread(r'C:\Users\sumit\OneDrive\Desktop\dataset\data\with_mask\with_mask_roi.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale',gray)

faces_rect=haar_cascade.detectMultiScale(gray,1.1,2)

for (x,y,w,h) in faces_rect:
    mask_roi = gray[y:y+h,x:x+h]

    label,percentage=face_recognizer.predict(mask_roi)
    print(f'{people[label]} with percentage of{percentage}')
    
    cv2.putText(img,str(people[label]),(20,20),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)

    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=1)

cv2.imshow('Deteted Image',img)
cv2.waitKey(0)