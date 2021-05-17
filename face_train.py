#training model
import numpy as np
import os
import cv2

people=["with_mask","without_mask"]

dir=r'C:\Users\sumit\OneDrive\Desktop\dataset\data'
haar_cascade=cv2.CascadeClassifier('haar_face.xml')
features=[]
labels=[]

def train():
    for type in people:
        path=os.path.join(dir,type)
        label=people.index(type)

        for img in os.listdir(path):
            img_path=os.path.join(path,img)

            imgarr=cv2.imread(img_path)
            

            gray=cv2.cvtColor(imgarr,cv2.COLOR_BGR2GRAY)
            faces_ret=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)

            for (x,y,w,h) in faces_ret:
                face=gray[y:y+h, x:x+w]
                features.append(face)
                labels.append(label)

train()
print(f'Length of feature list={len(features)}')
print(f'Length of feature list={len(labels)}')

