#training model
import numpy as np
import os
import cv2

people=["with_mask","without_mask"]

dir=r'C:\Users\sumit\OneDrive\Desktop\dataset\data'
haar_cascade=cv2.CascadeClassifier('haar_face.xml')
features=[]
labels=[]

def training():
    for type in people:
        path=os.path.join(dir,type)
        #print(path)
        label=people.index(type)
        print(label)
        for img in os.listdir(path):
            img_path=os.path.join(path,img)

            imgarr=cv2.imread(img_path)
            

            gray=cv2.cvtColor(imgarr,cv2.COLOR_BGR2GRAY)
            faces_ret=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)

            for (x,y,w,h) in faces_ret:
                face=gray[y:y+h, x:x+w]
                features.append(face)
                labels.append(label)

training()
print('Training Done ---------')

labels=np.array(labels)
features=np.array(features,dtype='object')

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.train(features,labels)


recognizer.save('face_trained.yml')

#Train the recognizer of the features list and labels

np.save('features.npy',features)
np.save('labels.npy',labels)

