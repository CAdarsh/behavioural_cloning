import keras 
import cv2
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/driving_log.csv')
data = []
imageData = []
images = [ df['center'].values,df['left'].values,df['right'].values ]
for i in range(len(df['center'].values)):
    for j in range(0,3):
        
        if j == 0:
            imageCenter = df['center'][i].split('/')[-1] 
            image = (cv2.imread('data/IMG/'+imageCenter))
            imageData.append(image)
            dataTemp = float(df['steering'][i])
            data.append(dataTemp)
            imageData.append(cv2.flip(image,1))
            data.append((dataTemp)*-1)
    
        
        elif j == 1:
            imageCenter = df['right'][i].split('/')[-1]
            image = (cv2.imread('data/IMG/'+imageCenter))
            imageData.append(image)
            dataTemp = float(df['steering'][i])-0.2
            data.append(dataTemp)
            imageData.append(cv2.flip(image,1))
            data.append((dataTemp-0.2)*-1)
    
    

        elif j == 2:
            imageCenter = df['left'][i].split('/')[-1] 
            image = (cv2.imread('data/IMG/'+imageCenter))
            imageData.append(image)
            dataTemp = float(df['steering'][i])+0.2
            data.append(dataTemp)
            imageData.append(cv2.flip(image,1))
            data.append((dataTemp+0.2)*-1)
       
#     print(i)
# y_data = sklearn.utils.shuffle(data)
# X_data = sklearn.utils.shuffle(imageData)
                     
X_train,X_test,Y_train,y_test = train_test_split(np.array(imageData),np.array(data),test_size=0.2)                    
                  
# print(X_train[0]) 
# print(type(X_train))    
print(X_train[0].shape)
from keras.layers import Dense,Conv2D,Dropout,Cropping2D
# from keras.layers.convolutional import Convolution2D 
model = keras.models.Sequential()
model.add(keras.layers.Lambda(lambda x : (x/255)-0.5,input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))  

model.add(Conv2D(24,5,5,activation='relu'))
                     
model.add(Conv2D(36,5,5,activation='relu'))
                     
model.add(Conv2D(48,5,5,activation='relu'))
                     
model.add(Conv2D(64,5,5,activation='relu'))
                     
model.add(keras.layers.Flatten())
                     
model.add(Dense(128,activation='relu'))

model.add(Dense(1,activation='relu'))
                     
model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
                     
result = model.fit(X_train,Y_train,nb_epoch=2,shuffle=True)
                     
model.save('model.h5')                     
                     
                     

    
