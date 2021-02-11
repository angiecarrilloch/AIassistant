#Imports
import os
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf

import time
from sklearn.model_selection import train_test_split
import random
from progressbar import ProgressBar

from sklearn.utils import shuffle
import h5py
import pickle


class PreProcessing():
    def __init__(self,inputFolder="",outputFolder="",epochs = 0):

        self.inputFolder = inputFolder
        self.outputFolder = outputFolder
        self.inputShape = (128,128)
        self.epochs = epochs

    def getPaths(self):
        train_folder = self.inputFolder + "train/"
        test_folder = self.inputFolder + "test/"

        #Get the paths of the images in each folder and get the label
        sub_folders_train=[x.path for x in os.scandir(train_folder) if x.is_dir()]
        sub_folders_test=[x.path for x in os.scandir(test_folder) if x.is_dir()]

        #Get people names
        names_train = os.listdir(train_folder)
        names_test = os.listdir(test_folder)
        #Create x and y
        x_train=[]
        y_train=[]
        for i in range(len(sub_folders_train)):
            for image in os.listdir(sub_folders_train[i]):
                x_train.append(os.path.join(sub_folders_train[i],image))
                y_train.append(names_train[i])  

        x_test=[]
        y_test=[]
        for i in range(len(sub_folders_test)):
            for image in os.listdir(sub_folders_test[i]):
                x_test.append(os.path.join(sub_folders_test[i],image))
                y_test.append(names_test[i]) 
        
        print((len(x_train),len(y_train),len(x_test),len(y_test)))

        return x_train,y_train,x_test,y_test

    def createPairs(self,x_train,y_train,x_test,y_test):
        print((len(x_train),len(y_train),len(x_test),len(y_test)))


        pair_test=[]
        label_test=[]
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(len(y_test),len(x_test))
        pbar=ProgressBar()
        for k in pbar(range(len(x_test)-1)):
            for l in range(k+1,len(x_test)):
                pair_test.append([x_test[k],x_test[l]])
                if y_test[k]==y_test[l]:
                    label_test.append(1)
                else:
                    label_test.append(0)
            time.sleep(1)

        pbar=ProgressBar()
        #Create pairs for train
        pair_train=[]
        label_train=[]
        for i in pbar(range(0,len(x_train)-1)):
            for j in range(i+1,len(x_train)):
                pair_train.append([x_train[i],x_train[j]])
                if y_train[i]==y_train[j]:
                    label_train.append(1)
                else:
                    label_train.append(0)
            time.sleep(1)
        

        return pair_train,label_train, pair_test, label_test

    def balanceDataset(self,pair_train,label_train, pair_test, label_test):

        #Balance dataset train
        pbar=ProgressBar()
        y_bal_train=[]
        x_bal_train=[]
        zeros=[]
        ones=[]
        for i,el in enumerate(label_train):
            if el==0:
                zeros.append(i)
            else:
                ones.append(i)

        random.shuffle(zeros)
        if len(ones)<1000:
            n=len(ones)
        else:
            n=1000
        sel_ones = ones[:n]
        sel_zeros=zeros[:n]

        sel_data=sel_zeros+sel_ones
        for i in sel_data:
            y_bal_train.append(label_train[i])
            x_bal_train.append(pair_train[i])

        #Balance dataset test
        y_bal_test=[]
        x_bal_test=[]
        zeros=[]
        ones=[]
        for i,el in enumerate(label_test):
            if el==0:
                zeros.append(i)
            else:
                ones.append(i)

        random.shuffle(zeros)
        if len(ones)<200:
            n=len(ones)
        else:
            n=200
        sel_ones = ones[:n]
        sel_zeros=zeros[:n]

        sel_data=sel_zeros+sel_ones
        for i in sel_data:
            y_bal_test.append(label_test[i])
            x_bal_test.append(pair_test[i])

        return x_bal_train,y_bal_train, x_bal_test, y_bal_test

    def splitData(self,x_bal_train,y_bal_train, x_bal_test, y_bal_test):
        random_state = 42

        train = list(zip(x_bal_train,y_bal_train))
        random.seed(random_state)
        
        test = list(zip(x_bal_test,y_bal_test))
        random.seed(random_state)
        
        random.shuffle(train)
        random.shuffle(test)
        #while (len(train)%20 != 0 ):
        #    train = train[:-1]
        #while (len(test)%20 != 0):
        #    test = test[:-1]
        train_x, train_y = zip(*train)
        test_x, test_y = zip(*test)
        

        return train_x,train_y,test_x,test_y

    def readImages(self,img):
        #Read images
        image = cv2.imread(img)
        image = cv2.resize(image, (self.inputShape[0],self.inputShape[1]))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = img_to_array(image)
        return image
        
    


    
        
    def generator(self,x_paths,y, batch_size):
        n_samples = len(x_paths)
        offset=0
        while True:
            for offset in range(0,n_samples,batch_size):
                batch_samples = x_paths[offset:offset+batch_size]
                labels = y[offset:offset+batch_size]
                X_train = []
                y_train = []

                for i in range(len(batch_samples)):
                    image_left = self.readImages(batch_samples[i][0])
                    image_right = self.readImages(batch_samples[i][1])
                    X_train.append([image_left,image_right])
                    y_train.append(labels[i])
                X_train = np.array(X_train, dtype="float") /255.0
                y_train = np.array(y_train)
                yield [X_train[:,0],X_train[:,1]],y_train

    def train(self,train_x,train_y,test_x,test_y,chunk_size,model):
        print(model.summary())
        num_train_samples = len(train_x)
        num_test_samples = len(test_x)
        print('#############################')
        print(num_train_samples)

        train_gen = self.generator(train_x, train_y, chunk_size)
        test_gen = self.generator(test_x, test_y, chunk_size)
        #FIT
    
        H = model.fit(train_gen,
                    steps_per_epoch=(num_train_samples) ,
                    epochs= self.epochs, # your desired number of epochs,
                    validation_data= test_gen,
                    verbose = 1,
                    max_queue_size=2,
                    workers=1,
                    validation_steps = (num_test_samples))
    
        
        f= open('history_image_similarity_pckl','wb')
        pickle.dump(H.history,f)
        f.close()

        model.save('model.h5') 
        model.save_weights('weights.h5')

        print("Models saved succesfully")

        return(model)