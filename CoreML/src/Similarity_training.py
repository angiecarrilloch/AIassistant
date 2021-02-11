#Imports

import os

import cv2

from tensorflow.keras.preprocessing.image import img_to_array

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf



import time

import random

from progressbar import ProgressBar



from sklearn.utils import shuffle

import pickle

import numpy.random as rng


from sklearn.preprocessing import LabelEncoder

from collections import Counter



from tensorflow.keras import Sequential, Input, Model

from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense, Lambda, concatenate, BatchNormalization, Activation

from tensorflow.keras.optimizers import Adam



#from sklearn.metrics import accuracy_score as accuracy

from tensorflow.keras.models import load_model

from  tensorflow.keras.applications import VGG16 


class train():

    def __init__(self,inputFolder="",outputFolder="",width=0, depth=0, epochs=0, lr=0, batch_size=0, loss = ""):



        self.inputFolder = inputFolder

        self.outputFolder = outputFolder

        self.width=width

        self.height=width

        self.depth=depth

        self.epochs = epochs

        self.lr=lr

        self.batch_size=batch_size

        self.loss = loss

    

    def readNumpy(self,numpy_file):

        with open(os.path.join(self.outputFolder,numpy_file), 'rb') as f:

            X_train = np.load(f)

            X_test = np.load(f)

            trainY = np.load(f)

            testY = np.load(f)

            print("Models read")

        return X_train, X_test, trainY, testY

        

    def modelInput(self):



        input_shape = (self.height, self.width, self.depth)

        chanDim = -1







        # Define the tensors for the two input images

        left_input = Input(input_shape)

        right_input = Input(input_shape)

        return left_input, right_input,input_shape

    
    def model(self, left_input,right_input,input_shape):

        def cosine_distance(vecs, normalize=False):
            x, y = vecs
            if normalize:
                x = K.l2_normalize(x, axis=0)
                y = K.l2_normalize(x, axis=0)
            return K.prod(K.stack([x, y], axis=1), axis=1)

        def cosine_distance_output_shape(shapes):
            return shapes[0]

        
        
        '''
        # Convolutional Neural Network
        
        
        model = Sequential()
        model.add(Conv2D(64, (7,7), 
                        activation='relu', 
                        input_shape=input_shape))
        model.add(MaxPooling2D())
        model.add(Dropout(0.7))
        model.add(Conv2D(128, (7,7), activation='relu'))
        model.add(MaxPooling2D())
        model.add(Conv2D(256, (4,4), activation='relu'))
        model.add(Flatten())
        model.add(Dense(16, 

                        activation='sigmoid'))


        print(model.summary())
        '''

        basemodel = VGG16(weights='imagenet',include_top=False,input_tensor=Input(shape=input_shape))

        headmodel = basemodel.output
        headmodel = MaxPooling2D()(headmodel)
        headmodel = Flatten()(headmodel)
        headmodel = Dense(16, activation='sigmoid')(headmodel)
        

        model = Model(inputs=basemodel.input, outputs=headmodel)
        

        # Generate the encodings (feature vectors) for the two images

        encoded_l = model(left_input)

        encoded_r = model(right_input)

        

        # Add a customized layer to compute the absolute difference between the encodings

        L1_layer = Lambda(lambda tensors:tf.abs(tensors[0] - tensors[1]))

        L1_distance = L1_layer([encoded_l, encoded_r])

    

        # Add a dense layer with a sigmoid unit to generate the similarity score

        prediction = Dense(1,activation='sigmoid')(L1_distance)

        

        

        # Connect the inputs with the outputs

        siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

        siamese_net.count_params()

        optimizer = Adam(self.lr)

        #//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking

        siamese_net.compile(loss=self.loss,optimizer=optimizer,metrics=['accuracy'])



        return siamese_net







    def train(self,siamese_net,X_train,trainY,X_test,testY):



        optimizer = Adam(self.lr)

        #//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking

        siamese_net.compile(loss=self.loss,optimizer=optimizer,metrics=['accuracy'])



        siamese_net.count_params()

        print(X_train[:,0].shape)





        H=siamese_net.fit([X_train[:, 0], X_train[:, 1]], trainY,

                batch_size=self.batch_size,

                epochs=self.epochs,

                verbose=1,

                validation_data=([X_test[:, 0], X_test[:, 1]],testY))





        # In[57]:





        f= open('history_image_similarity_pckl','wb')

        pickle.dump(H.history,f)

        f.close()



        siamese_net.save('model.h5') 

        siamese_net.save_weights('weights.h5')



        print("Models saved succesfully")



        return(siamese_net)







    # compute final accuracy on training and test sets
    '''
    def metrics(self,siamese_net,X_train,X_test,testY,trainY):

        tr_pred = siamese_net.predict([X_train[:, 0],X_train[:, 1]])



        tr_acc = accuracy(trainY, tr_pred.round())



        te_pred = siamese_net.predict([X_test[:, 0], X_test[:, 1]])

        te_acc = accuracy(testY, te_pred.round())



        print('* Accuracy on the training set: {:.2%}'.format(tr_acc))

        print('* Accuracy on the test set: {:.2%}'.format(te_acc))

    '''

    def predict(self,input1,input2):

        size = 128
        optimizer = Adam(self.lr)
        model_path= os.path.join(self.outputFolder, 'model.h5')
        weights_path=os.path.join(self.outputFolder, 'weights.h5')

        #Read left image
        img = []
        image1 = cv2.imread(input1)
        image1 = cv2.resize(image1, (size,size),interpolation = cv2.INTER_AREA)
        image1 = cv2.fastNlMeansDenoisingColored(image1,None,10,10,21,7)
        image1 = cv2.detailEnhance(image1, sigma_s = 10 , sigma_r = 0.15)
        image1 = img_to_array(image1)
        img.append(image1)

        #Read right image
        image2 = cv2.imread(input2)
        image2 = cv2.resize(image2, (size,size),interpolation = cv2.INTER_AREA)
        image2 = cv2.fastNlMeansDenoisingColored(image2,None,10,10,21,7)
        image2 = cv2.detailEnhance(image2, sigma_s = 10 , sigma_r = 0.15)
        image2 = img_to_array(image2)
        img.append(image2)
        r_img_pp = np.array([img], dtype="float") / 255.0

        #Load model
        with tf.device('/cpu:0'):
            model = load_model(model_path, compile=False, custom_objects={'tf': tf})
            model.load_weights(weights_path)
            model.compile(loss=self.loss, optimizer=optimizer, metrics=['accuracy'])
            input_shape = (size,size,3)
            chanDim = -1

            predict = model.predict([r_img_pp[:,0],r_img_pp[:,1]])

        return predict
