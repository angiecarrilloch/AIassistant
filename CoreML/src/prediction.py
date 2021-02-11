import sys
import os
from Similarity_training import train
from poc import Poc
from progressbar import ProgressBar
import time
import random
import cv2
from tensorflow.keras.preprocessing.image import img_to_array


POC = Poc.getInstance("config.json")

if __name__=='__main__':
    t = train(POC.inputFolder(),POC.outputFolder(),POC.width,POC.depth,POC.epochs,POC.lr,POC.batch_size,POC.loss)

    #Create ground truth
    #Get path where the test images are located
    path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    path_data = path + '/AIassistant/CoreML/data/lfw/data2/'

    #Get the paths of the images in each folder and get the label
    sub_folders=[x.path for x in os.scandir(path_data) if x.is_dir()]

    
    #Get people names
    names = os.listdir(path_data)

    #Create x and y
    x=[]
    y=[]
    for i in range(len(sub_folders)):
        for image in os.listdir(sub_folders[i]):
            x.append(os.path.join(sub_folders[i],image))
            y.append(names[i])  


    pair=[]
    label=[]
    
    pbar=ProgressBar()
    for k in pbar(range(len(x)-1)):
        for l in range(k+1,len(x)):
            pair.append([x[k],x[l]])
            if y[k]==y[l]:
                label.append(1)
            else:
                label.append(0)
        time.sleep(1)
    

    #Balance dataset train
    pbar=ProgressBar()
    y_bal=[]
    x_bal=[]
    zeros=[]
    ones=[]
    for i,el in enumerate(label):
        if el==0:
            zeros.append(i)
        else:
            ones.append(i)

    random.shuffle(zeros)
    if len(ones)<50:
        n=len(ones)
    else:
        n=50
    sel_ones = ones[:n]
    sel_zeros=zeros[:n]

    sel_data=sel_zeros+sel_ones
    for i in sel_data:
        y_bal.append(label[i])
        x_bal.append(pair[i])


    predictions_1 = []
    predictions_2 = []

    pos1= 0
    pos2= 0
    for i in range(len(x_bal)):
        predicted = t.predict(x_bal[i][0],x_bal[i][1])[0][0]
        print(y_bal[i], predicted)
        p1 = round(predicted)

        if predicted > 0.4:
            p2 = 1
        else:
            p2 = 0
        #prediction

        if p1 == y_bal[i]:
            pos1 = pos1 +1

        if p2 == y_bal[i]:
            pos2 = pos2 +1

        predictions_1.append(p1)
        predictions_2.append(p2)

    print("Accuracy 1: ", pos1/len(y_bal))
    print("Accuracy 2: ", pos2/len(y_bal))
    '''

    input1 = 'C:/Users/acarrillo/Documents/AI_repo/AIassistant/IR/output/cam_face.jpg'
    input2 = 'C:/Users/acarrillo/Documents/AI_repo/AIassistant/IR/output/id_face.jpg'
    message = t.predict(input1,input2)
    print(message)
    '''
