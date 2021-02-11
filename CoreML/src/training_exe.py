import sys

import os

from Similarity_training import train

from Similarity_data_preprocessing import PreProcessing

from poc import Poc



POC = Poc.getInstance("config.json")



if __name__=='__main__':

    pp = PreProcessing(POC.inputFolder(),POC.outputFolder(),POC.epochs)

    t = train(POC.inputFolder(),POC.outputFolder(),POC.width,POC.depth,POC.epochs,POC.lr,POC.batch_size,POC.loss)



    x_train,y_train,x_test,y_test = pp.getPaths()

    #X_img_pp, y_pp = pp.readImages((96,96,3),x,y)

    pair_train,label_train, pair_test, label_test = pp.createPairs(x_train,y_train,x_test,y_test)

    x_bal_train,y_bal_train, x_bal_test, y_bal_test = pp.balanceDataset(pair_train,label_train, pair_test, label_test)

    train_x,train_y,test_x,test_y = pp.splitData(x_bal_train,y_bal_train, x_bal_test, y_bal_test)





    left_input, right_input,input_shape = t.modelInput()

    siamese_net = t.model(left_input,right_input,input_shape)

    model = pp.train(train_x,train_y,test_x,test_y,1,siamese_net)

    #siamese_net = t.train(siamese_net,X_train,trainY,X_test,testY)

    #t.metrics(siamese_net,X_train,X_test,testY,trainY)

    



