import sys
import os
from Similarity_data_preprocessing import PreProcessing
from poc import Poc

POC = Poc.getInstance("config.json")

if __name__=='__main__':
    pp = PreProcessing(POC.inputFolder(),POC.outputFolder())
    
    x, y = pp.getPaths()
    #X_img_pp, y_pp = pp.readImages((96,96,3),x,y)
    pair,label = pp.createPairs(y,x)
    x_bal,y_bal = pp.balanceDataset(label,pair)
    train_x,train_y,test_x,test_y = pp.splitData(x_bal,y_bal)
    #pp.saveData(train_x,train_y,test_x,test_y)
    #p.split(x_bal,y_bal,"data.npy")
