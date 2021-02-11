import sys
import os
from getFace import detectFace
from poc import Poc

POC = Poc.getInstance("config.json")

if __name__=='__main__':
    input = 'pds_angie.jpg'
    f = detectFace(POC.inputFolder(),POC.outputFolder(),POC.scaleFactor,POC.minNeighbors,POC.minSize,input)
    face = f.readImage()
    print(face)