import sys
import os
from poc import Poc
import speech_recognition as sr
from pathlib import Path
#global_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
#global_path = global_path.replace(os.sep, '/')

global_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
global_path = global_path.replace(os.sep, '/')

sys.path.append(global_path)
from IR.src.getFace import detectFace


from CoreML.src.Similarity_training import train


POC = Poc.getInstance("config.json")


if __name__=='__main__':

    print("Hi there! I'll guide you through the registration process!")
    print("What's your name? Type it or press enter if you want to speak!")
    input_name = input()
    flag = False
    recognizer_instance = sr.Recognizer()
    if input_name == "":
        while(flag == False):
            with sr.Microphone() as source:
                recognizer_instance.adjust_for_ambient_noise(source)
                print("I am listening, you can talk!")
                audio = recognizer_instance.listen(source,timeout = 3)
            try:
                name = recognizer_instance.recognize_google(audio, language="it-IT")
                flag = True
            except Exception as e:
                print ("Sorry, I couldn't listen, could you repeat please?")
                flag = False
    else:
        name=input_name
    print("Okay, " , name , ", what is your surname?, type it or press Enter if you want to speak")
    input_surname = input()
    flag = False
    recognizer_instance = sr.Recognizer()
    if input_surname == "":
        while(flag == False):
            with sr.Microphone() as source:
                recognizer_instance.adjust_for_ambient_noise(source)
                print("I am listening, you can talk!")
                audio = recognizer_instance.listen(source,timeout = 3)
            try:
                surname = recognizer_instance.recognize_google(audio, language="it-IT")
                flag = True
            except Exception as e:
                print ("Sorry, I couldn't listen, could you repeat please?")
                flag = False
    else:
        surname=input_surname
    print("Okay, ",name, " ", surname, ", let's scan a document")
    print("Please upload the file by typing the path where it is located")
    path_id = input()
    #C:/Users/acarrillo/Documents/AI_repo/AIassistant/IR/data/pds_angie.jpg
    #C:/Users/acarrillo/Documents/AI_repo/AIassistant/IR/data/img_2.png
    f = detectFace(POC.inputFolderIR(),POC.outputFolderIR(),POC.scaleFactor,POC.minNeighbors,POC.minSize,path_id)
    face = f.readImage()
    print("Document uploaded succesfully!")
    print("Let's start verification")
    f = detectFace(POC.inputFolderIR(),POC.outputFolderIR(),POC.scaleFactor,POC.minNeighbors,POC.minSize,None)
    face = f.readImage()

    t = train(POC.inputFolderML(),POC.outputFolderML(),POC.width,POC.depth,POC.epochs,POC.lr,POC.batch_size,POC.loss)

    input1 = global_path + '/IR/output/cam_face.jpg'
    input2 = global_path + '/IR/output/id_face.jpg'
    message = t.predict(input1,input2)
    print(message)
