import os
import json
## This class contains the singleton, gets all the global variables and paths of the folders
## to be used in the main classes and auxiliar classes.
#sys.path.append(os.path.abspath('../'))

class Poc:

    __instance = None

    def __init__(self, config_file=""):
        """!
        @brief          Constructor of the class

        @details        Reserved method that initializes the attributes of the class

        @param [in]     self
        @param [in]     config_file: Configuration file with the specifications of the parameters that the program will use

        """

        if Poc.__instance != None:

            raise Exception("This class is a singleton!")

        else:
            self.path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            self.scaleFactor = 1.1
            self.minNeighbors = 5
            self.minSize = 30
            self.loss = "binary_crossentropy"
            self.lr = 0.00006
            self.batch_size = 8
            self.epochs = 100
            self.width = 96
            self.height = 96
            self.depth =3
            self.path = self.path.replace(os.sep, '/')
            if(config_file != ""):
                with open(self.path + '/AIassistant/NLP/src/config/' + config_file) as json_file:
                    data = json.load(json_file)
                    self.scaleFactor = data['face_cascade']['scaleFactor']
                    self.minNeighbors = data['face_cascade']['minNeighbors']
                    self.minSize = data['face_cascade']['minSize']
                    self.loss = data['siamese_network']['loss']
                    self.lr = data['siamese_network']['lr']
                    self.batch_size = data['siamese_network']['batch_size']
                    self.epochs = data['siamese_network']['epochs']
                    self.width = data['siamese_network']['width']
                    self.height = data['siamese_network']['height']
                    self.depth = data['siamese_network']['depth']

            Poc.__instance = self

    def getInstance(self, config_file=""):

        if Poc.__instance == None:
            Poc(config_file)

        return Poc.__instance

    def outputFolderML(self):

        return self.path + "/AIassistant/CoreML/output/"


    def inputFolderML(self):

        return self.path + "/AIassistant/CoreML/data/lfw/data2/"

    def outputFolderIR(self):

        return self.path + "/AIassistant/IR/output/"


    def inputFolderIR(self):

        return self.path + "/AIassistant/IR/data/"

    def configFolder(self):

        return self.path + "/AIassistant/NLP/src/config/"

