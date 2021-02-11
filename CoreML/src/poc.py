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
            self.loss = "binary_crossentropy"
            self.lr = 0.00001
            self.batch_size = 8
            self.epochs = 50
            self.width = 128
            self.height = 128
            self.depth =3
            self.path = self.path.replace(os.sep, '/')
            if(config_file != ""):
                with open(self.path + '/AIassistant/CoreML/src/config/' + config_file) as json_file:
                    data = json.load(json_file)
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

    def outputFolder(self):

        return self.path + "/AIassistant/CoreML/output/"


    def inputFolder(self):

        return self.path + "/AIassistant/CoreML/data/lfw/lfw/"

    def configFolder(self):

        return self.path + "/AIassistant/CoreML/src/config/"

