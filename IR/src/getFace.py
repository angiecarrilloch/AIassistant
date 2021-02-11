
import sys
import os
import cv2
from PIL import Image

class detectFace():

    def __init__(self,inputFolder="",outputFolder="",scaleFactor=0,minNeighbors=0,minSize=0,input=""):
        self.inputFolder = inputFolder
        self.outputFolder = outputFolder
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.minSize = minSize

        self.cascPath = os.path.join(self.inputFolder,'haarcascade_frontalface_default.xml')
        
        

        if input == None:
            self.flag = False
        else:
            self.flag = True 
            self.imagePath = input
        
    def readImage(self):
        face = []
        faceCascade = cv2.CascadeClassifier(self.cascPath)
        if self.flag == True:
            file_name='id_pic.jpg'
            #imagePath = 'C:/Users/acarrillo/Documents/AI_repo/AIassistant/IR/data/img_2.png'
            
            image = cv2.imread(self.imagePath)
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=self.scaleFactor,
                minNeighbors=self.minNeighbors,
                minSize=(self.minSize, self.minSize),
                flags = cv2.CASCADE_SCALE_IMAGE
            )
            


            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                roi_color = image[y:y + h, x:x + w]
                cv2.imwrite(os.path.join(self.outputFolder,"id_face.jpg"), roi_color)

                #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

                

            #cv2.imshow("Faces found", image)
            #cv2.waitKey(0)
            

        else:
            file_name = 'cam_face.jpg'
            video_capture = cv2.VideoCapture(0)
            while True:
                # Capture frame-by-frame
                ret, frame = video_capture.read()

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=self.scaleFactor,
                    minNeighbors=self.minNeighbors,
                    minSize=(self.minSize, self.minSize),
                    flags = cv2.CASCADE_SCALE_IMAGE
                )

                
                # Draw a rectangle around the faces
                for (x, y, w, h) in faces:
                    roi_color = frame[y:y + h, x:x + w]
                    cv2.imwrite(os.path.join(self.outputFolder,"cam_face.jpg"), roi_color)

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                
                # Display the resulting frame
                cv2.imshow('Video', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                
            # When everything is done, release the capture
            
            video_capture.release()
            cv2.destroyAllWindows()
            
       

        return face
    