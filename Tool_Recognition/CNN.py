import os
import cv2
import sys
import numpy
import tensorflow
import tensorflow.keras
from tensorflow.keras.applications import MobileNet,MobileNetV2
from tensorflow.keras.applications import InceptionV3,ResNet50
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json

#tensorflow.compat.v1.enable_eager_execution()

class CNN():
    def __init__(self):
        self.cnnModel = None
        self.cnnLabels = None

    def loadModel(self,modelFolder,modelName):
        self.cnnModel = self.loadCNNModel(modelFolder)

        with open(os.path.join(modelFolder,"cnn_labels.txt"),"r") as f:
            self.cnnLabels = f.readlines()
        self.cnnLabels = numpy.array([x.replace("\n","") for x in self.cnnLabels])

    def loadCNNModel(self,modelFolder):
        structureFileName = 'resnet50.json'
        weightsFileName = 'resnet50.h5'
        modelFolder = modelFolder.replace("'","")
        with open(os.path.join(modelFolder, structureFileName), "r") as modelStructureFile:
            JSONModel = modelStructureFile.read()
        model = model_from_json(JSONModel)
        model.load_weights(os.path.join(modelFolder, weightsFileName))
        return model

    def predict(self,image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image, (224, 224)) #MobileNet
        #resized = cv2.resize(image, (299, 299))  #InceptionV3
        normImage = cv2.normalize(resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        normImage = numpy.expand_dims(normImage, axis=0)

        toolClassification = self.cnnModel.predict(numpy.array(normImage))
        labelIndex = numpy.argmax(toolClassification)
        label = self.cnnLabels[labelIndex]
        networkOutput = str(label) + str(toolClassification)
        return networkOutput

    def createCNNModel(self,imageSize,num_classes):
        model = tensorflow.keras.models.Sequential()
        model.add(ResNet50(weights='imagenet',include_top=False,input_shape=imageSize,pooling='avg'))
        #model.add(InceptionV3(weights='imagenet', include_top=False, input_shape=imageSize))
        model.add(layers.Dense(512,activation='relu'))
        model.add(layers.Dense(num_classes,activation='softmax'))
        return model

    def saveModel(self,trainedCNNModel,saveLocation):
        JSONmodel = trainedCNNModel.to_json()
        structureFileName = 'resnet50.json'
        with open(os.path.join(saveLocation,structureFileName),"w") as modelStructureFile:
            modelStructureFile.write(JSONmodel)