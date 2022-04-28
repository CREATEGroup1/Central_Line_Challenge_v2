import os
import sys
import numpy
import random
import pandas
import argparse
import tensorflow
import tensorflow.keras
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,LearningRateScheduler
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json
import sklearn
import sklearn.model_selection
import sklearn.metrics
import cv2
import gc
from matplotlib import pyplot as plt
import CNN
from CNNSequence import CNNSequence

#tensorflow.compat.v1.disable_eager_execution()

FLAGS = None

class Train_CNN:

    def loadData(self,val_percentage,dataset):
        trainIndexes,valIndexes = sklearn.model_selection.train_test_split(dataset.index,test_size=val_percentage,shuffle=False)
        return trainIndexes,valIndexes

    def convertTextToNumericLabels(self,textLabels,labelValues):
        numericLabels =[]
        for i in range(len(textLabels)):
            label = numpy.zeros(len(labelValues))
            labelIndex = numpy.where(labelValues == textLabels[i])
            label[labelIndex] = 1
            numericLabels.append(label)
        return numpy.array(numericLabels)

    def saveTrainingInfo(self,saveLocation,trainingHistory,networkType,balanced=False):
        LinesToWrite = []
        modelType = "\nNetwork type: " + str(self.networkType)
        LinesToWrite.append(modelType)
        datacsv = "\nData CSV: " + str(FLAGS.data_csv_file)
        LinesToWrite.append(datacsv)
        numEpochs = "\nNumber of Epochs: " + str(len(trainingHistory["loss"]))
        numEpochsInt = len(trainingHistory["loss"])
        LinesToWrite.append(numEpochs)
        batch_size = "\nBatch size: " + str(self.batch_size)
        LinesToWrite.append(batch_size)
        LearningRate = "\nLearning rate: " + str(self.cnn_learning_rate)
        LinesToWrite.append(LearningRate)
        dataBalance = "\nData balanced: " + str(balanced)
        LinesToWrite.append(dataBalance)
        LossFunction = "\nLoss function: " + str(self.loss_Function)
        LinesToWrite.append(LossFunction)
        trainStatsHeader = "\n\nTraining Statistics: "
        LinesToWrite.append(trainStatsHeader)
        trainLoss = "\n\tFinal training loss: " + str(trainingHistory["loss"][numEpochsInt-1])
        LinesToWrite.append(trainLoss)
        for i in range(len(self.metrics)):
            trainMetrics = "\n\tFinal training " + self.metrics[i] + ": " + str(trainingHistory[self.metrics[i]][numEpochsInt-1])
            LinesToWrite.append(trainMetrics)
        valLoss = "\n\tFinal validation loss: " + str(trainingHistory["val_loss"][numEpochsInt - 1])
        LinesToWrite.append(valLoss)
        for i in range(len(self.metrics)):
            valMetrics = "\n\tFinal validation " + self.metrics[i] + ": " + str(trainingHistory["val_"+self.metrics[i]][numEpochsInt-1])
            LinesToWrite.append(valMetrics)

        with open(os.path.join(saveLocation,"trainingInfo_"+networkType+".txt"),'w') as f:
            f.writelines(LinesToWrite)

    def saveTrainingPlot(self,saveLocation,history,metric,networkType):
        fig = plt.figure()
        numEpochs =len(history[metric])
        plt.plot([x for x in range(numEpochs)], history[metric], 'bo', label='Training '+metric)
        plt.plot([x for x in range(numEpochs)], history["val_" + metric], 'b', label='Validation '+metric)
        plt.title(networkType+' Training and Validation ' + metric)
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.savefig(os.path.join(saveLocation, networkType+'_'+metric + '.png'))
        plt.close(fig)

    def balanceDataset(self,dataset):
        videos = dataset["Folder"].unique()
        balancedFold = pandas.DataFrame(columns=dataset.columns)
        for vid in videos:
            images = dataset.loc[dataset["Folder"] == vid]
            labels = sorted(images["Tool"].unique())
            counts = images["Tool"].value_counts()
            print(vid)
            smallestCount = counts[counts.index[-1]]
            print("Smallest label: " + str(counts.index[-1]))
            print("Smallest count: " + str(smallestCount))
            if smallestCount == 0:
                print("Taking second smallest")
                secondSmallest = counts[counts.index[-2]]
                print("Second smallest count: " + str(secondSmallest))
                reducedLabels = [x for x in labels if x != counts.index[-1]]
                print(reducedLabels)
                for label in reducedLabels:
                    toolImages = images.loc[images["Tool"] == label]
                    randomSample = toolImages.sample(n=secondSmallest)
                    balancedFold = balancedFold.append(randomSample, ignore_index=True)
            else:
                for label in labels:
                    toolImages = images.loc[images["Tool"] == label]
                    if label == counts.index[-1]:
                        balancedFold = balancedFold.append(toolImages, ignore_index=True)
                    else:
                        randomSample = toolImages.sample(n=smallestCount)
                        balancedFold = balancedFold.append(randomSample, ignore_index=True)
        print(balancedFold["Tool"].value_counts())
        return balancedFold

    def createBalancedCNNDataset(self,trainSet,valSet):
        newCSV = pandas.DataFrame(columns=self.dataCSVFile.columns)
        resampledTrainSet = self.balanceDataset(trainSet)
        sortedTrain = resampledTrainSet.sort_values(by=['FileName'])
        newCSV = newCSV.append(sortedTrain, ignore_index=True)
        resampledValSet = self.balanceDataset(valSet)
        sortedVal = resampledValSet.sort_values(by=['FileName'])
        newCSV = newCSV.append(sortedVal, ignore_index=True)
        print("Resampled Train Counts")
        print(resampledTrainSet["Tool"].value_counts())
        print("Resampled Validation Counts")
        print(resampledValSet["Tool"].value_counts())
        return newCSV

    def train(self):
        if FLAGS.save_location == "":
            print("No save location specified. Please set flag --save_location")
        elif FLAGS.data_csv_file == "":
            print("No dataset specified. Please set flag --data_csv_file")
        else:
            self.saveLocation = FLAGS.save_location
            self.networkType = "CNN"
            self.dataCSVFile = pandas.read_csv(FLAGS.data_csv_file)
            self.validation_percentage = FLAGS.validation_percentage

            self.numEpochs = FLAGS.num_epochs_cnn
            self.batch_size = FLAGS.batch_size
            self.cnn_learning_rate = FLAGS.cnn_learning_rate
            self.cnn_optimizer = tensorflow.keras.optimizers.Adam(learning_rate=self.cnn_learning_rate)
            self.loss_Function = FLAGS.loss_function
            self.metrics = FLAGS.metrics.split(",")
            network = CNN.CNN()
            if not os.path.exists(self.saveLocation):
                os.mkdir(self.saveLocation)
            toolLabelName = "Tool"

            TrainIndexes, ValIndexes = self.loadData(self.validation_percentage, self.dataCSVFile)

            cnnTrainDataSet = CNNSequence(self.dataCSVFile, TrainIndexes, self.batch_size, toolLabelName)
            cnnValDataSet = CNNSequence(self.dataCSVFile, ValIndexes, self.batch_size, toolLabelName)

            cnnLabelValues = numpy.array(sorted(self.dataCSVFile[toolLabelName].unique()))
            numpy.savetxt(os.path.join(self.saveLocation,"cnn_labels.txt"),cnnLabelValues,fmt='%s',delimiter=',')

            cnnModel = network.createCNNModel((224,224,3),num_classes=len(cnnLabelValues))
            cnnModel.compile(optimizer = self.cnn_optimizer, loss = self.loss_Function, metrics = self.metrics)

            earlyStoppingCallback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
            modelCheckPointCallback = ModelCheckpoint(os.path.join(self.saveLocation,'resnet50.h5'), verbose=1,monitor='val_accuracy', mode='max', save_weights_only = True,save_best_only=True)

            history = cnnModel.fit(x=cnnTrainDataSet,
                                   validation_data=cnnValDataSet,
                                   epochs=self.numEpochs,callbacks=[modelCheckPointCallback,earlyStoppingCallback])

            cnnModel.load_weights(os.path.join(self.saveLocation, 'resnet50.h5'))

            self.saveTrainingInfo(self.saveLocation, history.history,"CNN")
            self.saveTrainingPlot(self.saveLocation, history.history, "loss", "CNN")
            for metric in self.metrics:
                self.saveTrainingPlot(self.saveLocation,history.history,metric,"CNN")

            network.saveModel(cnnModel,self.saveLocation)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--save_location',
      type=str,
      default='',
      help='Name of the directory where the models and results will be saved'
  )
  parser.add_argument(
      '--validation_percentage',
      type=float,
      default=0.3,
      help='percent of data to be used for validation'
  )
  parser.add_argument(
      '--data_csv_file',
      type=str,
      default='',
      help='Path to the csv file containing locations for all data used in training'
  )
  parser.add_argument(
      '--num_epochs_cnn',
      type=int,
      default=50,
      help='number of epochs used in training the cnn'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=8,
      help='number of images in each batch'
  )
  parser.add_argument(
      '--cnn_learning_rate',
      type=float,
      default=0.000001,
      help='Learning rate used in training cnn network'
  )
  parser.add_argument(
      '--loss_function',
      type=str,
      default='categorical_crossentropy',
      help='Name of the loss function to be used in training (see keras documentation).'
  )
  parser.add_argument(
      '--metrics',
      type=str,
      default='accuracy',
      help='Metrics used to evaluate model.'
  )
  
FLAGS, unparsed = parser.parse_known_args()
tm = Train_CNN()
tm.train()
