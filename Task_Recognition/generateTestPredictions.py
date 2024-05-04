import os
from PIL import Image
import sys
import numpy
import pandas
pandas.options.mode.chained_assignment = None
import CNN_LSTM
import argparse

FLAGS = None

class Predict_CNN_LSTM:
    def getPredictions(self):
        network = CNN_LSTM.CNN_LSTM()
        if FLAGS.save_location == "":
            print("No save location specified. Please set flag --save_location")
        elif FLAGS.data_csv_file == "":
            print("No dataset specified. Please set flag --data_csv_file")
        else:
            self.saveLocation = FLAGS.save_location
            self.networkType = "CNN_LSTM"
            self.dataCSVFile = pandas.read_csv(FLAGS.data_csv_file)
            network.loadModel(self.saveLocation)
            network.cnn_model.cuda("cuda")
            network.cnn_model.return_head=False
            network.lstm_model.cuda("cuda")
            numClasses = network.num_classes
            self.sequence_length = network.sequence_length
            num_features = network.num_features
            print(numClasses)
            network.sequence = numpy.ones((self.sequence_length,num_features)).astype(float)*1e-6
            for task in network.task_class_mapping:
                if network.task_class_mapping[task]=="nothing":
                    nothingIndex = task
            columns =["FileName", "Time Recorded","Overall Task"]# ["FileName", "Time Recorded","Overall Task"] #+ [network.task_class_mapping[i] for i in range(network.num_classes)]
            predictions = pandas.DataFrame(columns=columns)
            predictions["FileName"] = self.dataCSVFile["FileName"]
            #predictions["Time Recorded"] = self.dataCSVFile["Time Recorded"]
            initialFolder = self.dataCSVFile["Folder"][0]
            for i in self.dataCSVFile.index:
                if i%500 == 0 or i==len(self.dataCSVFile.index)-1:
                    print("{}/{} predictions generated".format(i,len(self.dataCSVFile.index)))
                if self.dataCSVFile["Folder"][i] != initialFolder:
                    network.sequence = numpy.ones((self.sequence_length, num_features)).astype(float)*1e-6
                    initialFolder = self.dataCSVFile["Folder"][i]
                image = Image.open(os.path.join(self.dataCSVFile["Folder"][i],self.dataCSVFile["FileName"][i]))
                taskPrediction = network.predict(image)
                taskLabel,confidences = taskPrediction.split('[[')
                predictions["Overall Task"][i] = taskLabel
            predictions.to_csv(os.path.join(self.saveLocation,"Task_Predictions.csv"),index=False)
            print("Predictions saved to: {}".format(os.path.join(self.saveLocation,"Task_Predictions.csv")))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--save_location',
      type=str,
      default='',
      help='Name of the directory where the saved model is located'
  )
  parser.add_argument(
      '--data_csv_file',
      type=str,
      default='',
      help='Path to the csv file containing locations for all data used in testing'
  )
  parser.add_argument(
      '--sequence_length',
      type=int,
      default=100,
      help='number of images in the sequences for task prediction'
  )


FLAGS, unparsed = parser.parse_known_args()
tm = Predict_CNN_LSTM()
tm.getPredictions()

