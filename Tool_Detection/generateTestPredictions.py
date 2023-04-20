import tensorflow as tf
import argparse
import math
import os
import numpy as np
import json
import pandas
import cv2
tfVersion = tf.__version__
tfVersion = int(tfVersion[0])
if tfVersion >1:
    from tensorflow.keras.models import load_model
else:
    from keras.models import load_model
from utils.utils import get_yolo_boxes


FLAGS = None

class Predict_Yolov3:
    def loadData(self, dataset):
        dataset.index = [i for i in range(len(dataset.index))]
        test_textFile = os.path.join("{}".format(self.saveLocation), "Test.txt")
        labelFile = os.path.join("{}".format(self.saveLocation), "classes.txt")
        if not os.path.exists(test_textFile):
            self.writeDataToTextFile(dataset, test_textFile, labelFile)
        return test_textFile, labelFile

    def writeDataToTextFile(self, datacsv, textFile, labelFile):
        labels = []
        trainLines = []
        for i in datacsv.index:
            newLine = ''
            filePath = os.path.join(datacsv["Folder"][i], datacsv["FileName"][i])
            newLine += filePath
            boundingBoxes = eval(str(datacsv[self.labelName][i]))
            for boundingBox in boundingBoxes:
                x1 = boundingBox["xmin"]
                x2 = boundingBox["xmax"]
                y1 = boundingBox["ymin"]
                y2 = boundingBox["ymax"]
                xmin = min(x1, x2)
                xmax = max(x1, x2)
                ymin = min(y1, y2)
                ymax = max(y1, y2)
                if boundingBox["class"] != "nothing":
                    if not boundingBox["class"] in labels:
                        labels.append(boundingBox["class"])
                    bboxStr = " {},{},{},{},{}".format(boundingBox["class"], xmin, xmax, ymin, ymax)
                    newLine += bboxStr
                newLine += '\n'
                trainLines.append(newLine)

        trainLines[-1] = trainLines[-1].replace('\n', '')
        labels = sorted(labels)
        for i in range(len(trainLines)):
            matchingLabels = []
            matchingLabelLengths = []
            matchingLabelIndexes = []
            for j in range(len(labels)):
                labelName = labels[j]
                if labelName in trainLines[i]:
                    matchingLabels.append(labelName)
                    matchingLabelLengths.append(len(labelName))
                    matchingLabelIndexes.append(j)
            while len(matchingLabels) != 0:
                longestLength = max(matchingLabelLengths)
                longestLengthIndex = matchingLabelLengths.index(longestLength)
                longestLabel = matchingLabels[longestLengthIndex]
                longestLabelIndex = matchingLabelIndexes[longestLengthIndex]
                matchingLabels.remove(longestLabel)
                matchingLabelLengths.remove(longestLength)
                matchingLabelIndexes.remove(longestLabelIndex)
                trainLines[i] = trainLines[i].replace(longestLabel, str(longestLabelIndex))

        with open(textFile, "w") as f:
            f.writelines(trainLines)

        labels = [i + "\n" for i in labels]
        labels[-1] = labels[-1].replace("\n", "")

        if not os.path.exists(labelFile):
            with open(labelFile, "w") as f:
                f.writelines(labels)

    def getPredictions(self):
        if FLAGS.save_location == "":
            print("No save location specified. Please set flag --save_location")
        elif FLAGS.data_csv_file == "":
            print("No dataset specified. Please set flag --data_csv_file")
        else:
            self.saveLocation = FLAGS.save_location
            self.dataCSVFile = pandas.read_csv(FLAGS.data_csv_file)
            self.labelName = "Tool bounding box"
            _, labelFile = self.loadData(self.dataCSVFile)

            obj_thresh, nms_thresh = 0.5, 0.45
            net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster

            config_path = os.path.join(self.saveLocation,"config.json")

            with open(config_path) as config_buffer:
                config = json.loads(config_buffer.read())

            network  = load_model(config['train']['saved_weights_name'])

            columns = ["FileName", "Time Recorded", "Tool bounding box"]
            predictions = pandas.DataFrame(columns=columns)
            predictions["FileName"] = self.dataCSVFile["FileName"]
            predictions["Time Recorded"] = self.dataCSVFile["Time Recorded"]
            for i in self.dataCSVFile.index:
                if i % 10 == 0 or i == len(self.dataCSVFile.index) - 1:
                    print("{}/{} predictions generated".format(i, len(self.dataCSVFile.index)))
                image = cv2.imread(os.path.join(self.dataCSVFile["Folder"][i], self.dataCSVFile["FileName"][i]))
                pred_boxes = get_yolo_boxes(network, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]
                scores = np.array([box.classes for box in pred_boxes])

                score = np.array([box.get_score() for box in pred_boxes])
                pred_labels = np.array([box.label for box in pred_boxes])

                if len(pred_boxes) > 0:
                    pred_boxes = np.array(
                        [[box.xmin, box.ymin, box.xmax, box.ymax, box.get_score()] for box in pred_boxes])
                else:
                    pred_boxes = np.array([[]])

                    # sort the boxes and the labels according to scores
                score_sort = np.argsort(-score)
                pred_labels = pred_labels[score_sort]
                pred_boxes = pred_boxes[score_sort]
                scores = scores[score_sort]

                bboxList = []
                for j in range(len(pred_labels)):
                    if pred_boxes[j] != [] and pred_boxes[j][-1] > 0:
                        bbox = {"class": pred_labels[j], "xmin": int(pred_boxes[j][0]), "ymin": int(pred_boxes[j][1]),
                                "xmax": int(pred_boxes[j][2]), "ymax": int(pred_boxes[j][3]),
                                "conf": float(pred_boxes[j][4])}
                        bboxList.append(bbox)
                predictions["Tool bounding box"][i] = bboxList

            predictions.to_csv(os.path.join(self.saveLocation, "BoundingBox_Predictions.csv"), index=False)
            print("Predictions saved to: {}".format(os.path.join(self.saveLocation, "BoundingBox_Predictions.csv")))


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

FLAGS, unparsed = parser.parse_known_args()
tm = Predict_Yolov3()
tm.getPredictions()