import os
import cv2
import pandas
import argparse
from YOLOv5 import YOLOv5

def getPredictions(args):
    modelFolder = args.save_location
    model = YOLOv5()
    model.loadModel(modelFolder)
    data_csv = pandas.read_csv(args.data_csv_file)
    predictions = None
    for x in data_csv.index:
        image = cv2.imread(os.path.join(data_csv["Folder"][x],data_csv["FileName"][x]))
        bboxes = eval(model.predict(image))
        new_preds = pandas.DataFrame({"Folder":[data_csv["Folder"][x]],
                                      "FileName":[data_csv["FileName"][x]],
                                      "Tool bounding box":[bboxes]})
        if predictions is None:
            predictions = new_preds.copy()
        else:
            predictions = pandas.concat([predictions,new_preds])
        predictions.index = [i for i in range(len(predictions.index))]
    predictions.to_csv(os.path.join(modelFolder,"Results.csv"),index=False)

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

    args, unparsed = parser.parse_known_args()
    getPredictions(args)