import os
import random

import cv2
import pandas
import numpy
import yaml
import argparse
import gc
from YOLOv5 import YOLOv5


def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with VICReg", add_help=False)

    # Data
    parser.add_argument(
        '--save_location',
        type=str,
        default='',
        help='Name of the directory where the models and results will be saved'
    )
    parser.add_argument(
        '--data_csv_file',
        type=str,
        default='',
        help='Path to the csv file containing locations for all data used in training'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Number of images included in each batch'
    )
    # Optim
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help='Number of epochs'
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help='Epochs to wait for no observable improvement for early stopping of training'
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default='Adam',
        help='Choice of optimizer for training. Options include SGD, Adam, AdamW, NAdam, RAdam, RMSProp etc., or auto for automatic selection based on model configuration. Affects convergence speed and stability.'
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=1e-3,
        help='Initial learning rate'
    )
    parser.add_argument(
        "--lrf",
        type=float,
        default=1e-3,
        help='Final learning rate'
    )
    parser.add_argument(
        "--close_mosaic",
        type=int,
        default=0,
        help='Disables mosaic data augmentation in the last N epochs to stabilize training before completion. Setting to 0 disables this feature.'
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=None,
        help='Freezes the first N layers of the model or specified layers by index, reducing the number of trainable parameters. Useful for fine-tuning or transfer learning.'
    )
    parser.add_argument(
        "--box",
        type=float,
        default=7.5,
        help='Weight of the box loss component in the loss function, influencing how much emphasis is placed on accurately predicting bounding box coordinates.'
    )
    parser.add_argument(
        "--cls",
        type=float,
        default=0.5,
        help='Weight of the classification loss in the total loss function, affecting the importance of correct class prediction relative to other components.'
    )
    parser.add_argument(
        "--dfl",
        type=float,
        default=1.5,
        help='Weight of the distribution focal loss, used in certain YOLO versions for fine-grained classification.'
    )
    parser.add_argument(
        "--imgsz", #(***Hint: Larger values will train much slower but may preserve more detail)
        type=int,
        default=224,
        help='Size of input images as integer'
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help='Number of worker threads for data loading'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        help='device to use for training / testing'
    )
    parser.add_argument(
        '--include_blank',
        type=bool,
        default=False,
        help='Include images that have no labels for training'
    )
    parser.add_argument(
        '--balance',
        type=bool,
        default=False,
        help='Balance samples for training'
    )
    parser.add_argument(
        '--label_name',
        type=str,
        default='Tool bounding box',
        help='Name of the dataframe column containing labels'
    )
    parser.add_argument(
        '--val_percentage',
        type=float,
        default=0.3,
        help='Percent of data to be used for validation'
    )
    parser.add_argument(
        '--val_iou_threshold',
        type=float,
        default=0.5,
        help='IoU threshold used for NMS when evaluating model'
    )
    parser.add_argument(
        '--val_confidence_threshold',
        type=float,
        default=0.5,
        help='Confidence threshold used for NMS when evaluating model'
    )
    return parser

def xyxy_to_yolo(img_size,bbox):
    x_centre = ((int(bbox["xmin"]) + int(bbox["xmax"])) / 2) / img_size[1]
    y_centre = ((int(bbox["ymin"]) + int(bbox["ymax"])) / 2) / img_size[0]
    width = ((int(bbox["xmax"]) - int(bbox["xmin"]))) / img_size[1]
    height = ((int(bbox["ymax"]) - int(bbox["ymin"]))) / img_size[0]
    return x_centre, y_centre, width, height

def getMaxClassCounts(data,labelName):
    class_counts = {}
    maxCount = 0
    for i in data.index:
        try:
            bboxes = eval(str(data[labelName][i]))
            for bbox in bboxes:
                if not bbox["class"] in class_counts:
                    class_counts[bbox["class"]] = 1
                else:
                    class_counts[bbox["class"]] += 1
                    if class_counts[bbox["class"]] > maxCount:
                        maxCount = class_counts[bbox["class"]]
        except SyntaxError:
            print("{}: {}".format(data["FileName"][i],data[labelName][i]))
    return class_counts,maxCount

def determineNumDuplicatesNeeded(class_counts,max_count):
    numDuplicates = {}
    for key in class_counts:
        total_num = class_counts[key]
        fraction_max = round(max_count / total_num)
        numDuplicates[key] = fraction_max
    return numDuplicates

def writeLabelTextFiles(data, labelName, saveDir,inverted_class_mapping,include_blank=True,balance = True):
    linesToWrite = []
    class_counts, maxCount = getMaxClassCounts(data,labelName)
    foundCounts = dict(zip([key for key in class_counts],[0 for key in class_counts]))
    print(class_counts)
    if data["Set"][data.index[0]] == "Train" and balance:
        numDuplicates = determineNumDuplicatesNeeded(class_counts,maxCount)
    else:
        numDuplicates = dict(zip([key for key in class_counts],[1 for key in class_counts]))
    print(numDuplicates)
    for i in data.index:
        if (i- min(data.index)) %1000 == 0:
            print("\tparsed {}/{} samples".format(i- min(data.index),len(data.index)))
        filePath = os.path.join(data["Folder"][i],data["FileName"][i])
        bboxes = eval(str(data[labelName][i]))
        classNames = [bbox["class"] for bbox in bboxes]
        if len(classNames)>0:
            maxDuplicates = max([numDuplicates[class_name] for class_name in classNames])
            for class_name in classNames:
                foundCounts[class_name]+=maxDuplicates
        else:
            maxDuplicates = 1
        if len(bboxes) == 0 and ((data["Set"][i]=="Train" and include_blank) or data["Set"][i] != "Train"):
            for j in range(maxDuplicates):
                linesToWrite.append("{}\n".format(filePath))
        elif len(bboxes) > 0:
            for j in range(maxDuplicates):
                linesToWrite.append("{}\n".format(filePath))
            file,imgExtension = filePath.split(".",-1)
            labelFilePath = file+".txt"
            img = cv2.imread(filePath)
            img_shape = img.shape
            line = ""
            for bbox in bboxes:
                x_centre, y_centre, width, height = xyxy_to_yolo(img_shape,bbox)
                class_name = inverted_class_mapping[bbox["class"]]
                line += "{} {} {} {} {}\n".format(class_name,x_centre,y_centre,width,height)
            with open(labelFilePath, "w") as f:
                f.write(line)
    print(foundCounts)
    fileName = "{}.txt".format(data["Set"][data.index[0]])
    filePath = os.path.join(saveDir,fileName)
    with open(filePath,"w") as f:
        f.writelines(linesToWrite)

def labelmap_to_contour(labelmap):
    contour_coordinates_list = []

    present_classes = numpy.unique(labelmap)
    img_shape = labelmap.shape

    for class_label in present_classes:

        if class_label == 0:
            continue
        # Create a binary mask for the current class
        class_mask = numpy.uint8(labelmap == class_label)

        # Find contours in the binary mask
        num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(class_mask, 8, cv2.CV_32S)
        sizes = numpy.array(stats[:, -1])
        maxSize = numpy.max(sizes)
        currentIndex = 0

        # Loop through all components except background
        for i in range(num_components):
            if sizes[i] != maxSize:
                newImage = numpy.zeros(class_mask.shape)
                newImage[labels == i] = 1
                newImage = newImage.astype("uint8")

                contours, _ = cv2.findContours(newImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                maxindex = numpy.argmax([contours[i].shape[0] for i in range(len(contours))])
                contours = numpy.asarray(contours[maxindex])

                contour_string = "{}".format(class_label)
                for contour in contours:
                    for point in contour:
                        x = point[0] / img_shape[1]
                        y = point[1] / img_shape[0]
                        contour_string += " {} {}".format(x,y)
                contour_string+="\n"
                contour_coordinates_list.append(contour_string)
    return contour_coordinates_list

def invert_class_mapping(class_mapping):
    inverted_mapping = {}
    for key in class_mapping:
        inverted_mapping[class_mapping[key]] = key
    print(inverted_mapping)
    return inverted_mapping

def removeCache(data_dir):
    cache_path = os.path.dirname(data_dir)
    cache_name = os.path.basename(data_dir)
    cache_name = cache_name+".cache"
    print("removing existing cache: {}".format(os.path.join(cache_path,cache_name)))
    if os.path.exists(os.path.join(cache_path,cache_name)):
        os.remove(os.path.join(cache_path,cache_name))

def prepareData(datacsv, labelName,class_mapping,saveDir,val_percentage,include_blank):
    config = {}
    num_val_samples = round(val_percentage*len(datacsv.index))
    #Randomly selects a percentage of samples to use for validation (***Hint: This isn't always the best approach when working with videos)
    val_indexes = sorted(random.choices([i for i in datacsv.index],k=num_val_samples))
    train_indexes = [i for i in datacsv.index if not i in val_indexes]
    sets = {"Train":train_indexes, "Validation":val_indexes}
    print(len(val_indexes))
    for learning_set in sets:
        print("Parsing {} data".format(learning_set.lower()))
        data = datacsv.iloc[sets[learning_set]].copy()
        data.index = [i for i in range(len(data.index))]
        data["Set"] = [learning_set for i in data.index]
        removeCache(data["Folder"][data.index[0]])
        sample_file_path = os.path.join(saveDir,"{}.txt".format(learning_set))
        if not os.path.exists(sample_file_path):
            inverted_class_mapping = invert_class_mapping(class_mapping)
            writeLabelTextFiles(data, labelName, saveDir,inverted_class_mapping,include_blank)

        if learning_set == "Validation":
            config["val"] = sample_file_path
        else:
            config[learning_set.lower()] = sample_file_path
    config["names"] = class_mapping
    with open(os.path.join(saveDir,"data.yaml"),"w") as f:
        yaml.dump(config,f)
    return os.path.join(saveDir,"data.yaml")

def getClassMapping(data,labelName):
    class_names = []
    for i in data.index:
        bboxes = eval(str(data[labelName][i]))
        if len(bboxes) > 0:
            for bbox in bboxes:
                if not bbox["class"] in class_names:
                    class_names.append(bbox["class"])
    class_names = sorted(class_names)
    class_mapping = dict(zip([i for i in range(len(class_names))],class_names))
    return class_mapping

def saveMetrics(metrics,class_mapping,saveDir):
    class_indexes = metrics.box.ap_class_index
    maps = metrics.box.all_ap
    linesTo_write = []
    linesTo_write.append("mAP 50:\n")
    for i in range(len(class_indexes)):
        class_name = class_mapping[class_indexes[i]]
        map50 = maps[i][0]
        linesTo_write.append("{}: {}\n".format(class_name,map50))
    linesTo_write.append("\nmAP 95:\n")
    for i in range(len(class_indexes)):
        class_name = class_mapping[class_indexes[i]]
        map95 = maps[i][-1]
        linesTo_write.append("{}: {}\n".format(class_name,map95))
    with open(os.path.join(saveDir,"test_maps.txt"),"w") as f:
        f.writelines(linesTo_write)

def setAugmentationParams(saveDir):
    config_path = os.path.join(saveDir, os.pardir, "YOLOv5_config.yaml")
    with open (config_path,'r') as f:
        config = yaml.safe_load(f)
    #Define augmentation hyperparameters (***Hint: Finding the right balance of augmentations during training can improve generalizability of your results)
    config['flipud'] = 0.0 # image flip up-down (probability)
    config['fliplr']=0.0 # image flip left-right (probability)
    config['label_smoothing']=0.0 # label smoothing (fraction)
    config['hsv_h']=0.0 # image HSV-Hue augmentation (fraction)
    config['hsv_s']=0.0 # image HSV-Saturation augmentation (fraction)
    config['hsv_v']=0.0 # image HSV-Value augmentation (fraction)
    config['degrees']=0.0 # image rotation (+/- deg)
    config['translate']=0.0 # image translation (+/- fraction)
    config['scale']=0.0 # image scale (+/- gain)
    config['shear']=0.0 # image shear (+/- deg)
    config['perspective']=0.0 # image perspective (+/- fraction), range 0-0.001
    config['mosaic']=0.0 # image mosaic (probability)
    config['mixup']=0.0 # image mixup (probability)
    config['copy_paste']=0.0 # segment copy-paste (probability)

    new_config_path = os.path.join(saveDir,"model_config.yaml")
    with open (new_config_path,'w') as f:
        yaml.dump(config,f)
    return new_config_path

def updateModelConfig(args,saveDir):
    config_path = os.path.join(saveDir, "model_config.yaml")
    with open (config_path,'r') as f:
        config = yaml.safe_load(f)
    config['epochs'] = args.epochs
    config['optimizer'] = args.optimizer
    config['patience'] = args.patience
    config['lr0'] = args.lr0
    config['lrf'] = args.lrf
    config['freeze'] = args.freeze
    config['close_mosaic'] = args.close_mosaic
    config['box'] = args.box
    config['cls'] = args.cls
    config['dfl'] = args.dfl
    config['batch'] = args.batch_size
    config['imgsz'] = args.imgsz
    config['iou'] = args.val_iou_threshold
    config['conf'] = args.val_confidence_threshold
    config_path = os.path.join(saveDir, "model_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

def main(args):
    if args.save_location == "":
        print("No save location specified. Please set flag --save_location")
    elif args.data_csv_file == "":
        print("No dataset specified. Please set flag --data_csv_file")
    else:
        dataCSVFile = pandas.read_csv(args.data_csv_file)
        config = {}
        class_mapping = getClassMapping(dataCSVFile,args.label_name)
        config["class_mapping"] = class_mapping
        saveDir = args.save_location
        if not os.path.exists(saveDir):
            os.mkdir(saveDir)
        with open(os.path.join(saveDir,"config.yaml"),"w") as f:
            yaml.dump(config,f)
        dataPath = prepareData(dataCSVFile, args.label_name, class_mapping,saveDir,args.val_percentage,args.include_blank)
        model_config = setAugmentationParams(saveDir)
        updateModelConfig(args,saveDir)
        yolo = YOLOv5()
        if not os.path.exists(os.path.join(saveDir,"train/weights/best.pt")):
            model = yolo.createModel()
            model.train(data=dataPath,
                        device=args.device,
                        workers = args.workers,
                        verbose=True,
                        cache=False,
                        project=saveDir,
                        exist_ok=True,
                        amp=False,
                        cfg = model_config)
        yolo.loadModel(saveDir)
        model = yolo.model
        metrics = model.val(split="val",
                            device=args.device,
                            cfg = model_config)
        saveMetrics(metrics,class_mapping,saveDir)
        del metrics
        del model
        del yolo
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('YOLOv5 training script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)