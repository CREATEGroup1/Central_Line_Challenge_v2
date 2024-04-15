import os
import numpy
from pathlib import Path
import argparse
import json
import gc
import math
import os
import sys
import time
import cv2
import pandas
import numpy
import torch
import scipy
import yaml
import random
import sklearn
import sklearn.metrics
from matplotlib import pyplot as plt
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim
import torch.nn.functional as F
from CNN_LSTM import CNN_LSTM
from DatasetGenerator import LSTMDataset, CNNDataset

def get_arguments():
    parser = argparse.ArgumentParser(description="Train a CNN_LSTM model", add_help=False)
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
        '--validation_percentage',
        type=float,
        default=0.3,
        help='Percentage of samples to use for validation'
    )
    ## CNN Parameters ##
    parser.add_argument(
        '--cnn_epochs',
        type=int,
        default=100,
        help='number of epochs for training lstm'
    )
    parser.add_argument(
        '--cnn_batch',
        type=int,
        default=16,
        help='batch size for training / validation of lstm'
    )
    parser.add_argument(
        '--balance_cnn',
        type=bool,
        default=True,
        help='Balance number of samples in each class for training'
    )
    parser.add_argument(
        '--augment_cnn',
        type=bool,
        default=False,
        help='Use augmentations when training CNN'
    )
    parser.add_argument(
        '--cnn_lr',
        type=float,
        default=1e-5,
        help='Learning rate for CNN optimizer'
    )
    parser.add_argument(
        '--cnn_features',
        type=int,
        default=128,
        help='Number of features in last layer of CNN before the final softmax'
    )

    ## LSTM Parameters ##
    parser.add_argument(
        '--lstm_epochs',
        type = int,
        default=100,
        help='number of epochs for training lstm'
    )
    parser.add_argument(
        '--lstm_batch',
        type=int,
        default=8,
        help='batch size for training / validation of lstm'
    )
    parser.add_argument(
        '--lstm_sequence_length',
        type=int,
        default=100,
        help='number of frames to include in each sequence'
    )
    parser.add_argument(
        '--balance_lstm',
        type=bool,
        default=False,
        help='Balance number of samples in each class for training'
    )
    parser.add_argument(
        '--lstm_lr',
        type=float,
        default=1e-4,
        help='Learning rate for LSTM optimizer'
    )

    ## General Parameters ##
    parser.add_argument(
        '--device',
        default='cuda',
        help='device to use for training / testing'
    )
    return parser

def loadData(datacsv,val_percentage):
    num_val_samples = round(val_percentage*len(datacsv.index))
    val_indexes = datacsv.index[0:num_val_samples]
    train_indexes = datacsv.index[num_val_samples:]
    sets = {"Train": train_indexes, "Validation": val_indexes}
    datasets = []
    for learning_set in sets:
        print("Parsing {} data".format(learning_set.lower()))
        data = datacsv.iloc[sets[learning_set]].copy()
        data.index = [i for i in range(len(data.index))]
        data["Set"] = [learning_set for i in data.index]
        datasets.append(data)
    return datasets

def invert_class_mapping(class_mapping):
    inverted_mapping = {}
    for key in class_mapping:
        inverted_mapping[class_mapping[key]] = key
    print(inverted_mapping)
    return inverted_mapping

def writeLSTMConfig(foldDir,class_mapping, sequence_length, num_input_features,device):
    config = {"class_mapping":class_mapping,
              "sequence_length":sequence_length,
              "num_features":num_input_features,
              "device":device}
    with open(os.path.join(foldDir,"config.yaml"),"w") as f:
        yaml.dump(config,f)

def writeResultsToFile(saveLocation,resultsDict,confmat):
    linesToWrite = []
    for key in resultsDict:
        if "Train" in key or "Val" in key:
            linesToWrite.append("\n{}: {}".format(key, resultsDict[key][-1]))
        else:
            linesToWrite.append("\n{}: {}".format(key,resultsDict[key]))
    linesToWrite.append("\n\nConfusion matrix")
    linesToWrite.append("\n" + str(confmat))
    with open(os.path.join(saveLocation, "trainingInfo_" + resultsDict["Model name"] + ".txt"), 'w') as f:
        f.writelines(linesToWrite)

def saveTrainingPlot(saveLocation,resultsDict,metric):
    fig = plt.figure()
    numEpochs =len(resultsDict["Train " + metric])
    plt.plot([x for x in range(numEpochs)], resultsDict["Train " + metric], 'bo', label='Training '+metric)
    plt.plot([x for x in range(numEpochs)], resultsDict["Val " + metric], 'b', label='Validation '+metric)
    plt.title(resultsDict["Model name"]+' Training and Validation ' + metric)
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.savefig(os.path.join(saveLocation, resultsDict["Model name"]+'_'+metric + '.png'))
    plt.close(fig)

def getResnetPredictions(data,resnetModel,device):
    allPredictions = []
    images = []
    transforms = resnetModel.transforms
    resnetModel.cuda(device)
    for i in data.index:
        imgFilePath = os.path.join(data["Folder"][i],data["FileName"][i])
        img_tensor = transforms(Image.open(imgFilePath).resize((224,224)))
        images.append(img_tensor)
        if (i-min(data.index)) %10 ==0 or i==data.index[-1]:
            images = torch.from_numpy(numpy.array(images)).cuda(device)
            preds = resnetModel.forward(images)
            for pred in preds:
                pred = pred.cpu().detach().numpy()
                allPredictions.append(pred)
            print("Resnet predictions: {}/{} complete".format(i-min(data.index),len(data.index)))
            del images
            del preds
            gc.collect()
            images = []
    return numpy.array(allPredictions)

def trainResnet(foldDir, model,training_data,val_data,args,labelName="Overall Task"):
    gpu = torch.device(args.device)
    transforms = model.transforms
    resultsDict = {"Model name": "CNN",
                   "Num epochs": args.cnn_epochs,
                   "learning rate": args.cnn_lr,
                   "Train loss": [],
                   "Train accuracy": [],
                   "Val loss": [],
                   "Val accuracy": [],
                   "Final Val loss": None,
                   "Final Val accuracy": None}
    train_dataset = CNNDataset(training_data, labelName, transforms,balance=args.balance_cnn,augmentations = args.augment_cnn)
    val_dataset = CNNDataset(val_data, labelName, transforms,balance=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.cnn_batch,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.cnn_batch,
        num_workers=1,
        pin_memory=True,
    )
    classes = sorted(training_data[labelName].unique())
    lr = args.cnn_lr
    optimizer = optim.Adam(model.parameters(),lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    tool_loss = nn.BCEWithLogitsLoss()


    min_loss = 1e6
    num_epoch_without_improvement = 0
    for epoch in range(0, args.cnn_epochs):
        if num_epoch_without_improvement >= 4:
            print("Early stopping")
            break
        if num_epoch_without_improvement > 0 and num_epoch_without_improvement % 2 == 0:
            lr = adjust_learning_rate(lr, optimizer)#lr * 0.7
        model.train()
        loss_per_batch = []
        correct_instances = 0
        total_instances = 0
        start_time = last_logging = time.time()
        for step, (x, y) in enumerate(train_loader, start=epoch * len(train_loader)):
            x = x.cuda(gpu, non_blocking=True)
            y = y.cuda(gpu, non_blocking=True)
            y_pred = model.forward(x)
            loss = loss_fn(y_pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_per_batch.append(loss.item())
            y_pred = torch.softmax(y_pred, dim=1)
            categorical_y_pred = torch.argmax(y_pred, dim=1)
            categorical_y_true = y
            numCorrect = sum(categorical_y_pred == categorical_y_true).item()
            total_instances += len(y)
            correct_instances += numCorrect
            current_time = time.time()
            if current_time - last_logging > 30:
                stats = dict(
                    epoch=epoch,
                    step=step,
                    max_step=(epoch+1) * len(train_loader),
                    loss=loss.item(),
                    time=int(current_time - start_time),
                    estimated_time_remaining=int(((current_time - start_time)/(step-epoch*len(train_loader)))*((epoch+1)*len(train_loader)-step)),
                    lr=lr,
                )
                print(json.dumps(stats))
                last_logging = current_time
        state = dict(
            epoch=epoch + 1,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
        )
        train_accuracy = correct_instances / total_instances
        model.eval()
        val_loss_per_batch = []
        correct_instances = 0
        total_instances = 0
        start_time = last_logging = time.time()
        with torch.no_grad():
            for val_step, (x, y) in enumerate(val_loader, start=epoch * len(val_loader)):
                x = x.cuda(gpu, non_blocking=True)
                y = y.cuda(gpu, non_blocking=True)
                y_pred = model.forward(x)
                val_loss = loss_fn(y_pred, y)
                y_pred = torch.softmax(y_pred, dim=1)

                val_loss_per_batch.append(val_loss.item())
                categorical_y_pred = torch.argmax(y_pred, dim=1)
                categorical_y_true = y
                numCorrect = sum(categorical_y_pred == categorical_y_true).item()
                correct_instances += numCorrect
                total_instances += len(y)
                current_time = time.time()
                if current_time - last_logging > 30:
                    stats = dict(
                        epoch=epoch,
                        mode="Val",
                        step=val_step,
                        loss=val_loss.item(),
                        time=int(current_time - start_time),
                        estimated_time_remaining=int(
                            ((current_time - start_time) / (val_step - epoch * len(val_loader))) * ((epoch + 1) * len(val_loader) - val_step)),
                        lr=lr,
                    )
                    print(json.dumps(stats))
                    last_logging = current_time
        val_accuracy = correct_instances / total_instances
        train_loss = sum(loss_per_batch) / len(loss_per_batch)
        val_loss = sum(val_loss_per_batch) / len(val_loss_per_batch)
        resultsDict["Train loss"].append(train_loss)
        resultsDict["Train accuracy"].append(train_accuracy)
        resultsDict["Val loss"].append(val_loss)
        resultsDict["Val accuracy"].append(val_accuracy)
        if val_loss < min_loss:
            print("Val loss decreased from {} to {}. saving model.".format(min_loss, val_loss))
            torch.save(state, os.path.join(foldDir, "resnet.pth"))
            min_loss = val_loss
            num_epoch_without_improvement = 0
        else:
            num_epoch_without_improvement += 1
        print("Epoch: {} - Train loss: {}, Train accuracy: {}, Val loss: {}, Val accuracy: {}".format(epoch, train_loss,
                                                                                                      train_accuracy,
                                                                                                      val_loss,
                                                                                                      val_accuracy))
        if args.balance_cnn:
            train_dataset.balanceDataByVideo()

    ckpt = torch.load(os.path.join(foldDir, "resnet.pth"), map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    val_loss_per_batch = []
    correct_instances = 0
    total_instances = 0
    pred_labels = numpy.array([])
    true_labels = numpy.array([])
    with torch.no_grad():
        for val_step, (x, y) in enumerate(val_loader):
            x = x.cuda(gpu, non_blocking=True)
            y = y.cuda(gpu, non_blocking=True)
            y_pred = model.forward(x)
            val_loss = loss_fn(y_pred, y)
            y_pred = torch.softmax(y_pred, dim=1)
            val_loss_per_batch.append(val_loss.item())
            categorical_y_pred = torch.argmax(y_pred, dim=1)
            categorical_y_true = y
            numCorrect = sum(categorical_y_pred == categorical_y_true).item()
            correct_instances += numCorrect
            total_instances += len(x)
            pred_labels = numpy.concatenate([pred_labels, categorical_y_pred.cpu().numpy()], axis=None)
            true_labels = numpy.concatenate([true_labels, categorical_y_true.cpu().numpy()], axis=None)
    val_accuracy = correct_instances / total_instances
    val_loss = sum(val_loss_per_batch) / len(val_loss_per_batch)
    confMat = sklearn.metrics.confusion_matrix(true_labels, pred_labels)
    print("Final Testing Stats: loss: {}, accuracy: {}".format(val_loss, val_accuracy))
    print(confMat)
    resultsDict["Final Val loss"] = [val_loss]
    resultsDict["Final Val accuracy"] = [val_accuracy]
    writeResultsToFile(foldDir, resultsDict, confMat)
    saveTrainingPlot(foldDir, resultsDict, "loss")
    saveTrainingPlot(foldDir, resultsDict, "accuracy")

def getResnetFeatures(foldDir,set_name,data,resnetModel,device):
    if not os.path.exists(os.path.join(foldDir,"{}_resnet.npy".format(set_name))):
        res_preds = getResnetPredictions(data,resnetModel,device)
        numpy.save(os.path.join(foldDir,"{}_resnet.npy".format(set_name)),res_preds)
    else:
        res_preds = numpy.load(os.path.join(foldDir,"{}_resnet.npy".format(set_name)))
    return res_preds

def trainLSTM(foldDir,lstm_model,train_data,val_data,train_res_preds,val_res_preds,args,labelName="Overall Task"):
    gpu = torch.device(args.device)
    classes = sorted(list(set([x for x in train_data[labelName].unique()]+[x for x in val_data[labelName].unique()])))
    lstm_model.cuda(gpu)
    resultsDict = {"Model name": "LSTM",
                   "Num epochs": args.lstm_epochs,
                   "Sequence length": args.lstm_sequence_length,
                   "learning rate": args.lstm_lr,
                   "Train loss": [],
                   "Train accuracy": [],
                   "Val loss": [],
                   "Val accuracy": [],
                   "Final Val loss": [],
                   "Final Val accuracy": []}
    train_dataset = LSTMDataset(train_data, labelName, classes,train_res_preds, sequence_length=args.lstm_sequence_length,
                                balance=args.balance_lstm)
    val_dataset = LSTMDataset(val_data, labelName, classes,val_res_preds, sequence_length=args.lstm_sequence_length,
                              balance=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.lstm_batch,
        num_workers=1,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.lstm_batch,
        num_workers=1,
        pin_memory=True,
    )

    lr = args.lstm_lr
    optimizer = optim.Adam(lstm_model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()


    min_loss = 1e6
    num_epoch_without_improvement = 0
    if not os.path.exists(os.path.join(foldDir, "lstm.pth")):
        for epoch in range(0, args.lstm_epochs):
            if num_epoch_without_improvement >= 4:
                print("Early stopping")
                break
            if num_epoch_without_improvement > 0 and num_epoch_without_improvement % 2 == 0:
                lr = adjust_learning_rate(lr, optimizer)  # lr*0.7
            lstm_model.train()
            loss_per_batch = []
            correct_instances = 0
            total_instances = 0
            start_time = last_logging = time.time()
            for step, (x, y) in enumerate(train_loader, start=epoch * len(train_loader)):
                x = x.cuda(gpu, non_blocking=True)
                y = y.cuda(gpu, non_blocking=True)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    y_pred = lstm_model(x)
                    loss = customCELoss(loss_fn, y_pred, y)

                loss.backward()
                optimizer.step()
                loss_per_batch.append(loss.item())
                categorical_y_pred = torch.argmax(torch.softmax(y_pred[:, -1, :], dim=1), dim=1)
                categorical_y_true = torch.argmax(y[:, -1, :], dim=1)
                numCorrect = sum(categorical_y_pred == categorical_y_true).item()
                total_instances += len(y)
                correct_instances += numCorrect
                current_time = time.time()
                if current_time - last_logging > 30:  # args.log_freq_time:
                    stats = dict(
                        epoch=epoch,
                        step=step,
                        loss=loss.item(),
                        accuracy=correct_instances / total_instances,
                        time=int(current_time - start_time),
                        estimated_time_remaining=int(
                            ((current_time - start_time) / (step - epoch * len(train_loader))) * (
                                        (epoch + 1) * len(train_loader) - step)),
                        lr=lr,
                    )
                    print(json.dumps(stats))
                    last_logging = current_time
            state = dict(
                epoch=epoch + 1,
                model=lstm_model.state_dict(),
                optimizer=optimizer.state_dict(),
            )
            train_accuracy = correct_instances / total_instances
            lstm_model.eval()
            val_loss_per_batch = []
            correct_instances = 0
            total_instances = 0
            start_time = last_logging = time.time()
            with torch.no_grad():
                for val_step, (x, y) in enumerate(val_loader, start=epoch * len(val_loader)):
                    x = x.cuda(gpu, non_blocking=True)
                    y = y.cuda(gpu, non_blocking=True)
                    y_pred = lstm_model(x)

                    val_loss = customCELoss(loss_fn, y_pred, y)
                    val_loss_per_batch.append(val_loss.item())
                    categorical_y_pred = torch.argmax(torch.softmax(y_pred[:, -1, :], dim=1), dim=1)
                    categorical_y_true = torch.argmax(y[:, -1, :], dim=1)
                    numCorrect = sum(categorical_y_pred == categorical_y_true).item()
                    correct_instances += numCorrect
                    total_instances += len(y)
                    current_time = time.time()
                    if current_time - last_logging > 30:  # args.log_freq_time:
                        stats = dict(
                            epoch=epoch,
                            mode="Val",
                            step=val_step,
                            loss=val_loss.item(),
                            accuracy=correct_instances / total_instances,
                            time=int(current_time - start_time),
                            estimated_time_remaining=int(
                                ((current_time - start_time) / (val_step - epoch * len(val_loader))) * (
                                        (epoch + 1) * len(val_loader) - val_step)),
                            lr=lr,
                        )
                        print(json.dumps(stats))
                        last_logging = current_time
            val_accuracy = correct_instances / total_instances
            train_loss = sum(loss_per_batch) / len(loss_per_batch)
            val_loss = sum(val_loss_per_batch) / len(val_loss_per_batch)
            resultsDict["Train loss"].append(train_loss)
            resultsDict["Train accuracy"].append(train_accuracy)
            resultsDict["Val loss"].append(val_loss)
            resultsDict["Val accuracy"].append(val_accuracy)
            if val_loss < min_loss:
                print("Val loss decreased from {} to {}. saving model.".format(min_loss, val_loss))
                torch.save(state, os.path.join(foldDir, "lstm.pth"))
                min_loss = val_loss
                num_epoch_without_improvement = 0
            else:
                num_epoch_without_improvement += 1
            print("Epoch: {} - Train loss: {}, Train accuracy: {}, Val loss: {}, Val accuracy: {}".format(epoch,
                                                                                                          train_loss,
                                                                                                          train_accuracy,
                                                                                                          val_loss,
                                                                                                          val_accuracy))
            if args.balance_lstm:
                train_dataset.balanceDataByVideo()

    ckpt = torch.load(os.path.join(foldDir, "lstm.pth"), map_location="cpu")
    lstm_model.load_state_dict(ckpt["model"], strict=True)
    lstm_model.eval()
    val_loss_per_batch = []
    correct_instances = 0
    total_instances = 0
    pred_labels = numpy.array([])
    true_labels = numpy.array([])
    seq_preds = numpy.array([])
    seq_true = numpy.array([])
    with torch.no_grad():
        for val_step, (x, y) in enumerate(val_loader):
            x = x.cuda(gpu, non_blocking=True)
            y = y.cuda(gpu, non_blocking=True)
            y_pred = lstm_model(x)
            val_loss = customCELoss(loss_fn, y_pred, y)
            val_loss_per_batch.append(val_loss.item())
            categorical_y_pred = torch.argmax(torch.softmax(y_pred[:, -1, :], dim=1), dim=1)
            categorical_y_true = torch.argmax(y[:, -1, :], dim=1)
            numCorrect = sum(categorical_y_pred == categorical_y_true).item()
            correct_instances += numCorrect
            total_instances += len(x)
            pred_labels = numpy.concatenate([pred_labels, categorical_y_pred.cpu().numpy()], axis=None)
            true_labels = numpy.concatenate([true_labels, categorical_y_true.cpu().numpy()], axis=None)
            seq_y_pred = torch.argmax(torch.softmax(y_pred, dim=2), dim=2)
            seq_y_true = torch.argmax(y, dim=2)
            if seq_preds.size == 0:
                seq_preds = seq_y_pred.cpu().numpy()
                seq_true = seq_y_true.cpu().numpy()
            else:
                seq_preds = numpy.concatenate([seq_preds, seq_y_pred.cpu().numpy()], axis=0)
                seq_true = numpy.concatenate([seq_true, seq_y_true.cpu().numpy()], axis=0)
    print(seq_preds.shape)
    val_accuracy = correct_instances / total_instances
    val_loss = sum(val_loss_per_batch) / len(val_loss_per_batch)
    confMat = sklearn.metrics.confusion_matrix(true_labels, pred_labels)
    print("Final Testing Stats: loss: {}, accuracy: {}".format(val_loss, val_accuracy))
    print(confMat)
    jaccard = calculateJaccardSeq(val_data, seq_preds, true_labels)
    resultsDict["Final Val loss"] = [val_loss]
    resultsDict["Final Val accuracy"] = [val_accuracy]
    resultsDict["Jaccard"] = jaccard
    writeResultsToFile(foldDir, resultsDict, confMat)
    saveTrainingPlot(foldDir, resultsDict, "loss")
    saveTrainingPlot(foldDir, resultsDict, "accuracy")

def customCELoss(loss_function,y_pred,y_true):
    sequence_loss = 0
    for i in range(y_pred.size(1)):
        if y_pred.size(1)-1 == i:
            sequence_loss += 2*loss_function(y_pred[:, i, :], y_true[:, i, :])
        elif y_pred.size(1)- i < 5:
            sequence_loss += 1.2 * loss_function(y_pred[:, i, :], y_true[:, i, :])
        else:
            sequence_loss += loss_function(y_pred[:,i,:],y_true[:,i,:])
    return sequence_loss

def calculateJaccard(data,predictions,true_labels):
    videos = data["Folder"].unique()
    overall_min_idx = data.index[0]
    jacc_string = "Jaccard:"
    for vid in videos:
        vid_data = data.loc[data["Folder"]==vid]
        preds = predictions[vid_data.index[0]-overall_min_idx:vid_data.index[-1]-overall_min_idx+1]
        true_task = true_labels[vid_data.index[0] - overall_min_idx:vid_data.index[-1] - overall_min_idx + 1]
        jaccard = sklearn.metrics.jaccard_score(true_task,preds,average='weighted')
        print("{}: {}".format(vid,jaccard))
        jacc_string += "\n{}: {}".format(vid,jaccard)
    jaccard = sklearn.metrics.jaccard_score(predictions, true_labels, average='weighted')
    print("Overall: {}".format(jaccard))
    jacc_string += "\nOverall: {}".format(jaccard)
    return jacc_string

def calculateJaccardSeq(data,predictions,true_labels):
    videos = data["Folder"].unique()
    overall_min_idx = data.index[0]
    sequence_length = predictions.shape[-1]
    all_preds = []
    jacc_string = "Jaccard:"
    for vid in videos:
        vid_data = data.loc[data["Folder"]==vid]
        vid_preds = []
        true_task = true_labels[vid_data.index[0] - overall_min_idx:vid_data.index[-1] - overall_min_idx + 1]
        for i in range(vid_data.index[0]-overall_min_idx,vid_data.index[-1]-overall_min_idx+1):
            frame_preds = []
            initial_idx = -1
            for j in range(i,min(true_labels.shape[0],i+sequence_length)):
                frame_preds.append(predictions[j][initial_idx])
                initial_idx-=1
            most_common = scipy.stats.mode(numpy.array(frame_preds))[0]
            vid_preds.append(most_common)
            all_preds.append(most_common)

        jaccard = sklearn.metrics.jaccard_score(true_task,vid_preds,average='weighted')
        print("{}: {}".format(vid,jaccard))
        jacc_string += "\n{}: {}".format(vid, jaccard)
    jaccard = sklearn.metrics.jaccard_score(all_preds, true_labels, average='weighted')
    print("Overall: {}".format(jaccard))
    jacc_string += "\nOverall: {}".format(jaccard)
    return jacc_string

def main(args):
    if args.save_location == "":
        print("No save location specified. Please set flag --save_location")
    elif args.data_csv_file == "":
        print("No dataset specified. Please set flag --data_csv_file")
    else:
        torch.backends.cudnn.benchmark = True
        print(args)
        gpu = torch.device(args.device)
        labelName = "Overall Task"
        dataCSVFile = pandas.read_csv(args.data_csv_file)
        num_classes = len(dataCSVFile[labelName].unique())
        foldDir = args.save_location
        if not os.path.exists(foldDir):
            os.mkdir(foldDir)
        args.save_location = foldDir
        network = CNN_LSTM()

        train_data,val_data = loadData(dataCSVFile,args.validation_percentage)
        class_counts = train_data[labelName].value_counts()
        classes = sorted(dataCSVFile[labelName].unique())
        print(class_counts)
        class_mapping = dict(zip([i for i in range(len(dataCSVFile[labelName].unique()))],
                                 sorted(dataCSVFile[labelName].unique())))

        num_input_features = args.cnn_features
        resnetModel = network.createCNNModel(num_input_features, num_classes).cuda(gpu)
        if not os.path.exists(os.path.join(foldDir, "resnet.pth")):
            trainResnet(foldDir, resnetModel,train_data,val_data,args,labelName=labelName)
        resnetModel = network.loadCNNModel(foldDir)
        resnetModel.return_head = False
        train_res_preds = getResnetFeatures(foldDir,"Train",train_data,resnetModel,args.device)
        val_res_preds = getResnetFeatures(foldDir,"Validation",val_data,resnetModel,args.device)


        writeLSTMConfig(foldDir, class_mapping, args.lstm_sequence_length, num_input_features,args.device)
        lstm_model = network.createLSTMModel(num_input_features,len(classes))
        trainLSTM(foldDir,lstm_model, train_data, val_data, train_res_preds, val_res_preds, args, labelName)


def adjust_learning_rate(lr, optimizer):
    lr = lr*0.7
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

class FocalLoss(nn.Module):
    '''
    Multi-class Focal Loss
    '''
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        #self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C], float32
        target: [N, ], int64
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser('CNN_LSTM training script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
