import torch
import numpy
import random
import os
import cv2
import ast
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import math
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

TOOL_ORDER = ['anesthetic', 'catheter', 'dilator', 'guidewire', 'guidewire_casing', 'scalpel', 'syringe', 'ultrasound']
TOOL_VECTOR_LENGTH = len(TOOL_ORDER) * 4

def parseToolVector(tool_string):
        tool_vector = numpy.zeros((TOOL_VECTOR_LENGTH,), dtype=numpy.float32)
        try:
            tools = ast.literal_eval(tool_string)
            for t in tools:
                cls = t['class']
                if cls in TOOL_ORDER:
                    idx = TOOL_ORDER.index(cls) * 4
                    tool_vector[idx] = float(t['xmin'])
                    tool_vector[idx + 1] = float(t['ymin'])
                    tool_vector[idx + 2] = float(t['xmax'])
                    tool_vector[idx + 3] = float(t['ymax'])
        except:
            pass
        return tool_vector

class CNNDataset(Dataset):
    def __init__(self, datacsv, labelName, transforms, balance=True, img_size=224, augmentations=False):
        self.datacsv = datacsv.copy()
        self.transforms = transforms
        self.balanced = balance
        self.data = self.datacsv.index.copy()
        self.labelName = labelName
        self.labels = sorted(self.datacsv[self.labelName].unique())
        self.balanceDataByVideo()
        self.img_size = img_size
        self.augmentations = augmentations
        self.currentIndexes = dict(zip(range(len(self.labels)), [0 for _ in range(len(self.labels))]))

    def __len__(self):
        if self.balanced:
            minCount = min(len(self.sample_mapping[label]) for label in self.labels)
            return round(minCount * len(self.labels))
        else:
            return len(self.data)

    def convertCategoricalToOneHot(self, label):
        idx = self.labels.index(label)
        return idx

    def balanceDataByVideo(self, num_samples=100):
        print("Resampling data")
        self.sample_mapping = {}
        videos = sorted(self.datacsv["Folder"].unique())
        for vid in videos:
            for label in self.labels:
                entries = self.datacsv.loc[(self.datacsv[self.labelName] == label) & (self.datacsv["Folder"] == vid)]
                if not entries.empty:
                    sample_indexes = random.choices(entries.index.copy(), k=num_samples)
                    if label in self.sample_mapping:
                        self.sample_mapping[label] += shuffle(sample_indexes)
                    else:
                        self.sample_mapping[label] = shuffle(sample_indexes)

    def rotateImage(self, image, angle=-1):
        if angle < 0:
            angle = random.randint(1, 359)
        center = tuple(numpy.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    def getBalancedSample(self):
        idx = random.randint(0, len(self.labels) - 1)
        sample_idx = self.currentIndexes[idx]
        label = self.labels[idx]
        sample = self.sample_mapping[label][sample_idx]
        row = self.datacsv.loc[sample]
        imgFilePath = os.path.join(row["Folder"], row["FileName"])
        img_tensor = self.transforms(Image.open(imgFilePath).resize((224, 224)))
        tool_tensor = torch.tensor(parseToolVector(row["Tool bounding box"]))

        if self.augmentations:
            img = img_tensor.permute(1, 2, 0).numpy()
            preprocessing = random.randint(0, 10)
            if preprocessing == 0:
                img = cv2.flip(img, 1)
            elif preprocessing == 1:
                img = cv2.flip(img, 0)
            elif preprocessing == 2:
                img = self.rotateImage(img)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)

        self.currentIndexes[idx] = (self.currentIndexes[idx] + 1) % len(self.sample_mapping[label])
        if self.currentIndexes[idx] == 0:
            for key in self.sample_mapping:
                self.sample_mapping[key] = shuffle(self.sample_mapping[key])

        label_tensor = torch.tensor(self.convertCategoricalToOneHot(label))
        return (img_tensor, tool_tensor), label_tensor

    def getNextSample(self, idx):
        sample = self.data[idx]
        row = self.datacsv.loc[sample]
        imgFilePath = os.path.join(row["Folder"], row["FileName"])
        img_tensor = self.transforms(Image.open(imgFilePath).resize((224, 224)))
        tool_tensor = torch.tensor(parseToolVector(row["Tool bounding box"]))
        label = self.convertCategoricalToOneHot(row[self.labelName])
        label_tensor = torch.tensor(label)
        return (img_tensor, tool_tensor), label_tensor

    def __getitem__(self, idx):
        if self.balanced:
            return self.getBalancedSample()
        else:
            return self.getNextSample(idx)

from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
from sklearn.utils import shuffle
import torch

class LSTMDataset(Dataset):
    def __init__(self,datacsv,labelName,classes,resnet_predictions,sequence_length=50,balance=True):
        self.datacsv = datacsv.copy()
        self.balanced = balance
        self.data = self.datacsv.index.copy()
        self.labelName = labelName
        self.resnet_predictions = resnet_predictions
        self.sequence_length = sequence_length
        self.labels = classes
        self.balanceDataByVideo()
        self.currentIndexes = dict(zip([i for i in range(len(self.labels))],[0 for i in range(len(self.labels))]))
        self.minmaxscaler = MinMaxScaler(feature_range=(0, 1))
        self.minmaxscaler.fit(self.resnet_predictions)


    def __len__(self):
        if self.balanced:
            return len(self.labels)*len(self.sample_mapping[self.labels[0]])
        else:
            return len(self.data)

    def convertCategoricalToOneHot(self,label):
        one_hot_label = numpy.zeros((len(self.labels)))
        idx = self.labels.index(label)
        one_hot_label[idx] = 1
        return one_hot_label

    def splitDataByClass(self):
        self.sample_mapping = {}
        for label in self.labels:
            entries = self.datacsv.loc[self.datacsv[self.labelName]==label]
            self.sample_mapping[label] = shuffle(entries.index.copy())

    def balanceDataByVideo(self,num_samples = 100):
        if self.balanced:
            print("Resampling data")
        self.sample_mapping = {}
        videos = sorted(self.datacsv["Folder"].unique())
        class_counts = {}
        for vid in videos:
            for label in self.labels:
                entries = self.datacsv.loc[(self.datacsv[self.labelName] == label) & (self.datacsv["Folder"]==vid)]
                if not entries.empty:
                    sample_indexes = random.choices(entries.index.copy(),k=num_samples)
                    if label in self.sample_mapping:
                        self.sample_mapping[label] += shuffle(sample_indexes)
                    else:
                        self.sample_mapping[label] = shuffle(sample_indexes)
        for label in self.labels:
            try:
                self.sample_mapping[label] = shuffle(self.sample_mapping[label])
                class_counts[label] = len(self.sample_mapping[label])
            except KeyError:
                pass
        if self.balanced:
            print(class_counts)


    def getBalancedSample(self):
        idx = random.randint(0, len(self.labels) - 1)
        sample_idx = self.currentIndexes[idx]
        label = self.labels[idx]
        sample = self.sample_mapping[label][sample_idx]
        res_idx = sample - min(self.data)
        downsampling_rate = 1
        label = [self.convertCategoricalToOneHot(self.datacsv[self.labelName][sample])]
        videoID = self.datacsv["Folder"][sample]
        
        current_feat = self.resnet_predictions[res_idx].copy()
        current_bbox = parseToolVector(self.datacsv.loc[sample]["Tool bounding box"])
        combined = np.concatenate([current_feat, current_bbox])

        sequence = [combined]
        sample_shape = combined.shape
        next_idx = res_idx - downsampling_rate
        for i in range(self.sequence_length - 1):
            if next_idx >= 0:
                next_sample = self.data[next_idx]
                next_videoID = self.datacsv["Folder"][next_sample]
                if next_videoID == videoID:
                    nextSample = self.resnet_predictions[next_idx].copy()
                    label.insert(0,self.convertCategoricalToOneHot(self.datacsv[self.labelName][next_sample]))
                else:
                    nextSample = numpy.ones(sample_shape).astype(float)*1e-6
                    label.insert(0,self.convertCategoricalToOneHot("nothing"))
            else:
                nextSample = numpy.ones(sample_shape).astype(float)*1e-6
                label.insert(0,self.convertCategoricalToOneHot("nothing"))
            sequence.insert(0, nextSample)
            next_idx -= downsampling_rate
        self.currentIndexes[idx] = (self.currentIndexes[idx] + 1) % (len(self.sample_mapping[self.labels[numpy.argmax(label[-1])]]))
        sequence = numpy.array(sequence).astype(float)
        sequence_tensor = torch.from_numpy(sequence).float()
        label_tensor = torch.from_numpy(numpy.array(label))
        return sequence_tensor,label_tensor

    def getNextSample(self,idx):
        sample = self.data[idx]
        res_idx = sample - min(self.data)
        downsampling_rate = 1
        videoID = self.datacsv["Folder"][sample]

        current_feat = self.resnet_predictions[res_idx].copy()
        current_bbox = parseToolVector(self.datacsv.loc[sample]["Tool bounding box"])
        combined = np.concatenate([current_feat, current_bbox])

        sequence = [combined]
        sample_shape = combined.shape
        next_idx = res_idx - downsampling_rate
        label = [self.convertCategoricalToOneHot(self.datacsv[self.labelName][sample])]

        for i in range(self.sequence_length - 1):
            if next_idx >= 0:
                next_sample = self.data[next_idx]
                next_videoID = self.datacsv["Folder"][next_sample]
                if next_videoID == videoID:
                    next_feat = self.resnet_predictions[next_idx].copy()
                    next_bbox = parseToolVector(self.datacsv.loc[next_sample]["Tool bounding box"])
                    next_combined = np.concatenate([next_feat, next_bbox])
                    label.insert(0, self.convertCategoricalToOneHot(self.datacsv[self.labelName][next_sample]))
                else:
                    next_combined = numpy.zeros(sample_shape)
                    label.insert(0, self.convertCategoricalToOneHot("nothing"))
            else:
                next_combined = numpy.zeros(sample_shape)
                label.insert(0, self.convertCategoricalToOneHot("nothing"))

            sequence.insert(0, next_combined)
            next_idx -= downsampling_rate

        sequence = numpy.array(sequence).astype(float)
        sequence_tensor = torch.from_numpy(sequence).float()
        label_tensor = torch.from_numpy(numpy.array(label))

        return sequence_tensor, label_tensor


    def __getitem__(self,idx):
        if self.balanced == True:
            sequence_tensor,label_tensor = self.getBalancedSample()
        else:
            sequence_tensor, label_tensor = self.getNextSample(idx)
        return (sequence_tensor,label_tensor)