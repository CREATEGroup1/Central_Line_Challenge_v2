import numpy
import os
import cv2
import yaml
from ultralytics import YOLO

class YOLOv5:
    def __init__(self):
        self.model = None
        self.class_mapping = None

    def loadModel(self,modelFolder,modelName=None):
        self.model = YOLO(os.path.join(modelFolder,"train/weights/best.pt"))
        self.model_config = os.path.join(modelFolder,"model_config.yaml")
        with open(os.path.join(modelFolder,"config.yaml"),"r") as f:
            config = yaml.safe_load(f)
            self.class_mapping = config['class_mapping']

    def predict(self,image):
        results = self.model.predict(image,cfg=self.model_config)
        bboxes = results[0].boxes.xyxy.cpu().numpy()
        class_nums = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        resultList = []
        print(bboxes.shape)
        for i in range(bboxes.shape[0]):
            class_value = class_nums[i]
            class_name = self.class_mapping[class_value]
            xmin,ymin,xmax,ymax = bboxes[i]
            confidence = confs[i] #[class_value]
            bbox = {"class":class_name,
                    "xmin":round(xmin),
                    "ymin":round(ymin),
                    "xmax":round(xmax),
                    "ymax":round(ymax),
                    "conf":confidence}
            resultList.append(bbox)
        return str(resultList)

    def createModel(self,config=None):
        if config is None:
            self.model = YOLO("yolov8n.pt")
        else:
            self.model = YOLO(config)
        return self.model