import pandas as pd

import os
import cv2
import numpy as np

#argument parser
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("images", help="Path to input images")
parser.add_argument("output", help="Path to the output text files")
parser.add_argument("classes", help="Path to the classes csv file. Please create it using maskDetector_CreateCSV.py script")
parser.add_argument("model", help="Path to the .h5 model file")
parser.add_argument("threshold", help="Value in % of the threshold to use for the inference", type=int)

args = parser.parse_args()

imgPath = args.images
outpath = args.output
modelPath = args.model
threshold = args.threshold/100.0

#set memory growth to avoid errors
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

from keras_retinanet.utils.image import preprocess_image, resize_image

#get label names
label_map = {}
for line in open(args.classes):
  row = line.rstrip().split(',')
  label_map[int(row[1])] = row[0]

#create model
from keras_retinanet import models

model = models.load_model(modelPath, backbone_name='resnet50')
model = models.convert_model(model)

#for each image we make an inference
for img in os.listdir(imgPath):
    currentText = ""
    
    im = cv2.imread(imgPath+img, cv2.IMREAD_COLOR)
    
    im = im[:,:,:3]
    
    imp = preprocess_image(im)
    imp, scale = resize_image(im, 240, 1280)
      
    boxes, scores, labels = model.predict_on_batch(
      np.expand_dims(imp, axis=0)
    )
      
    # standardize box coordinates
    boxes /= scale
      
    # loop through each prediction for the input image
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can quit as soon
        # as we see a score below threshold
        if score < threshold:
          break
        
        box = box.astype(np.int32)
        
        #get the values needed
        class_name = label_map[label]
        currentText += "{} {} {} {} {} {}\n".format(class_name,
                                                 score,
                                                  box[0],
                                                  box[1],
                                                  box[2],
                                                  box[3])
    
    #write the values to txt file
    nameTxtFile = img.split(".")[0] + ".txt"
    print(outpath + nameTxtFile)
    with open(outpath + nameTxtFile, 'w') as f:
        f.write(currentText)
