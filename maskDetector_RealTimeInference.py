
#arguments parser
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("classes", help="Path to classes csv file of the dataset. Please create it using maskDetector_CreateCSV.py script")
parser.add_argument("model", help="Path to the .h5 model file")

args = parser.parse_args()

import numpy as np

from keras_retinanet.utils.visualization import draw_box, draw_caption , label_color
from keras_retinanet.utils.image import preprocess_image, resize_image
import cv2
import time

import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

#model_paths = glob('snapshots/resnet50_csv_*.h5')
#latest_path = sorted(model_paths)[-1]
latest_path = args.model
print("path:", latest_path)

from keras_retinanet import models

model = models.load_model(latest_path, backbone_name='resnet50')
model = models.convert_model(model)

label_map = {}
for line in open(args.classes):
  row = line.rstrip().split(',')
  label_map[int(row[1])] = row[0]
  
  
def real_time_prediction(im, threshold):
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
    #print(label)
    color = label_color(label)
    draw_box(im, box, color=color, thickness=2)

    class_name = label_map[label]
    caption = f"{class_name} {score:.3f}"
    draw_caption(im, box, caption)
  return im
  
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX

while(True):
    
    time1 = time.time()
    
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = real_time_prediction(frame, 0.5)

    time2 = time.time()
    
    #compute fps
    timeInSec = time2-time1
    
    fps = 1/timeInSec
    fps = int(fps)
    fps = str(fps)
    
    cv2.putText(img,fps, (7, 70), font, 1, (100, 255, 0), 3, cv2.LINE_AA)
    
    # Display the resulting frame
    cv2.imshow('frame',img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()