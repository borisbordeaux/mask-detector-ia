
#argument parser
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="Path to csv file of the dataset. Please create it using maskDetector_CreateCSV.py script")
parser.add_argument("classes", help="Path to classes csv file of the dataset. Please create it using maskDetector_CreateCSV.py script")
parser.add_argument("images", help="Path to the folder containing the images")
parser.add_argument("annotations", help="Path to the folder containing the annotations")
parser.add_argument("model", help="Path to the .h5 model file")
parser.add_argument("--no-annotation", help="Doesn't show annotation", action="store_true")

args = parser.parse_args()

import numpy as np
import pandas as pd

from keras_retinanet.utils.visualization import draw_box, draw_caption , label_color
from keras_retinanet.utils.image import preprocess_image, resize_image
import cv2

#to set memory growth
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

#model creation
from keras_retinanet import models

model_path = args.model
model = models.load_model(model_path, backbone_name='resnet50')
model = models.convert_model(model)

#read the data
jpgPath= args.images
annotPath=args.annotations

data=pd.read_csv(args.csv)
data.columns = ['fileName', 'xmin', 'ymin', 'xmax', 'ymax', 'class']

label_map = {}
for line in open(args.classes):
  row = line.rstrip().split(',')
  label_map[int(row[1])] = row[0]
  
  
def image_with_predictions(df, threshold=0.6):
  # choose a random image
  row = df.sample()
  filepath = row['fileName'].values[0]
  
  print("filepath:", filepath)
  # get all rows for this image
  df2 = df[df['fileName'] == filepath]
  
  im = cv2.imread(filepath, cv2.IMREAD_COLOR)
  
  #to get only 3 color chanels
  im = im[:,:,:3]

  print("im.shape:", im.shape)

  if not args.no_annotation:
    # plot true boxes
    for idx, row in df2.iterrows():
      box = [
        row['xmin'],
        row['ymin'],
        row['xmax'],
        row['ymax']
      ]
      
      draw_box(im, box, color=(255, 0, 0))

  ### plot predictions ###

  # get predictions
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
    
    #draw boxe and caption
    color = label_color(label)
    draw_box(im, box, color=color)

    class_name = label_map[label]
    print(f'{class_name} : {score:.3f}')
    caption = f"{class_name} {score:.3f}"
    draw_caption(im, box, caption)
  
  return im  

#create window  
cv2.namedWindow("test",cv2.WINDOW_AUTOSIZE)

#inferes while the user doesn't press 'q'
while True:
    img = image_with_predictions(data, threshold=0.25)
    
    cv2.imshow("test", img)
    
    if cv2.waitKey() & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()