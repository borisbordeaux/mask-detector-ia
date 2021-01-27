import pandas as pd
import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join
import argparse

#arguments parser
parser = argparse.ArgumentParser()
parser.add_argument("img", help="Path to images")
parser.add_argument("annotation", help="Path to annotations")
parser.add_argument("--output", help="Path to the output csv file", default="./maskDetectorData.csv")
parser.add_argument("--img-extension", help="Extension of images", choices=[".png", ".jpg"], default=".png")

args = parser.parse_args()

jpgPath = args.img
annotPath= args.annotation

#create the dataframe for the csv file
data=pd.DataFrame(columns=['fileName', 'xmin', 'ymin', 'xmax', 'ymax', 'class'])

#read the xml files
allfiles = [f for f in listdir(annotPath) if isfile(join(annotPath, f))]
for file in allfiles:
    if (file.split(".")[1]=='xml'):
        fileName = jpgPath + file.replace(".xml", args.img_extension)
        tree = ET.parse(annotPath+file)
        root = tree.getroot()
        for obj in root.iter('object'):
            cls_name = obj.find('name').text
            xml_box = obj.find('bndbox')
            xmin = xml_box.find('xmin').text.split(".")[0]
            ymin = xml_box.find('ymin').text.split(".")[0]
            xmax = xml_box.find('xmax').text.split(".")[0]
            ymax = xml_box.find('ymax').text.split(".")[0]
            data = data.append({'fileName': fileName, 'xmin': xmin, 'ymin':ymin,'xmax':xmax,'ymax':ymax,'class':cls_name}, ignore_index=True)
            
print(data.shape)

#save the csv file
data.to_csv(args.output ,index=False, header=False)

#save csv file for classes
classes = ['with_mask', 'mask_weared_incorrect', 'without_mask']
with open('maskDetectorClasses.csv', 'w') as f:
  for i, class_name in enumerate(classes):
    f.write(f'{class_name},{i}\n')
