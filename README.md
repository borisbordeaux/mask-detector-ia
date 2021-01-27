# Project Mask Detection

## Installation

1) Clone the repository https://github.com/fizyr/keras-retinanet.git
2) In the repository, execute `pip install . --user`.
3) Ensure that you have the pretrained model .h5

## How to use

Use the `maskDetector_CreateCSV.py` file to create the csv file based on a dataset given in parameters. You need to create both training and validation csv files.

### Calculate mean Average Precision

1) Use the `maskDetector_CreateTxt_mAP.py` file to create the the txt files for goundtruth used to calculate the mAP.
2) Use the `maskDetector_CreateTxt_mAP_inference.py` file to create the the txt files for detections used to calculate the mAP.
3) The the `main.py` file in the `mAP` folder in order to get the mAP.

### Inference on the dataset

Use the `maskDetector_DataSetInference.py` file to make inference on ramdom files of the dataset given in parameters.

### Realtime inferences

Use the `maskDetector_RealTimeInference.py` file to make real time inferences using camera.