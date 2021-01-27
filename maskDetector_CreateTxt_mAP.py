
#arguments parser
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("csv", help="Path to input csv file")
parser.add_argument("output", help="Path to the output text files")

args = parser.parse_args()

import pandas as pd

outpath = args.output

#read data
data=pd.read_csv(args.csv)
data.columns = ['fileName', 'xmin', 'ymin', 'xmax', 'ymax', 'class']

currentFile = ""
currentText = ""

#write txt files for each images
for idx, row in data.iterrows():
    print(currentFile)
    nameTxtFile = row['fileName'].split("/")[-1].split(".")[0] + ".txt"
    if currentFile != nameTxtFile:
        with open(outpath + (currentFile if currentFile != "" else nameTxtFile), 'w') as f:
            f.write(currentText)
        currentText = ""
        currentFile = nameTxtFile
        
    currentText += "{} {} {} {} {}\n".format(row['class'],
                                              row['xmin'],
                                              row['ymin'],
                                              row['xmax'],
                                              row['ymax'])

with open(outpath + currentFile, 'w') as f:
    f.write(currentText)
