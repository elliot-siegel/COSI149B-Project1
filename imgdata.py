import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import math
import pandas as pd

class ImgData:
    def __init__(self, dir, path_to_csv):
        self.dir = dir

        #the file names of the images
        self.imgList = os.listdir(dir)

        #the image data spreadsheet
        self.imgData = pd.read_csv(path_to_csv, sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')

    def __len__(self):
        return len(self.imgList)

    def getImgData(self, imgName):
        return np.array(Image.open(self.dir + "/" + imgName))

    def getImgType(self, imgName):
        return self.imgData.loc[self.imgData['filename'] == imgName, 'class'].values[0]

    def crossValSet(divisions):
        divLen = math.ceil(len(imgList) / divisions)
        dataSets = []
        for i in range(divisions - 1):
            dataSets.append(imgList[i*divLen : i*divLen + divLen])
        dataSets.append(imgList[divLen*divisions : len(imgList)])

    def getStd(self, imgName):
        #numpy array indexing is like this, I think. 54 to 74 should be the center since the images are 129x129
        return np.std(self.getImgData(imgName)[54:74, 54:74])

    def getMean(self, imgName):
        return np.mean(self.getImgData(imgName)[54:74, 54:74])

    def getImgClass(self, imgName):
        return

data = ImgData("Project1/train", "Project1/coordinates_train.csv")
#histSet = []
negSetX = []
negSetY = []
posSetX = []
posSetY = []

for i in range(5000):

    imgType = data.getImgType("train/" + data.imgList[i])

    std = data.getStd(data.imgList[i])
    #histSet.append(std)

    if(imgType == 0):
        negSetX.append(std)
        negSetY.append(data.getMean(data.imgList[i]))
    else:
        posSetX.append(std)
        posSetY.append(data.getMean(data.imgList[i]))

#plt.hist(histSet, bins=20)
#plt.show()

plt.scatter(negSetX, negSetY)
plt.scatter(posSetX, posSetY)
plt.show()
