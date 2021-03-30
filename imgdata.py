import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import math
import pandas as pd
from skimage.feature import hog

class ProcessImage:
    def __init__(self, dir, path_to_csv):
        self.dir = dir

        #the file names of the images
        self.imgList = os.listdir(dir)

        #the image data spreadsheet
        self.imgData = pd.read_csv(path_to_csv, sep=r'\s*,\s*', header=0, encoding='ascii', engine='python')

    def getImgList(self):
        return self.imgList

    def __len__(self):
        return len(self.imgList)

    def convertToArray(self, imgName):
        img = np.array(Image.open(self.dir + "/" + imgName))
        return np.uint8(img * 255)

    def getImgClass(self, imgName):
        return self.imgData.loc[self.imgData['filename'] == "train/"+imgName, 'class'].values[0]

    def getHogFeatures(self, image):
        return hog(image, orientations = 8, pixels_per_cell = (2,2), cells_per_block = (2,2), visualize=False, block_norm='L2-Hys')

    def extractImages(self, start, stop, length):
        images = []
        imageTypes = []
        for i in range(start, stop):
            images.append(self.getHogFeatures(self.convertToArray(self.imgList[i])))
            #type = self.getImgClass(self.imgList[i])
            # if(type == 0):
            #     imageTypes.append(0)
            # else:
            #     imageTypes.append(1)
            # print(i)
            if(self.imgList[i][0] == "N"):
                imageTypes.append(0)
            else:
                imageTypes.append(1)

            print(i)

        for i in range(length - stop, length - start):
            images.append(self.getHogFeatures(self.convertToArray(self.imgList[i])))
            #type = self.getImgClass(self.imgList[i])
            # if(type == 0):
            #     imageTypes.append(0)
            # else:
            #     imageTypes.append(1)
            # print(i)
            if(self.imgList[i][0] == "N"):
                imageTypes.append(0)
            else:
                imageTypes.append(1)

            print(self.imgList[i])

        return np.array(images), np.array(imageTypes)

    def crossValSet(divisions):
        divLen = math.ceil(len(imgList) / divisions)
        dataSets = []
        for i in range(divisions - 1):
            dataSets.append(imgList[i*divLen : i*divLen + divLen])
        dataSets.append(imgList[divLen*divisions : len(imgList)])
