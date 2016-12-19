import numpy as np
import os
import glob
import cv2.cv as cv
import cv2
import random
import cPickle
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn import svm
import time
from PIL import Image
from resizeimage import resizeimage

class HOGCompute:
    def __init__(self):

        FilePath = 'TrainingNew/'
        Folders = os.listdir(FilePath)
        LabelCount = 1
        FolderCheck = False
        Labels = np.array([])

        for FileName in Folders:
            pathToFolder = FilePath + FileName + "/"
            newEntry = os.listdir(pathToFolder)
            print "Label is: ", LabelCount

            for VideoEntry in newEntry:
                pathToVideoFile = pathToFolder + VideoEntry
                images = self.sort_files(pathToVideoFile)


                winSize = (128,64)
                blockSize = (16,16)
                blockStride = (8,8)
                cellSize = (8,8)
                nbins = 9
                derivAperture = 0
                winSigma = -1
                histogramNormType = 0
                L2HysThreshold = 2.0000000000000001e-01
                gammaCorrection = 0
                nlevels = 64
                hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
                winStride = (8,8)
                padding = (8,8)
                locations = ((10,20),)

                nInterval = 10
                BigCount = 1

                index = 1

                FirstEntryFlag = False

                while(index < len(images)):
                    hogCount = 0
                    for i in range(index,(index + nInterval)):
                        imgPath = pathToVideoFile + "/0 (" + str(i) + ").jpg"

                        #READ THE IMAGE HERE
                        img = cv2.imread(imgPath)
                        img = cv2.resize(img, (160, 120))
                        h1 = hog.compute(img,winStride,padding,locations).T
                        #print "Shape of HOG features is: ", h1.shape
                        temp = np.copy(h1)
                        #print "Shape of temp is: ", temp.shape

                        if(hogCount == 0):
                            hogTemp = np.zeros((nInterval, len(temp[0])))
                            #print "Shape of hogTemp is: ", hogTemp.shape
                            hogTemp[hogCount]= temp[0]
                            if (FirstEntryFlag == False):
                                FirstHOGEntry = np.copy(temp)
                                FirstEntryFlag = True
                        else:
                            hogTemp[hogCount]= temp

                        hogCount += 1


                    #HOGPH = self.computePCA(hogTemp)
                    HOGPH = self.computeHOGPH(hogTemp, FirstHOGEntry)
                    Labels = np.append(Labels, LabelCount)
                    #HOGPH = normalize(HOGPH)

                    #print "Shape of HOGPH is: ", HOGPH.shape

                    if (BigCount == 1):
                        bigArray = np.copy(HOGPH)
                    else:
                        bigArray = np.vstack((bigArray, HOGPH))
                    BigCount += 1
                    #print "Shape of bigArray is: ", bigArray.shape



                    index += nInterval
                    #print "Index value is: ", index

                if (FolderCheck == False):
                    TrainingData = np.copy(bigArray)
                    FolderCheck = True
                else:
                    TrainingData = np.vstack((TrainingData, bigArray))

            LabelCount += 1



        print "TrainingData Size is: ", TrainingData.shape
        Labels = Labels.T
        print "Labels shape is: ",Labels.shape


        clf = svm.SVC()
        clf.fit(TrainingData,Labels)

        with open('my_SVM_file.pkl', 'wb') as fid:
            cPickle.dump(clf, fid)



    def computePCA(self,array):
        pca = PCA()
        newData = pca.fit_transform(array)
        MeanArray = np.mean(newData, axis =0)
        #print "Size of Mean array: ", MeanArray.shape
        return MeanArray

    def computeHOGPH(self,array, firstEntry):
        hogph = firstEntry
        #hogph = np.copy(array[0])
        for j in range(1,len(array)):
            hogph += array[j-1] - array[j]

        return hogph


    def sort_files(self, index):
        self.fname=[]
        path = str(index) + "/*.*"
        for file in sorted(glob.glob(path)):
            s=file.split ('/')
            a=s[-1].split('\\')
            x=a[-1].split('.')
            o= x[0].split('(')[1]
            o = o.split(')')[0]
            self.fname.append(int(o))
        return(sorted(self.fname))

if __name__=='__main__':
    start_time = time.time()
    h= HOGCompute()
    print("--- %s seconds ---" % (time.time() - start_time))
