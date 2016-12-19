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
from collections import Counter

class HOGCompute:
    def __init__(self):
        FilePath = 'Test/'
        Folders = os.listdir(FilePath)

        with open('my_SVM_file.pkl', 'rb') as fid:
            clf = cPickle.load(fid)

        for FileName in Folders:
            images = self.sort_files(FilePath + FileName)

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
                    imgPath = FilePath + str(FileName) + "/0 (" + str(i) + ").jpg"
                    img = cv2.imread(imgPath,0)
                    #img = cv2.resize(img, (160, 120))
                    h1 = hog.compute(img,winStride,padding,locations).T
                    temp = np.copy(h1)


                    if(hogCount == 0):
                        hogTemp = np.zeros((nInterval, len(temp[0])))
                        #print "Shape of hogTemp is: ", hogTemp.shape
                        hogTemp[hogCount]= temp[0]
                        if (FirstEntryFlag == False):
                            FirstHOGEntry = np.copy(temp)
                            FirstEntryFlag = True
                    else:
                        hogTemp[hogCount]= temp

                    #print "Shape of hogTemp is: ", hogTemp.shape
                    hogCount += 1

                HOGPH = self.computeHOGPH(hogTemp, FirstHOGEntry)
                #HOGPH = normalize(HOGPH)

                if (BigCount == 1):
                    bigArray = np.copy(HOGPH)
                else:
                    bigArray = np.vstack((bigArray, HOGPH))
                BigCount += 1

                index += nInterval

            print "Shape of Big array is: ", bigArray.shape
            print (clf.predict(bigArray))
            most_common,num_most_common = Counter(clf.predict(bigArray)).most_common(1)[0]
            print "Action is: ",self.DisplayAction(most_common)
            #self.WriteAction(self.DisplayAction(most_common))


    def computePCA(self,array):
        pca = PCA()
        newData = pca.fit_transform(array)
        MeanArray = np.mean(newData, axis =0)
        print "Size of Mean array: ", MeanArray.shape
        return MeanArray

    def computeHOGPH(self,array, firstEntry):
        hogph = firstEntry
        for j in range(1,len(array)):
            hogph += array[j-1] - array[j]

        return hogph

    def DisplayAction(self,actionIndex):
        if(actionIndex == 1):
            Action = "Handwaving"
        elif(actionIndex== 2):
            Action = "Running"
        elif(actionIndex == 3):
            Action = "Walking"
        return Action

    def WriteAction(self, string):
        FramePath = "FramesFinalFull/"
        #entries=os.listdir(FramePath)
        entries = self.sort_files(FramePath)
        for frame in entries:
            pic = FramePath + "0 (" + str(frame) + ").jpg"
            img = cv2.imread(pic)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,string,(10,20), font, 1,(0,0,255),1)
            cv2.imwrite(pic, img)

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
