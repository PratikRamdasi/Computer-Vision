import numpy as np
import os
import glob
import math
import cv2.cv as cv
import cv2
import random
import cPickle
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn import svm
import time
from PIL import Image
from scipy import linalg as LA

class HOGCompute:
    def __init__(self):

        FilePath = 'Training/'
        Folders = os.listdir(FilePath)
        LabelCount = 1
        FolderCheck = False
        Labels = np.array([])

        output = self.GetTestData()

        #FPA = np.zeros((48,2))
        #FPACount = 0
        FPA = np.array([])



        for FileName in Folders:
            pathToFolder = FilePath + FileName + "/"
            newEntry = os.listdir(pathToFolder)
            print "Label is: ", LabelCount

            for VideoEntry in newEntry:
                pathToVideoFile = pathToFolder + VideoEntry
                images = self.sort_files(pathToVideoFile)

                Number_of_Frames = len(images)
                height = 64
                width = 64

                samples = np.array([[0 for x in range(0,Number_of_Frames)] for x in range(0,(height*width))])
                BigCount = 1
                sampleIndex = 0

                for i in range(1,(Number_of_Frames+1)):
                    imgPath = pathToVideoFile + "/0 (" + str(i) + ").jpg"

                    #Read the image
                    img = cv2.imread(imgPath,0)
                    img = cv2.resize(img, (height, width))

                    #Vectorize the image
                    tempArray = np.reshape(img,(height*width)).T

                    #add to the big array
                    samples[:,sampleIndex]= np.copy(tempArray)
                    sampleIndex+=1

                Labels = np.append(Labels, LabelCount)
                data, eigenValues, eigenVectors = self.PCA(samples)

                input = eigenVectors
                print input.shape

                mat = np.dot(input.T, output)

                U, s, Vh = LA.svd(mat, full_matrices= False)


                angles = np.array([np.arccos(e) for e in s])
                #print "Principal angles: ", angles

                FPA = np.append(FPA, angles)
                #FPA[FPACount,:] = angles
                #FPACount += 1

                #PA = self.angle_between(v1,v2)
                #dotProduct = sum((a*b) for a, b in zip(v1, v2))
                #print FPA.shape
            LabelCount += 1

        #print FPA

        indx = np.argmin(FPA)

        '''
        print min(FPA[:,0]), min(FPA[:,1])
        indx1 = np.argmin(FPA[:,0])
        indx2 = np.argmin(FPA[:,1])
        print self.DisplayAction(Labels[indx1]),self.DisplayAction(Labels[indx2])
        '''
        print self.DisplayAction(Labels[indx])

    def unit_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def angle_between(self,v1, v2):
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def GetTestData(self):
        FilePath = 'Test/'
        Folders = os.listdir(FilePath)

        for FileName in Folders:
            images = self.sort_files(FilePath + FileName)

            Number_of_Frames = len(images)
            height = 64
            width = 64

            Testsamples = np.array([[0 for x in range(0,Number_of_Frames)] for x in range(0,(height*width))])
            BigCount = 1
            sampleIndex = 0

            for i in range(1,(Number_of_Frames+1)):
                TestimgPath = FilePath + str(FileName) + "/0 (" + str(i) + ").jpg"

                #Read the image
                img = cv2.imread(TestimgPath,0)
                img = cv2.resize(img, (height, width))

                #Vectorize the image
                tempArray = np.reshape(img,(height*width)).T

                #add to the big array
                Testsamples[:,sampleIndex]= np.copy(tempArray)
                sampleIndex+=1

            data, eigenValues, eigenVectors = self.PCA(Testsamples)

        return eigenVectors


    def PCA(self,data):
        """
        returns: data transformed in 2 dims/columns + regenerated original data
        pass in: data as 2D NumPy array
        """
        dims_rescaled_data=2
        m, n = data.shape

        data -= data.mean(axis=0)

        # calculate the covariance matrix
        R = np.cov(data, rowvar=False)
        # calculate eigenvectors & eigenvalues of the covariance matrix
        # use 'eigh' rather than 'eig' since R is symmetric,
        # the performance gain is substantial
        evals, evecs = LA.eig(R)
        # sort eigenvalue in decreasing order
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:,idx]
        # sort eigenvectors according to same index
        evals = evals[idx]
        # select the first n eigenvectors (n is desired dimension
        # of rescaled data array, or dims_rescaled_data)
        evecs = evecs[:, :dims_rescaled_data]
        # carry out the transformation on the data using eigenvectors
        # and return the re-scaled data, eigenvalues, and eigenvectors
        return np.dot(evecs.T, data.T).T, evals, evecs

    def DisplayAction(self,actionIndex):
        if(actionIndex == 1):
            Action = "Boxing"
        elif(actionIndex== 2):
            Action = "Handclapping"
        elif(actionIndex== 3):
            Action = "Handwaving"
        elif(actionIndex == 4):
            Action = "Running"
        elif(actionIndex == 5):
            Action = "Walking"
        return Action

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
