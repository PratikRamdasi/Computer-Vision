import numpy as np
import glob
import cv2
import cv2.cv as cv
from PIL import Image
import os, sys

class Video:
    def __init__(self):
        global FilePath, count
        if not os.path.exists("Images"): os.makedirs("Images")
        FilePath = "Training/"
        count = 1
        #for path in os.listdir(FilePath):
        #    self.framing(path)
        self.writeOutputFile()

    def framing(self,path):
        global FilePath, count
        Newpath = FilePath + path
        cap = cv2.VideoCapture(Newpath)
        success,frame=cap.read(cv.CV_IMWRITE_JPEG_QUALITY)      #handle of the Video Capture is required for obtaining frame.

        while success:
          cv2.imwrite("Images/%d.jpg" % count, frame)           # save frame as JPEG file
          count += 1
          success,frame = cap.read(cv.CV_IMWRITE_JPEG_QUALITY)  # to read the last frame

        cap.release()

    def writeOutputFile(self):
        self.height,self.width=cv2.imread("/Users/pratikramdasi/Desktop/frames/0 (1).jpg").shape[:2]
        out = cv2.VideoWriter("/Users/pratikramdasi/Desktop/vtest.mp4",cv.CV_FOURCC('a','v','c','1'), 30.0, (self.width, self.height))
        folder=self.sort_files()

        for i in folder:
            pic="/Users/pratikramdasi/Desktop/frames/0 ("+str(i)+").jpg"
            img=cv2.imread(pic)
            out.write(img)
        out.release()

                     
    def sort_files(self):
        self.fname=[]

        for file in sorted(glob.glob("/Users/pratikramdasi/Desktop/frames/*.*")):
            s=file.split('/')
            a=s[-1].split('.')
            temp=a[0].split(' ')
            x=temp[-1].strip('()')
            self.fname.append(int(x))
        return(sorted(self.fname)) 

if __name__ == "__main__": 
    v=Video()
