'''
Project: Applying similarity, affine and perspective transformations on a video.

Author: Pratik Ramdasi

Date: 02/ 10/ 2016

'''

# Import Libraries

import numpy as np
import glob
import cv2
import cv2.cv as cv
import os

# Define class sturcture Video including transform functions

class Video:
    def __init__(self,input_path,transformIndex,coordinates,output_path):
        self.fname=[]
        self.path=input_path
        self.output=output_path
        self.transformIndex=transformIndex
        self.coordinates=coordinates
        
        # Directory is checked for the existence of the required folders.
        # Folder contains the extracted input video frames
        # call Processing Functions to get transforms 
        newpath = r'Frames'
        if not os.path.exists(newpath): os.makedirs(newpath)
        self.framing(self.path)
        self.decideTransform()
        self.applyTransform()
        
    #  Method for similarity transformation
        
    def similarityTransform(self):
        self.height,self.width=cv2.imread("Frames/1.jpg").shape[:2]
        A = abs(float(self.coordinates[1][0]) - float(self.coordinates[0][0]))
        B = abs(float(self.coordinates[1][1]) - float(self.coordinates[0][1]))
        C = abs(float(self.coordinates[1][2]) - float(self.coordinates[0][2]))
        D = abs(float(self.coordinates[1][3]) - float(self.coordinates[0][3]))

        a = np.float32(float(C*A*(A+B*D))/float(A**2 + B))
        b = np.float32(float(C-D*A)/float(A**2 + B))
        c = np.float32(abs(float(self.coordinates[0][2]) - a*float(self.coordinates[0][0]) - b*float(self.coordinates[0][1])))
        d = np.float32(abs(float(self.coordinates[0][3]) - b*float(self.coordinates[0][0]) - a*float(self.coordinates[0][1])))

        M=np.array([[a ,b ,c],[-b ,a ,d]])
        #print "M is: ",M
        
        # sort the input frames
        folder=self.sort_files()
        
        # Process next frames
        for i in folder:
            pic="Frames/"+str(i)+".jpg"
            img = cv2.imread(pic)
            dst=cv2.warpAffine(img,M,(self.width,self.height))
            cv2.imwrite("Frames/%d.jpg" % i, dst)
            
            
    # Method for affine transformation: OpenCV Module
            
    def affineTransform(self):
        folder=self.sort_files()
        P=self.get_points()
        self.height,self.width=cv2.imread("Frames/1.jpg").shape[:2]
        # Process frames
        for i in folder:
            pic="Frames/"+str(i)+".jpg"
            img = cv2.imread(pic)
            pts1 = np.float32([[P[0][0],P[0][1]],[P[1][0],P[1][1]],[P[2][0],P[2][1]]])
            pts2 = np.float32([[P[0][2],P[0][3]],[P[1][2],P[1][3]],[P[2][2],P[2][3]]])
            M = cv2.getAffineTransform(pts1,pts2)
            dst = cv2.warpAffine(img,M,(self.width,self.height))
            cv2.imwrite("Frames/%d.jpg" % i, dst)


    # Method for Perspective transformation: OpenCV Module

    def perspectiveTransform(self):
        folder=self.sort_files()
        P=self.get_points()
        self.height,self.width=cv2.imread("Frames/1.jpg").shape[:2]
        # Process frames  
        for i in folder:
            pic="Frames/"+str(i)+".jpg"
            img = cv2.imread(pic)
            pts1 = np.float32([[P[0][0],P[0][1]],[P[1][0],P[1][1]],[P[2][0],P[2][1]],[P[3][0],P[3][1]]])
            pts2 = np.float32([[P[0][2],P[0][3]],[P[1][2],P[1][3]],[P[2][2],P[2][3]],[P[3][2],P[3][3]]])
            M = cv2.getPerspectiveTransform(pts1,pts2)
            dst = cv2.warpPerspective(img,M,(self.width,self.height))
            cv2.imwrite("Frames/%d.jpg" % i, dst)

    # Get x,y co-ordinates 

    def get_points(self):
        P=np.array(self.coordinates)
        return P

    # Extract frames from the video

    def framing(self,path): 
        cap = cv2.VideoCapture(path)
        success,frame=cap.read(cv.CV_IMWRITE_JPEG_QUALITY)      #handle of the Video Capture is required for obtaining frame.

        count = 1
        while success:
          cv2.imwrite("Frames/%d.jpg" % count, frame)           # save frame as JPEG file
          count += 1
          success,frame = cap.read(cv.CV_IMWRITE_JPEG_QUALITY)  # to read the last frame

        cap.release()

    # select transformation 

    def decideTransform(self): 
        if self.transformIndex == 2:
            self.similarityTransform()
        elif self.transformIndex == 3:
            self.affineTransform()
        elif self.transformIndex == 4:
            self.perspectiveTransform()
        else:
            print ("Not correct number of pair of points")
        
        self.writeOutputFile(self.output)

    # Apply selected transformation
        
    def applyTransform(self):
        self.framing(self.path)
        self.height,self.width=cv2.imread("Frames/1.jpg").shape[:2]
        
        # write transformed video
        
        out = cv2.VideoWriter("changedOutput.mp4",cv.CV_FOURCC('a','v','c','1'), 30.0, (self.width, self.height))
        folder=self.sort_files()
        
        # write Transformed video frames
        
        for i in folder:
            pic="Frames/"+str(i)+".jpg"
            Newpic=cv2.imread(pic,0)
            frame=cv2.Canny(Newpic,100,200)
            cv2.imwrite(pic,frame)
            Newpic=cv2.imread(pic)
            img=cv2.flip(Newpic,0)
            out.write(img)
        out.release()
              
    # Writing output video file
              
    def writeOutputFile(self,output):
        self.height,self.width=cv2.imread("Frames/1.jpg").shape[:2]
        out = cv2.VideoWriter(output,cv.CV_FOURCC('a','v','c','1'), 30.0, (self.width, self.height))
        folder=self.sort_files()
            
        for i in folder:
            pic="Frames/"+str(i)+".jpg"
            img=cv2.imread(pic) 
            out.write(img)
        out.release()

    # Method to sort the files (here, frames!)
                     
    def sort_files(self):
        '''Files in python are not sorted normally, they are sorted in the order in which the numbers appear:
        1. That means 1, 10, 100, 2, 20, 200...and so on...
        2. so we obtain the ending part of the filenames and then sort that array and return it.
        '''
        for file in sorted(glob.glob("Frames/*.*")):
            s=file.split('/')
            a=s[-1].split('\\')
            x=a[-1].split('.')
            self.fname.append(int(x[0]))
        return(sorted(self.fname)) 

# Main Function

if __name__ == "__main__":  
    flag=False
    while(flag!=True):       
        # User input arguments in the format mentioned in Readme document.    
        string=raw_input("Enter the arguments: ")
        tokens=string.split(" ")
        if(len(tokens)==3):
            inputfile,textfile,outputfile = tokens[0], tokens[1], tokens[2]
            print "Input file is: ", inputfile
            print "Test file is: ", textfile
            print "Output file is: ",outputfile
            flag=True
        else:
            flag=False
        
    count = 0
    coordinates = []
    with open(textfile) as txtfile:
        for line in txtfile:
            count += 1
            val = list(line.split())
            coordinates.append(val)

    v=Video(inputfile,count,coordinates,outputfile) 

