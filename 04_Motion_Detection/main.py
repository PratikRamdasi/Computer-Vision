# import the necessary packages
import numpy as np
import os
import glob
import cv2.cv as cv
import cv2
import random

class video:
    def __init__(self,path):
        global newpath
        self.numberOfSamples = 20
        self.requiredMatches = 2
        self.distanceThreshold = 20
        self.subsamplingFactor = 16
        self.fname=[]
        self.path=path
        newpath = r'Frames'
        if not os.path.exists(newpath): os.makedirs(newpath)
        newpath = r'NewFrames'
        if not os.path.exists(newpath): os.makedirs(newpath)
        bigSampleArray = self.initialFraming(self.path)
        self.processVideo(bigSampleArray)

    def sort_files(self):
        for file in sorted(glob.glob("Frames/*.*")):
            s=file.split ('/')
            a=s[-1].split('\\')
            x=a[-1].split('.')
            self.fname.append(int(x[0]))
        return(sorted(self.fname)) 

    def initialFraming(self,path):
        global cap
        global success
        global frame
        
        sampleIndex=0
        cap = cv2.VideoCapture(path)
        success,frame=cap.read(cv.CV_IMWRITE_JPEG_QUALITY)       
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        height,width = gray.shape[:2]
        print "Dimension of the image is: ",height, width, (height*width)

        samples = np.array([[0 for x in range(0,self.numberOfSamples)] for x in range(0,(height*width))])

        tempArray = np.reshape(gray,(height*width)).T
        
        samples[:,sampleIndex]= np.copy(tempArray)
        sampleIndex+=1

        while (success and sampleIndex!=(self.numberOfSamples)):
            success,frame = cap.read(cv.CV_IMWRITE_JPEG_QUALITY)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            tempArray = (np.reshape(gray,(height*width))).T
            samples[:,sampleIndex]= np.copy(tempArray)
            sampleIndex+=1

        return samples

    def writeVideo(self):
        height,width=cv2.imread("Frames/1.jpg").shape[:2]
        out = cv2.VideoWriter("changedOutput.ogv",cv.CV_FOURCC('t','h','e','0'), 25.0, (width,height))
        folder=self.sort_files()
            
        for i in folder:
            pic="Frames/"+str(i)+".jpg"
            img=cv2.imread(pic) 
            out.write(img)
        out.release()

    def getNeighbours(self,arrayX,arrayY, height, width):
        neighbourX = [(arrayX-1),arrayX,(arrayX+1),(arrayX-1),(arrayX+1),(arrayX-1),arrayX,(arrayX+1)]
        neighbourY = [(arrayY-1),(arrayY-1),(arrayY-1),arrayY,arrayY,(arrayY+1),(arrayY+1),(arrayY+1)]
##        print "neighbourX , neighburY is: ",neighbourX, neighbourY
        finalX = []
        finalY = []
        for i in range(0,len(neighbourX)):
            if(neighbourX[i]>=height or neighbourY[i]>=width or neighbourX[i]<0 or neighbourY[i]<0):
                temp = 0
            else:
                finalX.append(neighbourX[i])
                finalY.append(neighbourY[i])
       
        return np.array(finalX),np.array(finalY)
    

    def findValues(self,neighbourX, neighbourY, width):
        valueArray =  np.zeros(len(neighbourX))
        for i in range(0,len(neighbourX)):
            valueArray[i] = (width* neighbourX[i]) + neighbourY[i]
            
        return valueArray

    def getPixelLocation(self,p, h, w):
        arrayX=p/w
        arrayY=p%w
        nX, nY = self.getNeighbours(arrayX, arrayY, h, w)
        values = self.findValues(nX, nY, w)
##        print "values are: ",values
        randomPixel = int(values[random.randint(0,len(values)-1)])
        return randomPixel
        
    def processVideo(self,bigSampleArray):
        global success
        global frame
        global cap
        
        Finalcount=1
        samples= bigSampleArray

        i=0
        while success:
##            success,frame = cap.read(cv.CV_IMWRITE_JPEG_QUALITY)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            height,width = gray.shape[:2]
            tempArray = np.reshape(gray,(height*width)).T
            segmentationMap = np.copy(tempArray)*0
            for p in range(0,len(bigSampleArray)):
##                print "Value of p is: ",p
                count = index = distance = 0

                while((count < self.requiredMatches) and (index < self.numberOfSamples)):
                    distance = np.linalg.norm(tempArray[p]-samples[p][index])
##                    print "Euclidean distance is: ",distance
                    if (distance < self.distanceThreshold):
                        count += 1
##                        print "count reached" ,count
                    index += 1

                if(count<self.requiredMatches):
                    segmentationMap[p]=255   
                else:
                    segmentationMap[p]=0
                    randomNumber= random.randint(0,self.subsamplingFactor-1)
                    if(randomNumber==0):
                        randomNumber= random.randint(0,self.numberOfSamples-1)
                        samples[p][randomNumber] = tempArray[p]
                    randomNumber = random.randint(0, self.subsamplingFactor-1)
##                    print "Random number detected is: ",randomNumber
                    if(randomNumber==0):
##                        print "Enters randomNumber section"
                        q = self.getPixelLocation(p,height,width)
##                        print "Returned q value is: ",q
                        randomNumber = random.randint(0,self.numberOfSamples-1)
                        samples[q][randomNumber] = tempArray[p]
		#if the `q` key is pressed, break from the loop
		key = cv2.waitKey(1) & 0xFF
            	if key == ord("q"):
                    break
                    
            segmentationMap= np.reshape(segmentationMap,(height,width))
            NewPath="NewFrames/"+ str(i+1) + ".jpg"
            cv2.imwrite(NewPath,segmentationMap)
            thresh = cv2.dilate(segmentationMap, None, iterations=2)
            (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                       cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imwrite("Frames/%d.jpg" % Finalcount, frame)           # save frame as JPEG file
            Finalcount += 1
            success,frame = cap.read(cv.CV_IMWRITE_JPEG_QUALITY)
            i+=1

        cv2.destroyAllWindows()
        self.writeVideo()
        

if __name__ == "__main__": 
    path_file='movie.ogv'
    v = video(path_file)

       
'''

            
##            print len(cnts)
	# loop over the contours
            
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
##		if(len(c) > 0):
##                    m= np.mean(c[0],axis=0)
##                    measuredTrack[count-1,:]=m[0]
##                    plt.plot(m[0,0],m[0,1],'ob')
                
##                cv2.drawContours(frame, c, -1, (0,255,0), 3)
    def framing(self,path):
        global newpath
        cap = cv2.VideoCapture(path)
        success,frame=cap.read(cv.CV_IMWRITE_JPEG_QUALITY)
              
        count = 1;
        
        firstFrame = None
# loop over the frames of the video
        while success:
	# resize the frame, convert it to grayscale, and blur it
##            frame = imutils.resize(frame, width=500)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (51, 51), 0)

##            print newpath, len(os.listdir(newpath))
##            print len([name for name in os.listdir(newpath) if os.path.isfile(name)])

	# if the first frame is None, initialize it
            if firstFrame is None:
                   firstFrame = gray
                   continue

	# compute the absolute difference between the current frame and
	# first frame
            frameDelta = cv2.absdiff(firstFrame, gray)
            
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
            thresh = cv2.dilate(thresh, None, iterations=2)
            (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                       cv2.CHAIN_APPROX_SIMPLE)
##            print len(cnts)
	# loop over the contours
            for c in cnts:
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
##		if(len(c) > 0):
##                    m= np.mean(c[0],axis=0)
##                    measuredTrack[count-1,:]=m[0]
##                    plt.plot(m[0,0],m[0,1],'ob')
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
##                cv2.drawContours(frame, c, -1, (0,255,0), 3)
                 
	# show the frame and record if the user presses a key
            cv2.imshow("Feed", frame)
            cv2.imshow("Thresh", thresh)
##            cv2.imshow("Frame Delta", frameDelta)
            

	# 
                
            cv2.imwrite("Frames/%d.jpg" % count, frame)           # save frame as JPEG file
            count += 1
            success,frame = cap.read(cv.CV_IMWRITE_JPEG_QUALITY)

        array=self.sort_files()
        print array
        cv2.destroyAllWindows()
        cap.release()
        plt.show()
'''


 

