
import cv2
#import cv2.cv as cv

cap = cv2.VideoCapture("v1.mp4")
success,frame=cap.read(0)      #handle of the Video Capture is required for obtaining frame.
count = 1
while success:
    cv2.imwrite("/Users/pratikramdasi/Desktop/frames/%d.jpg" % count, frame)# save frame as JPEG file
    count += 1
    success,frame = cap.read(0)  # to read the last frame


