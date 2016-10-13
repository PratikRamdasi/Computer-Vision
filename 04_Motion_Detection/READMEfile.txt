Programming Assignment 2
Object Detection in videos:
3/2/2016
--------------------------------------------

NOTE:
1. The OpenCV version required for this program is 2.4.x. the department machines have different versions, while the machine where the code was developed and tested has OpenCV version 2.4.12.
2. The video format compatible with the Linux system is .ogv and the codec 'theo'.
 

Submitted Files:
--------------------------------------------
Main code: vibe.py
Accuracy measure code: accuracy_measure.py
Videos : movie_cars.ogv 
Folders: groundtruth

Methods applied and improvements:
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
1. Basic foreground-background segmentation and detection of contours using opencv functions.
2. Modification over basic foreground-background segmentation: considering previous samples for estimation of new pixel value (by using mean of previous samples and deciding threshold manually). Results were satisfactory for specific type of video where background is stationary for some initial frames. Ghosts  were the biggest problem.
3. ViBe implementation: ViBe algorithm is implemented and output is compared with previous methods. Most useful advantage as can be seen from the 'changedOutput.ogv' video is elimination of ghosts. Also the accuracy obtained is satisfactory. 

Attached code shows the implementation of ViBe.   

Instructions for running the attached codes:
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
To run the algorithm type:python vibe.py (You might have to change the the path of the video, ).
After running the code, two folders will be generated - 'Frames' (contains output frames after object detection) and 'NewFrames' (contains corresponding binary output frames) in the working directory.Also, output video containing object detection and tracking will be generated named: 'changedOutput.ogv'. 

For accuracy you will have to run another file named accuracy_measure.py.
For running it:
python accuracy_measure.py
Ground truth frames for "movie_cars.ogv" are included in "groundtruth" folder.
This folder is from change detection dataset but we renamed the images in the form "0 (i).png". So, kindly use the attached (renamed) groundtruth folder for verification. 

We have computed accuracy for number of correctly detected objects (Comparing number of contours from ground truth and the obtained results).
Accuracy = No of objects detected correctly / (no of objects detected correctly + no of objects not detected correctly).
Obtained accuracy is: 88% for this specific video.

References:
--------------
1. http://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
2. Reference links for the referred papers:
   http://orbi.ulg.ac.be/bitstream/2268/145853/1/Barnich2011ViBe.pdf
   http://orbi.ulg.ac.be/bitstream/2268/117561/1/VanDroogenbroeck2012Background.pdf
   http://orbi.ulg.ac.be/bitstream/2268/157176/1/VanDroogenbroeck2014ViBe.pdf 
3. OpenCV documentation
4. Lecture Notes.



