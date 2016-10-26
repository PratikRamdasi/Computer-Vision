Human Activity Recognition in Videos
-------------------------------------

Objective:
----------
Identify types of human activity in a video by classification based on appearance.

Datasets used for training and testing:
---------------------------------------
1. KTH human activity - Boxing, Hand clapping, Running, Walking
2. Weizmann - Bending, One hand waving

Methodology:
------------
Method-1 : HOG feature vecots from n consecutive video frames are analyzed to generate HOGPH (history of HOG features over                past frames). HOGPH feature vectors are used to train the multi-class SVM classifier model for all activities.
           For testing, HOGPH vector is generated for each sample video and SVM used for prediction of the class.

Method-2 : Video Matching using PCA

Results:
--------
Method 1 does not account for motion information and not suitable for large dataset.
Method 2, on the other hand, is easy to implement but time consuming. Method 2 accuracy found out to be around 63%. 
           

