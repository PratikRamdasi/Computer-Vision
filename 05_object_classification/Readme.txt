Objective:
-----------
Identify types of moving objects in a video to track and classify them based on their type. 
We have selected three catagories, 'Person', 'People(group of persons)' and 'Car'.

Methodology:
------------
step 1: Template Generation and Detection
       - applied ViBe foreground subtraction to generate the templates.
       
step 2: Feature Extraction and Classification
       - training database contains images for 'Car', 'People','Chair' and 'Bike' objects.
       - SIFT features are extracted with K-means clustering and SVM classification.

step 3: Testing 
       - Tested algorithm on several input videos for which ground truth is already generated.

Results:
---------
PETS dataset is used for person vs people classification.
Based on Ground Truth alreay available, the average accuracy for pedestrians video 
calculated to be 75%.
