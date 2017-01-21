Objective:
----------
For each probe image in probe.txt, write a script that can:
(1) identify the brand logo in the probe image (if any), 
(2) draw the bounding box around the recognized logo (if any),
And finally, the script should tally the number of correct vs incorrect matches and summarize the accuracy.

Method Implemented: 
============================================

Edge and Scale Based Template Matching
---------------------------------------
Since the most distinguishing part between the logos is their shape, edge based template matching proves to be more useful than other methods mentioned below.

Steps:
------ 
1. Convert both logo template and image containing the same logo into edge images using sobel operator.
2. Since the image may contain scaled logo template, vary the scale of the logo and match each scaled template to the image to get correlation coefficient value.
3. Return the scaled template for which correlation coefficient value is maximum.
4. Draw the bounding box around the maximum match location.

Scripts:
------------------------------
"edgeTemplateMatching_single.cpp"

-> Script for single test image with certain logo (decided visually) vs same logo template
-> out of 133 test images with logos, 80 identified with correct logo and bounding box around it.
-> I renamed the test images by  - "(logoname).png". There are average 4 images for each logo.
 
-> Accuracy: (80 / 133) * 100 = 60.15%

Advantages:
-----------
-> Template Scaling proves to better matching strategy than using single sized template.
-> Easier to implement.
-> Works well for single test image containing specific logo with same logo template.

Disadvantages:
--------------
-> Threshold values for correlation coefficient need to be adjusted by trial and error.
-> Very unreliable for matching test image having certain logo with different logo template.
-> Very difficult to check if the test image has any logo or none.
-> Not suitable to work with million images containing logos due to time constraints. Computation time will increase with number of test images or logo images.
-> It does not handle multiple occurences of the same logo in test image.

