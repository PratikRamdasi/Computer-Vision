Readme File

Submission folder contains C++ scripts for generating Eigenfaces
---------------------------------------------------------------------
'eigenfaces.cpp' - Generate first 10 eigenfaces for given BioID face recognition dataset.

System config and external libraries used: 
-------------------------------------------
1. LINUX UBUNTU 14.04 - 4GB RAM
2. OpenCV 2.4.13
3. Geany IDE

To run the scripts follow:
---------------------------
From the folder where scripts are located,
(using terminal)
1. g++ eignfaces.cpp `pkg-config --cflags --libs opencv` -o output
2. ./output
