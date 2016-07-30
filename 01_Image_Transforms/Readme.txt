Working with OpenCV:

Main code is " imageTransforms.py ". 
The file contains the functions for Similarity, Affine and Perspective transform.

The code takes three arguments as follows:
<Path to input video.mp4> <inputfile.txt> <Path to output video.mp4>

The input file takes Floating point values.

The extension mp4 is required for input as well as output videos, as the video codec for MacOS is 'avc1'
and 'mp4' format.

The code generates two output videos:
First video (Pathtooutputvideo.mp4) shows the output corresponding to the given transform coordinates in the input file.

Second video (changedOutput.mp4) shows the extra work on the video.
This includes the flipping of the video and applying edge detection to it.
 
An example of the input given to the program will be as:
inputvideo.mp4 inputfile.txt outputvideo.mp4

