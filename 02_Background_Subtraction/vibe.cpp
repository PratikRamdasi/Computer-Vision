/* PROJECT : Person Detection 
   Author: Pratik Ramdasi 
   TITLE:  VIBE background subtraction.
   Date: 07/ 14/ 2016                      
*/

#include "opencv2/core/core.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include "opencv2/core/types_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>

using namespace cv;
using namespace std;

// define structure vibe

struct Vibe
{
    int width;                     // width of the image
    int height;                    // height of the image
    int nbSamples;                 // number of samples per pixel
    int reqMatches;                // #_min matches
    int radius;                    // R^2
    int bogo_radius;               // adaptive radius when resizing /initializing the samples ( my addition ;] )
    int subsamplingFactor;         // amount of random subsampling

    vector< Mat_<uchar> > samples;   // the 'model'
    Mat_<uchar> segmentation;      // 0:bg , 255:fg

    RNG rng;

	// select the parameter values
	
    Vibe (int w, int h, int nbSamples=20, int reqMatches=2, int radius=400, int subsamplingFactor=8)
        : width(w)
        , height(h)
        , nbSamples(nbSamples)
        , reqMatches(reqMatches)
        , radius(radius) // R^2
        , bogo_radius(200000)
        , subsamplingFactor(subsamplingFactor)
        , rng(getTickCount())
        , segmentation(height,width)
    {
        clear();
    };

    void clear()
    {
        samples.clear();
        for ( int i=0; i<nbSamples; i++ )
            samples.push_back( Mat_<uchar>(height,width,128) );
        bogo_radius= 200000;
    };

	// VIBE segmentation
	
    void segment(const Mat & img, Mat & segmentationMap)
    {
        if ( nbSamples != samples.size() )
            clear();

        bogo_radius = bogo_radius > radius
                    ? bogo_radius *= 0.8
                    : radius;

        Mat_<uchar> image(img);
        for (int x=1; x<width-1; x++) // spare a 1 pixel border for the neighbour sampling
        {
            for (int y=1; y<height-1; y++)
            {
                uchar pixel = image(y,x);

                // comparison with the model
                int count = 0;
                for ( int i=0; (i<nbSamples)&&(count<reqMatches); i++ )
                {
                    int distance = pixel - samples[i](y,x);
                    count += (distance*distance < bogo_radius);
                }
                // pixel classification according to reqMatches
                if (count >= reqMatches) // the pixel belongs to the background
                {
                    // store 'bg' in the segmentation map
                    segmentation(y,x) = 0;
                    // gets a random number between 0 and subsamplingFactor-1
                    int randomNumber = rng.uniform(0, subsamplingFactor);
                    // update of the current pixel model
                    if (randomNumber == 0) // random subsampling
                    {
                        // other random values are ignored
                        randomNumber = rng.uniform(0, nbSamples);
                        samples[randomNumber](y,x) = pixel;
                    }
                    // update of a neighboring pixel model
                    randomNumber = rng.uniform(0, subsamplingFactor);
                    if (randomNumber == 0) // random subsampling
                    {
                        // chooses a neighboring pixel randomly
                        const static int nb[8][2] = {-1,0, -1,1, 0,1, 1,1, 1,0, 1,-1, 0,-1, -1,-1};
                        int n = rng.uniform(0,8);
                        int neighborX = x + nb[n][1], neighborY = y + nb[n][0];
                        // chooses the value to be replaced randomly
                        randomNumber = rng.uniform(0, nbSamples);
                        samples[randomNumber](neighborY,neighborX) = pixel;
                    }
                }
                else // the pixel belongs to the foreground
                {    // store 'fg' in the segmentation map
                    segmentation(y,x) = 255;
                }
            }
        }
        segmentationMap = segmentation;
    }
};

Rect compressROI(Mat frm, Rect boundingBox, int padding) {
    Rect returnRect = Rect(boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height - padding);
    if (returnRect.x < 0)returnRect.x = 0;
    if (returnRect.y < 0)returnRect.y = 0;
    if (returnRect.x+returnRect.width >= frm.cols)returnRect.width = frm.cols-returnRect.x;
    if (returnRect.y+returnRect.height >= frm.rows)returnRect.height = frm.rows-returnRect.y;
    return returnRect;
};
	
int main()
{
    // get the input video
    VideoCapture cap;
    cap.open(0); // input video path
    if ( !cap.isOpened() )
        return -1;

	// input parameters
    int w = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT);  
    int ct = 0;

    stringstream ss;
    string folderName = "cropped";
    string folderCreateCommand = "mkdir " + folderName;
    system(folderCreateCommand.c_str());
					
    vector < vector < Point > >contours;
    vector < Point > points;
    vector<Vec4i> hierarchy;
		
    Vibe vibe(w,h);
    
    while(1)
    {
		// read the input frame
        Mat frame;
        if ( !cap.read(frame) ) continue;
		
		// convert to Gray image for segmentation
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        
        // size of the frame
        Size s = gray.size();
        
        // get segmented image
        
        Mat seg;
        vibe.segment(gray,seg);
        
        // removal of noise by median filtering   
             
        medianBlur(seg, seg, 5);
        
		// morphology
		
        dilate(seg, seg, Mat(10,5,CV_8U));
        
		// define horizontal line parameters  
		             
        Point mid_left, mid_right;
		mid_left.y = s.height/2;
		mid_left.x = 0;
		mid_right.x = s.width;
		mid_right.y = s.height/2;
        
        // find the contours
        
        findContours (seg, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		
        vector<vector<Point> > contours_poly( contours.size() );
        vector<Rect> boundRect( contours.size() );
        
        // get the moments 
        
        vector<Moments> mu(contours.size() );
        for( size_t i = 0; i < contours.size(); i++ )
           {
             mu[i] = moments( contours[i], false );
           }
	     	
        // define bounding rectangle object
        
        for( int i = 0; i < contours.size(); i++ )
                 {
                     approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );                   
                     boundRect[i] = boundingRect( Mat(contours_poly[i]) );
                 }
							     

		// define threshold values - specific to application video

	    int min_area  = 100; // area thresholding for contours, value can be changed    
	    int max_height = 100; // maximum height of the contour
		int line_thresh = 10; // contours above this line are ignored
	    
        for( int i = 0; i< contours.size(); i++ )
          {           
             if (contourArea(contours[i]) > min_area && boundRect[i].y < mid_left.y-line_thresh) {	
                  							              
                 if (boundRect[i].height  >= max_height) {
					 
					 boundRect[i] = compressROI(frame, boundRect[i], boundRect[i].height*3/4);
				 }
				 
				 // ROI 				  
				 Rect R = boundRect[i]; 
			     Mat ROI = frame(R);
			     
			     ss << folderName <<"/"<< "cropped_" << (ct + 1) << ".jpg";
				 string fullPath = ss.str();
				 ss.str("");
				 imwrite(fullPath, ROI);
				 ct += 1;
				 
				 rectangle(frame, boundRect[i].tl(), boundRect[i].br(), Scalar(0,255,0), 2, 8, 0 );
				 
				}
			}
			
		
			
		// show output frame
		
        imshow("vibe",frame);
            
        int k = waitKey(10);
        if ( k == ' ' ) vibe.clear();
        if ( k == 27  ) break;
    }

    return 0;
}
