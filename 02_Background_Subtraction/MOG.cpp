/* PROJECT : Person Detection  
   Author: Pratik Ramdasi 
   TITLE:  MOG based background subtraction 
   Date: 07/ 19/ 2016                      
*/

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <stdlib.h>

using namespace cv;
using namespace std;

// Method to perform morphological filtering

Mat filter_image(Mat& img)
{
    int morph_size1 = 2;
    int morph_size2 = 2;
    Mat kernel = cv::getStructuringElement(MORPH_RECT, Size(2*morph_size1 +1, 2*morph_size2 +1));
    Mat filtered;
    dilate(img, filtered, kernel, cv::Point(morph_size1, morph_size2), 2);

    return filtered;
};

// Method to reduce bounding box height 

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
    /*  read input video */
    VideoCapture cap;
    cap.open(0); // input video path
    
    int frame_no = 0;

    /* define Bg model parameters */
    const int nmixtures = 5;
    const bool bShadowDetection = false;
    const int history = 100;
    BackgroundSubtractorMOG bg(history, nmixtures, bShadowDetection);
    
   
    vector < vector < Point > >contours;
    vector < Point > points;
    vector<Vec4i> hierarchy;

    Mat frame, fgmask, fgimg, backgroundImage;
   
    while(1)
    {
	bool bSuccess = cap.read(frame); // read a new frame from video
        if (!bSuccess) // if not success, break loop
        {
            cout << "Cannot read a frame from video file" << endl;
            break;
        }
        
        frame_no ++;     
             // increase the frame count                             
        Size s = frame.size();        // size of the frame
                     
        // Pre process the frame, denoising 
  
        medianBlur(frame, frame, 5);              
        Mat blur_out;
        GaussianBlur(frame, blur_out, Size(5,5),0,0);
        
        
        /*  Strategy: apply learning rate > 0 for first frame to learn the background and for next successive frames,
         apply learning rate = 0. */
         
        // Get intial frame
        
        if (frame_no == 1)
        {
	    const double learningRate = 0.8;
            bg.operator()(blur_out, fgimg, learningRate);
	}
		
	// Process next frames	
	else {
	    const double learningRate = 0;
	    bg.operator()(blur_out, fgimg, learningRate);
	}
		
        bg.getBackgroundImage (backgroundImage);
        
        // filter the segmented image using morphology

       fgimg = filter_image(fgimg);
       
       // define horizontal line parameters  
		             
        Point mid_left, mid_right;
	mid_left.y = s.height/2;
	mid_left.x = 0;
	mid_right.x = s.width;
	mid_right.y = s.height/2;
      
		// find the contours 
                  
        findContours (fgimg, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

        vector<vector<Point> > contours_poly( contours.size() );
        vector<Rect> boundRect( contours.size() );
        
        // Get the moments 
        
        vector<Moments> mu(contours.size() );
        for( size_t i = 0; i < contours.size(); i++ )
           {
             mu[i] = moments( contours[i], false );
           }
      
        // Approx. contours to polygons and get bounding boxes 
        
        for( int i = 0; i < contours.size(); i++ )
         {
             approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
             boundRect[i] = boundingRect( Mat(contours_poly[i]) );
             
         };

        // define threshold values - specific to application video

	int min_area  = 100; // area thresholding for contours, value can be changed    
	int max_height = 100; // maximum height of bounding box 
	int line_thresh = 70; // contours above this horizontal line are ignored 
	    
        for( int i = 0; i< contours.size(); i++ )
          {           
             if (contourArea(contours[i]) > min_area && boundRect[i].y < mid_left.y-line_thresh) {	
                  							              
                 if (boundRect[i].height  >= max_height) {
			boundRect[i] = compressROI(frame, boundRect[i], boundRect[i].height*3/4);
		  }
				 
		  rectangle(frame, boundRect[i].tl(), boundRect[i].br(), Scalar(0,255,0), 2, 8, 0 );
				 
		  }
	   }
		
	// show output frame
		
        imshow ("Frame", frame);
            
        char k = (char)waitKey(30);
        if( k == 27 ) break;

    }

    return 0;
}

