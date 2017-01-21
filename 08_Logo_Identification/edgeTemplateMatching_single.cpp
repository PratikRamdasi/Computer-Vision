
/* Title : Logo identification in given image
 * Author: Pratik Mohan Ramdasi
 * Date: 1/17/2017
 * Methodology:  Edge and Scale based Template Matching
*/   

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <fstream>

using namespace cv;
using namespace std;

/*  Function to preprocess the image: smoothing and conversion to gray
 *  Inputs: Original test image or template (logo) image
 *  Output: Processed image 
 */
Mat preProcessImage(Mat& img)
{
	Mat gray;
	// Gaussian smoothing
	GaussianBlur( img, img, Size(3, 3), 0, 0);
	// convert image to greyscale
	Mat imgGrey;
	cvtColor(img, gray, CV_BGR2GRAY);
	
	return gray;
}


/*  Function to get gradient (edge) image using Sobel operator
 *  Inputs: Original test image or template (logo) image
 *  Output: Edged image 
 */
Mat getGradientImage(Mat& img)
{
	/// sobel edge operator parameters
	int ddepth = CV_16S;
	int scale = 1;
	int delta = 0;
	Mat abs_grad_x, abs_grad_y, grad_x, grad_y;
	Mat edged;
	
	// Edge detection using SOBEL operator
	Sobel(img, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	// gradient y
	Sobel(img, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	// total gradient
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, edged);
	
	return edged;
}


/*  Function for Scaled Template Matching
 *  Inputs: Original test image, Gradients of test image and template
 *  Output: Bounding box around the logo if found. 
 */
void scaledTemplateMatching(Mat& img, Mat& grad_img, Mat& grad_template)
{
	/// loop over different template image scales
	Mat resized; // for resizing the template to each scale 
	double found_maxVal; // to sore max correlation coefficient value obtained
	Point found_maxLoc;  // location of max correlation point
	double found_ratio;  // Useful to scale the template back to original
	double ccoeffThreshold = 0.10;  // correlation coefficient threshoold
	double initScale;    // starting template scale
	double bestVal = 0.0;
	double ratio;
	
	// resize very small image to template size and adjust initial scale for the template
	if (grad_img.cols < grad_template.cols or grad_img.rows < grad_template.rows) 
	{
		resize(grad_img, grad_img, Size(grad_template.cols, grad_template.rows ));
		initScale = grad_img.cols / (float)grad_template.cols;
	}
	else 
	{
		initScale = 1.4; // decided by trying and testing
	}
	
	// loop through different template scales			
	for (double scale = initScale; scale >= 0.6; scale -= 0.1)
	{
		
		resize( grad_template, resized, Size(grad_template.cols * scale, grad_template.rows ) );
		ratio = resized.cols / (float) grad_template.cols;
			
		// if resized shape is less than image, break from the loop
		if ( resized.rows > grad_img.rows or resized.cols > grad_img.cols )
				continue;
		
		/// template matching	
		// create result matrix
		Mat result;
		int result_cols = grad_img.cols - resized.cols + 1;
		int result_rows = grad_img.rows - resized.rows + 1;
		
		result.create( result_rows, result_cols, CV_32FC1);
		
		int match_method = CV_TM_CCOEFF_NORMED;
			
		// do the matching
		matchTemplate( grad_img, resized, result, match_method );
			
		//localize best match
		double minVal; double maxVal; Point minLoc; Point maxLoc; Point matchLoc;	
		minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
		
		//best matches values
		if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED )
			{ matchLoc = minLoc; }
		else
			{ matchLoc = maxLoc; }

		// find max correlation value
		if ( maxVal > bestVal) { 
			bestVal = maxVal ;
			// store them
			found_maxVal = bestVal;
			found_maxLoc = maxLoc;
			found_ratio = ratio;
		}
	   
	}
	
	/// If maximum value obtained is greater than threshold, logo is identified.
	if (found_maxVal > ccoeffThreshold) 
	{
		
		// find bounding box cooredinates based on ratio
		Point start, end;
		start.x = ((int) found_maxLoc.x );
		start.y = ((int) found_maxLoc.y );	
		
		end.x = ((int) (found_maxLoc.x +  ( grad_template.cols * found_ratio ) ));
		end.y = ((int) (found_maxLoc.y +  ( grad_template.rows * found_ratio ) ));
		
		// display outputs	
		cout << "LOGO detected!" << endl;
		rectangle( img, start, end, Scalar(0,255,0), 2, 8, 0 );
		
		// display input template name on the image
        //putText(img, "Template: Apple", Point(start.x , start.y - 40), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(20,150,20), 2);
			
		imshow("output", img);
		//imwrite("incorrectResult_3.jpg", img);
	}
	else 
	{
		cout << "LOGO is not present!" << endl; 
	} 
	
}


int main()
{
	/// get input image and preprocess it
	Mat img = imread("/home/pratikramdasi/comp_inter/trademarkVision/logo_spotting_problem/test/porsche_4.jpg", 1);
	Mat imgGray;
	imgGray = preProcessImage(img);
	
	/// read template image and preprocess it
	Mat logo;
	logo = imread("/home/pratikramdasi/comp_inter/trademarkVision/logo_spotting_problem/logos/porsche.png", 1);
	Mat logoGray;
	logoGray = preProcessImage(logo);
	
	// resize very small image to template size
	if (img.cols < logo.cols or img.rows < logo.rows) {
		resize(img, img, Size(logo.cols, logo.rows ));
	}
	
	/// Perform edge detection on original and template image for matching
	Mat grad_img, grad_template;
	grad_img = getGradientImage(imgGray);
	grad_template = getGradientImage(logoGray);
	
	imshow("image scene", grad_img);
	imshow("Logo template", grad_template);
	
	/// perform scaled template matching
	scaledTemplateMatching(img, grad_img, grad_template);
	
	waitKey(0);
	
	return 0;
}
