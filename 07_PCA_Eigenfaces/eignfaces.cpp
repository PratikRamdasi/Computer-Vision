/* PROJECT: Generate first 10 priciple componenets of faces.
 * Dataset : BioID face dataset - https://www.bioid.com/About/BioID-Face-Database
 * Author: Pratik Mohan Ramdasi
 * Date: 12/14/2016
 * 
 * Contents: 
 * ==========
 * 1. Principle Component Analysis (PCA) to detect the eigenfaces for given face dataset. 
 * 2. Alignment of the input face images for improved results. First 101 images in the dataset are considered.
*/ 

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

/* Function to compute distance between two points
 */
double Distance(Point p1, Point p2)
{
	double dx = p2.x - p1.x;
	double dy = p2.y - p1.y;
	return sqrt( dx * dx + dy * dy );
}

/* Function to rotate an image along given center and rotation angle
 * Input: original image
 * Output: rotated image
 */
Mat rotate(Mat &image, Point2f center, double angle, double scale)
{
	// get 2x3 rotation matrix
	Mat rot_matrix = getRotationMatrix2D( center, angle, scale );
	Mat rotated_img( Size( image.size().height, image.size().width ), image.type());
	// Perform affine transform
	warpAffine( image, rotated_img, rot_matrix, image.size());
	return rotated_img;
}

/* Function to align the face images based on eye locations
 * Input: original face image with eye positions
 * Output: cropped aligned face image  
 */	
Mat cropFaces(Mat &img, Point e_left = Point(0, 0), Point e_right = Point(0, 0))
{
	//calculate offsets in the original image
	//offset perventage is selected to be 0.2 both horizontally and vertically. Destination image size
	//is selected to be : (100, 100)
	int offset_h = floor(float(0.2 * 100));    
	int offset_v = floor(float(0.2 * 100));	 
	
	//get the direction
	Point eye_direction;
	eye_direction.x = (e_right.x - e_left.x);
	eye_direction.y = (e_right.y - e_left.y);
	
	//calculate rotation angle in radians
	double rotation = atan2(float(eye_direction.y), float(eye_direction.x));
	
	//distance between them
	double dist = Distance(e_left, e_right);
	
	//calculate reference eye width
	double ref = 100 - 2.0 * offset_h;
	
	//scale factor
	double scale  = float(dist)/float(ref);
	//cout << "scale: " << scale << endl;
	
	//rotate image around the left eye
	Mat rotated_img;
	rotated_img = rotate(img, e_left, rotation, scale);
	
	//crop the rotated image
	Rect crop;
	crop.x = e_right.x - scale * offset_h;
	crop.y = e_right.y - scale * offset_v;
	crop.width = 100 * scale;
	crop.height = 100 * scale;
	crop = Rect(crop.x, crop.y, crop.x + crop.width, crop.y + crop.height) & crop;
	Mat cropped;
	cropped = img(crop);
	
	//resize the image 
	resize(cropped, cropped, Size(100, 100));
	
	return cropped;
}

/* function to normalize the image between 0-255
 * Input: original image
 * Output: normalized image
 */
Mat norm_0_255(const Mat &src)
{
	Mat dst;
	switch(src.channels()) {
	case 1:
		normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	
	return dst;
}

/* Function to read eye file 
 * Input:'.eye' filename
 * Output: Extracted left and right eye positions
 */    
vector<double> readEyeCoordinates(const string& filename)
{
	vector<double> co_ordinates;
	ifstream file(filename.c_str());
	string line;
	while(getline(file, line)){
		istringstream ss(line);	
		int value;
		while (ss >> value){
			co_ordinates.push_back(value);
		}
	}

	return co_ordinates;
}

/* Function to read eye files from the directory and store them
 * Input: path to folder containing '.eye' files 
 * Output: vector storing all the '.eye' files 
 */
vector<String> readEyeFiles(const string& folder)
{
	vector<String> files;
	glob(folder, files);
	
	return files;
}

/* Function to read input images from the directory and store them
 * Input: path to folder containing '.pgm' image files 
 * Output: vector storing all the image files 
 */
vector<Mat> readIpImages(const string& folder)
{
	vector<String> files;
	glob(folder, files);
	//Store all the images into vector of images
	vector<Mat> images;
	for (size_t i = 0; i < files.size(); i++){
		//read the image
		Mat image = imread(files[i], 0);
		if(image.empty()){
			cerr << "Could not load image!";
		}
		//store it into the vector of images
		images.push_back(image);
	}
	return images;
}

/* Function to get all aligned images for PCA processing
 * Input: vector of all the input images and '.eye' files 
 * Output: vector storing all the aligned images 
 */
vector<Mat> alignIpImages(vector<Mat> ipImages, vector<String> eyeFiles)
{
	vector<Mat> aligned_images;
	for(size_t i = 0; i < ipImages.size(); i++)
	{
		//read the txt file
		String filename = eyeFiles[i];
		//get left and right eye locations
		vector<double> locs;
		locs = readEyeCoordinates(filename);
		Point eye_left = Point(locs[0], locs[1]);
		Point eye_right = Point(locs[2], locs[3]);
		//get aligned images
		Mat aligned;
		aligned = cropFaces(ipImages[i], eye_left, eye_right);
		
		aligned_images.push_back(aligned);
	}
	return aligned_images;
}


/* Function applying PCA to get eigenvalues and eigenvectors (eigenfaces)
 * Input: aligned images, number of principle components
 * Output: eigenvectors for given number of principle components
 */ 
Mat pcaProcessing(vector<Mat> alignedImages, int num_comps)
{
	//reshape the images to generate dataset for PCA
	Mat dst(static_cast<int>(alignedImages.size()), alignedImages[0].rows * alignedImages[0].cols, CV_32F);
	for (unsigned int i = 0; i < alignedImages.size(); i++){
		Mat image_row = alignedImages[i].clone().reshape(1,1);
		Mat row_i = dst.row(i);
		image_row.convertTo(row_i, CV_32F);
	}
	
	cout << "Size of training set: " << dst.cols << "," << dst.rows << endl;
	
	// copy the dataset 
	Mat data;
	dst.copyTo(data);
	
	//perform PCA
	PCA pca( data, Mat(), CV_PCA_DATA_AS_ROW, num_comps);
	
	//compute and copy PCA results
	Mat mean = pca.mean.clone();
	Mat evals = pca.eigenvalues.clone();
	Mat evecs = pca.eigenvectors.clone();
	
	return evecs;
}
	
int main()
{
	
	clog <<  "Reading input images and eye files ... " << endl;
	
	//read input images in the dataset
	vector<Mat> ipImages;
	string img_folder = "/home/pratikramdasi/comp_inter/Koh-young/dataset/*.pgm";
	ipImages = readIpImages(img_folder);
	
	//read eye locations files
	vector<String> eyeFiles;
	string eye_folder = "/home/pratikramdasi/comp_inter/Koh-young/dataset/*.eye";
	eyeFiles = readEyeFiles(eye_folder);
	
	clog <<  "Aligning input images ... " << endl;
	
	//align the input face images 
	vector<Mat> alignedIp;
	alignedIp = alignIpImages(ipImages, eyeFiles); 

	//get number of principle components 
	int num_comps = 10;
	
	clog << "Getting PCA results ... " << endl;
	
	//PCA processing - display first 10 eigenfaces
	Mat eigenVectors;
	eigenVectors = pcaProcessing(alignedIp, num_comps);
	
	clog << "Displaying PCA results ... " << endl;
	
	//display eigenfaces in a single window
	Mat win_mat(Size(1000, 100), CV_8UC3);
	for(int i = 0; i < num_comps; i++){
		//get ith eigenvector
		Mat ev = eigenVectors.row(i);
		//reshape it to normal size and normallize it to 0-255
		Mat out = norm_0_255(ev.reshape(1, alignedIp[i].rows));
		//apply colormap - jet
		Mat cout;
		applyColorMap(out, cout, COLORMAP_JET);
		cout.copyTo(win_mat(Rect(100 * i, 0, 100, 100)));
	}
	
	imshow("Eigenfaces", win_mat);
	imwrite("Eigenfaces.jpg", win_mat);
	waitKey(0); //press any key to continue...
	
	return 0;
}
