/* File: context_classifier.cpp
 * ----------------------------
 * for training a classifier that will discern what context the phone is currently in, based on
 * its background model.
 */

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace cv;
using namespace std;

#define ORIGINAL_HEIGHT 640
#define ORIGINAL_WIDTH 480
#define REDUCTION_CONST 8

#define REDUCED_SIZE (Size(40, 30))
#define EXAMPLE_SIZE (40*30)


/* Function: get_histograms
 * ------------------------
 * given an image of dimensions 40*30, this will return a vector of mats that are its histogram
 */
 Mat get_histogram_representation (Mat image) {

 	/*### Step 1: Split the image into b/g/r planes ###*/
	vector<Mat> bgr_planes;
	split (image, bgr_planes);
	

	/*### Step 2: calculate the histograms ###*/
  	int histSize = 256; 					//number of bins in histogram
	float range[] = { 0, 256 }; 			//ranges for b,g,r
	const float* histRange = { range };		

	bool uniform = true; 					//it is uniform
	bool accumulate = false;				//it does not accumulate
	Mat b_hist, g_hist, r_hist;

	calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

	/*### join all 3 channels into a long row vector... ###*/
	Mat total_histogram;
	transpose (b_hist, b_hist);
	transpose (g_hist, g_hist);
	transpose (r_hist, r_hist);
	hconcat (b_hist, g_hist, total_histogram);
	hconcat (total_histogram, r_hist, total_histogram);

	return total_histogram;


 }

/* Function: get_training_data
 * ---------------------------
 * given a filename of positive examples and a filename of negative example locations,
 * this function will fill in the mats training_data and training_labels
 */
void get_training_data (Mat& training_data, Mat& training_labels, char* positive_filenames, char* negative_filenames) {


	/*### Step 1: fill vectors with lists of all positive filenames, negative filenames ###*/
	vector <string> positive_examples;
	vector <string> negative_examples;
	ifstream positive_infile (positive_filenames);
	ifstream negative_infile (negative_filenames);
	string filename;
	while (getline (positive_infile, filename)) {
		positive_examples.push_back (filename);
	}
	while (getline (negative_infile, filename)) {
		negative_examples.push_back (filename);
	}

	/*### Step 2: for each positive example, load it in and add it to the matrix 'training data' ###*/
	for (int i=0;i<positive_examples.size();i++) {

		/*### Step 3: read in the image, resize it ###*/
		Mat image = imread(positive_examples[i], CV_LOAD_IMAGE_COLOR);
		resize(image, image, REDUCED_SIZE);

		/*### Step 4: get its histogram ###*/
		Mat histogram_representation = get_histogram_representation (image);


		/*### Step 5: add data to training data ###*/
		if (i == 0) {
			training_data = histogram_representation.clone ();
		}
		else {
			vconcat(training_data, histogram_representation, training_data);
		}

	}

	cout << "training data rows, cols = " << training_data.rows << ", " << training_data.cols << endl;

}


int main (int argc, char ** argv) {

	/*### Step 1: Get the image ###*/
	Mat image = imread ("data/inside.jpg");
	if (!image.data) {
		cout << "Error: could not load the image" << endl;
	}



	/*### get training_data, labels ###*/
	Mat training_data, training_labels;	
	get_training_data (training_data, training_labels, "positive_examples.txt", "negative_examples.txt");
	return 0;


	/*### Step 2: Split the image into b/g/r planes ###*/
	vector<Mat> bgr_planes;
	split (image, bgr_planes);
	


	/*### Step 3: calculate the actual histograms ###*/
  	int histSize = 256; 					//number of bins in histogram
	float range[] = { 0, 256 }; 			//ranges for b,g,r
	const float* histRange = { range };		

	bool uniform = true; 					//it is uniform
	bool accumulate = false;				//it does not accumulate
	Mat b_hist, g_hist, r_hist;

	calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );



	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
  	int bin_w = cvRound( (double) hist_w/histSize );

  	Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

  	/// Normalize the result to [ 0, histImage.rows ]
  	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

  	/// Draw for each channel
  	for( int i = 1; i < histSize; i++ )
  	{
		line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
                         Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                       	 Scalar( 255, 0, 0), 2, 8, 0  );
      	line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
                       	 Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                       	 Scalar( 0, 255, 0), 2, 8, 0  );
      	line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
                       	 Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
                       	 Scalar( 0, 0, 255), 2, 8, 0  );
  	}	

  	/// Display
  	namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
  	imshow("calcHist Demo", histImage );

  	waitKey(0);

}




