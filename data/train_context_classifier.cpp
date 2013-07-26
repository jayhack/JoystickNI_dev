/* File: context_classifier.cpp
 * ----------------------------
 * for training a classifier that will discern what context the phone is currently in, based on
 * its background model.
 */

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"

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
void get_training_data (Mat& training_data, Mat& training_labels, char* training_examples_filename, char* training_labels_filename) {

	/*### Step 1: fill vectors with lists of all positive filenames, negative filenames ###*/
	vector <string> file_names;
	vector <int> file_labels;
	ifstream infile_examples (training_examples_filename);
	ifstream infile_labels (training_labels_filename);

	string line;
	while (getline (infile_examples, line))
		file_names.push_back (line);

	while (getline (infile_labels, line))
		file_labels.push_back (atoi(line.c_str()));

	if (file_names.size () != file_labels.size ())  {
		cout << "ERROR: not the same number of labels as there are examples. exiting" << endl;
	}

	/*### Step 2: for each positive example, load it in and add it to the matrix 'training data' ###*/
	for (int i=0;i<file_names.size();i++) {

		/*### Step 3: read in the image, resize it ###*/
		Mat image = imread(file_names[i], CV_LOAD_IMAGE_COLOR);
		cout << file_names[i] << endl;
		if (!image.data) cout << "ERROR: failed to load " << file_names[i] << endl;
		resize(image, image, REDUCED_SIZE);

		/*### Step 4: get its histogram ###*/
		Mat histogram_representation = get_histogram_representation (image);

		/*### Step 5: add data to training data ###*/
		if (i == 0) training_data = histogram_representation.clone ();
		else vconcat(training_data, histogram_representation, training_data);

		/*### Step 6: add a positive to training_labels ###*/
		int current_label = file_labels[i];
		if (i == 0) {
			if (current_label == 0) 	training_labels = Mat::zeros(1, 1, CV_32FC1);
			else 						training_labels = Mat::ones(1, 1, CV_32FC1);
		}
		else {
			Mat concat_mat;
			if (current_label == 0)		concat_mat = Mat::zeros(1, 1, CV_32FC1);
			else 						concat_mat = Mat::ones(1, 1, CV_32FC1);
			vconcat(training_labels, concat_mat, training_labels);
		}
	}
}


int main (int argc, char ** argv) {


	/*### Step 1: get training_data, labels ###*/
	Mat training_data, training_labels;	
	get_training_data (training_data, training_labels, "training_examples.txt", "training_labels.txt");


	/*### Step 2: set up SVM params ###*/
    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);


    /*### Step 3: create/train the SVM ###*/
    CvSVM SVM;
    SVM.train(training_data, training_labels, Mat(), Mat(), params);


    /*### Step 4: test it!!! ###*/
    /* 1 = outside, 0 = inside */
	// Mat test_example = imread ("../data/inside_stills/4.jpg", CV_LOAD_IMAGE_COLOR);
	// if (!test_example.data) {
		// cout << "Error: could not load the image" << endl;
		// return 0;
	// }
	// resize(test_example, test_example, REDUCED_SIZE);
	// Mat features = get_histogram_representation (test_example);
	// cout << "Prediction: " << SVM.predict (features) << endl;

    cout << "--- success! ---" << endl;
    cout << "---> Status: saving..." << endl;
    SVM.save ("classifiers/context_svm.xml");

    return 0;
}




