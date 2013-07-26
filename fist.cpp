#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fist.h"

using namespace std;
using namespace cv;


#define ORIGINAL_HEIGHT 640
#define ORIGINAL_WIDTH 480
#define SIZE_REDUCTION_CONST 8

#define GAUSSIAN_BLUR_KERNEL (Size(30,30))
#define GAUSSIAN_BLUR_SIGMA_X 2

#define NUM_FRAMES_IN_BG_MODEL 20



/*########################################################################################################################*/
/*###################################[ --- UTILITIES --- ]################################################################*/
/*########################################################################################################################*/
/* Function: convert_to_raw_coords 
 * -------------------------------
 * given coordinates on the reduced image, this will return coordinates on
 * the raw image.
 */
Point Fist::convert_to_raw_coords (Point reduced_coords) {

	return Point (0, 0);
}

/* Function: background_model_set
 * ------------------------------
 * returns true if the background model is complete, false otherwise
 */
bool Fist::background_model_set () {
	if (num_frames_seen <= NUM_FRAMES_IN_BG_MODEL) return false;
	else return true;
}

/* Function: get_histogram_representation
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




/*########################################################################################################################*/
/*###################################[ --- CONSTRUCTOR/DESTRUCTOR --- ]###################################################*/
/*########################################################################################################################*/
/* Function: constructor 
 * ---------------------
 * initializes all values 
 */
Fist::Fist () {}
Fist::Fist (Mat frame, char* svm_location) {

 	cout << "--- Fist: Constructor ----" << endl;

 	/*--- load context_svm ---*/
 	context_svm = new CvSVM;
 	context_svm->load (svm_location);

 	/*--- determine image dimensions ---*/
 	raw_size = Size(frame.cols, frame.rows);
 	reduced_size = Size(raw_size.width/SIZE_REDUCTION_CONST, raw_size.height/SIZE_REDUCTION_CONST);
 	cout << "	- raw size width, height: " << raw_size.width << ", " << raw_size.height << endl;
	cout << "	- reduced size width, height: " << reduced_size.width << ", " << reduced_size.height << endl;

	/*--- determine our current context ---*/
	preprocess (frame); //current_frame now has the resized version
	Mat histogram_rep = get_histogram_representation(current_frame);
	if (context_svm->predict (histogram_rep) == 1) is_outside = true;
	else is_outside = false;
	cout << "	- is_outside = " << is_outside << endl;




	/*--- background model initialization ---*/
	background_totals = Mat(reduced_size, CV_32FC3);
	background_model = Mat (reduced_size, CV_8UC3);


 	/*--- fist existence/location ---*/
 	fist_exists = false;
 	num_frames_seen = 0;
 	center = Point (-1, -1);


 }




/*########################################################################################################################*/
/*###################################[ --- BACKGROUND MODEL --- ]#########################################################*/
/*########################################################################################################################*/
/* Function: get_abs_diff
 * ----------------------
 * gets absolute difference between background_model and current frame
 * results ---> abs_diff
 */
void Fist::get_abs_diff () {
	absdiff (background_model, current_frame, abs_diff);
	cvtColor (abs_diff, abs_diff, CV_BGR2GRAY);
}


/* Function: get_canny_edges
 * -------------------------
 * gets the edges from the current_frame
 * results ---> canny_edges
 */
void Fist::get_canny_edges () {
	
	/*### blur the image ###*/
	blur (current_frame, canny_edges, Size(3, 3));

	/*### then run canny edge detection on it ###*/
	Canny(current_frame, canny_edges, 50, 150, 3);
}


/* Function: get_edges_diff
 * ------------------------
 * finds the diff between edges in current_frame and in the background model
 * results ---> edges_abs_diff
 */
void Fist::get_edges_diff () {
 	canny_edges.convertTo (canny_edges, CV_8UC1);
 	background_canny_edges.convertTo(background_canny_edges, CV_8UC1);

 	Mat background_edges_blurred, edges_blurred;
 	blur (canny_edges, edges_blurred, Size(3, 3));
 	blur (background_canny_edges, background_edges_blurred, Size(3, 3));

 	absdiff (edges_blurred, background_edges_blurred, edges_abs_diff);
}
	


/* Function: update_background_model 
 * ---------------------------------
 * will update the background model with the current_frame
 */
void Fist::update_background_model () {

	/*### convert frame to the correct depth ###*/
	Mat deep_frame; 
    current_frame.convertTo(deep_frame, CV_32FC3);

	/*### add it to the totals ###*/
	add (deep_frame, background_totals, background_totals);

	/*### scale the background_totals appropriately and add to num_frames_seen ###*/
	background_totals.convertTo (background_model, CV_8UC3, 1/((float)num_frames_seen));

	/*### blur the image ###*/
	blur (background_model, background_canny_edges, Size(3, 3));

	/*### then run canny edge detection on it ###*/
	Canny(background_canny_edges, background_canny_edges, 50, 150, 3);
}




/*########################################################################################################################*/
/*###################################[ --- UPDATING --- ]#################################################################*/
/*########################################################################################################################*/
/* Function: preprocess
 * --------------------
 * this function will take care of all preprocessing of the frame, including downsampling, etc.
 * stores the resulting, 'preprocesed' frame in current_frame;
 */
void Fist::preprocess (Mat raw_frame) {
	resize (raw_frame, current_frame, reduced_size);
}



/* Function: udpdate_outside
 * -------------------------
 * update for outside context
 * might have to train for fist color when outside... shouldn't be too hard though?
 * can't you just calculate p(in fist) for each pixel in the image here, or something like it?
 * you aren't dealing with as much outside shit as you are otherwise
 */
void Fist::update_outside () {

	Mat hsv_mat, skin_mat;
	cvtColor(current_frame, hsv_mat, COLOR_BGR2HSV);
    inRange(hsv_mat, Scalar(0, 10, 60), Scalar(20, 150, 255), skin_mat);    //these parameters work MUCH better than (0, 10, 60) and (20, 150, 255)

    resize (skin_mat, skin_mat, raw_size);
    imshow ("Skin", skin_mat);
    int key = 0;
    while (key != 'q')
    	key = waitKey(30);

}

/* Function: update_inside
 * -----------------------
 * update for inside context
 */
void Fist::update_inside () {

	get_abs_diff ();		//absolute difference between images
	get_canny_edges ();		//canny edge detection
	get_edges_diff ();		//edges difference

	resize (edges_abs_diff, edges_abs_diff, raw_size);
	resize (abs_diff, abs_diff, raw_size);
	imshow ("edges_diff", edges_abs_diff);
	imshow ("abs_diff", abs_diff);

	int key = 0;
	while (key != 'q') 
		key = waitKey (30);
}


/* Function: update
 * ----------------
 * preprocesses, updates background model (if appropriate), then otherwise 
 * calls functions to update appropriately given the context
 */
void Fist::update (Mat raw_frame) {

	/*### Step 1: preprocess the frame (downsize, etc) ###*/
	preprocess (raw_frame);

	/*### Step 2: add the frame to the background model if appropriate, otherwise update ###*/
	if (!background_model_set ()) {
		update_background_model ();
	}
	else {
		if (outside ()) update_outside ();
		else update_inside ();
	}

	/*### Step 3: update the number of frames seen ###*/
	num_frames_seen++;
}








/*########################################################################################################################*/
/*###################################[ --- GETTERS/SETTERS/INDICATORS --- ]###############################################*/
/*########################################################################################################################*/
/* Getter: get_center
 * ------------------
 * returns center of fist
 */
Point Fist::get_center () {
	return center;
}


/* Indicator: fist_exists
 * ----------------------
 * returns wether the fist is detected on the screen
 */
bool Fist::fist () {
	return fist_exists;
}

/* Function: outside
 * -----------------
 * returns wether the scene is outside or not
 */
bool Fist::outside () {
	return is_outside;
}



