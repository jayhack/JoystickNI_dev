#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

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
/*###################################[ --- CONSTRUCTOR/DESTRUCTOR --- ]###################################################*/
/*########################################################################################################################*/
/* Function: constructor 
 * ---------------------
 * initializes all values 
 */
Fist::Fist () {}
Fist::Fist (Mat frame) {

 	cout << "--- Fist: Constructor ----" << endl;

 	/*--- image sizes ---*/
 	raw_size = Size(frame.cols, frame.rows);
 	reduced_size = Size(raw_size.width/SIZE_REDUCTION_CONST, raw_size.height/SIZE_REDUCTION_CONST);

 	cout << "	- raw size width, height: " << raw_size.width << ", " << raw_size.height << endl;
	cout << "	- reduced size width, height: " << reduced_size.width << ", " << reduced_size.height << endl;;


	/*--- background model initialization ---*/
	background_totals = Mat(reduced_size, CV_32FC3);
	background_model = Mat (reduced_size, CV_8UC3);

 	/*--- fist existence/location ---*/
 	fist_exists = false;
 	num_frames_seen = 0;
 	center = Point (-1, -1);

 }





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



/*########################################################################################################################*/
/*###################################[ --- BACKGROUND MODEL --- ]#########################################################*/
/*########################################################################################################################*/
/* Function: get_abs_diff
 * ----------------------
 * this function will get the abs_diff image current_frame and background_model
 */
void Fist::get_abs_diff () {
	Mat difference;
	absdiff (background_model, current_frame, abs_diff);
	cvtColor (abs_diff, abs_diff, CV_BGR2GRAY);
}


/* Function: get_canny_edges
 * -------------------------
 * gets the edges from the current_frame
 */
void Fist::get_canny_edges () {
	
	/*### blur the image ###*/
	blur (current_frame, edges, Size(3, 3));

	/*### then run canny edge detection on it ###*/
	Canny(current_frame, edges, 50, 150, 3);

}


/* Function: get_edges_diff
 * ------------------------
 * finds the diff between edges in current_frame and in the background 
 * model
 */
 void Fist::get_edges_diff () {
 	edges.convertTo (edges, CV_8UC1);
 	background_edges.convertTo(background_edges, CV_8UC1);

 	Mat background_edges_blurred, edges_blurred;
 	blur (edges, edges_blurred, Size(3, 3));
 	blur (background_edges, background_edges_blurred, Size(3, 3));

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
	blur (background_model, background_edges, Size(3, 3));

	/*### then run canny edge detection on it ###*/
	Canny(background_edges, background_edges, 50, 150, 3);
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


void Fist::update (Mat raw_frame) {

	/*### Step 1: preprocess the frame (downsize, etc) ###*/
	preprocess (raw_frame);

	/*### Step 2: add the frame to the background model if appropriate ###*/
	if (!background_model_set ()) {
		update_background_model ();
	}


	/*### Step 3: otherwisefind the absdiff with the background model ###*/
	else {

		// get_abs_diff ();		//absolute difference between images
		// get_canny_edges ();		//canny edge detection
		// get_edges_diff ();		//edges difference

		// resize (abs_diff, abs_diff, raw_size);
		// imshow ("frame", abs_diff);

		/*### experiment for outside: get edges + contours ###*/
		// get_canny_edges ();
		get_abs_diff ();
		// blur (abs_diff, abs_diff, Size(3, 3));
		/*### then run canny edge detection on it ###*/
		Canny(abs_diff, edges, 50, 150, 3);


		vector <vector <Point> > contours;
		vector<Vec4i> hierarchy;


		RNG rng(12345);

  		findContours(edges, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
  		Mat drawing = Mat::zeros(edges.size(), CV_8UC3);
  		for( int i = 0; i< contours.size(); i++ )
     	{
       		Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       		drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
     	}

     	resize (drawing, drawing, raw_size);
     	imshow ("contours", drawing);

		// resize (edges_abs_diff, edges_abs_diff, raw_size);
		// Mat background_edges_display, edges_display, diff_display;
		// resize (background_edges, background_edges_display, raw_size);
		// resize (edges, edges_display, raw_size);
		// resize (edges_abs_diff, diff_display, raw_size);

		int key = 0;
		while (key != 'q')
			key = waitKey(30);

	}


	num_frames_seen++;
}








/*########################################################################################################################*/
/*###################################[ --- GETTERS/SETTERS --- ]##########################################################*/
/*########################################################################################################################*/
/* Getter: get_center
 * ------------------
 * returns center of fist
 */
Point Fist::get_center () {
	return center;
}