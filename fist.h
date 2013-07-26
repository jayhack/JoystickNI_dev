#ifndef _FIST_H
#define _FIST_H

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace std;
using namespace cv;

#define ORIGINAL_HEIGHT 640
#define ORIGINAL_WIDTH 480
#define REDUCTION_CONST 8

#define GAUSSIAN_BLUR_KERNEL (Size(30,30))
#define GAUSSIAN_BLUR_SIGMA_X 2

class Fist {
private:
	
	bool 	fist_exists;		/*indicator for existence of fist */
	Point 	center;				/*center of fist*/
	int 	num_frames_seen;	/*total number of frames encountered */
	Size	raw_size;			/*size of raw frames*/
	Size 	reduced_size;		/*size of frames we deal with*/

	Mat 	current_frame;
	Mat 	background_totals;
	Mat		background_model;

	/*--- edges ---*/
	Mat 	background_canny_edges;
	Mat 	canny_edges;
	/*--- diff w/ background model ---*/	
	Mat 	abs_diff;
	Mat		edges_abs_diff;


	/*--- Context Classification ---*/
	CvSVM *context_svm;
	bool is_outside;


	/*--- Utilities ---*/
	Point convert_to_raw_coords (Point reduced_coords);
	bool background_model_set ();


	/*--- Background Model ---*/
	void get_abs_diff ();
	void get_canny_edges ();
	void get_edges_diff ();
	void update_background_model ();


public:

	/*--- constructors/destructors ---*/
	Fist ();
	Fist (Mat frame, char* svm_location);


	/*--- updating the fist ---*/
	void preprocess (Mat raw_frame);
	void update_inside ();
	void update_outside ();
	void update (Mat raw_frame);

	/*--- getters/setters ---*/
	Point get_center ();
	bool fist ();
	bool outside ();



};
#endif 
