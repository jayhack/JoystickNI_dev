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
#define REDUCTION_CONST 8

#define GAUSSIAN_BLUR_KERNEL (Size(30,30))
#define GAUSSIAN_BLUR_SIGMA_X 2


int main (int argc, char ** argv) {

	/*### sanitize arguments being passed in ###*/
	if (argc != 2) {
		cout << "Error: enter the name of the file you want to work with. exiting." << endl;	
		return 0;
	}

	/*### set up displays ###*/
	namedWindow ("frame");

	/*### get the image capture going ###*/
	VideoCapture video(argv[1]);
	if (!video.isOpened()) {
		cout << "ERROR: could not open the video: " << argv[1] << endl;
		return 0;
	}



	/*### create the fist object ###*/
	Mat frame;
	video >> frame;
	Fist fist (frame, "data/classifiers/context_svm.xml");


	/*### Start working with frames ###*/
	while (true) {

		/* --- get frame and update ---*/
		video >> frame;
		fist.update (frame);

	}


	return 0;
}