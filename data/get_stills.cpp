#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
using namespace cv;


int main (int argc, char ** argv) {

	/*### Step 2: get the filename arguments (input/output) ###*/
	if (argc != 3) {
		cout << "ERROR: enter an infile then an outfile" << endl;
	}

	/*### get the image capture going ###*/
	VideoCapture video(argv[1]);
	if (!video.isOpened()) {
		cout << "ERROR: could not open the video: " << argv[1] << endl;
		return 0;
	}

	/*### create the fist object ###*/
	Mat frame;
	video >> frame;

	imwrite (argv[2], frame);
	
	cout << "---> successfully wrote " << argv[2] << endl;

	return 0;
}