#include "stdafx.h"
#include "face_feature_extract.h"

// Function main
int main(void)
{
	std::string dir = "C:/Users/Austin Pursley/Desktop/ECEN-Senior-Design-Smart-Mirror-Image-Processing/face_feature_extract_(James)/";
	std::string input_dir = dir + "input/";
	std::string output_dir = dir + "output/";


	string face_cascade_name = "C:/Users/Austin Pursley/Documents/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml";
	string eyes_cascade_name = "C:/Users/Austin Pursley/Documents/opencv/build/etc/haarcascades/haarcascade_eye.xml";
	string mouth_cascade_name = "C:/Users/Austin Pursley/Documents/opencv/build/etc/haarcascades/Mouth.xml";
	CascadeClassifier face_cascade;
	CascadeClassifier eyes_cascade;
	CascadeClassifier mouth_cascade;

	// Load the face cascade
	if (!face_cascade.load(face_cascade_name)) {
		printf("--(!)Error loading face cascade\n");
		return (-1);
	}

	// Load the eyes cascade
	if (!eyes_cascade.load(eyes_cascade_name)) {
		printf("--(!)Error loading eyes cascade\n");
		return (-1);
	}

	// Load the mouth cascade
	if (!mouth_cascade.load(mouth_cascade_name)) {
		printf("--(!)Error loading mouth cascade\n");
		return (-1);
	}

	// Read the image file
	std::string img_file = "austin (2).jpg"; //e.g. name.jpg
	std::string img_dir(input_dir.c_str() + img_file);
	Mat frame = imread(img_dir, cv::IMREAD_COLOR);

	// Apply the classifier to the frame
	if (!frame.empty()) {
		detectAndDisplay(frame, face_cascade, eyes_cascade, mouth_cascade);
	}
	else {
		printf(" --(!) No captured frame -- Break!");
		exit(1);
	}

	int c = waitKey(10);

	if (27 == char(c)) {
		exit(1);
	}

	return 0;
}