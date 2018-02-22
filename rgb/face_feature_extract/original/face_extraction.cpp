#include "stdafx.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

// Function Headers
void detectAndDisplay(Mat frame);

// Global variables
// Copy this file from opencv/data/haarscascades to target folder
string face_cascade_name = "c:/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml";
string eyes_cascade_name = "c:/opencv/build/etc/haarcascades/haarcascade_eye.xml";
string mouth_cascade_name = "c:/opencv/build/etc/haarcascades/Mouth.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier mouth_cascade;
string window_name = "Capture - Face detection";
int filenumber; // Number of file to be saved
string filename;

// Function main
int main(void)
{
	// Load the face cascade
	if (!face_cascade.load(face_cascade_name)) {
		printf("--(!)Error loading\n");
		return (-1);
	}

	// Load the eyes cascade
	if (!eyes_cascade.load(eyes_cascade_name)) {
		printf("--(!)Error loading\n");
		return (-1);
	}

	// Load the mouth cascade
	if (!mouth_cascade.load(mouth_cascade_name)) {
		printf("--(!)Error loading\n");
		return (-1);
	}

	// Read the image file
	Mat frame = imread("austin.jpg");

	// Apply the classifier to the frame
	if (!frame.empty()) {
		detectAndDisplay(frame);
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

// Function detectAndDisplay
void detectAndDisplay(Mat frame)
{
	std::vector<Rect> faces; //all detected faces
	std::vector<Rect> eyes; //all detected eyes
	std::vector<Rect> mouths; //all detected mouths
	Mat frame_gray; //the original image, converted to grayscale
	Mat crop; //cropped face, color
	Mat crop_eyes; //cropped eyes, color
	Mat crop_eyes2;
	Mat crop_mouth;
	Mat res;
	Mat gray; //cropped face, grayscale
	string text;
	stringstream sstm;

	//convert the image to grayscale
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	// Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	// Set Region of Interest
	cv::Rect roi_b;
	cv::Rect roi_c;

	size_t ic = 0; // ic is index of current element
	int ac = 0; // ac is area of current element

	size_t ib = 0; // ib is index of biggest element
	int ab = 0; // ab is area of biggest element


	for (ic = 0; ic < faces.size(); ic++) // Iterate through all current elements (detected faces)

	{
		roi_c.x = faces[ic].x;
		roi_c.y = faces[ic].y;
		roi_c.width = (faces[ic].width);
		roi_c.height = (faces[ic].height);

		ac = roi_c.width * roi_c.height; // Get the area of current element (detected face)

		roi_b.x = faces[ib].x;
		roi_b.y = faces[ib].y;
		roi_b.width = (faces[ib].width);
		roi_b.height = (faces[ib].height);

		ab = roi_b.width * roi_b.height; // Get the area of biggest element, at beginning it is same as "current" element

		if (ac > ab)
		{
			ib = ic;
			roi_b.x = faces[ib].x;
			roi_b.y = faces[ib].y;
			roi_b.width = (faces[ib].width);
			roi_b.height = (faces[ib].height);
		}

		crop = frame(roi_b);
		resize(crop, res, Size(128, 128), 0, 0, INTER_LINEAR); // This will be needed later while saving images
		cvtColor(crop, gray, CV_BGR2GRAY); // Convert cropped image to Grayscale

										   // Form a filename
		
		//filename = "";
		//stringstream ssfn;
		//ssfn << filenumber << ".png";
		//filename = ssfn.str();
		//filenumber++;

		imwrite("face.png", crop);

		Point pt1(faces[ic].x, faces[ic].y); // Display detected faces on main window - live stream from camera
		Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y + faces[ic].width));
		rectangle(frame, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
	}

	// Show image
	sstm << "Crop area size: " << roi_b.width << "x" << roi_b.height << " Filename: " << filename;
	text = sstm.str();

	putText(frame, text, cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
	imshow("original", frame);

	if (!crop.empty())
	{
		imshow("detected", crop);
	}
	else
		destroyWindow("detected");

	//extract eyes
	//TODO: extract both eyes instead of one
	//to do this, instead of extracting the biggest element, extract the two biggest elements
	//update 12/5/7 (insight): extract the biggest element e1, then run the same algorithm again to extract e2, with the condition
	//that e2 != e1
	eyes_cascade.detectMultiScale(gray, eyes, 1.05, 4, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	// Set Region of Interest
	cv::Rect roi_b1; //biggest element
	cv::Rect roi_b2; //second biggest element
	cv::Rect roi_c1; //current element

	size_t ic1 = 0; // ic is index of current element
	int ac1 = 0; // ac is area of current element

	size_t ib1 = 0; // ib is index of biggest element
	int ab1 = 0; // ab is area of biggest element

	size_t ib2 = 0;
	int ab2 = 0;

	int pic = 0;

	//EYE DETECTION
	for (ic1 = 0; ic1 < eyes.size(); ic1++) // Iterate through all current elements (detected eyes)

	{
		roi_c1.x = eyes[ic1].x;
		roi_c1.y = eyes[ic1].y;
		roi_c1.width = (eyes[ic1].width);
		roi_c1.height = (eyes[ic1].height);

		ac1 = roi_c1.width * roi_c1.height; // Get the area of current element (detected eye)

		roi_b1.x = eyes[ib1].x;
		roi_b1.y = eyes[ib1].y;
		roi_b1.width = (eyes[ib1].width);
		roi_b1.height = (eyes[ib1].height);

		ab1 = roi_b1.width * roi_b1.height; // Get the area of biggest element, at beginning it is same as "current" element

											//roi_b2.x = eyes[ib2].x;
											//roi_b2.y = eyes[ib2].y;
											//roi_b2.width = (eyes[ib2].width);
											//roi_b2.height = (eyes[ib2].height);

											//ab2 = roi_b2.width * roi_b2.height; // Get the area of second biggest element, at beginning it is same as "current" element



		if (ac1 > ab1)
		{
			ib1 = ic1;
			roi_b1.x = eyes[ib1].x;
			roi_b1.y = eyes[ib1].y;
			roi_b1.width = (eyes[ib1].width);
			roi_b1.height = (eyes[ib1].height);
		}
	}
		//if current element is bigger than 2nd biggest element, and the current element is NOT equal to the biggest element
		//2nd biggest element = current element
		//if ((ac1 > ab2) && (ac1 != ab1))
		//{
//
		//}

		//DIFFERENT!
		crop_eyes = crop(roi_c1);
		resize(crop_eyes, res, Size(128, 128), 0, 0, INTER_LINEAR); // This will be needed later while saving images
							
										//save extracted eyes

		imwrite(std::to_string(pic) + ".png", crop_eyes);
		imshow(std::to_string(pic), crop_eyes);
		pic = pic++;

		//produce image with eyes cropped out

		//TODO: detect and extract mouth
		mouth_cascade.detectMultiScale(gray, mouths, 1.05, 8, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

		// Set Region of Interest
		cv::Rect roi_b_m; //biggest element
		cv::Rect roi_c_m; //current element

		size_t ic_m = 0; // ic is index of current element
		int ac_m = 0; // ac is area of current element

		size_t ib_m = 0; // ib is index of biggest element
		int ab_m = 0; // ab is area of biggest element

		for (ic_m = 0; ic_m < mouths.size(); ic_m++) // Iterate through all current elements (detected mouths)

		{
			roi_c_m.x = mouths[ic_m].x;
			roi_c_m.y = mouths[ic_m].y;
			roi_c_m.width = (mouths[ic_m].width);
			roi_c_m.height = (mouths[ic_m].height);

			ac_m = roi_c_m.width * roi_c_m.height; // Get the area of current element (detected mouth)

			roi_b_m.x = mouths[ib_m].x;
			roi_b_m.y = mouths[ib_m].y;
			roi_b_m.width = (mouths[ib_m].width);
			roi_b_m.height = (mouths[ib_m].height);

			ab_m = roi_b_m.width * roi_b_m.height; // Get the area of biggest element, at beginning it is same as "current" element

			if (ac_m > ab_m)
			{
				ib_m = ic_m;
				roi_b_m.x = mouths[ib_m].x;
				roi_b_m.y = mouths[ib_m].y;
				roi_b_m.width = (mouths[ib_m].width);
				roi_b_m.height = (mouths[ib_m].height);
			}
		}
			crop_mouth = crop(roi_c_m);
			resize(crop_mouth, res, Size(128, 128), 0, 0, INTER_LINEAR); // This will be needed later while saving images

																		//save extracted mouth
			imwrite("mouth.png", crop_mouth);
			imshow("mouth", crop_mouth);
		
	

	
	//imshow("eyes", crop_eyes);
	//imshow("eyes2", crop_eyes2);

	

	waitKey(0);
}