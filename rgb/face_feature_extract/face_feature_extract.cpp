#include "stdafx.h"
#include "face_feature_extract.h"

// Function detectAndDisplay
void get_face_features(cv::Mat frame, cv::CascadeClassifier &face_cascade, cv::CascadeClassifier &eyes_cascade, cv::CascadeClassifier &mouth_cascade, std::map<std::string, cv::Rect>& features)
{
	std::vector<cv::Rect> faces; //all detected faces
	std::vector<cv::Rect> eyes; //all detected eyes
	std::vector<cv::Rect> mouths; //all detected mouths
	cv::Mat frame_gray; //the original image, converted to grayscale
	cv::Mat face_gray; //cropped face, color
	cv::Mat crop_eyes; //cropped eyes, color
	cv::Mat crop_eyes2;
	cv::Mat crop_mouth;
	cv::Mat res;
	cv::Mat gray; //cropped face, grayscale
	std::string text;
	std::stringstream sstm;
	//convert the image to grayscale
	cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
	//equalizeHist(frame_gray, frame_gray);

	///----------------------FACE------------------------------
	face_cascade.detectMultiScale(frame_gray, faces, 1.05, 3, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(100, 100));
	cv::Rect roi_face;
	if (faces.size() >= 1) {
		printf("face detect\n");
		roi_face = faces[0];
		features["face"] = roi_face;
		face_gray = frame_gray(roi_face);
	}
	else {
		printf("ERROR: no fadce detected\n");
		return;
	}

	///----------------------EYES------------------------------
	
	cv::Mat upper_face_gray;
	cv::Rect roi_upper_face;
	roi_upper_face.x = roi_face.x;
	roi_upper_face.y = roi_face.y;
	roi_upper_face.width = roi_face.width;
	roi_upper_face.height = roi_face.height*0.7;
	upper_face_gray = frame_gray(roi_upper_face);
	//imshow("upperFace", upper_face_gray);
	eyes_cascade.detectMultiScale(upper_face_gray, eyes, 1.05, 3, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(5, 5), cv::Size(60, 60));
	
	// Set Region of Interest
	cv::Rect roi_c1; //current element
	size_t ic1 = 0; // ic is index of current element
	int ac1 = 0; // ac is area of current element
	if (eyes.size() >= 2) {
		features["eye1"] = eyes[0];
		features["eye2"] = eyes[1];
		//upper_face_gray(eyes_roi[0]) = 0;
		//upper_face_gray(eyes_roi[1]) = 0;
	}
	else if (eyes.size() == 2) {
		features["eye1"] = eyes[0];
		//upper_face_gray(eyes_roi[0]) = 0;
		printf("only one eye detected \n");
	}
	else {
		printf("ERROR: no eyes detected \n");
	}
	
	///----------------------MOUTH------------------------------
	cv::Rect roi_lower_face;
	roi_lower_face.x = roi_face.x;
	roi_lower_face.y = roi_face.y + roi_face.height*0.6;
	roi_lower_face.width = roi_face.width;
	roi_lower_face.height = roi_face.height*0.6;
	cv::Mat  lower_face = frame_gray(roi_lower_face);
	
	mouth_cascade.detectMultiScale(lower_face, mouths, 1.05, 3, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(20, 20), cv::Size(75, 75));
	cv::Rect roi_mouth;
	if (mouths.size() >= 1) {
		 roi_mouth = mouths[0];
		 roi_mouth.y = roi_mouth.y + roi_face.height*0.55;
		 features["mouth"] = roi_mouth;
		//imshow("lowerFace", lower_face);
	}
	else {
		printf("ERROR: no mouth detected \n");
	}
	
	return;
}

void mask_features(cv::Mat face, std::map<std::string, cv::Rect> features, cv::Mat& masked_face) {
	printf("here \n");
	//resize(face, face, cv::Size(128, 128), 0, 0, cv::INTER_LINEAR);
	face.copyTo(masked_face);
	double escale = 0.4;
	cv::Point center(features["eye1"].x + features["eye1"].width / 2, features["eye1"].y + features["eye1"].height / 2);
	cv::Size axes(features["eye1"].height * escale, features["eye1"].width * escale);
	ellipse(masked_face, center, axes, 90, 0, 360, 0, -1);
	
	center = cv::Point(features["eye2"].x + features["eye2"].width / 2, features["eye2"].y + features["eye2"].height / 2);
	axes = cv::Size(features["eye2"].height * escale, features["eye2"].width * escale);
	ellipse(masked_face, center, axes, 90, 0, 360, 0, -1);

	double mscale = 0.5;
	center = cv::Point(features["mouth"].x + features["mouth"].width / 2, features["mouth"].y + features["mouth"].height / 2);
	axes = cv::Size(features["mouth"].height * mscale, features["mouth"].width * mscale);
	ellipse(masked_face, center, axes, 90, 0, 360, 0, -1);
	
	return;
}