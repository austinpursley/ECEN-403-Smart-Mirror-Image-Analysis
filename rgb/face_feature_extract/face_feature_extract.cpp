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
	face_cascade.detectMultiScale(frame_gray, faces, 1.05, 3, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(300, 300));
	cv::Rect roi_face(0,0,0,0);
	if (faces.size() >= 1) {
		//printf("face detect\n");
		roi_face = faces[0];
		features["face"] = roi_face;
		face_gray = frame_gray(roi_face);
	}
	else {
		//printf("ERROR: no fadce detected\n");
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
	eyes_cascade.detectMultiScale(upper_face_gray, eyes, 1.05, 3, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(75, 75));
	
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
		//printf("only one eye detected \n");
	}

	else {
		//printf("ERROR: no eyes detected \n");
	}
	
	///----------------------MOUTH------------------------------
	cv::Rect roi_lower_face;
	roi_lower_face.x = roi_face.x;
	roi_lower_face.y = roi_face.y + roi_face.height*0.6;
	roi_lower_face.width = roi_face.width;
	roi_lower_face.height = roi_face.height*0.4;
	cv::Mat  lower_face = frame_gray(roi_lower_face);
	
	mouth_cascade.detectMultiScale(lower_face, mouths, 1.05, 3, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(75, 75));
	cv::Rect roi_mouth;
	if (mouths.size() >= 1) {
		 roi_mouth = mouths[0];
		 roi_mouth.y = roi_mouth.y + roi_face.height*0.55;
		 features["mouth"] = roi_mouth;
		//imshow("lowerFace", lower_face);
	}
	else {
		//printf("ERROR: no mouth detected \n");
	}

	return;
}