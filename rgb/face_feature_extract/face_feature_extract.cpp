#include "stdafx.h"
#include "face_feature_extract.h"

// Function detectAndDisplay
void get_face_features(cv::Mat frame, cv::CascadeClassifier &face_cascade, cv::CascadeClassifier &eyes_cascade, cv::CascadeClassifier &mouth_cascade, std::map<std::string, cv::Rect>& features)
{
	std::vector<cv::Rect> faces; //all detected faces
	std::vector<cv::Rect> eyes; //all detected eyes
	std::vector<cv::Rect> mouths; //all detected mouths
	cv::Mat frame_gray; //the original image, converted to grayscale
	cv::Mat crop; //cropped face, color
	cv::Mat crop_eyes; //cropped eyes, color
	cv::Mat crop_eyes2;
	cv::Mat crop_mouth;
	cv::Mat res;
	cv::Mat gray; //cropped face, grayscale
	std::string text;
	std::stringstream sstm;
	//convert the image to grayscale
	cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	///----------------------FACE------------------------------
	// Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

	// Set Region of Interest
	cv::Rect roi_c;
	size_t ic = 0; // ic is index of current element
	int ac = 0; // ac is area of current element

	std::map<int, cv::Rect> face_areas;
	for (ic = 0; ic < faces.size(); ic++) // Iterate through all current elements (detected faces)
	{
		roi_c.x = faces[ic].x;
		roi_c.y = faces[ic].y;
		roi_c.width = (faces[ic].width);
		roi_c.height = (faces[ic].height);
		
		ac = roi_c.width * roi_c.height; // Get the area of current element (detected face)
		face_areas[ac] = roi_c;
	}
	
	int face_num = 1;
	cv::Rect roi_face;
	for (auto it = face_areas.rbegin(); it != face_areas.rend(); ++it) {
		cv::Mat frame_copy;
		frame.copyTo(frame_copy);
		crop = frame_copy(it->second);
		resize(crop, crop, cv::Size(128, 128), 0, 0, cv::INTER_LINEAR); // This will be needed later while saving images

		//imshow(std::to_string(face_num), crop);
		cvtColor(crop, gray, CV_BGR2GRAY); // Convert cropped image to Grayscale
										   // Form a filename
		face_num--;
		if (face_num == 0) {
			roi_face = it->second;
			features["face"] = roi_face;
			printf("face detect\n");
			break;
		}
	}

	if (!crop.empty())
	{
		//imshow("detected", crop);
	}


	///----------------------MOUTH------------------------------
	mouth_cascade.detectMultiScale(gray, mouths, 1.6, 8, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

	// Set Region of Interest
	cv::Rect roi_c_m; //current element
	size_t ic_m = 0; // ic is index of current element
	int ac_m = 0; // ac is area of current element

	std::map<int, cv::Rect> mouth_areas;

	for (ic_m = 0; ic_m < mouths.size(); ic_m++) // Iterate through all current elements (detected mouths)
	{
		roi_c_m.x = mouths[ic_m].x;
		roi_c_m.y = mouths[ic_m].y;
		roi_c_m.width = (mouths[ic_m].width);
		roi_c_m.height = (mouths[ic_m].height);
		//crop_mouth = crop(roi_c_m);
		//printf("roi_c_m.x: %d \n", roi_c_m.x);
		ac_m = roi_c_m.width * roi_c_m.height; // Get the area of current element (detected mouth)
		mouth_areas[ac_m] = roi_c_m;
		//imshow("mouth", crop_mouth);
	}

	int mouth_num = 1;
	cv::Rect roi_mouth;
	for (auto itm = mouth_areas.rbegin(); itm != mouth_areas.rend(); ++itm) {
		//std::cout << itm->first << '\n';
		crop_mouth = crop(itm->second);
		mouth_num--;
		if (mouth_num == 0) {
			roi_mouth = itm->second;
			features["mouth"] = roi_mouth;
			printf("mouth detected\n");
			crop(roi_mouth) = 0;
			break;
		}
	}

	cvtColor(crop, gray, CV_BGR2GRAY); // Convert cropped image to Grayscale

	///----------------------EYES------------------------------
	//TODO: extract both eyes instead of one
	//to do this, instead of extracting the biggest element, extract the two biggest elements
	//update 12/5/7 (insight): extract the biggest element e1, then run the same algorithm again to extract e2, with the condition
	//that e2 != e1
	eyes_cascade.detectMultiScale(gray, eyes, 1.01, 3, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(10, 10), cv::Size(100,100));

	// Set Region of Interest
	cv::Rect roi_c1; //current element
	size_t ic1 = 0; // ic is index of current element
	int ac1 = 0; // ac is area of current element
	
	std::map<int, cv::Rect> eye_areas;
	for (ic1 = 0; ic1 < eyes.size(); ic1++) // Iterate through all current elements (detected eyes)

	{
		roi_c1.x = eyes[ic1].x;
		roi_c1.y = eyes[ic1].y;
		roi_c1.width = (eyes[ic1].width);
		roi_c1.height = (eyes[ic1].height);
		ac1 = roi_c1.width * roi_c1.height; // Get the area of current element (detected eye)

		eye_areas[ac1] = roi_c1;
	}

	int eye_num = 2;
	cv::Rect roi_eye1;
	cv::Rect roi_eye2;
	for (auto ite = eye_areas.rbegin(); ite != eye_areas.rend(); ++ite) {
		crop_eyes = crop(ite->second);
		resize(crop_eyes, res, cv::Size(128, 128), 0, 0, cv::INTER_LINEAR); // This will be needed later while saving images
																	//save extracted eyes
		//imshow(std::to_string(eye_num), crop_eyes);
		eye_num--;
		if (eye_num == 0) {
			printf("eye 2 found\n");
			roi_eye2 = ite->second;
			features["eye2"] = roi_eye2;
			crop(roi_eye2) = 0;
			break;
		}
		else if (eye_num == 1) {
			printf("eye 1 found\n");
			roi_eye1 = ite->second;
			features["eye1"] = roi_eye1;
			crop(roi_eye1) = 0;
		}
	}

	/*
	roi_mouth.x = roi_mouth.x + roi_mouth.width/4;
	roi_mouth.y = roi_mouth.y + roi_mouth.height/4;
	roi_mouth.width = roi_mouth.width/2;
	roi_mouth.height = roi_mouth.height/2;
	
	roi_eye1.x = roi_eye1.x + roi_eye1.width / 4;
	roi_eye1.y = roi_eye1.y + roi_eye1.height / 4;
	roi_eye1.width = roi_eye1.width / 2;
	roi_eye1.height = roi_eye1.height / 2;

	roi_eye2.x = roi_eye2.x + roi_eye2.width / 4;
	roi_eye2.y = roi_eye2.y + roi_eye2.height / 4;
	roi_eye2.width = roi_eye2.width / 2;
	roi_eye2.height = roi_eye2.height / 2;
	*/

	//crop(roi_mouth) = 0;
	//crop(roi_eye1) = 0;
	//crop(roi_eye2) = 0;

	cv::waitKey(0);
}