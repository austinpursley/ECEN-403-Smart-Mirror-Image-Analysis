#include "stdafx.h"
#include "face_feature_extract.h"

// Function detectAndDisplay
void get_face_features(Mat frame, CascadeClassifier &face_cascade, CascadeClassifier &eyes_cascade, extern CascadeClassifier &mouth_cascade, std::vector<Rect>& features)
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
		//std::cout << it->first << " => " << it->second << '\n';
		crop = frame(it->second);
		resize(crop, res, Size(128, 128), 0, 0, INTER_LINEAR); // This will be needed later while saving images
		//imshow(std::to_string(face_num), crop);
		cvtColor(crop, gray, CV_BGR2GRAY); // Convert cropped image to Grayscale
										   // Form a filename
		face_num--;
		if (face_num == 0) {
			roi_face = it->second;
			features.push_back(roi_face);
			printf("face detect\n");
			break;
		}
	}

	if (!crop.empty())
	{
		//imshow("detected", crop);
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
	cv::Rect roi_c1; //current element
	size_t ic1 = 0; // ic is index of current element
	int ac1 = 0; // ac is area of current element
	
	std::map<int, cv::Rect> eye_areas;
	//EYE DETECTION
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
		resize(crop_eyes, res, Size(128, 128), 0, 0, INTER_LINEAR); // This will be needed later while saving images
																	//save extracted eyes
		//imshow(std::to_string(eye_num), crop_eyes);
		eye_num--;
		if (eye_num == 0) {
			printf("eye 2 found\n");
			roi_eye2 = ite->second;
			features.push_back(roi_eye2);
			break;
		}
		else if (eye_num == 1) {
			printf("eye 1 found\n");
			features.push_back(roi_eye1);
			roi_eye1 = ite->second;
		}
	}

	//produce image with eyes cropped out
	//TODO: detect and extract mouth
	mouth_cascade.detectMultiScale(gray, mouths, 1.5, 15, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

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
		resize(crop_mouth, res, Size(128, 128), 0, 0, INTER_LINEAR); // This will be needed later while saving images
																	//save extracted eyes
		//imshow("mouth", crop_mouth);
		mouth_num--;
		if (mouth_num == 0) {
			roi_mouth = itm->second;
			//roi_mouth.x = roi_mouth.x + roi_face.x;
			//roi_mouth.y = roi_mouth.y + roi_face.y;
			features.push_back(roi_mouth);
			printf("mouth detected\n");
			break;
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
	double escale = 0.4;
	cv::Point center(roi_eye1.x + roi_eye1.width / 2, roi_eye1.y + roi_eye1.height / 2);
	cv::Size axes(roi_eye1.height * escale, roi_eye1.width * escale);
	ellipse(crop, center, axes, 90, 0, 360, 0, -1);

	center = cv::Point(roi_eye2.x + roi_eye2.width / 2, roi_eye2.y + roi_eye2.height / 2);
	axes = cv::Size(roi_eye2.height * escale, roi_eye2.width * escale);
	ellipse(crop, center, axes, 90, 0, 360, 0, -1);

	double mscale = 0.4;
	center = cv::Point(roi_mouth.x + roi_mouth.width/2,roi_mouth.y + roi_mouth.height/2);
	axes = cv::Size(roi_mouth.height * mscale, roi_mouth.width * mscale);
	ellipse(crop, center, axes, 90,0, 360, 0, -1);

	imshow("crop", crop);
	imshow("frame", frame);
	waitKey(0);
}