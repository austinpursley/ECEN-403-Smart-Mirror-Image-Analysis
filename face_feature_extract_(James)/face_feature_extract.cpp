#include "stdafx.h"
#include "face_feature_extract.h"

// Function detectAndDisplay
void detectAndDisplay(Mat frame, CascadeClassifier &face_cascade, CascadeClassifier &eyes_cascade, extern CascadeClassifier &mouth_cascade)
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
	sstm << "Crop area size: " << roi_b.width << "x" << roi_b.height << " Filename: " << "no file name";
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
	cv::Rect roi_c1; //current element

	size_t ic1 = 0; // ic is index of current element
	int ac1 = 0; // ac is area of current element

	size_t ib1 = 0; // ib is index of biggest element
	int ab1 = 0; // ab is area of biggest element

	size_t ib2 = 0;
	int ab2 = 0;
	
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
	for (auto it = eye_areas.rbegin(); it != eye_areas.rend(); ++it) {
		//std::cout << it->first << " => " << it->second << '\n';
		

		crop_eyes = crop(it->second);
		resize(crop_eyes, res, Size(128, 128), 0, 0, INTER_LINEAR); // This will be needed later while saving images
																	//save extracted eyes
		imwrite(std::to_string(eye_num) + ".png", crop_eyes);
		imshow(std::to_string(eye_num), crop_eyes);
		eye_num--;
		if (eye_num == 0) {
			break;
			printf("works\n");
		}
	}

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