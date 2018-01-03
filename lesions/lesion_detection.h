#include "stdafx.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "lesions.h"
#include "lesion_filter.h"
#include <time.h>

std::vector<std::vector<cv::Point>> lesion_detection(const cv::Mat & image, std::string img_name) {
	///VARIABLES / SETTINGS
	//all tuning/performance parameters in one place.
	int gauss_ksize = 25;
	int blocksize = 27;
	int size_close = 2;
	int size_open = 2;

	int size_close2 = 4;
	int size_open2 = 2;
	
	//guassian blur
	cv::Size ksize;
	ksize.height = gauss_ksize;
	ksize.width = ksize.height;
	//thresholding
	int thresh = 0;
	int maxValue = 255;
	int thresholdType = cv::THRESH_BINARY_INV;
	int adaptMethod = cv::ADAPTIVE_THRESH_GAUSSIAN_C;
	//morphology
	int shape = cv::MORPH_ELLIPSE;
	int open = cv::MORPH_OPEN;
	int close = cv::MORPH_CLOSE;
	cv::Mat elem_close = cv::getStructuringElement(shape,
		cv::Size(2 * size_close + 1, 2 * size_close + 1),
		cv::Point(size_close, size_close));
	cv::Mat elem_open = cv::getStructuringElement(shape,
		cv::Size(2 * size_open + 1, 2 * size_open + 1),
		cv::Point(size_open, size_open));

	cv::Mat elem_open2 = cv::getStructuringElement(shape,
		cv::Size(2 * size_open2 + 1, 2 * size_open2 + 1),
		cv::Point(size_open2, size_open2));
	cv::Mat elem_close2 = cv::getStructuringElement(shape,
		cv::Size(2 * size_close2 + 1, 2 * size_close2 + 1),
		cv::Point(size_close2, size_close2));

	cv::Mat elem_erode = cv::getStructuringElement(shape,
		cv::Size(3, 3),
		cv::Point(1, 1));

	//countour
	std::vector<std::vector<cv::Point>> contours0;
	std::vector<std::vector<cv::Point>> contours1;
	std::vector<std::vector<cv::Point>> contours2;
	//matrices for each step of process
	cv::Mat gr_img, blur_img, gray_img, thresh_img, close_img, open_img;
	cv::Mat morph0_img, morph1_img, morph2_img;
	cv::Mat lab_img, yrb_img, not_B, not_Cr;
	
	///PROCESS
	/*
		Each line of code below corresponds to steps 1-8.
		1: removing blue channel removes noise (green-red image)
		2: guassin blur filter to reduce image noise
	 	3: convert to gray scale
	 	4: adaptive thresholding because lighting may not be totallly uniform
	 	5: close to fill in gaps
	 	6: open removes smaller blobs
	 	7: dilate what's left to make them more prominent
	 	8: find contours, the points that make up border of area on the original image 
	*/
	cv::GaussianBlur(image, blur_img, ksize, 0);
	gr_img = blur_img & cv::Scalar(0, 255, 255);      

	cv::cvtColor(gr_img, gray_img, CV_BGR2GRAY, 3);
	cv::cvtColor(gr_img, lab_img, CV_BGR2Lab, 3);
	cv::cvtColor(gr_img, yrb_img, CV_BGR2YCrCb, 3);
	std::vector<cv::Mat> lab(3);
	std::vector<cv::Mat> yrb(3);
	cv::split(lab_img, lab);
	cv::split(lab_img, yrb);
	cv::bitwise_not(yrb[1], not_Cr);
	cv::bitwise_not(lab[2], not_B);

	///group 0: grayscale
	cv::adaptiveThreshold(gray_img, thresh_img, maxValue, adaptMethod, thresholdType, blocksize, 2);
	cv::morphologyEx(thresh_img, close_img, close, elem_close);
	cv::morphologyEx(close_img, close_img, open, elem_open);
	//throw in erode in there
	cv::morphologyEx(close_img, morph0_img, cv::MORPH_ERODE, elem_erode);
	///group 1: not Cr channel (from YCrCb color space)
	cv::adaptiveThreshold(not_Cr, thresh_img, maxValue, adaptMethod, thresholdType, blocksize, 2);  
	cv::morphologyEx(thresh_img, close_img, close, elem_close);
	cv::morphologyEx(close_img, morph1_img, open, elem_open);
	///group 2: not B channel (LAB color space)
	cv::adaptiveThreshold(not_B, thresh_img, maxValue, adaptMethod, thresholdType, blocksize, 2);
	cv::morphologyEx(thresh_img, close_img, close, elem_close);
	cv::morphologyEx(close_img, morph2_img, open, elem_open);
	
	cv::findContours(morph0_img, contours0, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	cv::findContours(morph1_img, contours1, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	cv::findContours(morph2_img, contours2, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	std::vector<std::vector<cv::Point>> contours0_fil;
	lesion_filter(image, contours0, contours0_fil, (img_name + "_0"));
	std::vector<std::vector<cv::Point>> contours1_fil;
	lesion_filter(image, contours1, contours1_fil, (img_name + "1"));
	std::vector<std::vector<cv::Point>> contours2_fil;
	lesion_filter(image, contours2, contours2_fil, (img_name + "2"));

	///------------- TESTING / DEBUG ---------------------
	std::string img_out_dir = output_dir + "/detection/";
	//_mkdir(img_out_dir.c_str());
	//img_out_dir = img_out_dir + img_name + "/";
	//_mkdir(img_out_dir.c_str());

	/*
	FILE * file;
	std::string out_file = img_out_dir + "/num_of_lesions.txt";
	file = fopen(out_file.c_str(), "w");
	*/



	///COLOR SPACE EXPERIMENTS
	/*
	cv::Mat ab_img, cr_img, yrb_img, hsv_img;
	cv::cvtColor(blur_img, lab_img, CV_BGR2Lab, 3);
	cv::cvtColor(blur_img, yrb_img, CV_BGR2YCrCb, 3);
	cv::cvtColor(blur_img, hsv_img, CV_BGR2HSV, 3);
	std::vector<cv::Mat> bgr(3);
	std::vector<cv::Mat> lab(3);
	std::vector<cv::Mat> yrb(3);
	std::vector<cv::Mat> hsv(3);
	cv::split(lab_img, bgr);
	cv::split(lab_img, lab);
	cv::split(yrb_img, yrb);
	cv::split(hsv_img, hsv);
	
	//lab[1] = hsv[1];
	//lab[1] = cv::Scalar(0);
	lab[2] = yrb[1];
	//cv::bitwise_not(lab[1], lab[1]);
	//cv::bitwise_not(lab[2], lab[2]);
	//cv::bitwise_not(lab[1], lab[1]);
	//cv::bitwise_not(lab[2], lab[2]);

	cv::merge(lab, ab_img);
	//cv::cvtColor(ab_img, gray_img, CV_Lab2BGR, 3);
	//cv::cvtColor(ab_img, gray_img, CV_BGR2GRAY, 1);
	
	cv::bitwise_not(lab[1], lab[1]);
	gray_img = lab[1];

	//cv::bitwise_not(yrb[1], yrb[1]);
	//gray_img = yrb[1];

	cv::bitwise_not(gray_img, gray_img);

	cv::adaptiveThreshold(gray_img, thresh_img, maxValue, adaptMethod, thresholdType, blocksize, 2);
	//cv::bitwise_not(thresh_img, thresh_img);
	//cv::morphologyEx(thresh_img, thresh_img, open, elem_open);
	cv::morphologyEx(thresh_img, close_img, close, elem_close);
	cv::morphologyEx(close_img, open_img, open, elem_open);
	cv::morphologyEx(open_img, close_img, close, elem_close2);
	cv::morphologyEx(close_img, open_img, open, elem_open2);
	cv::findContours(open_img, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	*/
	///---------------------
	cv::Mat masked0, masked1, masked2, color;
	cv::bitwise_not(morph0_img, morph0_img);
	cv::cvtColor(morph0_img, color, CV_GRAY2BGR);
	cv::bitwise_and(color, image, masked0);

	cv::bitwise_not(morph1_img, morph1_img);
	cv::cvtColor(morph1_img, color, CV_GRAY2BGR);
	cv::bitwise_and(color, image, masked1);

	cv::bitwise_not(morph2_img, morph2_img);
	cv::cvtColor(morph2_img, color, CV_GRAY2BGR);
	cv::bitwise_and(color, image, masked2);

	cv::Mat masked0_filter;
	image.copyTo(masked0_filter);
	printf("cont vec size: %d \n", contours0_fil.size());
	for (int i = 0; i < contours0_fil.size(); i++) {
		//mask is a black image
		//on black background, draw white contour area corresponding to lesion
		cv::drawContours(masked0_filter, contours0_fil, i, cv::Scalar(255), -1);
	}

	cv::Mat masked1_filter;
	image.copyTo(masked1_filter);
	printf("cont vec size: %d \n", contours1_fil.size());
	for (int i = 0; i < contours1_fil.size(); i++) {
		//mask is a black image
		//on black background, draw white contour area corresponding to lesion
		cv::drawContours(masked1_filter, contours1_fil, i, cv::Scalar(255), -1);
	}

	cv::Mat masked2_filter;
	image.copyTo(masked2_filter);
	printf("cont vec size: %d \n", contours2_fil.size());
	for (int i = 0; i < contours2_fil.size(); i++) {
		//mask is a black image
		//on black background, draw white contour area corresponding to lesion
		cv::drawContours(masked2_filter, contours2_fil, i, cv::Scalar(255), -1);
	}

	cv::imwrite(img_out_dir + img_name + "_0_bgr.jpg", image);
	
	//cv::imwrite(img_out_dir + img_name + "_1_gr_" +  ".jpg", gr_img);
	//cv::imwrite(img_out_dir + img_name + "_2_blur_" + ".jpg", blur_img);
	//cv::imwrite(img_out_dir + img_name + "_3_gray_" + ".jpg", gray_img);
	/*
	cv::imwrite(img_out_dir + img_name + "_2_lab_" +  ".jpg", lab_img);
	cv::imwrite(img_out_dir + "_3_ab_" + img_name + ".jpg", ab_img);
	cv::imwrite(img_out_dir + img_name + "_4_thresh_" +  ".jpg", thresh_img);
	cv::imwrite(img_out_dir + img_name + "_5_close_" + ".jpg",  close_img);
	cv::imwrite(img_out_dir + img_name + "_6_open_" + ".jpg",   open_img);
	*/
	cv::imwrite(img_out_dir + img_name + "_0_masked_value_" + ".jpg", masked0);
	cv::imwrite(img_out_dir + img_name + "_0_masked_value_filter_" + ".jpg", masked0_filter);

	cv::imwrite(img_out_dir + img_name + "_1_masked_value_" + ".jpg", masked1);
	cv::imwrite(img_out_dir + img_name + "_1_masked_value_filter_" + ".jpg", masked1_filter);

	cv::imwrite(img_out_dir + img_name + "_2_masked_value_" + ".jpg", masked2);
	cv::imwrite(img_out_dir + img_name + "_2_masked_value_filter_" + ".jpg", masked2_filter);
	//cv::imwrite(img_out_dir + img_name + "_8_masked_Cr_" + ".jpg", masked1);
	//cv::imwrite(img_out_dir + img_name + "_9_masked_notB_" + ".jpg", masked2);
	
	//int num_lesions = contours.size() - 1;
	//fprintf(file, "# lesions: %d \n", num_lesions);
	///--------------------------------------------------------

	return contours0;
}
