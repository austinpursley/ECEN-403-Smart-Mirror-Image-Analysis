#include "stdafx.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "lesions.h"
#include "lesion_filter.h"
#include <time.h>

///Function source: https://stackoverflow.com/questions/35668074/how-i-can-take-the-average-of-100-image-using-opencv
cv::Mat1b getMean(const std::vector<cv::Mat1b>& images)
{
	if (images.empty()) return cv::Mat1b();
	// Create a 0 initialized image to use as accumulator
	cv::Mat m(images[0].rows, images[0].cols, CV_64FC1);
	m.setTo(cv::Scalar(0));

	// Use a temp image to hold the conversion of each input image to CV_64FC1
	// This will be allocated just the first time, since all your images have
	// the same size.
	cv::Mat temp;
	
	for (int i = 0; i < images.size(); ++i)
	{
		// Convert the input images to CV_64FC3 ...
		images[i].convertTo(temp, CV_64FC1);

		// ... so you can accumulate
		m += temp;

	}
	// Convert back to CV_8UC1 type, applying the division to get the actual mean
	m.convertTo(m, CV_8U, 1. / images.size());
	return m;
}

std::vector<std::vector<cv::Point>> lesion_detection(const cv::Mat & image, std::string img_name) {
	///VARIABLES / SETTINGS
	//all tuning/performance parameters in one place.
	/*
	//light red
	int gauss_ksize = 9;
	int blocksize = 39;
	int size_close = 1;
	int size_open = 2;
	int size_close2 = 4;
	int size_open2 = 3;
	*/

	int gauss_ksize = 9;
	int blocksize = 39;
	int size_close = 1;
	int size_open = 2;
	int size_close2 = 4;
	int size_open2 = 3;
	
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
	//countour
	std::vector<std::vector<cv::Point>> contours;
	//matrices for each step of process
	cv::Mat gr_img, blur_img, gray_img, thresh_img, close_img, open_img;
	
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
	cv::cvtColor(gr_img, gray_img, CV_BGR2GRAY); 
	/*
	cv::adaptiveThreshold(gray_img, thresh_img, maxValue, adaptMethod, thresholdType, blocksize, 2);  
	cv::morphologyEx(thresh_img, close_img, close, elem_close);
	cv::morphologyEx(close_img, open_img, open, elem_open); 
	cv::findContours(open_img, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	*/


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
	

	cv::Mat3b lab_img(image.rows, image.cols, CV_8UC3);
	cv::Mat3b yrb_img(image.rows, image.cols, CV_8UC3);
	cv::Mat3b hsv_img(image.rows, image.cols, CV_8UC3);
	cv::Mat ab_img, cr_img;
	cv::cvtColor(gr_img, lab_img, CV_BGR2Lab, 3);
	cv::cvtColor(gr_img, yrb_img, CV_BGR2YCrCb, 3);
	cv::cvtColor(blur_img, hsv_img, CV_BGR2HSV, 3);
	std::vector<cv::Mat1b> bgr(3);
	std::vector<cv::Mat1b> lab(3);
	std::vector<cv::Mat1b> yrb(3);
	std::vector<cv::Mat1b> hsv(3);

	cv::split(gr_img, bgr);
	cv::split(lab_img, lab);
	cv::split(yrb_img, yrb);
	cv::split(hsv_img, hsv);
	
	//------- LIGHT RED -----------//
	cv::Scalar mean_color = cv::mean(image);
	cv::Mat mean_gray_img, light_red;
	cv::Mat mean_color_img(image.rows, image.cols, CV_8UC3, mean_color);
	cv::cvtColor(mean_color_img, mean_gray_img, CV_BGR2GRAY, 1);
	cv::bitwise_not(yrb[1], yrb[1]);
	cv::addWeighted(yrb[1], 0.7, mean_gray_img, 0.3, 0, light_red);

	//---------- ALL ------------//
	cv::Mat AB, LAB;
	cv::addWeighted(lab[1], 0.5, lab[2], 0.5, 0, AB);
	cv::addWeighted(AB, 0.6, lab[0], 0.4, 0, LAB);
	
	std::vector<std::vector<cv::Point>> contours1;
	std::vector<std::vector<cv::Point>> contours2;
	cv::Mat morph1;
	cv::Mat morph2;

	cv::adaptiveThreshold(light_red, thresh_img, maxValue, adaptMethod, thresholdType, blocksize, 2);
	cv::morphologyEx(thresh_img, close_img, close, elem_close);
	cv::morphologyEx(close_img, open_img, open, elem_open);
	cv::morphologyEx(open_img, close_img, close, elem_close2);
	cv::morphologyEx(close_img, morph1, open, elem_open2);
	cv::findContours(morph1, contours1, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	cv::adaptiveThreshold(LAB, thresh_img, maxValue, adaptMethod, thresholdType, blocksize, 2);
	cv::morphologyEx(thresh_img, close_img, close, elem_close);
	cv::morphologyEx(close_img, open_img, open, elem_open);
	cv::morphologyEx(open_img, close_img, close, elem_close2);
	cv::morphologyEx(close_img, morph2, open, elem_open2);
	cv::findContours(morph2, contours2, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	
	//entropy filtering
	cv::GaussianBlur(image, blur_img, ksize, 0);
	cv::cvtColor(blur_img, gray_img, CV_BGR2GRAY);
	std::vector<std::vector<cv::Point>> contours1_filt;
	std::vector<std::vector<cv::Point>> contours2_filt;
	filter_lesions_by_entropy(gray_img, contours1, contours1_filt, (img_name + "_0"), 5, 0.3);
	filter_lesions_by_entropy(gray_img, contours2, contours2_filt, (img_name + "_1"), 5, 0.3);

	//combining?
	cv::Mat comb = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
	cv::drawContours(comb, contours1_filt, -1, cv::Scalar(255), -1);
	cv::drawContours(comb, contours2_filt, -1, cv::Scalar(255), -1);

	//color filtering
	//std::vector<std::vector<cv::Point>> filtered_cnts;
	//color_filter(blur_img, contours, filtered_cnts, (img_name + "_0"), 5, 0.25);

	///---------------------
	cv::Mat masked1_filter, masked2_filter, masked3_combined;
	cv::Mat masked1, masked2, color;
	image.copyTo(masked1_filter);
	image.copyTo(masked2_filter);
	image.copyTo(masked3_combined);
	cv::drawContours(masked1_filter, contours1_filt, -1, cv::Scalar(255), -1);
	cv::drawContours(masked2_filter, contours2_filt, -1, cv::Scalar(255), -1);

	cv::drawContours(masked3_combined, contours1_filt, -1, cv::Scalar(255), -1);
	cv::drawContours(masked3_combined, contours2_filt, -1, cv::Scalar(255), -1);
	
	cv::bitwise_not(morph1, morph1);
	cv::cvtColor(morph1, color, CV_GRAY2BGR);
	cv::bitwise_and(color, image, masked1);
	cv::bitwise_not(morph2, morph2);
	cv::cvtColor(morph2, color, CV_GRAY2BGR);
	cv::bitwise_and(color, image, masked2);
	
	cv::imwrite(img_out_dir + img_name + "_0_bgr.jpg", image);
	//cv::imwrite(img_out_dir + img_name + "_1_masked1_" + ".jpg", masked1);
	//cv::imwrite(img_out_dir + img_name + "_2_masked2_" + ".jpg", masked2);
	cv::imwrite(img_out_dir + img_name + "_3_masked1_entropy_filter_" + ".jpg", masked1_filter);
	cv::imwrite(img_out_dir + img_name + "_4_masked2_entropy_filter_" + ".jpg", masked2_filter);
	cv::imwrite(img_out_dir + img_name + "_5_combined_masks_" + ".jpg", masked3_combined);
	//cv::imwrite(img_out_dir + img_name + "_1_hsv_" + ".jpg", hsv[1]);
	//cv::imwrite(img_out_dir + img_name + "_4_thresh_" + ".jpg", thresh_img);
	/*
	cv::imwrite(img_out_dir + img_name + "_1_gr_" +  ".jpg", gr_img);
	cv::imwrite(img_out_dir + img_name + "_2_blur_" + ".jpg", blur_img);
	cv::imwrite(img_out_dir + img_name + "_3_gray_" + ".jpg", gray_img);
	cv::imwrite(img_out_dir + img_name + "_2_lab_" +  ".jpg", lab_img);
	cv::imwrite(img_out_dir + "_3_ab_" + img_name + ".jpg", ab_img);
	cv::imwrite(img_out_dir + img_name + "_4_thresh_" +  ".jpg", thresh_img);
	cv::imwrite(img_out_dir + img_name + "_5_close_" + ".jpg",  close_img);
	cv::imwrite(img_out_dir + img_name + "_6_open_" + ".jpg",   open_img);
	*/
	
	
	//int num_lesions = contours.size() - 1;
	//fprintf(file, "# lesions: %d \n", num_lesions);
	///--------------------------------------------------------

	return contours;
}
