#include "stdafx.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "fix_me_lesions.h"
#include "lesion_filter.h"
#include <time.h>
#include "lesions.hpp"

///Function source: https://stackoverflow.com/questions/35668074/how-i-can-take-the-average-of-100-image-using-opencv
cv::Mat1b getMean(const std::vector<cv::Mat1b>& images)
{
	if (images.empty()) return cv::Mat1b();
	// Create a 0 initialized image to use as accumulator
	cv::Mat m(images[0].rows, images[0].cols, CV_64FC1);
	m.setTo(cv::Scalar(0));

	// Use a temp image to hold the conversion of each input image to CV_64FC1
	// This will be allocated just the first time, since mix your images have
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
	//mix tuning/performance parameters in one place.
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
	int size_open = 3;
	int size_close2 = 3;
	int size_open2 = 3;
	int size_erode = 1;
	
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
	int erode = cv::MORPH_ERODE;
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
		cv::Size(2 * size_erode + 1, 2 * size_erode + 1),
		cv::Point(size_erode, size_erode));
	//countour
	std::vector<std::vector<cv::Point>> contours;
	std::vector<std::vector<cv::Point>> contours2;
	std::vector<std::vector<cv::Point>> contours3;
	//matrices for each step of process
	cv::Mat gr_img, lab_img, mix_img, gray_img, blur_img, thresh_img, close_img, open_img;
	cv::Mat morph1, morph2, morph_comb;
	cv::Mat AB;
	cv::Mat mean_gray_img, red_mix;

	/*
	cv::Mat3b hsv_img(image.rows, image.cols, CV_8UC3);
	cv::cvtColor(image, hsv_img, CV_BGR2HSV, 3);
	std::vector<cv::Mat1b> hsv(3);
	cv::split(hsv_img, hsv);
	*/

	///PROCESS
	/*

		1: removing blue channel removes noise (green-red image)
		2: guassin blur filter to reduce image noise
	 	3: convert to gray scale
	 	4: adaptive thresholding because lighting may not be totallly uniform
	 	5: close to fill in gaps
	 	6: open removes smaller blobs
	 	7: dilate what's left to make them more prominent
	 	8: find contours, the points that make up border of area on the original image 
	*/

	//general, all lesions
	gr_img = image & cv::Scalar(0, 255, 255);
	lab_img = cv::Mat(image.rows, image.cols, CV_8UC3);
	cv::cvtColor(gr_img, lab_img, CV_BGR2Lab, 3);
	std::vector<cv::Mat1b> lab(3);
	cv::split(lab_img, lab);
	cv::addWeighted(lab[1], 0.5, lab[2], 0.5, 0, AB);
	cv::addWeighted(AB, 0.6, lab[0], 0.4, 0, mix_img);
	cv::GaussianBlur(mix_img, blur_img, ksize, 0);
	cv::adaptiveThreshold(blur_img, thresh_img, maxValue, adaptMethod, thresholdType, blocksize, 2);  
	cv::morphologyEx(thresh_img, close_img, close, elem_close);
	cv::morphologyEx(close_img, open_img, open, elem_open); 
	cv::morphologyEx(open_img, morph1, erode, elem_erode);
	cv::findContours(morph1, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	//redder lesions
	
	cv::Mat3b yrb_img(image.rows, image.cols, CV_8UC3);
	cv::cvtColor(gr_img, yrb_img, CV_BGR2YCrCb, 3);
	std::vector<cv::Mat1b> yrb(3);
	cv::split(yrb_img, yrb);
	
	cv::bitwise_not(yrb[1], yrb[1]);
	cv::bitwise_not(lab[0], lab[0]);
	cv::addWeighted(yrb[1], 0.8, lab[0], 0.2, 0, red_mix);
	cv::GaussianBlur(red_mix, blur_img, ksize, 0);
	cv::adaptiveThreshold(blur_img, thresh_img, maxValue, adaptMethod, thresholdType, blocksize, 2);
	cv::morphologyEx(thresh_img, close_img, close, elem_close);
	cv::morphologyEx(close_img, open_img, open, elem_open);
	cv::morphologyEx(open_img, morph2, erode, elem_erode);
	cv::findContours(morph2, contours2, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	///LESION CLASS TESTING
	std::vector<Lesion> lesions;
	
	//combine two above
	//morph1 = cv::Mat(image.rows, image.cols, CV_8UC1, cv::Scalar(0));
	cv::bitwise_not(morph1, morph1);
	cv::bitwise_not(morph2, morph2);
	cv::bitwise_and(morph1, morph2, morph_comb);
	cv::findContours(morph2, contours3, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	

	for (int i = 0; i < contours3.size(); i++) {
		int id_num = i;
		lesions.push_back(Lesion (contours3[i], image, morph_comb, id_num));
		/*
		std::string img_out_dir = output_dir + "/lesion_class/";
		cv::Mat show = image.clone();
		lesions[i].draw(show);
		cv::imwrite(img_out_dir + std::to_string(i) + "_0les" + ".jpg", show);
		*/
	}
	///-------------------

	///------------- TESTING / DEBUG ---------------------
	cv::Mat color, masked;
	//cv::bitwise_not(morph_comb, morph_comb);
	cv::cvtColor(morph_comb, color, CV_GRAY2BGR);
	
	cv::bitwise_and(color, image, masked);

	std::string img_out_dir = output_dir + "/detection/";
	//_mkdir(img_out_dir.c_str());
	//img_out_dir = img_out_dir + img_name + "/";
	//_mkdir(img_out_dir.c_str());

	/*
	FILE * file;
	std::string out_file = img_out_dir + "/num_of_lesions.txt";
	file = fopen(out_file.c_str(), "w");
	*/

	cv::imwrite(img_out_dir + img_name + "_10_image_" + ".jpg", image);
	cv::imwrite(img_out_dir + img_name + "_6_morph1_" + ".jpg", morph1);
	cv::imwrite(img_out_dir + img_name + "_7_morph2_" + ".jpg", morph2);
	cv::imwrite(img_out_dir + img_name + "_8_morph_" + ".jpg", morph_comb);
	cv::imwrite(img_out_dir + img_name + "_9_mask_" + ".jpg", masked);
	/*
	cv::imwrite(img_out_dir + img_name + "_0_image_" + ".jpg", image);
	cv::imwrite(img_out_dir + img_name + "_1_gr_" +  ".jpg", gr_img);
	cv::imwrite(img_out_dir + img_name + "_2_mix_" + ".jpg", mix_img);
	cv::imwrite(img_out_dir + img_name + "_3_blur_" + ".jpg", blur_img);
	//cv::imwrite(img_out_dir + img_name + "_4_thresh_" +  ".jpg", thresh_img);
	//cv::imwrite(img_out_dir + img_name + "_5_close_" + ".jpg",  close_img);
	cv::imwrite(img_out_dir + img_name + "_6_morph_" + ".jpg",   morph_comb);
	cv::imwrite(img_out_dir + img_name + "_7_mask_" + ".jpg", masked);
	*/

	//int num_lesions = contours.size() - 1;
	//fprintf(file, "# lesions: %d \n", num_lesions);
	///--------------------------------------------------------

	return contours;
}
