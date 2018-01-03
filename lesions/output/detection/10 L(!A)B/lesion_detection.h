#include "stdafx.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "lesions.h"
#include <time.h>

std::vector<std::vector<cv::Point>> lesion_detection(const cv::Mat & image, std::string img_name) {
	///VARIABLES / SETTINGS
	//all tuning/performance parameters in one place.
	int gauss_ksize = 19;
	int blocksize = 45;
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
	
	gr_img = image & cv::Scalar(0, 255, 255);      
	cv::GaussianBlur(gr_img, blur_img, ksize, 0); 
	cv::cvtColor(blur_img, gray_img, CV_BGR2GRAY); 
	cv::adaptiveThreshold(gray_img, thresh_img, maxValue, adaptMethod, thresholdType, blocksize, 2);  
	cv::morphologyEx(thresh_img, close_img, close, elem_close);
	cv::morphologyEx(close_img, open_img, open, elem_open); 
	cv::findContours(open_img, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));


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
	cv::Mat lab_img, ab_img, cr_img, yrb_img, hsv_img;
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
	
	//lab[0] = hsv[1];
	//lab[0] = cv::Scalar(0);
	//lab[2] = cv::Scalar(0);
	//cv::bitwise_not(lab[1], lab[1]);
	//cv::bitwise_not(lab[2], lab[2]);
	cv::bitwise_not(lab[1], lab[1]);
	cv::merge(lab, ab_img);
	//cv::cvtColor(ab_img, gray_img, CV_Lab2BGR, 3);
	cv::cvtColor(ab_img, gray_img, CV_BGR2GRAY, 1);
	
	//gray_img = lab[1];

	//cv::bitwise_not(gray_img, gray_img);

	cv::adaptiveThreshold(gray_img, thresh_img, maxValue, adaptMethod, thresholdType, blocksize, 2);
	//cv::bitwise_not(thresh_img, thresh_img);
	//cv::morphologyEx(thresh_img, thresh_img, open, elem_open);
	cv::morphologyEx(thresh_img, close_img, close, elem_close);
	cv::morphologyEx(close_img, open_img, open, elem_open);
	cv::morphologyEx(open_img, close_img, close, elem_close2);
	cv::morphologyEx(close_img, open_img, open, elem_open2);
	cv::findContours(open_img, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	
	///---------------------
	cv::bitwise_not(open_img, open_img);
	cv::Mat masked, color;
	cv::cvtColor(open_img, color, CV_GRAY2BGR);
	cv::bitwise_and(color, image, masked);
	
	cv::imwrite(img_out_dir + img_name + "_0_bgr.jpg", image);
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
	cv::imwrite(img_out_dir + img_name + "_8_masked_" + ".jpg", masked);
	
	//int num_lesions = contours.size() - 1;
	//fprintf(file, "# lesions: %d \n", num_lesions);
	///--------------------------------------------------------

	return contours;
}
