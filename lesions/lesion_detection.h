#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "lesions.h"
#include "stdafx.h"

std::vector<std::vector<cv::Point>> lesion_detection(const cv::Mat & image, std::string img_name) {
	//all tuning/performance parameters in one place.
	int gauss_ksize = 45;
	int blocksize = 57;
	int size_close = 7;
	int size_open = 4;
	int size_dilate = 0;
	//guassian Blur
	cv::Size ksize;
	ksize.height = gauss_ksize;
	ksize.width = ksize.height;
	// thresholding
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
	cv::Mat elem_dilate = cv::getStructuringElement(shape,
		cv::Size(2 * size_dilate + 1, 2 * size_dilate + 1),
		cv::Point(size_dilate, size_dilate));
	//countour
	std::vector<std::vector<cv::Point>> contours;
	//Mats for each step of process
	cv::Mat gr_img, blur_img, gray_img, thresh_img, close_img, open_img, dilate_img;
	
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
	cv::dilate(open_img, dilate_img, elem_dilate); 
	cv::findContours(dilate_img, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
		
	///PRESENTATION / DEBUG
	std::string img_out_dir = output_dir + "/detection/";
	_mkdir(img_out_dir.c_str());
	img_out_dir = img_out_dir + img_name + "/";
	_mkdir(img_out_dir.c_str());

	FILE * file;
	std::string out_file = img_out_dir + "/num_of_lesions.txt";
	file = fopen(out_file.c_str(), "w");

	cv::bitwise_not(dilate_img, dilate_img);
	cv::Mat masked, color;
	cv::cvtColor(dilate_img, color, CV_GRAY2BGR);
	cv::bitwise_and(color, image, masked);
	
	cv::imwrite(img_out_dir + "_0_bgr.jpg", image);
	cv::imwrite(img_out_dir + "_1_gr_" + img_name + ".jpg", gr_img);
	cv::imwrite(img_out_dir + "_2_blur_" + img_name + ".jpg", blur_img);
	cv::imwrite(img_out_dir + "_3_gray_" + img_name + ".jpg",  gray_img);
	cv::imwrite(img_out_dir + "_4_thresh_" + img_name + ".jpg", thresh_img);
	cv::imwrite(img_out_dir + "_5_close_" + img_name + ".jpg",  close_img);
	cv::imwrite(img_out_dir + "_6_open_" + img_name + ".jpg",   open_img);
	cv::imwrite(img_out_dir + "_7_dilate_" + img_name + ".jpg", dilate_img);
	cv::imwrite(img_out_dir + "_8_masked_" + img_name + ".jpg", masked);
	
	int num_lesions = contours.size() - 1;
	fprintf(file, "# lesions: %d \n", num_lesions);
	
	return contours;
}
