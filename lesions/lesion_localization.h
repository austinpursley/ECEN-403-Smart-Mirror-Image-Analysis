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

std::vector<std::vector<cv::Point>> lesion_localization(const cv::Mat & image, int type = 0, std::string img_name = "") {

	cv::Mat mix_img;
	if (type == 0) {
		//by value, all dark lesions on light background
		cv::Mat3b lab_img(image.rows, image.cols, CV_8UC3);
		cv::cvtColor(image, lab_img, CV_BGR2Lab, 3);
		std::vector<cv::Mat1b> lab(3);
		cv::split(lab_img, lab);
		cv::Mat gr_img = image & cv::Scalar(0, 255, 255);
		gr_img = image & cv::Scalar(0, 255, 255);
		lab_img = cv::Mat(image.rows, image.cols, CV_8UC3);
		cv::cvtColor(gr_img, lab_img, CV_BGR2Lab, 3);
		std::vector<cv::Mat1b> lab(3);
		cv::split(lab_img, lab);
		cv::Mat AB;
		cv::addWeighted(lab[1], 0.5, lab[2], 0.5, 0, AB);
		cv::addWeighted(AB, 0.6, lab[0], 0.4, 0, mix_img);
	}
	else if (type == 1) {
		/*
		//redder lesions
		cv::Mat3b yrb_img(image.rows, image.cols, CV_8UC3);
		cv::cvtColor(gr_img, yrb_img, CV_BGR2YCrCb, 3);
		std::vector<cv::Mat1b> yrb(3);
		cv::split(yrb_img, yrb);
		cv::bitwise_not(yrb[1], yrb[1]);
		cv::bitwise_not(lab[0], lab[0]);
		cv::addWeighted(yrb[1], 0.8, lab[0], 0.2, 0, red_mix);
		///
		*/
	}
	else if (type == 2) {
		/*
		cv::Mat3b hsv_img(image.rows, image.cols, CV_8UC3);
		cv::cvtColor(image, hsv_img, CV_BGR2HSV, 3);
		std::vector<cv::Mat1b> hsv(3);
		cv::split(hsv_img, hsv);
		*/
	}
	else {
		printf("defaulted case! \n");
	}

	//blob detection on our mixed, single channel image
	cv::Mat bin_mask;
	std::vector<std::vector<cv::Point>> les_contours;
	blob_detect(mix_img, bin_mask, les_contours);

	//Lesion class, stores properties of lesion like color and area
	std::vector<Lesion> lesions;
	for (int i = 0; i < les_contours.size(); i++) {
		int id_num = i;
		lesions.push_back(Lesion (les_contours[i], image, bin_mask, id_num));
		/*
		std::string img_out_dir = output_dir + "/lesion_class/";
		cv::Mat show = image.clone();
		lesions[i].draw(show);
		cv::imwrite(img_out_dir + std::to_string(i) + "_0les" + ".jpg", show);
		*/
	}

	///DEBUG -------------------------------------------------
	std::string img_out_dir = output_dir + "/lesion_class/";
	//_mkdir(img_out_dir.c_str());
	//img_out_dir = img_out_dir + img_name + "/";
	//_mkdir(img_out_dir.c_str());

	/*
	FILE * file;
	std::string out_file = img_out_dir + "/num_of_lesions.txt";
	file = fopen(out_file.c_str(), "w");
	*/

	cv::Mat color, masked;
	//cv::bitwise_not(morph_comb, morph_comb);
	cv::cvtColor(bin_mask, color, CV_GRAY2BGR);
	cv::bitwise_and(color, image, masked);
	cv::imwrite(img_out_dir + img_name + "_0_image_" + ".jpg", image);
	cv::imwrite(img_out_dir + img_name + "_0_bin_mask_" + ".jpg", bin_mask);
	cv::imwrite(img_out_dir + img_name + "_3_masked_" + ".jpg", masked);

	//int num_lesions = contours.size() - 1;
	//fprintf(file, "# lesions: %d \n", num_lesions);
	///--------------------------------------------------------

	return les_contours;
}

void blob_detect(const cv::Mat1b & src_1b, const cv::Mat1b & bin_mask, std::vector<std::vector<cv::Point>> contours_output) {
	///VARIABLES / SETTINGS
	//mix tuning/performance parameters in one place.

	int gauss_ksize = 9;
	int blocksize = 39;
	int size_close = 1;
	int size_open = 3;
	//int size_close2 = 3;
	//int size_open2 = 3;
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
	cv::Mat elem_close = cv::getStructuringElement(shape,
		cv::Size(2 * size_close + 1, 2 * size_close + 1),
		cv::Point(size_close, size_close));
	cv::Mat elem_open = cv::getStructuringElement(shape,
		cv::Size(2 * size_open + 1, 2 * size_open + 1),
		cv::Point(size_open, size_open));
	/*
	cv::Mat elem_open2 = cv::getStructuringElement(shape,
		cv::Size(2 * size_open2 + 1, 2 * size_open2 + 1),
		cv::Point(size_open2, size_open2));
	cv::Mat elem_close2 = cv::getStructuringElement(shape,
		cv::Size(2 * size_close2 + 1, 2 * size_close2 + 1),
		cv::Point(size_close2, size_close2));
	*/

	cv::Mat elem_erode = cv::getStructuringElement(shape,
		cv::Size(2 * size_erode + 1, 2 * size_erode + 1),
		cv::Point(size_erode, size_erode));
	cv::Mat gr_img, lab_img, mix_img, gray_img, blur_img, bin_img, close_img, open_img, morph;
	std::vector<std::vector<cv::Point>> contours;
	
	///PROCESS
	/*
	1: guassian blur filter to reduce image noise
	2: adaptive thresholding, binarization
	3: close to fill in gaps
	4: open removes smaller blobs
	5: dilate what's left to make them more prominent
	6: find contours, the points that make up border of area on the original image
	*/

	cv::GaussianBlur(src_1b, blur_img, ksize, 0);
	cv::adaptiveThreshold(blur_img, bin_img, maxValue, adaptMethod, thresholdType, blocksize, 2);
	cv::morphologyEx(bin_img, close_img, cv::MORPH_CLOSE, elem_close);
	cv::morphologyEx(close_img, open_img, cv::MORPH_OPEN, elem_open);
	cv::morphologyEx(open_img, bin_mask, cv::MORPH_ERODE, elem_erode);
	cv::findContours(bin_mask, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	///OUTPUT / DEBUG
	/*
	std::string img_out_dir = output_dir + "/blob_detect/";
	cv::imwrite(img_out_dir + img_name + "_1_src1b_" + ".jpg", src_1b);
	cv::imwrite(img_out_dir + img_name + "_2_blur_" + ".jpg", blur_img);
	cv::imwrite(img_out_dir + img_name + "_3_thresh_" +  ".jpg", thresh_img);
	cv::imwrite(img_out_dir + img_name + "_4_morph_" + ".jpg",   morph);
	
	*/
	return;
}