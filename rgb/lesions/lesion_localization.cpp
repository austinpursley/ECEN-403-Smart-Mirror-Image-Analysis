/*
Name: lesion_localization.cpp
Date: 01/2018
Author: Austin Pursley
Course: ECEN 403, Senior Design Smart Mirror

Purpose: Image processing functions to find skin lesions.
*/

#include "stdafx.h"
#include "lesion_localization.hpp"

void mask_image(const cv::Mat &mask, cv::Mat &masked_out) {
	cv::Mat mask_copy;
	mask.copyTo(mask_copy);
	cv::Mat color;
	cv::bitwise_not(mask_copy, mask_copy);
	cv::cvtColor(mask_copy, color, CV_GRAY2BGR);
	cv::bitwise_and(color, masked_out, masked_out);
}

void lesion_draw_contours(const std::vector<Lesion > &lesions, cv::Mat &img) {
	std::vector<std::vector<cv::Point> > lesion_contours;
	for (int i = 0; i < lesions.size(); i++) {
		lesion_contours.push_back(lesions[i].get_contour());
	}
	cv::drawContours(img, lesion_contours, -1, cv::Scalar(255), -1);
}

//Find "blobs" of the an image, dark spots on lighter background.
//to do: some sort of struct for parameters, and make it an input.
void blob_detect(const cv::Mat1b &src_1b, cv::Mat1b &bin_mask, std::vector<std::vector<cv::Point>> &contours_output) {
	///VARIABLES / SETTINGS
	//mix tuning/performance parameters in one place.
	int gauss_ksize = 9;
	int blocksize = 49;
	int size_close = 1;
	int size_open = 3;
	int size_close2 = 0;
	int size_open2 = 0;
	int size_erode = 1;
	//guassian blur
	cv::Size ksize;
	ksize.height = gauss_ksize;
	ksize.width = ksize.height;
	//morphology
	int shape = cv::MORPH_ELLIPSE;
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
	cv::Mat gr_img, lab_img, mix_img, gray_img, blur_img, bin_img, close_img, open_img;
	///PROCESS
	/*
	1: guassian blur filter to reduce image noise and detail
	2: adaptive thresholding, binarization
	3: close to fill in gaps
	4: open removes smaller blobs
	5: erosion to make blobs smaller, fit better
	6: find contours, the points that make up border of area on the original image
	*/
	cv::GaussianBlur(src_1b, blur_img, ksize, 0);
	cv::adaptiveThreshold(blur_img, bin_img, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, blocksize, 2);
	cv::morphologyEx(bin_img, close_img, cv::MORPH_CLOSE, elem_close);
	cv::morphologyEx(close_img, open_img, cv::MORPH_OPEN, elem_open);
	cv::morphologyEx(open_img, close_img, cv::MORPH_CLOSE, elem_close2);
	cv::morphologyEx(close_img, open_img, cv::MORPH_OPEN, elem_open2);
	cv::morphologyEx(open_img, bin_mask, cv::MORPH_ERODE, elem_erode);
	cv::findContours(bin_mask, contours_output, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	
	///OUTPUT / DEBUG
	
	std::string img_out_dir = output_dir + "/3_lesion_localization/";
	_mkdir(img_out_dir.c_str());
	img_out_dir = img_out_dir + "/blob_detect/";
	_mkdir(img_out_dir.c_str());
	/*
	cv::imwrite(img_out_dir + std::to_string(Lesion::img_id) + "_1_src1b_" + ".jpg", src_1b);
	cv::imwrite(img_out_dir + std::to_string(Lesion::img_id) + "_2_blur_" + ".jpg", blur_img);
	cv::imwrite(img_out_dir + std::to_string(Lesion::img_id) + "_3_thresh_" +  ".jpg", bin_img);
	*////--------------- */
	cv::imwrite(img_out_dir + std::to_string(Lesion::img_id) + "_4_morph_" + ".jpg",   bin_mask);
	return;
}

//Filter a vector of Lesion objects by how big or small their contours are.
void lesion_area_filter(std::vector<Lesion > &lesions, const double min_area, const double max_area) {
	std::vector<Lesion> new_lesions;
	for (int i = 0; i < lesions.size(); i++) {
		double crnt_les_area = lesions[i].get_area();
		if (crnt_les_area <= min_area || crnt_les_area >= max_area) {
			continue;
		}
		else {
			new_lesions.push_back(lesions[i]);
		}
	}
	lesions = new_lesions;
}

//Filter a vector of Lesion objects by how elongated they are.
void lesion_intertia_filter(std::vector<Lesion > &lesions, const double min_inertia_ratio, const double max_inertia_ratio) {
	std::vector<Lesion> new_lesions;
	for (int i = 0; i < lesions.size(); i++) {
		double ratio = lesions[i].get_inertia_ratio();
		if (ratio <= min_inertia_ratio || ratio >= max_inertia_ratio) {
			continue;
		}
		else {
			new_lesions.push_back(lesions[i]);
		}
	}
	lesions = new_lesions;
}

//Filter a vector of Lesion objects by their color similarity  to background
/*
void lesion_color_filter(std::vector<Lesion > &lesions) {
	std::vector<Lesion> new_lesions;
	for (int i = 0; i < lesions.size(); i++) {
		cv::Scalar color = lesions[i].get_color();
		cv::Scalar bg_color = lesions[i].get_bg_color();
		double ratio = lesions[i].get_inertia_ratio();
		if () {
			continue;
		}
		else {
			new_lesions.push_back(lesions[i]);
		}
	}
	lesions = new_lesions;
}
*/
//Function to find lesions spots from an image of skin.
void lesion_localization(const cv::Mat &image, std::vector<Lesion> &lesions, int type) {
	cv::Mat mix_img;
	if (type == 1) {
		cv::Mat lab_img = cv::Mat(image.rows, image.cols, CV_8UC3);
		std::vector<cv::Mat1b> lab(3);
		cv::cvtColor(image, lab_img, CV_BGR2Lab, 3);
		cv::split(lab_img, lab);
		bitwise_not(lab[2], lab[2]);
		bitwise_not(lab[1], lab[1]);
		cv::addWeighted(lab[1], 0.95, lab[2], 0.05, 0, mix_img);
	}
	else {
		//defualt, dark lesions on light background
		cv::Mat lab_img = cv::Mat(image.rows, image.cols, CV_8UC3);
		std::vector<cv::Mat1b> lab(3);
		cv::cvtColor(image, lab_img, CV_BGR2Lab, 3);
		cv::split(lab_img, lab);
		cv::Mat AB;
		cv::addWeighted(lab[1], 0.5, lab[2], 0.5, 0, AB);
		cv::addWeighted(AB, 0, lab[0], 1, 0, mix_img);
	}

	//blob detection on our mixed, single channel image
	cv::Mat1b bin_mask;
	std::vector<std::vector<cv::Point>> les_contours;
	blob_detect(mix_img, bin_mask, les_contours);

	//Lesion class, stores properties of lesion like color and area
	for (int i = 0; i < les_contours.size(); i++) {
		int id_num = i;
		lesions.push_back(Lesion(les_contours[i], image, bin_mask, id_num));
		
		///OUTPUT / DEBUG: each individual lesion 
		/*
		std::string img_out_dir = output_dir + "/3_lesion_localization/";
		_mkdir(img_out_dir.c_str());
		img_out_dir = img_out_dir + "/lesions/";
		_mkdir(img_out_dir.c_str());
		img_out_dir = img_out_dir + std::to_string(Lesion::img_id) + "/";
		_mkdir(img_out_dir.c_str());
		cv::Mat show = image.clone();
		lesions[i].draw(show);
		cv::imwrite(img_out_dir + std::to_string(i) + "_les" + ".jpg", show);
		*/
		///--------------- 
	}
	cv::Mat drawn_lesions = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
	//double min_area = std::sqrt(image.rows * image.cols)*0.04;
	double min_area = 5;
	double max_area = 300;
	lesion_area_filter(lesions, min_area, max_area);
	lesion_intertia_filter(lesions, 0.04);
	lesion_draw_contours(lesions, drawn_lesions);
	
	///OUTPUT / DEBUG
	std::string img_out_dir = output_dir + "/3_lesion_localization/";
	_mkdir(img_out_dir.c_str());
	cv::Mat masked, filter_masked;
	image.copyTo(masked);
	mask_image(bin_mask, masked);
	image.copyTo(filter_masked);
	mask_image(drawn_lesions, filter_masked);
	cv::imwrite(img_out_dir + std::to_string(Lesion::img_id) + "_0_image_" + ".jpg", image);
	cv::imwrite(img_out_dir + std::to_string(Lesion::img_id) + "_1_masked" + ".jpg", masked);
	cv::imwrite(img_out_dir + std::to_string(Lesion::img_id) + "_2_filtered_les" + ".jpg", filter_masked);
	///---------------
}