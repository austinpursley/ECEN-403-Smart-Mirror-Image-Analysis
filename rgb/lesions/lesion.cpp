/*
Name: lesion.cpp
Date: 01/2018
Author: Austin Pursley
Course: ECEN 403, Senior Design Smart Mirror

Purpose: Implementation of Lesion class.
*/
#define _CRT_SECURE_NO_WARNINGS
#include "stdafx.h"
#include "lesion.hpp"

//constructor
Lesion::Lesion(std::vector<cv::Point> init_contour, const cv::Mat &mat, cv::Mat &mask, int id_num, const double roi_scale) {
	contour = init_contour;
	id = id_num;
	find_area();
	find_inertia_ratio();
	roi_size = ((mat.cols+mat.rows)/2)*roi_scale;
	find_roi(mat);
	find_colors(mat, mask);
	find_class();
}

//copy constructor
Lesion::Lesion(const Lesion& copy_from) {
	contour = copy_from.contour;
	id = copy_from.id;
	roi_size = copy_from.roi_size;
	roi = copy_from.roi;
	color = copy_from.color;
	bg_color = copy_from.bg_color;
	area = copy_from.area;
	inertia_ratio = copy_from.inertia_ratio;
	les_class = copy_from.les_class;
}

//copy assignment
Lesion& Lesion::operator=(const Lesion &copy_from) {
	contour = copy_from.contour;
	id = copy_from.id;
	roi_size = copy_from.roi_size;
	roi = copy_from.roi;
	color = copy_from.color;
	bg_color = copy_from.bg_color;
	area = copy_from.area;
	inertia_ratio = copy_from.inertia_ratio;
	les_class = copy_from.les_class;
	return *this;
}

///-------------------------------------------------------

//accessor functions
std::vector<cv::Point> Lesion::get_contour() const {
	return contour;
}
int Lesion::get_id() const {
	return id;
}
int Lesion::get_lesion_class() const {
	return les_class;
}
cv::Rect Lesion::get_roi() const {
	return roi;
}
cv::Scalar Lesion::get_color() const {
	return color;
}
cv::Scalar Lesion::get_bg_color() const {
	return bg_color;
}
double Lesion::get_area() const {
	return area;
}
double Lesion::get_inertia_ratio() const {
	return inertia_ratio;
}

///-------------------------------------------------------

//update lesions according to new contour
void Lesion::update(cv::Mat &mat, std::vector<cv::Point> new_contour, cv::Mat &mask) {
	contour = new_contour;
	//update other members according to new contour
	find_area();
	find_inertia_ratio();
	find_roi(mat);
	find_colors(mat, mask);
	find_class();
}

//draw a contour onto image mat
void Lesion::draw(cv::Mat &mat) {
	std::vector < std::vector<cv::Point>> contour_vec;
	contour_vec.push_back(get_contour());
	cv::drawContours(mat, contour_vec, 0, cv::Scalar(155, 255, 255), 1, 1);
}

///-------------------------------------------------------

void Lesion::find_area() {
	area = cv::contourArea(contour);
}

//adapted from blobdetector.cpp OpenCV source
//info on intertia ratio: https://www.learnopencv.com/blob-detection-using-opencv-python-c/
void Lesion::find_inertia_ratio() {
	cv::Moments moms = moments(cv::Mat(contour));
	double denominator = std::sqrt(std::pow(2 * moms.mu11, 2) + std::pow(moms.mu20 - moms.mu02, 2));
	const double eps = 1e-2;
	if (denominator > eps)
	{
		double cosmin = (moms.mu20 - moms.mu02) / denominator;
		double sinmin = 2 * moms.mu11 / denominator;
		double cosmax = -cosmin;
		double sinmax = -sinmin;

		double imin = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmin - moms.mu11 * sinmin;
		double imax = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmax - moms.mu11 * sinmax;
		inertia_ratio = imin / imax;
	}
	else
	{
		inertia_ratio = 1;
	}
}

void Lesion::find_roi(const cv::Mat &mat) {
	roi = cv::boundingRect(contour);
	cv::Size inflationSize(roi_size * 2, roi_size * 2);
	roi += inflationSize;
}

//find the average color of the area within the lesion contour and the local backgroun color
void Lesion::find_colors(const cv::Mat &img, const cv::Mat &mask) {	
	//pad image and mask so roi doesn't overbound
	cv::Mat pad_mask, not_mask;
	cv::bitwise_not(mask, not_mask);
	cv::copyMakeBorder(not_mask, pad_mask, roi_size, roi_size, roi_size, roi_size, cv::BORDER_CONSTANT, 0);
	cv::Mat pad_img;
	cv::copyMakeBorder(img, pad_img, roi_size, roi_size, roi_size, roi_size, cv::BORDER_REPLICATE, 0);

	//should make this a seperate function, filter lesion that do not pass this test
	if (roi.x >= 0 && roi.y >= 0 && roi.width + roi.x <= pad_img.cols && roi.height + roi.y <= pad_img.rows) {
		//vector because drawContours only works with vector of contours
		std::vector < std::vector<cv::Point>> contour_vec;
		contour_vec.push_back(contour);
		//lesion color
		cv::Mat single_mask = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
		cv::drawContours(single_mask, contour_vec, 0, cv::Scalar(255), -1);
		color = cv::mean(img, single_mask);
		//background color
		cv::Mat mat_roi = pad_img(roi);
		cv::Mat mat_roi_mask = pad_mask(roi);
		bg_color = cv::mean(mat_roi, mat_roi_mask);

		///OUTPUT / DEBUG
		//use one image at a time in input directory (not passing image name / id, yet)
		/*
		std::string img_out_dir = output_dir + "/lesion_class/";
		_mkdir(img_out_dir.c_str());
		img_out_dir = img_out_dir + std::to_string(img_id) + "/";
		_mkdir(img_out_dir.c_str());
		cv::Mat color, masked;
		cv::cvtColor(mat_roi_mask, color, CV_GRAY2BGR);
		cv::bitwise_and(color, mat_roi, masked);
		cv::imwrite(img_out_dir  + std::to_string(id) + "_0_les_roi" + ".jpg", mat_roi);
		cv::imwrite(img_out_dir + std::to_string(id) + "_1_les_mask" + ".jpg", mat_roi_mask);
		cv::imwrite(img_out_dir + std::to_string(id) + "_2_masked" + ".jpg", masked);
		*/
		///-------------- 
	}

	else {
		printf("SKIPPED LESION \n");
		/*///DEBUG
		printf("mat.cols: %d \n", mat.cols);
		printf("size: %f \n", roi_size);
		printf("roi x: %d \n", roi.x);
		printf("roi width: %d \n", roi.width);
		printf("roi x + width: %d \n", roi.width + roi.x);
		printf("pad_mat.cols: %d \n", pad_mat.cols);
		printf("roi y: %d \n", roi.y);
		printf("roi height: %d \n", roi.height);
		printf("roi y + height: %d \n", roi.height + roi.y);
		printf("pad_mat.rows: %d \n", pad_mat.rows);
		///---- */
	}
	return;
}

void Lesion::find_class() {
	std::string img_out_dir = output_dir + "/4_lesion_classify/" + std::to_string(Lesion::img_id) + "/";
	_mkdir(img_out_dir.c_str());
	FILE * pFile;
	std::string text_file = img_out_dir + std::to_string(Lesion::img_id) + "_color_data.txt";
	pFile = fopen(text_file.c_str(), "w");
	cv::Mat les_color_mat(64, 64, CV_32FC3);
	les_color_mat = color;

	cv::Mat les_hsv_mat(64, 64, CV_32FC3);
	cv::cvtColor(les_color_mat, les_hsv_mat, CV_RGB2HSV);
	cv::Vec3f hsv_color = les_hsv_mat.at<cv::Vec3f>(cv::Point(0, 0));

	cv::Mat les_bg_color_mat(64, 64, CV_32FC3);
	les_bg_color_mat = bg_color;

	cv::Mat les_bg_hsv_mat(64, 64, CV_32FC3);
	cv::cvtColor(les_bg_color_mat, les_bg_hsv_mat, CV_RGB2HSV);
	cv::Vec3f bg_hsv_color = les_bg_hsv_mat.at<cv::Vec3f>(cv::Point(0, 0));

	double perc_diff_hue = ((hsv_color[0] - bg_hsv_color[0]) / bg_hsv_color[0]) * 100;
	double perc_diff_sat = ((hsv_color[1] - bg_hsv_color[1]) / bg_hsv_color[1]) * 100;
	double perc_diff_val = ((hsv_color[2] - bg_hsv_color[2]) / bg_hsv_color[2]) * 100;
	//----------------------testing----------------------------
	/*
	fprintf(pFile, "------lesion%d------ \n", (i));
	fprintf(pFile, "lesion%d size is %.2f \n", (i), lesions[i].get_area());
	cv::imwrite(img_out_dir + "les_color_" + std::to_string(i) + ".jpg", les_bg_color_mat);
	cv::imwrite(img_out_dir + "les_color_" + std::to_string(i) + ".jpg", les_color_mat);

	fprintf(pFile, "hue:     %f \n", hsv_color[0]);
	fprintf(pFile, "satur:   %f \n", hsv_color[1]);
	fprintf(pFile, "value:   %f \n", hsv_color[2]);
	fprintf(pFile, "perc_diff_hue: %f \n", perc_diff_hue);
	fprintf(pFile, "perc_diff_sat: %f \n", perc_diff_sat);
	fprintf(pFile, "perc_diff_val: %f \n\n", perc_diff_val);
	*/
	//--------------------testing-----------------------------
	if (perc_diff_sat > 12.0 || perc_diff_val > 25.0) {
		if ((perc_diff_hue > 3.25)) {
			if (perc_diff_val < 10.0) {
				//red
				les_class = 1;
			}
			else if ((perc_diff_sat < 25.0) && (perc_diff_val < 30.0)) {
				//red
				les_class = 1;
			}
			else {
				//dark
				les_class = 0;
			}
		}
		else {
			if (perc_diff_val < 5.0) {
				//red
				les_class = 1;
			}
			else {
				//dark
				les_class = 0;
			}
		}
	}
	else {
		//lesion is similar to skin (but we'll say it's dark for now)
		les_class = 0;
	}
}