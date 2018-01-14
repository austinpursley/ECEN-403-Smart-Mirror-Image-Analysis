#include "stdafx.h"
#include "lesions.hpp"

//constructor
Lesion::Lesion( std::vector<cv::Point> init_contour, const cv::Mat &mat, cv::Mat &mask, int id_num, const double roi_scale) {
	contour = init_contour;
	id = id_num;
	find_area();
	roi_size = ((mat.cols+mat.rows)/2)*roi_scale;
	find_roi(mat);
	find_colors(mat, mask);
	
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
	find_roi(mat);
	find_colors(mat, mask);
	
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

void Lesion::find_inertia_ratio() {
		cv::Moments moms = moments(cv::Mat(contour));
		double denominator = std::sqrt(std::pow(2 * moms.mu11, 2) + std::pow(moms.mu20 - moms.mu02, 2));
		const double eps = 1e-2;
		double ratio;
		if (denominator > eps)
		{
			double cosmin = (moms.mu20 - moms.mu02) / denominator;
			double sinmin = 2 * moms.mu11 / denominator;
			double cosmax = -cosmin;
			double sinmax = -sinmin;

			double imin = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmin - moms.mu11 * sinmin;
			double imax = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmax - moms.mu11 * sinmax;
			ratio = imin / imax;
		}
		else
		{
			ratio = 1;
		}

		inertia_ratio = ratio;
}

void Lesion::find_roi(const cv::Mat &mat) {
	roi = cv::boundingRect(contour);
	cv::Size inflationSize(roi_size * 2, roi_size * 2);
	roi += inflationSize;
}

//find the average color of the area within the lesion contour and the local backgroun color
void Lesion::find_colors(const cv::Mat &mat, const cv::Mat &mask) {
	//calculate the mean color of entire image
	//int size = mat.cols*roi_scale;
	
	//mask for getting local background color
	cv::Mat pad_mask, not_mask;
	cv::bitwise_not(mask, not_mask);
	cv::copyMakeBorder(not_mask, pad_mask, roi_size, roi_size, roi_size, roi_size, cv::BORDER_CONSTANT, 0);

	cv::Mat pad_mat;
	//cv::Rect cnt_roi = cv::boundingRect(contour);
	cv::copyMakeBorder(mat, pad_mat, roi_size, roi_size, roi_size, roi_size, cv::BORDER_REPLICATE, 0);
	//cv::Size inflationSize(size * 2, size * 2);
	//cnt_roi += inflationSize;

	//should make this a seperate function, filter lesion that do not pass this test
	if (roi.x >= 0 && roi.y >= 0 && roi.width + roi.x <= pad_mat.cols && roi.height + roi.y <= pad_mat.rows) {
		//vector because drawContours only works with vector of contours
		std::vector < std::vector<cv::Point>> contour_vec;
		contour_vec.push_back(contour);
		//lesion color
		cv::Mat single_mask = cv::Mat::zeros(mat.rows, mat.cols, CV_8UC1);
		cv::drawContours(single_mask, contour_vec, 0, cv::Scalar(255), -1);
		color = cv::mean(mat, single_mask);
		//background color
		cv::Mat mat_roi = pad_mat(roi);
		cv::Mat mat_roi_mask = pad_mask(roi);
		bg_color = cv::mean(mat_roi, mat_roi_mask);

		///DEBUG
		/*
		std::string output_dir("C:/Users/Austin Pursley/Desktop/ECEN-403-Smart-Mirror-Image-Analysis/lesions/output/lesion_class/");
		cv::Mat color, masked;
		cv::cvtColor(mat_roi_mask, color, CV_GRAY2BGR);
		cv::bitwise_and(color, mat_roi, masked);
		cv::imwrite(output_dir  + std::to_string(id) + "_1les_roi" + ".jpg", mat_roi);
		//cv::imwrite(output_dir + std::to_string(id) + "_les_mask" + ".jpg", mat_roi_mask);
		cv::imwrite(output_dir + std::to_string(id) + "_2masked" + ".jpg", masked);
		*/
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