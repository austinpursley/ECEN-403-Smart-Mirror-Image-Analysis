#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "lesions.h"
#include "stdafx.h"

std::vector<cv::Scalar> lesion_colors(const cv::Mat & image, const std::vector<std::vector<cv::Point> > & contours, std::string img_name) {
	std::vector<cv::Scalar> contour_colors;
	cv::Scalar color;
	cv::Mat mask;
	//calculate the mean color of entire image
	cv::Scalar mean_color = cv::mean(image);
	contour_colors.push_back(mean_color);

	for (int i = 0; i < contours.size(); i++) {
		//mask is a black image
		mask = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
		//on black background, draw white contour area corresponding to lesion
		cv::drawContours(mask, contours, i, cv::Scalar(255), -1);
		//mask means we are only considering white contour section
		color = cv::mean(image, mask);
		contour_colors.push_back(color);

		///------------------ PRESENTATION / DEBUG ---------------
		std::string img_out_dir = output_dir + "/classification/";
		_mkdir(img_out_dir.c_str());
		img_out_dir = img_out_dir + img_name + "/";
		_mkdir(img_out_dir.c_str());

		cv::Mat show = image.clone();
		cv::drawContours(show, contours, i, cv::Scalar(255), 1, 1);
		cv::imwrite(img_out_dir  + std::to_string(i) + "_les" +  ".jpg", show);
		///--------------------------------------------------------
	}

	return(contour_colors);
}

 std::vector<double> lesion_areas(const std::vector<std::vector<cv::Point> > & contours) {
	 //each lesion has an associated contour and here we calculate the area of contours
	 std::vector<double> contour_areas;
	 double area;
	 for (int i = 0; i < contours.size(); i++) {
		 area = cv::contourArea(contours[i]);
		 contour_areas.push_back(area);
	 }
	 return(contour_areas);
 }