#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "lesions.h"
#include "lesion_entropy_detect.h"
#include "stdafx.h"

std::vector<cv::Scalar> lesion_colors(const cv::Mat & image, const std::vector<std::vector<cv::Point> > & contours) {
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
		//img_out_dir = img_out_dir + img_name + "/";
		img_out_dir = img_out_dir + "/";
		_mkdir(img_out_dir.c_str());

		cv::Mat show = image.clone();
		cv::drawContours(show, contours, i, cv::Scalar(255), 1, 1);
		cv::imwrite(img_out_dir  + std::to_string(i) + "_les" +  ".jpg", show);
		
		///--------------------------------------------------------
	}

	return(contour_colors);
}

std::vector<cv::Scalar> lesion_entropies(const cv::Mat1b & gray, const std::vector<std::vector<cv::Point> > & contours) {
	cv::Mat g, entropy;
	gray.copyTo(g);
	cv::Rect roi(0, 0, g.cols, g.rows);
	cv::Mat dst = cv::Mat::zeros(g.rows, g.cols, CV_32F);
	getLocalEntropyImage(g, roi, dst);
	cv::normalize(dst, dst, 0.0, 255.0, cv::NORM_MINMAX, CV_32FC1);
	dst.convertTo(entropy, CV_8U);
	
	
	std::vector<cv::Scalar> contour_entrops;
	cv::Scalar les_entropy;
	cv::Mat mask;
	//calculate the mean color of entire image
	cv::Scalar mean_entropy = cv::mean(entropy);
	contour_entrops.push_back(mean_entropy);

	for (int i = 0; i < contours.size(); i++) {
		//mask is a black image
		mask = cv::Mat::zeros(entropy.rows, entropy.cols, CV_8UC1);
		//on black background, draw white contour area corresponding to lesion
		cv::drawContours(mask, contours, i, cv::Scalar(255), -1);
		//mask means we are only considering white contour section
		les_entropy = cv::mean(entropy, mask);
		printf("les_entropy: %f \n", les_entropy.val[0]);
		contour_entrops.push_back(les_entropy);

		///------------------ PRESENTATION / DEBUG ---------------

		std::string img_out_dir = output_dir + "/classification/";
		_mkdir(img_out_dir.c_str());
		//img_out_dir = img_out_dir + img_name + "/";
		img_out_dir = img_out_dir + "/";
		_mkdir(img_out_dir.c_str());

		cv::Mat show = entropy.clone();
		cv::drawContours(show, contours, i, cv::Scalar(255), 1, 1);
		cv::imwrite(img_out_dir + std::to_string(i) + "_les" + ".jpg", show);

		///--------------------------------------------------------
	}

	return(contour_entrops);
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


