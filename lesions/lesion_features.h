#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "lesions.h"
#include "lesion_entropy_detect.h"
#include "stdafx.h"

///to do: get rid of return (more like other opencv function). Use this to get local mean color as well (instead of global mean)
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

//Input array of contours/lesions, grayscale. Output array of entropy values corresponding to lesions, as well as image of entropy image.
void lesion_entropies(const cv::Mat &gray, const std::vector<std::vector<cv::Point> > &contours, std::vector<double> &lesion_entrops, cv::Mat &entropy_mat) {
	cv::Mat g(gray);
	cv::Mat dst = cv::Mat::zeros(g.rows, g.cols, CV_32F);

	entropy_filter(g, dst);
	cv::normalize(dst, dst, 0.0, 255.0, cv::NORM_MINMAX, CV_32FC1);
	dst.convertTo(entropy_mat, CV_8U);
	
	//calculate the mean entropy of entire image
	//this is first element of output vector
	double mean_entropy = cv::mean(entropy_mat).val[0];
	lesion_entrops.push_back(mean_entropy);

	for (int i = 0; i < contours.size(); i++) {
		//mask is a black image
		cv::Mat mask = cv::Mat::zeros(entropy_mat.rows, entropy_mat.cols, CV_8UC1);
		//on black background, draw white contour area corresponding to lesion
		cv::drawContours(mask, contours, i, cv::Scalar(255), -1);
		//mask means we are only considering white contour section
		double les_entropy = cv::mean(entropy_mat, mask).val[0];
		
		lesion_entrops.push_back(les_entropy);

		///------------------ PRESENTATION / DEBUG ---------------
		//printf("les_entropy: %f \n", lesion_entrops[i]);
		std::string img_out_dir = output_dir + "/lesion_entropies/";
		_mkdir(img_out_dir.c_str());
		//img_out_dir = img_out_dir + img_name + "/";
		img_out_dir = img_out_dir + "/";
		_mkdir(img_out_dir.c_str());

		cv::Mat show;
		entropy_mat.copyTo(show);
		cv::drawContours(show, contours, i, cv::Scalar(0), 1, 1);
		cv::imwrite(img_out_dir + std::to_string(i) + "_les" + ".jpg", show);
		
		///--------------------------------------------------------
	}
	return;
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


