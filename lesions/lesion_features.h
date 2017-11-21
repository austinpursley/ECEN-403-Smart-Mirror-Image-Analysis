#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "stdafx.h"

std::vector<cv::Scalar> lesion_colors(const cv::Mat & image, const std::vector<std::vector<cv::Point> > & contours) {
	std::vector<cv::Scalar> contour_colors;
	cv::Scalar color;
	cv::Mat mask;

	for (int i = 0; i < contours.size(); i++) {
		//clear mask (set it all black)
		mask = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
		//on black background, draw white contour area corresponding to lesion
		cv::drawContours(mask, contours, i, cv::Scalar(255), -1);
		//calculate mean lesion color
		//mask means we are only considering white contour section
		color = cv::mean(image, mask);

		//------------------for output testing delete later ---------------
		std::string image_out("C:/Users/Austin Pursley/Desktop/ECEN-403-Smart-Mirror-Image-Analysis/data/lesions/output/");
		cv::Mat show = image.clone();
		cv::drawContours(show, contours, i, cv::Scalar(255), 1, 1);
		cv::imwrite(image_out + "les_" + std::to_string(i) + ".jpg", show);
		//---------------------delete---------------------------------------
		

		contour_colors.push_back(color);
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