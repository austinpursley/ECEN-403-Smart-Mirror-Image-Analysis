/*
Name: main.cpp
Date: 12/2017
Author: Austin Pursley
Course: ECEN 403, Senior Design Smart Mirror

Purpose: Uses computer vision to analyze thermal images of someone
to get a skin temperature metric.
*/

#include "stdafx.h"
#include "thermal.h"

int main() {
	//location of text file with thermal imaging data matrix
	std::string dir = "C:/Users/Austin Pursley/Desktop/ECEN-Senior-Design-Smart-Mirror-Image-Processing/thermal/data/";
	std::string file = dir + "input/snapshotdata1.txt";

	//Mat to hold thermal image pixel values
	int thermal_width = 80;  // FLiR Lepton thermal image size: 80x60
	int thermal_height = 60;
	cv::Mat thermal(thermal_width, thermal_height, CV_32SC1);
	//convert text file to thermal image
	thermal = txt_to_mat(file, thermal_width, thermal_height);
	cv::Mat thermal_show(60, 80, CV_8UC1);
	cv::normalize(thermal, thermal_show, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::imshow("Thermal Image", thermal_show);
	//image processing function to get temperature metric
	double skin_temp = temp_from_thermal_img(thermal);
	skin_temp = 0.0487*skin_temp - 312.98;
	printf("skin temp: %f \n", skin_temp);
	
	cv::waitKey();

}