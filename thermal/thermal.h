/*
	Name: thermal.h
	Date: 12/2017
	Author: Austin Pursley
	Course: ECEN 403, Senior Design Smart Mirror
*/


#include <fstream>
#include "stdafx.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

/*
	Convert text file with matrix of int values to an OpenCV Mat
	txt_file is the name of the txt_file that contain data
	width and height are the size of the matrix inside the text file.

	Specific purpose is to convert thermal imaging data given by Flir 
	Lepton so that it can be analyzed by OpenCV.
*/
cv::Mat txt_to_mat(std::string txt_file, int width, int height) {
	//The data in txt will be integer values, hence CV_32SC1
	cv::Mat mat(height, width, CV_32S);
	std::ifstream ifs(txt_file);
	std::string row;
	int tempint;
	int i = 0;
	int j = 0;
	while (std::getline(ifs, row)) {
		j = 0;
		std::istringstream iss(row);
		std::vector<int> tempv;
		while (iss >> tempint) {
			mat.at<int>(i, j) = tempint;
			j++;
		}
		i++;
	}
	return mat;
}

/*
	Finds the median value of an OpenCV mat (Input)
	nVals is the max value of pixel in Matrix. e.g. in CV_8U it's 256.

	Specific purpose here is find a threshold value for thermal image analysis.
	SOURCE: stackoverflow.com/a/30093862
*/

double medianMat(cv::Mat Input, int nVals) {
	// COMPUTE HISTOGRAM OF SINGLE CHANNEL MATRIX
	float range[] = { 0, nVals };
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;
	cv::Mat hist;
	calcHist(&Input, 1, 0, cv::Mat(), hist, 1, &nVals, &histRange, uniform, 
		accumulate);

	// COMPUTE CUMULATIVE DISTRIBUTION FUNCTION (CDF)
	cv::Mat cdf;
	hist.copyTo(cdf);
	for (int i = 1; i <= nVals - 1; i++) {
		cdf.at<float>(i) += cdf.at<float>(i - 1);
	}
	cdf /= Input.total();

	// COMPUTE MEDIAN
	double medianVal;
	for (int i = 0; i <= nVals - 1; i++) {
		if (cdf.at<float>(i) >= 0.5) { medianVal = i;  break; }
	}
	return medianVal;
}
