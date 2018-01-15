/*
Name: thermal.h
Date: 01/2018
@author Austin Pursley
Course: ECEN 403, Senior Design Smart Mirror

Purpose: header file for thermal image processing functions. Document how to use.
*/

#include <fstream>
#include "stdafx.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
/** @brief Convert text file with matrix of int values to an OpenCV Mat.

@param txt_file Name or direcotory of text file with matrix of numbers.
@param width Width of number matrix in text file.
@param height Height of number matrix in text file.

Specific purpose is to convert thermal imaging data given by Flir
Lepton so that it can be analyzed by OpenCV.
 */
cv::Mat txt_to_mat(std::string txt_file, const int width, const int height);

/** @brief Finds the median value of an OpenCV Mat.

@param mat_in Input image / OpenCV Mat object.
@param nVals Max value of pixel in Mat e.g. in CV_8U it's 256.
@return medianVal The median pixel value of image / Mat.

Specific purpose here is find a threshold value for thermal image analysis.
 */
double medianMat(cv::Mat mat_in, const int nVals);

/** @brief Takes a thermal image of a person and finds the median
pixel value of the a region of interest that corresponds to the person.

@param thermal OpenCV Mat of the thermal image (grayscale).
@param nVals Max value of pixel in Matrix. e.g. in CV_8U it's 256.
@return skin_temp A double value skin temperature metric, value from mat_in.

The purpose is to extract an approximate skin temperature metric.
 */
double temp_from_thermal_img(const cv::Mat &thermal);