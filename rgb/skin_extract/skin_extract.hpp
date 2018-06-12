/*
Name: skin_extract.hpp
Date: 06/2018
Author: Austin Pursley
Course: ECEN 403/404, Senior Design Smart Mirror

Purpose: Header, documentation for skin extraction functions.
*/

#define NOMINMAX
#include "stdafx.h"
#include "../my_in_out_directory.hpp"
#include <opencv2/opencv.hpp>

/** @brief Cover up lips and eyes of a face with black ellipses.

@param face An image of a face.
@param features Vector of extracted face features associated with the image.
@param masked_face Output image of face with eyes and lips covered with black ellipses.

Purpose is to remove the eyes and lips as part of skin extraction of a face.
*/
void mask_features(cv::Mat face, std::map<std::string, cv::Rect> features, cv::Mat& masked_face);

/** @brief Apply some morphological operations to binary skin mask.

@param mask Extracted binary skin mask.

Purpose clean up the extracted skin area, which might contain holes or outlying blobs.
*/
void skin_morph(cv::Mat& mask);

/** @brief Determine skin area based on known face color.

@param img An image of a person or face.
@param features Vector of extracted face features associated with the image.
@param skin_mask Output a binary image where the white area corresponds to skin.

Purpose is to determine which part of an image of a person is skin.
*/
void extract_skin(cv::Mat img, std::map<std::string, cv::Rect> features, cv::Mat& skin_mask);