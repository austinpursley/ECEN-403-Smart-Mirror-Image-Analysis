/*
Name: lesion_localization.hpp
Date: 01/2018
Author: Austin Pursley
Course: ECEN 403, Senior Design Smart Mirror

Purpose: Header, documentation for lesion localization functions.
*/

#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "my_in_out_directory.hpp"
#include "lesion.hpp"

void mask_image(const cv::Mat &mask, cv::Mat &masked_out);
void lesion_draw_contours(const std::vector<Lesion > &lesions, cv::Mat &img);

/** @brief Find "blobs" of the an image, dark spots on lighter background.

@param src_1b A single 8-bit channel OpenCV Mat.
@param bin_mask Output binary image mask.
@param contours_output Output vector of blob contours.

Purpose here is that the blobs we care about are lesions and the grayscale
input image is going to be from various single or mixed channels to get
different types of lesions.
*/
void blob_detect(const cv::Mat1b &src_1b, cv::Mat1b &bin_mask, std::vector<std::vector<cv::Point>> &contours_output);

/** @brief Filter a vector of Lesion objects by how big or small their contours are.

@param lesions The vector of Lesion objects.
@param min_area Minimum size that lesion contour can be.
@param max_area Maxiumum size that lesion contour can be.

Purpose is get ride of lesions that are too small or too big. 
*/
void lesion_area_filter(std::vector<Lesion> &lesions, const double min_area = 25, const double max_area = 5000);

/** @brief Filter a vector of Lesion objects by how elongated they are.

@param lesions The vector of Lesion objects.
@param min_inertia_ratio Minimum size that the intertia ratio of the lesion contour can be.
@param max_inertia_ratio Maximum size that the intertia ratio of the lesion contour can be.

The purpose is to filter out lesions that are too long or look too much like lines.
The more elongated a contour is, the closer to 0 it's intertia ratio will be.
*/
void lesion_intertia_filter(std::vector<Lesion> &lesions, const double min_inertia_ratio = 0.1f, const double max_inertia_ratio = 100000);

/** @brief Function to find lesions spots from an image of skin.

@param image The input image of skin. Assumes skin is only thing in image.
@param type Setting for what type of lesion want to locate. Coming soon.
@return contours Contours of lesions (std::vector<std::vector<cv::Point> >).

The purpose is find skin lesions, those being area of skin that are discolored and stand out, such as acne or moles.
*/
std::vector<std::vector<cv::Point>> lesion_localization(const cv::Mat &image, int type = 0);