/*
Name: lesion.hpp
Date: 01/2018
Author: Austin Pursley
Course: ECEN 403, Senior Design Smart Mirror

Purpose: Lesion class helps with filtering and classifying skin lesions.
*/


#include "stdafx.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "../my_in_out_directory.hpp"
#include "direct.h"

/** @brief Skin lesion class that contains its properties like shape and color. Helps with filtering and classifying skin lesions.

Lesions are defined by contours. The contour itself has properties about its 
shape e.g. area and intertia ratio. The area of the image associated with 
the contour also has other properties e.g. color.
*/

#ifndef LESION_H
#define LESION_H
class Lesion
{
public:
	Lesion( std::vector<cv::Point> init_contour, const cv::Mat &mat, cv::Mat &mask, int id = 0, const double roi_scale = 0.25); //constructor
	Lesion(const Lesion& copy_from);              //copy constructor
	Lesion& operator=(const Lesion &copy_from); //copy assignment
	//~Lesion();                                    //destrucctor
	
	//image id number, remeber to use in .cpp
	static int img_id;
	//accessor functions
	std::vector<cv::Point> get_contour() const;
	int get_id() const;
	int get_lesion_class() const;
	cv::Rect get_roi() const;
	cv::Scalar get_color() const;
	cv::Scalar get_bg_color() const;
	double get_area() const;
	double get_inertia_ratio() const;
	//draw lesion on mat
	void draw(cv::Mat &mat);
	//member function to update class according to new contours
	void update(cv::Mat &mat, std::vector<cv::Point> new_contour, cv::Mat &mask);

private:
	//member variables
	std::vector<cv::Point> contour;
	int id;
	int roi_size;
	int les_class;
	cv::Rect roi;
	cv::Scalar color;
	cv::Scalar bg_color; //local roi background color
	double area;
	double inertia_ratio;
	//functions calculate member variables with lesion contours;
	void find_area();
	void find_inertia_ratio();
	void find_colors(const cv::Mat &mat, const cv::Mat &mask);
	void find_roi(const cv::Mat &mat);
	void find_class();
};

#endif /* lLESION_H */