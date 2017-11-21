#define _CRT_SECURE_NO_DEPRECATE
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "stdafx.h"
#include <iostream>

#include "lesion_detection.h"
#include "lesion_features.h"
#include "lesion_id.h"


int main(int argc, char** argv) {
	std::string image_name("les_skin2.jpg");
	std::string image_file("C:/Users/Austin Pursley/Desktop/ECEN-403-Smart-Mirror-Image-Analysis/data/lesions/" + image_name);
	std::string image_out("C:/Users/Austin Pursley/Desktop/ECEN-403-Smart-Mirror-Image-Analysis/data/lesions/output/");
	
	FILE * pFile;
	pFile = fopen("C:/Users/Austin Pursley/Desktop/ECEN-403-Smart-Mirror-Image-Analysis/data/lesions/output/mean_color_data.txt", "w");

	cv::Mat matImage = cv::imread(image_file, cv::IMREAD_COLOR);
	cv::imwrite(image_out + "0_orig.jpg", matImage);
	if (!matImage.data) {
		std::cout << "Unable to open the file: " << image_file;
		return 1;
	}

	//---------------------delete l8r-----------------------------------
	cv::Scalar mean_color = cv::mean(matImage);
	cv::Mat show(64, 64, CV_8UC3);
	
	show = cv::Scalar(mean_color);
	cv::imwrite(image_out + "mean_skin_color.jpg", show);

	fprintf(pFile, "------MEAN------ \n");

	cv::Vec3b bgr_color = show.at<cv::Vec3b>(cv::Point(0, 0));
	fprintf(pFile, "blue:    %d \n", bgr_color[0]);
	fprintf(pFile, "green:   %d \n", bgr_color[1]);
	fprintf(pFile, "red:     %d \n\n", bgr_color[2]);

	cv::Mat show_lab(64, 64, CV_8UC3);
	cv::cvtColor(show, show_lab, CV_BGR2Lab);
	cv::Vec3b lab_color = show_lab.at<cv::Vec3b>(cv::Point(0, 0));
	fprintf(pFile, "lab0_L:  %d \n", lab_color[0]);
	fprintf(pFile, "lab1_GR: %d \n", lab_color[1]);
	fprintf(pFile, "lab2_BY: %d \n\n", lab_color[2]);

	cv::Mat show_hsv(64, 64, CV_8UC3);
	cv::cvtColor(show, show_hsv, CV_BGR2HSV);
	cv::Vec3b hsv_color = show_hsv.at<cv::Vec3b>(cv::Point(0, 0));
	fprintf(pFile, "hue:     %d \n", hsv_color[0]);
	fprintf(pFile, "satur:   %d \n", hsv_color[1]);
	fprintf(pFile, "value:   %d \n\n", hsv_color[2]);
	//---------------------delete---------------------------------------

	//lesion detection: find the contours/borders of the lesions.
	std::vector<std::vector<cv::Point> > lesion_contours;
	lesion_contours = lesion_detection(matImage);

	//lesion features: area/size and mean color
	std::vector<double> les_areas;
	les_areas = lesion_areas(lesion_contours);
	std::vector<cv::Scalar> les_colors;
	les_colors = lesion_colors(matImage, lesion_contours);

	//lesion identification: 
	//which are dark spots?  which are red spots?
	std::vector<int> lesion_ids;
	lesion_ids = lesion_id(les_colors);

	cv::waitKey();
	
	return 0;
}