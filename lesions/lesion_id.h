#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "lesions.h"
#include "stdafx.h"

std::vector<int> lesion_id(std::vector<cv::Scalar> & lesion_colors) {
	std::vector<int> lesion_id;
	
	//std::string image_out("C:/Users/Austin Pursley/Desktop/ECEN-403-Smart-Mirror-Image-Analysis/data/lesions/output/");
	
	//file stuff for testing
	FILE * pFile;
	pFile = fopen ("C:/Users/Austin Pursley/Desktop/ECEN-403-Smart-Mirror-Image-Analysis/lesions/output/color_data.txt", "w");

	for (int i = 0;i < lesion_colors.size();i++) {
		//----------------------testing----------------------------
		cv::Mat show(64, 64, CV_8UC3);
		show = cv::Scalar(lesion_colors[i]);
		cv::imwrite(image_out + "les_color_" + std::to_string(i) + ".jpg", show);

		fprintf(pFile, "------lesion%d------ \n", i);
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
		//--------------------testing-----------------------------
	}

	//thinking color matrix is easier to work with
	cv::Mat m(1, 1, CV_8UC3); 
	
	m = cv::Scalar(lesion_colors[0]);
	
	return  lesion_id;
}