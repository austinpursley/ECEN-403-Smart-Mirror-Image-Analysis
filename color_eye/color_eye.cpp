#include "stdafx.h"
#include <string>
#include <iostream>
#include "../dominant_color.h"

/*This program does...*/
int main(int argc, char** argv) {


	std::string imageName("C:/Users/Austin Pursley/Desktop/ECEN-403-Smart-Mirror-Image-Analysis/data/eyes/eye_b0.jpg"); // by default

	//if (argc > 1)
	//{
	//	//I confingured this in Visual Studios debugging command arguments
	//	imageName = argv[1];
	//}

	cv::Mat matImage = cv::imread(imageName, cv::IMREAD_COLOR);
	//cv::Mat gray_image = cv::imread(imageName, 0);

	if (!matImage.data) {
		std::cout << "Unable to open the file: " << imageName;
		return 1;

	}

	//display original image
	cv::namedWindow("Original Image", cv::WINDOW_AUTOSIZE);
	cv::imshow("Original Image", matImage);

	//median filter, to reduce noise
	int ksize = 3;
	cv::medianBlur(matImage, matImage, ksize);
	cv::namedWindow("Median Blurred", cv::WINDOW_AUTOSIZE);
	cv::imshow("Median Blurred", matImage);

	//adaptive threshold
	cv::Mat adpt_threshold_img = matImage.clone();
	cv::cvtColor(adpt_threshold_img, adpt_threshold_img, cv::COLOR_BGR2GRAY);
	int maxValue = 255;
	int adaptiveMethod = cv::ADAPTIVE_THRESH_MEAN_C;
	int thresholdType = cv::THRESH_BINARY;
	int blockSize = 103;
	double C = 2;
	cv::adaptiveThreshold(adpt_threshold_img, adpt_threshold_img, maxValue, adaptiveMethod, thresholdType, blockSize, C);
	cv::namedWindow("Adaptive Threshold", cv::WINDOW_AUTOSIZE);
	cv::imshow("Adaptive Threshold", adpt_threshold_img);

	//mask blurred original image with thresholded image
	cv::cvtColor(adpt_threshold_img, adpt_threshold_img, cv::COLOR_GRAY2BGR);
	cv::Mat masked_img = adpt_threshold_img.clone();
	cv::bitwise_and(matImage, adpt_threshold_img, masked_img);
	cv::namedWindow("Masked", cv::WINDOW_AUTOSIZE);
	cv::imshow("Masked", masked_img);
	
	//hue threshold
	cv::Mat hsv_img1 = matImage.clone();
	cv::Mat hsv_img2 = matImage.clone();
	cv::cvtColor(hsv_img1, hsv_img1, cv::COLOR_BGR2HSV);
	cv::cvtColor(hsv_img2, hsv_img2, cv::COLOR_BGR2HSV);

	//hue in
	cv::inRange(hsv_img1, cv::Scalar(0, 0, 128), cv::Scalar(8, 255, 255), hsv_img1);
	cv::namedWindow("1", cv::WINDOW_AUTOSIZE);
	cv::imshow("1", hsv_img1);
	cv::cvtColor(hsv_img1, hsv_img1, cv::COLOR_GRAY2BGR);
	
	cv::inRange(hsv_img2, cv::Scalar(175, 0, 128), cv::Scalar(180, 255, 255), hsv_img2);
	cv::namedWindow("2", cv::WINDOW_AUTOSIZE);
	cv::imshow("2", hsv_img2);
	cv::cvtColor(hsv_img2, hsv_img2, cv::COLOR_GRAY2BGR);

	cv::Mat masked_img2 = hsv_img1.clone();
	cv::bitwise_and(hsv_img2, hsv_img1, masked_img2);
	cv::namedWindow("3", cv::WINDOW_AUTOSIZE);
	cv::imshow("3", masked_img2);

	cv::bitwise_and(matImage, masked_img2, masked_img2);

	/*
	//reduce image to 10 dominate colors to help us pick out white part of eye
	int count = 8;

	//This function returns a vector that contains dominantc colors (see header for how it works)
	std::vector<cv::Vec3b> colors = find_dominant_colors(matImage, count);

	//create pallette image from the two extracted dominant colors
	cv::Mat palette = get_dominant_palette(colors);
	cv::imwrite("C:/Users/Austin Pursley/Desktop/ECEN-403-Smart-Mirror-Image-Analysis/color_eye/output/palette.png", palette);

	//create image that's just a square the dominant color of white part of eye
	//cv::Mat dom_color = get_color_rect(skin_color);
	//cv::imwrite("C:/Users/Austin Pursley/Desktop/ECEN-403-Smart-Mirror-Image-Analysis/color_eye/output/eyecolor.png", dom_color);
	//display dominate face color square
	//cv::namedWindow("Face Color", cv::WINDOW_AUTOSIZE);
	//cv::imshow("Face Color", dom_face);
	
	
	//rgb to hsv color
	cv::Mat dom_hsv;
	//cv::cvtColor(dom_color, dom_hsv, CV_BGR2HSV);

	//get the eye color
	cv::Vec3b eye_color;

	//for (int i = 0; i < count;i++) {
	//	if (colors[i][0] > 200 && colors[i][0] > 200 && colors[i][0] > 200) {
	//		printf("colors[i]: %d\n", colors[i][0]);
	//		skin_color = colors[i];
	//		printf("skin_color[0]: %d\n", skin_color[0]);
	//	}
	//}
	*/

	//wait to close display windows
	cv::waitKey();
	
	return 0;
}
