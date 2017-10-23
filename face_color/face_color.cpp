#include "stdafx.h"
#include <stdio.h>
#include "face_color.h"
#include <iostream>
#include <string>
#include <queue>
#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

/*This program does...*/
int main(int argc, char** argv) {
	

	std::string imageName("input/skin0_scaled.jpg"); // by default

	//if (argc > 1)
	//{
	//	//I confingured this in Visual Studios debugging command arguments
	//	imageName = argv[1];
	//}

	cv::Mat matImage = cv::imread(imageName);

	if (!matImage.data) {
		printf("Unable to open the file: %s\n", imageName);
		return 1;
	}

	//two colors: color of image and color of background (white)
	int count = 2;

	//This function returns a vector that contains two color vectors
	std::vector<cv::Vec3b> colors = find_dominant_colors(matImage, count);

	//grab face color (TODO: add check to make sure get non-white color)
	std::vector<cv::Vec3b> face_color;
	face_color.push_back(colors[0]);

	//display original image
	cv::namedWindow("Original Image", cv::WINDOW_AUTOSIZE);
	cv::imshow("Original Image", matImage);

	//output palette image with two colors, should be white and skin color.
	cv::Mat dom = get_dominant_palette(colors);
	cv::imwrite("./output/palette.png", dom);

	/*//display dominate color palette
	namedWindow("Palette", WINDOW_AUTOSIZE);
	imshow("Palette", dom); */

	//matrix with dominate face color
	cv::Mat dom_face = get_dominant_palette(face_color);
	cv::imwrite("./output/facecolor.png", dom_face);
	//display dominate face color in window
	cv::namedWindow("Face Color", cv::WINDOW_AUTOSIZE);
	cv::imshow("Face Color", dom_face);

	//wait to close display windows
	cv::waitKey();

	return 0;
}