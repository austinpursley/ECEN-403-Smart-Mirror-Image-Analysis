#include "stdafx.h"
#include <string>
#include <iostream>
#include "../dominant_color.h"

void get_class_mean_var(cv::Mat img, cv::Mat classes, t_color_node *node) {
	const int width = img.cols;
	const int height = img.rows;
	const uchar classid = node->classid;
	//3x1 matrix and 3x3 matrix of 0s for mean and variance, repectively.
	cv::Mat mean = cv::Mat(3, 1, CV_64FC1, cv::Scalar(0));
	cv::Mat variance = cv::Mat(3, 3, CV_64FC1, cv::Scalar(0));

	double pixcount = 0; //track number of pixels in class
	for (int y = 0;y<height;y++) {
		//pointer to row
		cv::Vec3b* ptr = img.ptr<cv::Vec3b>(y);
		uchar* ptrClass = classes.ptr<uchar>(y);
		for (int x = 0;x<width;x++) {
			//skip pixel at (x,y) if it is not in the class associated with node
			if (ptrClass[x] != classid)
				continue;
			//The RGB vector of the pixel
			cv::Vec3b color = ptr[x];
			//scale RGB values to 0-1 range to avoid overflows
			cv::Mat scaled = cv::Mat(3, 1, CV_64FC1, cv::Scalar(0));
			scaled.at<double>(0) = color[0] / 255.0f;
			scaled.at<double>(1) = color[1] / 255.0f;
			scaled.at<double>(2) = color[2] / 255.0f;
			//See aishack tutorial for mean and variance forumals.
			//From that tutorial, these are Rn, mn, and Nn
			mean += scaled;
			variance = variance + (scaled * scaled.t());
			pixcount++;
		}
	}
	variance = variance - (mean * mean.t()) / pixcount;
	mean = mean / pixcount;

	// The node mean and variance
	node->mean = mean.clone();
	node->variance = variance.clone();

	return;
}

/*This program does...*/
int main(int argc, char** argv) {
	

	std::string imageName("C:/Users/Austin Pursley/Desktop/ECEN-403-Smart-Mirror-Image-Analysis/color_face/input/skin1.jpg"); // by default

	//if (argc > 1)
	//{
	//	//I confingured this in Visual Studios debugging command arguments
	//	imageName = argv[1];
	//}

	cv::Mat matImage = cv::imread(imageName);

	if (!matImage.data) {
		std::cout << "Unable to open the file: " << imageName;
		return 1;
	}

	//two colors: color of skin and color of background (white)
	int count = 2;

	//This function returns a vector that contains two color vectors
	std::vector<cv::Vec3b> colors = find_dominant_colors(matImage, count);

	//select skin color, making sure it's not the white background color
	cv::Vec3b skin_color;
	for (int i = 0; i < count;i++) {
		if (colors[i][0] < 245 && colors[i][0] < 245 && colors[i][0] != 245) {
			skin_color = colors[i];
			//printf("skin_color[0]: %d\n", skin_color[0]);
		}
	}

	//display original image
	cv::namedWindow("Original Image", cv::WINDOW_AUTOSIZE);
	cv::imshow("Original Image", matImage);

	//create pallette image from the two extracted dominant colors
	cv::Mat dom = get_dominant_palette(colors);
	cv::imwrite("C:/Users/Austin Pursley/Desktop/ECEN-403-Smart-Mirror-Image-Analysis/color_face/output/palette.png", dom);

	//create image that's just a square the dominant color of skin
	cv::Mat dom_face = get_color_rect(skin_color);
	cv::imwrite("C:/Users/Austin Pursley/Desktop/ECEN-403-Smart-Mirror-Image-Analysis/color_face/output/facecolor.png", dom_face);
	//display dominate face color square
	cv::namedWindow("Face Color", cv::WINDOW_AUTOSIZE);
	cv::imshow("Face Color", dom_face);

	//wait to close display windows
	cv::waitKey();

	return 0;
}