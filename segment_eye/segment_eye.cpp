#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "stdafx.h"
#include <iostream>


int main(int argc, char** argv) {
	std::string image_name("eye1.jpg");
	std::string image_file("C:/Users/Austin Pursley/Desktop/ECEN-403-Smart-Mirror-Image-Analysis/data/eyes/" + image_name);
	std::string image_out("C:/Users/Austin Pursley/Desktop/ECEN-403-Smart-Mirror-Image-Analysis/data/eyes/output/" + image_name);
	cv::Mat matImage = cv::imread(image_file, cv::IMREAD_COLOR);

	if (!matImage.data) {
		std::cout << "Unable to open the file: " << image_file;
		return 1;

	}

	//display original image
	//cv::namedWindow("Original Image", cv::WINDOW_AUTOSIZE);
	//cv::imshow("Original Image", matImage);
	cv::Mat src_gray, edge, draw, blur;
	
	//convert to gray
	cvtColor(matImage, src_gray, CV_BGR2GRAY);
	
	//apply blur
	GaussianBlur(src_gray, blur, cv::Size(5, 5), 2, 2);
	cv::namedWindow("blurred", CV_WINDOW_AUTOSIZE);
	cv::imshow("blurred", blur);
	
	//canny edge
	Canny(blur, edge, 50, 150, 3);
	//convert to image
	edge.convertTo(draw, CV_8U);
	cv::namedWindow("Canny", CV_WINDOW_AUTOSIZE);
	cv::imshow("Canny", draw);

	//GaussianBlur(draw, src_gray, cv::Size(9, 9), 2, 2);

	std::vector<cv::Vec3f> circles;

	/// Apply the Hough Transform to find the circles
	HoughCircles(draw, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows / 8, 200, 100, 0, 0);
	printf("circles: %x", circles.size());
	/// Draw the circles detected
	for (size_t i = 0; i < circles.size(); i++)
	{
		cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		cv::circle(matImage, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
		// circle outline
		circle(matImage, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);
	}

	/// Show your results
	cv::namedWindow("Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE);
	cv::imshow("Hough Circle Transform Demo", matImage);
	
	cv::waitKey();
	return 0;
}