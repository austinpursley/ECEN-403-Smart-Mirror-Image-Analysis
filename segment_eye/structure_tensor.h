#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <queue>
#include <vector>

void struct_tensor(cv::Mat img) {
	std::string image_out("C:/Users/Austin Pursley/Desktop/ECEN-403-Smart-Mirror-Image-Analysis/data/eyes/output/");
	const int width = img.cols;
	const int height = img.rows;
	cv::Mat e1_img = cv::Mat(height, width, CV_64FC1, cv::Scalar(0));
	cv::Mat e2_img = cv::Mat(height, width, CV_64FC1, cv::Scalar(0));
	double a, b, c, d, f, g, h;
	double e1, e2;
	cv::Mat dx, dy;
	cv::Mat dx2, dy2, dxy;
	cv::Mat dx2_e1, dy2_e1, dxy_e1, dx2_e2, dy2_e2, dxy_e2;

	//Calculate image derivatives
	//These are 2D tensor structure values without weighted window 
	cv::Sobel(img, dx, CV_64FC1, 1, 0);
	cv::Sobel(img, dy, CV_64FC1, 0, 1);
	cv::multiply(dx, dx, dx2, CV_64FC1);
	cv::multiply(dy, dy, dy2, CV_64FC1);
	cv::multiply(dx, dy, dxy, CV_64FC1);

	//Guassian blur on derivate matrices
	//This is the "window weight" or "gaussian weighted window"
	//ksize is the window size

	//Params
	
	int e1_win_sz = 3;
	int e2_win_sz = 13;
	int e1_thresh = 10;
	int e2_thresh = 125;
	int size_open = 1;

	cv::Size ksize_e1;
	ksize_e1.height = e1_win_sz;
	ksize_e1.width = ksize_e1.height;
	cv::GaussianBlur(dx2, dx2_e1, ksize_e1, 0);
	cv::GaussianBlur(dy2, dy2_e1, ksize_e1, 0);
	cv::GaussianBlur(dxy, dxy_e1, ksize_e1, 0);
	cv::Size ksize_e2;
	ksize_e2.height = e2_win_sz;
	ksize_e2.width = ksize_e2.height;
	cv::GaussianBlur(dx2, dx2_e2, ksize_e2, 0);
	cv::GaussianBlur(dy2, dy2_e2, ksize_e2, 0);
	cv::GaussianBlur(dxy, dxy_e2, ksize_e2, 0);

	for (int y = 0;y < height;y++) {
		for (int x = 0;x < width;x++) {
			//caculate the eigen values of 2D structure tensor Sw[p]
			//e1
			a = dx2_e1.at<double>(y, x); //Sw[0,0]
			b = dxy_e1.at<double>(y, x); //Sw[1,1]
			c = dy2_e1.at<double>(y, x); //Sw[1,0] & Sw[0,1]
			f = std::pow((a-b), 2);
			g = std::pow((4*c), 2);
			h = std::sqrt(f + g);
			e1 = 0.5*(a + b + h);
			//e2
			a = dx2_e2.at<double>(y, x); //Sw[0,0]
			b = dxy_e2.at<double>(y, x); //Sw[1,1]
			c = dy2_e2.at<double>(y, x); //Sw[1,0] & Sw[0,1]
			f = std::pow((a - b), 2);
			g = std::pow((4 * c), 2);
			h = std::sqrt(f + g);
			e2 = 0.5*(a + b - h);
			//place eigen values in corresponding img matrix
			e1_img.at<double>(y, x) = e1;
			e2_img.at<double>(y, x) = e2;
		}
		
	}
	//normalize eigen values between 0 and 255
	cv::normalize(e1_img, e1_img, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::normalize(e2_img, e2_img, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	
	cv::Mat e2_max, otsu, spec, rmv_spec;

	//e2
	cv::bitwise_not(e2_img, e2_img);
	cv::imwrite(image_out + "e2_0.jpg", e2_img);
	cv::threshold(e2_img, e2_max, e2_thresh, 255, cv::THRESH_BINARY);
	//cv::threshold(e2_img, otsu, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
	cv::bitwise_and(e2_max, e2_max, spec);
	cv::imwrite(image_out + "e2_1spec.jpg", spec);

	//e1
	cv::imwrite(image_out + "e1_0.jpg", e1_img);
	//cv::bitwise_not(e1_img, e1_img);
	cv::bitwise_not(spec, spec);

	//cv::threshold(e1_img, otsu, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
	cv::threshold(e1_img, otsu, e1_thresh, 255, cv::THRESH_BINARY);
	cv::imwrite(image_out + "e1_2otsu.jpg", otsu);

	cv::bitwise_and(otsu, spec, rmv_spec);
	cv::imwrite(image_out + "e1_1rmv_spec.jpg", rmv_spec);

	
	int open = cv::MORPH_OPEN;
	int shape = cv::MORPH_ELLIPSE;
	cv::Mat elem_open = cv::getStructuringElement(shape,
		cv::Size(2 * size_open + 1, 2 * size_open + 1),
		cv::Point(size_open, size_open));
	cv::morphologyEx(rmv_spec, rmv_spec, open, elem_open);
	cv::imwrite(image_out + "e1_dilate.jpg", rmv_spec);
	
	/*
	int thresh = 0;
	int maxValue = 255;
	int thresholdType = cv::THRESH_BINARY_INV;
	int adaptMethod = cv::ADAPTIVE_THRESH_GAUSSIAN_C;
	int blocksize = 15;
	cv::adaptiveThreshold(rmv_spec, otsu, maxValue, adaptMethod, thresholdType, blocksize, 2);
	cv::imwrite(image_out + "e1_2otsu.jpg", otsu);
	*/
	/*v
	//e2
	cv::Mat e2_max, otsu, spec;
	//cv::bitwise_not(e2_img, e2_img);
	cv::threshold(e2_img, e2_max, 150, 255, cv::THRESH_BINARY);
	cv::threshold(e2_img, otsu, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
	cv::bitwise_and(e2_max, otsu, spec);
	
	cv::imwrite(image_out + "e2_spec.jpg", spec);
	//e1
	cv::Mat e1_rmv_spec;
	//cv::bitwise_not(spec, spec);
	cv::bitwise_not(e1_img, e1_img);
	cv::bitwise_and(e1_img, spec, e1_rmv_spec);
	cv::threshold(e1_rmv_spec, otsu, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
	
	cv::imwrite(image_out + "e1_otsu.jpg", otsu);
	cv::imwrite(image_out + "e1.jpg", e1_img);
	cv::imwrite(image_out + "e1_rmv_spec.jpg", e1_rmv_spec);
	*/

	//cv::namedWindow("e2", CV_WINDOW_AUTOSIZE);
	//cv::imshow("e2", e2_img);
	//cv::namedWindow("white", CV_WINDOW_AUTOSIZE);
	//cv::imshow("white", white);
	//cv::namedWindow("max", CV_WINDOW_AUTOSIZE);
	//cv::imshow("max", max);
	
	/*
	cv::bitwise_not(e2_img, e2_img);
	cv::namedWindow("e2", CV_WINDOW_AUTOSIZE);
	cv::imshow("e2", e2_img);
	*/

	//printf("count: %d \n", count)
	return;
}