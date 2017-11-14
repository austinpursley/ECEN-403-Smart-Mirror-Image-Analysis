#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <queue>
#include <vector>

void struct_tensor(cv::Mat img) {
	const int width = img.cols;
	const int height = img.rows;
	printf("width: %d \n", width);
	printf("height: %d \n", height);
	// Calculate image derivatives 
	cv::Mat dx, dy;
	cv::Mat dx2, dy2, dxy;
	cv::Sobel(img, dx, CV_64FC1, 1, 0);
	cv::Sobel(img, dy, CV_64FC1, 0, 1);
	cv::multiply(dx, dx, dx2, CV_64FC1);
	cv::multiply(dy, dy, dy2, CV_64FC1);
	cv::multiply(dx, dy, dxy, CV_64FC1);
	cv::Mat eigen1 = cv::Mat(height, width, CV_64FC1, cv::Scalar(0));
	cv::Mat eigen2 = cv::Mat(height, width, CV_64FC1, cv::Scalar(0));
	
	//Every pixel in image going to be associated with 2x2 matrix
	//so we need 4D matrix
	const int num_of_dim = 4;
	const int dimensions[num_of_dim] = { height, width, 2, 2 };
	cv::Mat struct_tensors = cv::Mat(num_of_dim, dimensions, CV_64FC1);
	//placeholder matrix, makes clear what's going on in loop
	cv::Mat s0 = cv::Mat(2, 2, CV_64FC1, cv::Scalar(0));

	int count = 0; //track number of pixels in class
	for (int y = 0;y < height;y++) {
		for (int x = 0;x < width;x++) {
			//s0 matrix
			s0.at<double>(0, 0) = dx2.at<double>(y, x);
			s0.at<double>(0, 1) = dxy.at<double>(y, x);
			s0.at<double>(1, 0) = dxy.at<double>(y, x);
			s0.at<double>(1, 1) = dy2.at<double>(y, x);
			//assign s0 to corresponding pixel
			for (int i = 0;i < 1;i++) {
				for (int j = 0;j < 1;j++) {
					const int coords1[4] = { y, x, i, j };
					struct_tensors.at<double>(coords1) = s0.at<double>(i, j);
				}
			}
		}
		
	}
	printf("count: %d \n", count);
	

	return;
}