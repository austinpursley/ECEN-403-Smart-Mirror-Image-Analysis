#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "lesions.h"
#include "stdafx.h"

std::vector<std::vector<cv::Point>> lesion_detection(const cv::Mat & image, int n) {
	// put all tuning parameters in one placee
	int gauss_ksize = 45;
	int blocksize = 57;
	int size_close = 7;
	int size_open = 4;
	int size_dilate = 0;
	// guassian Blur setting
	cv::Size ksize;
	ksize.height = gauss_ksize;
	ksize.width = ksize.height;
	// thresholding setting
	int thresh = 0;
	int maxValue = 255;
	int thresholdType = cv::THRESH_BINARY_INV;
	int adaptMethod = cv::ADAPTIVE_THRESH_GAUSSIAN_C;
	// morphology
	int shape = cv::MORPH_ELLIPSE;
	int open = cv::MORPH_OPEN;
	int close = cv::MORPH_CLOSE;
	cv::Mat elem_close = cv::getStructuringElement(shape,
		cv::Size(2 * size_close + 1, 2 * size_close + 1),
		cv::Point(size_close, size_close));
	cv::Mat elem_open = cv::getStructuringElement(shape,
		cv::Size(2 * size_open + 1, 2 * size_open + 1),
		cv::Point(size_open, size_open));
	cv::Mat elem_dilate = cv::getStructuringElement(shape,
		cv::Size(2 * size_dilate + 1, 2 * size_dilate + 1),
		cv::Point(size_dilate, size_dilate));
	// output matrix for each progression of processing
	cv::Mat gray_gr_img, blurr_img, thresh_img, close_img, open_img, dilate_img;

	// removing blue channel removes noise, making lesions clearer
	// green-red image
	cv::Mat gr_image = image & cv::Scalar(0, 255, 255);

	//convert color image to gray scale
	cv::cvtColor(gr_image, gray_gr_img, CV_BGR2GRAY);

	//guassin blur filter to reduce image noise
	cv::GaussianBlur(gray_gr_img, blurr_img, ksize, 0);

	//adaptive thresholding because lighting may not be totallly uniform
	cv::adaptiveThreshold(blurr_img, thresh_img, maxValue, adaptMethod, thresholdType, blocksize, 2);

	//close to fill in gaps
	cv::morphologyEx(thresh_img, close_img, close, elem_close);

	//open removes smaller blobs
	cv::morphologyEx(close_img, open_img, open, elem_open);

	//dilate what's left to make them more prominent
	cv::dilate(open_img, dilate_img, elem_dilate);

	//----------debug/output, delete later-------------------
	//output this for now
	/*
	cv::imwrite(image_out + "0_grb.jpg", gr_image);
	cv::imwrite(image_out + "1_gr.jpg", gr_image);
	cv::imwrite(image_out + "2_gr_gray.jpg", gray_gr_img);
	cv::imwrite(image_out + "3_blurr.jpg",  blurr_img);
	cv::imwrite(image_out + "4_thresh.jpg", thresh_img);
	cv::imwrite(image_out + "5_close.jpg",  close_img);
	cv::imwrite(image_out + "6_open.jpg",   open_img);
	cv::imwrite(image_out + "7_dilate.jpg", dilate_img);
	*/
	cv::bitwise_not(dilate_img, dilate_img);
	cv::Mat masked, color;
	cv::cvtColor(dilate_img, color, CV_GRAY2BGR);
	cv::bitwise_and(color, image, masked);

	//cv::imwrite(image_out + "5_masked.jpg", masked);
	//-------------------------------------------------------

	//find contours
	//contours points that make up border of area on the original image
	int compCount = 0;
	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(dilate_img, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	int size = contours.size() - 1;
	printf("# lesions: %d \n", size);
	//cv::imwrite(image_out + std::to_string(n) + "_size" + std::to_string(size) + "_m.jpg", masked);
	cv::imwrite(image_out + std::to_string(n) + "_test.jpg", masked);

	return contours;
}
