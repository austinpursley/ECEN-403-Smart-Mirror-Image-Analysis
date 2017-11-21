#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "stdafx.h"

std::vector<std::vector<cv::Point>> lesion_detection(const cv::Mat & image) {
	//guassian Blur
	cv::Size ksize;
	ksize.height = 39;
	ksize.width = ksize.height;
	//thresholding
	int thresh = 0;
	int maxValue = 255;
	int blocksize = 25;
	int thresholdType = cv::THRESH_BINARY_INV;
	int adaptMethod = cv::ADAPTIVE_THRESH_GAUSSIAN_C;
	//morphology
	int shape = cv::MORPH_ELLIPSE;
	int open = cv::MORPH_OPEN;
	int close = cv::MORPH_CLOSE;
	int size_close = 4;
	cv::Mat elem_close = cv::getStructuringElement(shape,
		cv::Size(2 * size_close + 1, 2 * size_close + 1),
		cv::Point(size_close, size_close));
	int size_open = 2;
	cv::Mat elem_open = cv::getStructuringElement(shape,
		cv::Size(2 * size_open + 1, 2 * size_open + 1),
		cv::Point(size_open, size_open));
	int size_dilate = 0;
	cv::Mat elem_dilate = cv::getStructuringElement(shape,
		cv::Size(2 * size_dilate + 1, 2 * size_dilate + 1),
		cv::Point(size_dilate, size_dilate));
	//output matrix for each progression of processing
	cv::Mat gray_gr_img, blurr_img, thresh_img, close_img, open_img, dilate_img;

	//removing blue channel removes noise, making lesions clearer
	//green-reg image
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
	cv::erode(open_img, dilate_img, elem_dilate);

	//----------debug/output, delete later-------------------
	//output this for now
	std::string image_out("C:/Users/Austin Pursley/Desktop/ECEN-403-Smart-Mirror-Image-Analysis/data/lesions/output/");
	//cv::imwrite(image_out + "0_blurr.jpg",  blurr_img);
	//cv::imwrite(image_out + "1_thresh.jpg", thresh_img);
	//cv::imwrite(image_out + "2_close.jpg",  close_img);
	//cv::imwrite(image_out + "3_open.jpg",   open_img);
	//cv::imwrite(image_out + "4_dilate.jpg", dilate_img);

	cv::bitwise_not(dilate_img, dilate_img);
	cv::Mat masked, color;
	cv::cvtColor(dilate_img, color, CV_GRAY2BGR);
	cv::bitwise_and(color, image, masked);

	cv::imwrite(image_out + "5_masked.jpg", masked);
	//-------------------------------------------------------

	//find contours
	//contours points that make up border of area on the original image
	int compCount = 0;
	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(dilate_img, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	return contours;
}