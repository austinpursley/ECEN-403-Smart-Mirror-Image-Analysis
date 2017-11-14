#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "stdafx.h"
#include <iostream>


int main(int argc, char** argv) {
	std::string image_name("les_skin1.jpg");
	std::string image_file("C:/Users/Austin Pursley/Desktop/ECEN-403-Smart-Mirror-Image-Analysis/data/lesions/" + image_name);
	std::string image_out("C:/Users/Austin Pursley/Desktop/ECEN-403-Smart-Mirror-Image-Analysis/data/lesions/output/");
	
	cv::Mat matImage = cv::imread(image_file, cv::IMREAD_COLOR);
	cv::imwrite(image_out + "orig.jpg", matImage);
	if (!matImage.data) {
		std::cout << "Unable to open the file: " << image_file;
		return 1;

	}

	//display original image
	//cv::namedWindow("Original Image", cv::WINDOW_AUTOSIZE);
	//cv::imshow("Original Image", matImage);

	//Guassian Blur
	cv::Size ksize;
	ksize.height = 31;
	ksize.width = ksize.height;
	//Thresholding
	int thresh = 0;
	int maxValue = 255;
	int thresholdType = cv::THRESH_BINARY_INV;
	int adaptMethod = cv::ADAPTIVE_THRESH_GAUSSIAN_C;
	int blocksize = 25;
	//Morphology
	int shape = cv::MORPH_ELLIPSE;
	int open = cv::MORPH_OPEN;
	int close = cv::MORPH_CLOSE;
	int size_close = 4; 
	cv::Mat elem_close = cv::getStructuringElement(shape,
		cv::Size(2 * size_close + 1, 2 * size_close + 1),
		cv::Point(size_close, size_close));
	int size_open = 3;
	cv::Mat elem_open = cv::getStructuringElement(shape,
		cv::Size(2 * size_open + 1, 2 * size_open + 1),
		cv::Point(size_open, size_open));
	int size_dilate = 3;
	cv::Mat elem_dilate = cv::getStructuringElement(shape,
		cv::Size(2 * size_dilate + 1, 2 * size_dilate + 1),
		cv::Point(size_dilate, size_dilate));
	/*
	 * Green and Red Image
	 */
	
	//removing blue channel removes noise, making lesions clearer
	cv::Mat gr_image = matImage & cv::Scalar(0, 255, 255);
	//convert color image to gray scale
	cv::cvtColor(gr_image, gr_image, CV_BGR2GRAY);
	//guassin blur filter to reduce image noise
	cv::GaussianBlur(gr_image, gr_image, ksize, 0);
	cv::imwrite(image_out + "orig_gr_blur.jpg", gr_image);
	//adaptive thresholding because lighting may not be totallly uniform
	cv::adaptiveThreshold(gr_image, gr_image, maxValue, adaptMethod, thresholdType, blocksize, 2);
	cv::imwrite(image_out + "greenred_de0.jpg", gr_image);
	//close to fill in gaps
	cv::morphologyEx(gr_image, gr_image, close, elem_close);
	cv::imwrite(image_out + "greenred_de1.jpg", gr_image);
	//open removes smaller blobs
	cv::morphologyEx(gr_image, gr_image, open, elem_open);
	cv::imwrite(image_out + "greenred_de2.jpg", gr_image);
	//dilate what's left to make them more prominent
	cv::dilate(gr_image, gr_image, elem_dilate);
	cv::imwrite(image_out + "greenred_de3.jpg", gr_image);
	//to see results, apply mask to original image
	cv::bitwise_not(gr_image, gr_image);
	cv::Mat masked, color;
	cv::cvtColor(gr_image, color, CV_GRAY2BGR);
	cv::bitwise_and(color, matImage, masked);
	cv::imwrite(image_out + "masked.jpg", masked);
	
	/*
	 * Gray image
	/*
	cv::Mat gray;
	cvtColor(matImage, gray, CV_BGR2GRAY);
	cv::GaussianBlur(gray, gray, ksize, 0);
	cv::threshold(gray, gray, thresh, maxValue, thresholdType + cv::THRESH_OTSU);
	cv::adaptiveThreshold(gray, gray, maxValue, adaptMethod, thresholdType, blocksize, 2);
	cv::imwrite(image_out + "gray.jpg", gray);
	*/


	/*
	cv::namedWindow("Threshold", cv::WINDOW_AUTOSIZE);
	cv::imshow("Threshold", gray);
	cv::imshow("blue", cv::WINDOW_AUTOSIZE);
	cv::imshow("blue", rgbChannels[0]);
	cv::imshow("green", cv::WINDOW_AUTOSIZE);
	cv::imshow("green", rgbChannels[1]);
	cv::imshow("red", cv::WINDOW_AUTOSIZE);
	cv::imshow("red", rgbChannels[2]);
	*/

	/*
	cv::Mat src_gray, edge, draw, blur;

	//convert to gray
	cvtColor(matImage, src_gray, CV_BGR2GRAY);

	//apply blur
	GaussianBlur(src_gray, blur, cv::Size(5, 5), 2, 2);
	cv::namedWindow("blurred", CV_WINDOW_AUTOSIZE);
	cv::imshow("blurred", blur);


	/// Show your results
	cv::namedWindow("Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE);
	cv::imshow("Hough Circle Transform Demo", matImage);
	*/

	cv::waitKey();
	return 0;
}

















/*
//split image
std::vector<cv::Mat> rgbChannels(3);
cv::split(matImage, rgbChannels);

//rgb channels
for (unsigned int i = 0; i < 3;i++) {
cv::GaussianBlur(rgbChannels[i], rgbChannels[i], ksize, 0);
}

//rgb channels
for (unsigned int i = 0; i < 3;i++) {
cv::adaptiveThreshold(rgbChannels[i], rgbChannels[i], maxValue, adaptMethod, thresholdType, 11, 2);
//cv::threshold(rgbChannels[i], rgbChannels[i], thresh, maxValue, thresholdType + cv::THRESH_OTSU);
cv::imwrite(image_out + "rgbchannel" + std::to_string(i) + ".jpg", rgbChannels[i]);
}
*/