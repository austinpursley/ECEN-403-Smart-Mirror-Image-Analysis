#include "stdafx.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//to-do:
class Lesion
{
public:
	Lesion( std::vector<cv::Point> init_contour, const cv::Mat &mat, cv::Mat &mask, int id = 0, const double roi_scale = 0.25); //constructor
	Lesion(const Lesion& copy_from);              //copy constructor
	Lesion& operator=(const Lesion &copy_from); //copy assignment
	//~Lesion();                                    //destrucctor

	//accessor functions
	std::vector<cv::Point>  get_contour() const;
	int get_id() const;
	cv::Rect Lesion::get_roi() const;
	cv::Scalar get_color() const;
	cv::Scalar get_bg_color() const;
	double get_area() const;
	double get_inertia_ratio() const;

	//draw lesion on mat
	void draw(cv::Mat &mat);
	//member function to update class according to new contours
	void update(cv::Mat &mat, std::vector<cv::Point> new_contour, cv::Mat &mask);

private:
	//member variables
	std::vector<cv::Point> contour;
	int id;
	int roi_size;
	cv::Rect roi;
	cv::Scalar color;
	cv::Scalar bg_color; //local, background color
	double area;
	double inertia_ratio;
	//functions calculate member variables with lesion contours
	void find_area();
	void find_inertia_ratio();
	void find_colors(const cv::Mat &mat, const cv::Mat &mask);
	void find_roi(const cv::Mat &mat);
	
};
