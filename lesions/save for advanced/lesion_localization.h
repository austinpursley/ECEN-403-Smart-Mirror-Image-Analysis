#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "my_in_out_directory.hpp"
#include "lesions.hpp"

void blob_detect(const cv::Mat1b &src_1b, cv::Mat1b &bin_mask, std::vector<std::vector<cv::Point>> &contours_output);
void lesion_area_filter(std::vector<Lesion > &lesions, const double min_area = 25, const double max_area = 5000);
void lesion_intertia_filter(std::vector<Lesion > &lesions, const double min_intertia_ratio = 0.1f, const double max_intertia_ratio = std::numeric_limits<float>::max());
void lesion_draw_contours(const std::vector<Lesion > &lesions, cv::Mat &img);
void mask_image(const cv::Mat &mask, cv::Mat &masked_out);
std::vector<std::vector<cv::Point>> lesion_localization(const cv::Mat & image, int type = 0, std::string img_name = ""); 
