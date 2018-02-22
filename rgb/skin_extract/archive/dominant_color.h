#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <queue>
#include <vector>

/*
 * Passive object that holds collection of data associated with a node. 
 * Nodes in this case will comprise the color data for a portion of an image. 
 */
typedef struct t_color_node {
	//The mean will be the average RGB color of each pixel in node, represented 
	//in an 3x1 matrix. It is used to calculate variance and will be the color
	//assigned to node.
	cv::Mat       mean;
	//The variance is a measures how spread out values are from their average.
	cv::Mat       variance;
	//Each node is associated with a class ID number. Each pixel starts out in 
	//class one. On first split, each pixel belongs to class 2 or 3. And this 
	// pattern continues on.
	uchar         classid; 
	//each node will split into a "left" and "right" node
	t_color_node  *left;
	t_color_node  *right;
} t_color_node;

/*
 * Returns "leaves" of a node i.e. two nodes it split into.
 */
std::vector<t_color_node*> get_leaves(t_color_node *root) {
	std::vector<t_color_node*> ret;
	std::queue<t_color_node*> queue;
	queue.push(root);

	while (queue.size() > 0) {
		t_color_node *current = queue.front();
		queue.pop();

		if (current->left && current->right) {
			queue.push(current->left);
			queue.push(current->right);
			continue;
		}

		ret.push_back(current);
	}

	return ret;
}

/*
 * Return the mean/dominant color of node.
 */
std::vector<cv::Vec3b> get_dominant_colors(t_color_node *root) {
	std::vector<t_color_node*> leaves = get_leaves(root);
	std::vector<cv::Vec3b> ret;

	for (int i = 0;i<leaves.size();i++) {
		cv::Mat mean = leaves[i]->mean;
		//multiply by 255 to make a viable opencv color.
		ret.push_back(cv::Vec3b(mean.at<double>(0)*255.0f,
			mean.at<double>(1)*255.0f,
			mean.at<double>(2)*255.0f));
	}

	return ret;
}

/*
 * Return the IDs of node's two classes i.e. two nodes split into.
 */
int get_next_classid(t_color_node *root) {
	int maxid = 0;
	std::queue<t_color_node*> queue;
	queue.push(root);

	while (queue.size() > 0) {
		t_color_node* current = queue.front();
		queue.pop();

		if (current->classid > maxid)
			maxid = current->classid;

		if (current->left != NULL)
			queue.push(current->left);

		if (current->right)
			queue.push(current->right);
	}

	return maxid + 1;
}

/*
 * Calculates the mean and variance for each class/segment of image. 
 * Does not return mean or variance associated with node.
 */
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

/*
 * Given a node split it into two classes, or in other words just two new nodes.
 */
void split_class(cv::Mat img, cv::Mat classes, uchar nextid, t_color_node *node) {
	const int width = img.cols;
	const int height = img.rows;
	const int classid = node->classid;

	const uchar newidleft = nextid;
	const uchar newidright = nextid + 1;

	cv::Mat mean = node->mean;
	cv::Mat variance = node->variance;
	cv::Mat eigenvalues, eigenvectors;
	//opencv's function for caluclting egienvalue and eigenvectors. Notice we are using
	// the variance matrix associated with node.
	cv::eigen(variance, eigenvalues, eigenvectors);

	cv::Mat eig = eigenvectors.row(0);
	//scaled by mean color of node
	cv::Mat comparison_value = eig * mean;

	node->left = new t_color_node();
	node->right = new t_color_node();

	node->left->classid = newidleft;
	node->right->classid = newidright;
	//go through each pixel and determine if it's in the left or right side of split.
	for (int y = 0;y<height;y++) {
		cv::Vec3b* ptr = img.ptr<cv::Vec3b>(y);
		uchar* ptrClass = classes.ptr<uchar>(y);
		for (int x = 0;x<width;x++) {
			if (ptrClass[x] != classid)
				continue;

			cv::Vec3b color = ptr[x];
			cv::Mat scaled_color = cv::Mat(3, 1,
				CV_64FC1,
				cv::Scalar(0));

			scaled_color.at<double>(0) = color[0] / 255.0f;
			scaled_color.at<double>(1) = color[1] / 255.0f;
			scaled_color.at<double>(2) = color[2] / 255.0f;

			//scaled by color of pixel
			cv::Mat this_value = eig * scaled_color;
			
			//split comparison
			if (this_value.at<double>(0, 0) <= comparison_value.at<double>(0, 0)) {
				ptrClass[x] = newidleft;
			}
			else {
				ptrClass[x] = newidright;
			}
		}
	}
	return;
}

/*
 * When splitting the classes, this funciton determines which class to split next
 */
t_color_node* get_max_eigenvalue_node(t_color_node *current) {
	double max_eigen = -1;
	cv::Mat eigenvalues, eigenvectors;

	std::queue<t_color_node*> queue;
	queue.push(current);
	
	t_color_node *ret = current;
	if (!current->left && !current->right)
		return current;

	while (queue.size() > 0) {
		t_color_node *node = queue.front();
		queue.pop();

		if (node->left && node->right) {
			queue.push(node->left);
			queue.push(node->right);
			continue;
		}

		cv::eigen(node->variance, eigenvalues, eigenvectors);
		double val = eigenvalues.at<double>(0);
		if (val > max_eigen) {
			max_eigen = val;
			ret = node;
		}
	}

	return ret;
}

/*
* Helps visualize the dominant colors. It outputs a open CV
* matrix that is a palette visual of the dominant colors.
*/
cv::Mat get_dominant_palette(std::vector<cv::Vec3b> colors) {
	const int tile_size = 64;
	cv::Mat ret = cv::Mat(tile_size, tile_size*colors.size(), CV_8UC3, cv::Scalar(0));

	for (int i = 0;i<colors.size();i++) {
		cv::Rect rect(i*tile_size, 0, tile_size, tile_size);
		cv::rectangle(ret, rect, cv::Scalar(colors[i][0], colors[i][1], colors[i][2]), CV_FILLED);
	}

	return ret;
}

/*
 * Outputs what it looks like when each a pixel is associated with mean color of its respective class.
 */
cv::Mat get_quantized_image(cv::Mat classes, t_color_node *root) {
	std::vector<t_color_node*> leaves = get_leaves(root);

	const int height = classes.rows;
	const int width = classes.cols;
	cv::Mat ret(height, width, CV_8UC3, cv::Scalar(0));

	for (int y = 0;y<height;y++) {
		uchar *ptrClass = classes.ptr<uchar>(y);
		cv::Vec3b *ptr = ret.ptr<cv::Vec3b>(y);
		for (int x = 0;x<width;x++) {
			uchar pixel_class = ptrClass[x];
			for (int i = 0;i<leaves.size();i++) {
				if (leaves[i]->classid == pixel_class) {
					ptr[x] = cv::Vec3b(leaves[i]->mean.at<double>(0) * 255,
						leaves[i]->mean.at<double>(1) * 255,
						leaves[i]->mean.at<double>(2) * 255);
				}
			}
		}
	}

	return ret;
}

cv::Mat get_color_rect(cv::Vec3b color) {
	const int tile_size = 64;
	cv::Mat ret = cv::Mat(tile_size, tile_size, CV_8UC3, cv::Scalar(0));

	cv::Rect rect(0, 0, tile_size, tile_size);
	cv::rectangle(ret, rect, cv::Scalar(color[0], color[1], color[2]), CV_FILLED);

	return ret;
}

/* 
 * Use statistics and linear algebra to find the most dominant color of an image.
 * Can also be thought of reducing the number of colors of an image.
 */
std::vector<cv::Vec3b> find_dominant_colors(cv::Mat img, int count) {
	const int width = img.cols;
	const int height = img.rows;
	//Indication that program may take awhile to run
	printf("Color Count: %d\n", count);
	printf("waiting...\n");
	//classes is matrix that encodes which pixel has what class. At first each
	//pixel belongs to class 1.
	cv::Mat classes = cv::Mat(height, width, CV_8UC1, cv::Scalar(1));
	//node0
	t_color_node *root = new t_color_node();
	root->classid = 1;
	root->left = NULL;
	root->right = NULL;
	t_color_node *next = root;
	get_class_mean_var(img, classes, root);
	//Each iteration creates a node.
	for (int i = 0;i<count - 1;i++) {
		next = get_max_eigenvalue_node(root);
		split_class(img, classes, get_next_classid(root), next);
		get_class_mean_var(img, classes, next->left);
		get_class_mean_var(img, classes, next->right);
	}
	std::vector<cv::Vec3b> colors = get_dominant_colors(root);
	printf("Dominate color(s) found!\n");
	//images to illustrate results
	//cv::Mat viewable = get_viewable_image(classes);
	cv::Mat quantized = get_quantized_image(classes, root);

	//cv::imwrite("./output/classification.png", viewable);	
	cv::imwrite("C:/Users/Austin Pursley/Desktop/ECEN-403-Smart-Mirror-Image-Analysis/color_face/output/quantized.jpg", quantized);

	return colors;
}