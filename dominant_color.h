#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <queue>
#include <vector>

/*Passive object that holds collection of data associated with a node. 
  Nodes in this case will comprise the color data for a portion of an image. */
typedef struct t_color_node {
	//TODO: PURPOSE OF EACH DATA TYPE
	//The mean will be the average RGB color of each pixel in node, represented 
	//in an 3x1 matrix. [TODO:PURPOSE]
	cv::Mat       mean;
	//The covariance is a measure to the extent to which corresponding elements
	//from two sets of orrdered data move in the same direction. [TODO: PURPOSE]
	cv::Mat       cov;        // The covariance of this node
	//Each node is associated with a class ID number. Each pixel starts out in 
	//class one. On first split, each pixel belongs to class 2 or 3. And this 
	// pattern continues on.
	uchar         classid; 

	t_color_node  *left;
	t_color_node  *right;
} t_color_node;

/*What the function does, how to use it*/
cv::Mat get_dominant_palette(std::vector<cv::Vec3b> colors) {
	const int tile_size = 64;
	cv::Mat ret = cv::Mat(tile_size, tile_size*colors.size(), CV_8UC3, cv::Scalar(0));

	for (int i = 0;i<colors.size();i++) {
		cv::Rect rect(i*tile_size, 0, tile_size, tile_size);
		cv::rectangle(ret, rect, cv::Scalar(colors[i][0], colors[i][1], colors[i][2]), CV_FILLED);
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

/*what the function does, how to use it*/
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
/*what the function does, how to use it*/
std::vector<cv::Vec3b> get_dominant_colors(t_color_node *root) {
	std::vector<t_color_node*> leaves = get_leaves(root);
	std::vector<cv::Vec3b> ret;

	for (int i = 0;i<leaves.size();i++) {
		cv::Mat mean = leaves[i]->mean;
		ret.push_back(cv::Vec3b(mean.at<double>(0)*255.0f,
			mean.at<double>(1)*255.0f,
			mean.at<double>(2)*255.0f));
	}

	return ret;
}

/*what the function does, how to use it*/
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

/*what the function does, how to use it*/
void get_class_mean_cov(cv::Mat img, cv::Mat classes, t_color_node *node) {
	const int width = img.cols;
	const int height = img.rows;
	const uchar classid = node->classid;
	//3x1 matrix and 3x3 matrix of 0s for mean and covariance, repectively.
	cv::Mat mean = cv::Mat(3, 1, CV_64FC1, cv::Scalar(0));
	cv::Mat cov = cv::Mat(3, 3, CV_64FC1, cv::Scalar(0));
	
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
			//See aishack tutorial for mean and covariance forumals.
			//From that tutorial, these are Rn, mn, and Nn
			mean += scaled;
			cov = cov + (scaled * scaled.t());
			pixcount++;
		}
	}
	cov = cov - (mean * mean.t()) / pixcount;
	mean = mean / pixcount;

	// The node mean and covariance
	node->mean = mean.clone();
	node->cov = cov.clone();

	return;
}

/*what the function does, how to use it*/
void partition_class(cv::Mat img, cv::Mat classes, uchar nextid, t_color_node *node) {
	const int width = img.cols;
	const int height = img.rows;
	const int classid = node->classid;

	const uchar newidleft = nextid;
	const uchar newidright = nextid + 1;

	cv::Mat mean = node->mean;
	cv::Mat cov = node->cov;
	cv::Mat eigenvalues, eigenvectors;
	cv::eigen(cov, eigenvalues, eigenvectors);

	cv::Mat eig = eigenvectors.row(0);
	cv::Mat comparison_value = eig * mean;

	node->left = new t_color_node();
	node->right = new t_color_node();

	node->left->classid = newidleft;
	node->right->classid = newidright;

	// We start out with the average color
	for (int y = 0;y<height;y++) {
		cv::Vec3b* ptr = img.ptr<cv::Vec3b>(y);
		uchar* ptrClass = classes.ptr<uchar>(y);
		for (int x = 0;x<width;x++) {
			if (ptrClass[x] != classid)
				continue;

			cv::Vec3b color = ptr[x];
			cv::Mat scaled = cv::Mat(3, 1,
				CV_64FC1,
				cv::Scalar(0));

			scaled.at<double>(0) = color[0] / 255.0f;
			scaled.at<double>(1) = color[1] / 255.0f;
			scaled.at<double>(2) = color[2] / 255.0f;

			cv::Mat this_value = eig * scaled;

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

/*what the function does, how to use it*/
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

/*what the function does, how to use it*/
cv::Mat get_viewable_image(cv::Mat classes) {
	const int height = classes.rows;
	const int width = classes.cols;

	const int max_color_count = 12;
	cv::Vec3b *palette = new cv::Vec3b[max_color_count];
	palette[0] = cv::Vec3b(0, 0, 0);
	palette[1] = cv::Vec3b(255, 0, 0);
	palette[2] = cv::Vec3b(0, 255, 0);
	palette[3] = cv::Vec3b(0, 0, 255);
	palette[4] = cv::Vec3b(255, 255, 0);
	palette[5] = cv::Vec3b(0, 255, 255);
	palette[6] = cv::Vec3b(255, 0, 255);
	palette[7] = cv::Vec3b(128, 128, 128);
	palette[8] = cv::Vec3b(128, 255, 128);
	palette[9] = cv::Vec3b(32, 32, 32);
	palette[10] = cv::Vec3b(255, 128, 128);
	palette[11] = cv::Vec3b(128, 128, 255);

	cv::Mat ret = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
	for (int y = 0;y<height;y++) {
		cv::Vec3b *ptr = ret.ptr<cv::Vec3b>(y);
		uchar *ptrClass = classes.ptr<uchar>(y);
		for (int x = 0;x<width;x++) {
			int color = ptrClass[x];
			if (color >= max_color_count) {
				printf("You should increase the number of predefined colors!\n");
				continue;
			}
			ptr[x] = palette[color];
		}
	}

	return ret;
}

/*what the funciton does, how to use it*/
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

		cv::eigen(node->cov, eigenvalues, eigenvectors);
		double val = eigenvalues.at<double>(0);
		if (val > max_eigen) {
			max_eigen = val;
			ret = node;
		}
	}

	return ret;
}

/*what the function does, how to use it*/
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
	get_class_mean_cov(img, classes, root);
	//Each iteration creates a node.
	for (int i = 0;i<count - 1;i++) {
		next = get_max_eigenvalue_node(root);
		partition_class(img, classes, get_next_classid(root), next);
		get_class_mean_cov(img, classes, next->left);
		get_class_mean_cov(img, classes, next->right);
	}
	std::vector<cv::Vec3b> colors = get_dominant_colors(root);
	printf("Dominate color(s) found!\n");
	//images to illustrate results
	//cv::Mat viewable = get_viewable_image(classes);
	cv::Mat quantized = get_quantized_image(classes, root);

	//cv::imwrite("./output/classification.png", viewable);	
	cv::imwrite("C:/Users/Austin Pursley/Desktop/ECEN-403-Smart-Mirror-Image-Analysis/color_eye/output/quantized.png", quantized);

	return colors;
}