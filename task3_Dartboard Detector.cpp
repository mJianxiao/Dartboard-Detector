#include <iostream>
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <math.h>
#include <opencv2/objdetect/objdetect.hpp>
#include <string.h>

using namespace cv;
using namespace std;

void GaussianBlur(
	cv::Mat &input,
	int size,
	cv::Mat &blurredOutput);

/** Function Headers */
void detectAndDisplay(Mat frame);

void detectAndDisplay2(Mat frame, int point_x, int point_y);

/** Global variables */
String cascade_name = "cascade.xml";
CascadeClassifier cascade;

int hough[2000][2000] = {};  // store theta and r
int cross[2000][2000] = {};  // point of intersection

int main()
{
	/* load image */
	string imagename;
	printf("Please input image name: ");
	cin >> imagename;

	Mat image, d_image;
	image = imread(imagename, 1);
	d_image = imread(imagename, 1);
	imshow("original", image);

	Mat gray_image;
	cvtColor(image, gray_image, CV_BGR2GRAY);
	//imshow("gray", gray_image);

	Mat th_image;
	th_image.create(gray_image.size(), gray_image.type());

	/* noise reduction */
	Mat blur_image;
	GaussianBlur(gray_image, 5, blur_image);
	//imshow("blur", blur_image);

	/* edge detection */
	int x, y;
	int gmax = 0;
	for (x = 1; x < image.rows; x++)
	{
		for (y = 1; y < image.cols; y++)
		{
			int a = blur_image.at<uchar>(x, y) - blur_image.at<uchar>(x - 1, y);
			int b = blur_image.at<uchar>(x, y) - blur_image.at<uchar>(x, y - 1);
			int c = a * a + b * b;
			th_image.at<uchar>(x, y) = sqrt(c);

			if (th_image.at<uchar>(x, y) > gmax)
			{
				gmax = th_image.at<uchar>(x, y);
			}

		}
	}
	printf("max gradient = %d\n", gmax);

	/* thresholding */
	int th = 0.7 * gmax;
	for (x = 1; x < image.rows; x++)
	{
		for (y = 1; y < image.cols; y++)
		{
			if (th_image.at<uchar>(x, y) > th)
			{
				th_image.at<uchar>(x, y) = 255;
			}
			else
			{
				th_image.at<uchar>(x, y) = 0;
			}
		}
	}
	for (x = 0; x < image.rows; x++)
	{
		y = 0;
		th_image.at<uchar>(x, y) = 0;
	}
	for (y = 0; y < image.cols; y++)
	{
		x = 0;
		th_image.at<uchar>(x, y) = 0;
	}
	imshow("thresholded", th_image);

	/* create hough space */
	int a = th_image.rows;
	int b = th_image.cols;
	int centerX = a / 2;
	int centerY = b / 2;
	int hough_size = sqrt(((a / 2) *(a / 2) + (b / 2) *(b / 2)));
	printf("hough size = %d\n", hough_size);
	Mat hough_image(b, 2 * hough_size, th_image.type());
	for (int x = 0; x < hough_image.rows; x++)
	{
		for (int y = 0; y < hough_image.cols; y++)
		{
			hough_image.at<uchar>(x, y) = 0;
		}
	}

	/* dart board detection (original) */
	if (!cascade.load(cascade_name)) { printf("--(!)Error loading\n"); return -1; };
	detectAndDisplay(image);
	imshow("detected_1", image);

	/* hough transform */
	float hough_intervals = 2 * 3.1415926 / b; // 2*pi
	int max_hough = 0;
	for (int x = 0; x < th_image.rows; x++)
	{
		for (int y = 0; y < th_image.cols; y++)
		{
			int temp = (int)th_image.at<uchar>(x, y);
			if (temp == 0) continue;
			else
			{
				for (int degree = 0; degree < th_image.cols; degree++)
				{
					int r = (x - centerX) * cos(hough_intervals*degree) + (y - centerY) * sin(hough_intervals*degree);
					int r1 = hough_size + r;

					hough_image.at<uchar>(degree, r1) = 255;

					hough[degree][r1]++;
					if (max_hough < hough[degree][r1])
					{
						max_hough = hough[degree][r1];
					}
				}
			}
		}
	}
	imshow("hough transform", hough_image);
	printf("max_hough = %d\n", max_hough);

	/* line detection */
	int hough_threshold = int(max_hough * 0.6);
	for (int degree = 0; degree < th_image.rows; degree++)
	{
		for (int r2 = 0; r2 < 2 * hough_size; r2++)
		{
			if (hough[degree][r2] < hough_threshold) continue;
			else
			{
				int x1 = (r2 - hough_size) / cos(degree*hough_intervals) - (1 - centerY) * tan(degree*hough_intervals);
				int x2 = (r2 - hough_size) / cos(degree*hough_intervals) - (centerY - 1) * tan(degree*hough_intervals);

				Point b = Point(2 * centerY - 1, x2 + centerX);
				Point a = Point(1, x1 + centerX);
				line(image, b, a, Scalar(255, 120, 0));

				// find intersection point
				Mat line_image;
				line_image.create(image.size(), image.type());
				line(line_image, b, a, Scalar(255, 0, 0));
				for (int x = 0; x < line_image.rows; x++)
				{
					for (int y = 0; y < line_image.cols; y++)
					{
						if (line_image.at<Vec3b>(x, y)[0] == 255)
						{
							cross[x][y]++;
						}
					}
				}
			}
		}
	}

	/* the max number of lines intersect in the same point */
	int max_cross = 5;
	for (x = 0; x < image.rows; x++)
	{
		for (y = 0; y < image.cols; y++)
		{
			if (cross[x][y] > max_cross)
			{
				max_cross = cross[x][y];
			}
		}
	}
	printf("max_cross = %d\n", max_cross);

	/* point of intersection */
	int c1 = 0;
	int c2 = 0;

	for (x = 0; x < image.rows; x++)
	{
		for (y = 0; y < image.cols; y++)
		{
			if (cross[x][y] == max_cross)
			{
				printf("line_intersection_point = (%d,%d)\n", y, x);
				c1 = y;
				c2 = x;
			}
		}
	}
	Point line_cross(c1, c2);
	circle(image, line_cross, 3, Scalar(180, 105, 255), 5);
	imshow("line", image);

	/* hough circles detection */
	vector<Vec3f>circles;
	int circle_g = 200; // threshold value of edge detection
	int circle_h = 160; // threshold value of hough circle
	HoughCircles(gray_image, circles, CV_HOUGH_GRADIENT, 1, 1, circle_g, circle_h, 0);
	for (int i = 0; i<circles.size(); i++)
	{
		Point circleCenter(circles[i][0], circles[i][1]);
		int radius = circles[i][2];
		circle(image, circleCenter, radius, Scalar(60, 20, 220), 2);
		circle(image, circleCenter, 3, Scalar(6, 229, 249), 2);
		c1 = circles[i][0];
		c2 = circles[i][1];
		printf("circle_center_point = (%d,%d)\n", c1, c2);
	}
	/* if no circle detected, decrease threshold value */
	for (; circles.size() == 0; circle_g = circle_g - 20, circle_h = circle_h - 10)
	{
		if (circle_g < 110) break;
		HoughCircles(gray_image, circles, CV_HOUGH_GRADIENT, 1, 1, circle_g, circle_h, 0);
		for (int i = 0; i<circles.size(); i++)
		{
			Point circleCenter(circles[i][0], circles[i][1]);
			int radius = circles[i][2];
			circle(image, circleCenter, radius, Scalar(60, 20, 220), 2);
			circle(image, circleCenter, 3, Scalar(6, 229, 249), 2);
			c1 = circles[i][0];
			c2 = circles[i][1];
			printf("circle_center_point = (%d,%d)\n", c1, c2);
		}
	}
	imshow("circle", image);
	printf("line / circle_point_for_detection= (%d,%d)\n", c1, c2);

	/* dart board detection (improved) */
	if (!cascade.load(cascade_name)) { printf("--(!)Error loading\n"); return -1; };
	if (c1 == 0 && c2 == 0)
	{
		detectAndDisplay(d_image);
	}
	else
	{
		detectAndDisplay2(d_image, c1, c2);
	}
	imshow("detected_2", d_image);
	imwrite("detected.jpg", d_image);

	waitKey();

	return 0;
}

void GaussianBlur(cv::Mat &input, int size, cv::Mat &blurredOutput)
{
	blurredOutput.create(input.size(), input.type());

	cv::Mat kX = cv::getGaussianKernel(size, -1);
	cv::Mat kY = cv::getGaussianKernel(size, -1);

	cv::Mat kernel = kX * kY.t();

	int kernelRadiusX = (kernel.size[0] - 1) / 2;
	int kernelRadiusY = (kernel.size[1] - 1) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder(input, paddedInput,
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE);

	for (int i = 0; i < input.rows; i++)
	{
		for (int j = 0; j < input.cols; j++)
		{
			double sum = 0.0;
			for (int m = -kernelRadiusX; m <= kernelRadiusX; m++)
			{
				for (int n = -kernelRadiusY; n <= kernelRadiusY; n++)
				{
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					int imageval = (int)paddedInput.at<uchar>(imagex, imagey);
					double kernalval = kernel.at<double>(kernelx, kernely);

					sum += imageval * kernalval;
				}
			}
			blurredOutput.at<uchar>(i, j) = (uchar)sum;
		}
	}
}

void detectAndDisplay(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale(frame_gray, faces, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500, 500));

	// 3. Print number of dartboards found
	std::cout << faces.size() << std::endl;

	// 4. Draw box around dartboards found
	for (int i = 0; i < faces.size(); i++)
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(0, 255, 0), 2);
	}

}

void detectAndDisplay2(Mat frame, int point_x, int point_y)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale(frame_gray, faces, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500, 500));

	// 3. Draw box around dartboard found
	int min = 500;
	for (int i = 0; i < faces.size(); i++)
	{
		if (point_x<faces[i].x || point_x>faces[i].x + faces[i].width) continue;
		else if (point_y<faces[i].y || point_y>faces[i].y + faces[i].height) continue;
		else
		{
			if (abs(point_x - faces[i].x - 0.5*faces[i].width) < min)
				min = point_x - faces[i].x;
		}
	}

	for (int i = 0; i < faces.size(); i++)
	{
		if (point_x - faces[i].x == min)
			rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(0, 255, 0), 2);
	}

	// dart board detection based on intersection point
	if (faces.size() == 0)
		rectangle(frame, Point(point_x - 100, point_y - 100), Point(point_x + 100, point_y + 100), Scalar(0, 255, 0), 2);

}