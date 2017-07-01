// Car_plate.cpp : 定义控制台应用程序的入口点。  

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
using namespace cv::ml;
using namespace std;
using namespace cv;


//对minAreaRect获得的最小外接矩形，用纵横比进行判断  
bool verifySizes(RotatedRect mr)
{
	float error = 0.4;
	//Spain car plate size: 52x11 aspect 4,7272  
	float aspect = 4.7272;
	//Set a min and max area. All other patchs are discarded  
	int min = 15 * aspect * 15; // minimum area  
	int max = 125 * aspect * 125; // maximum area  
								  //Get only patchs that match to a respect ratio.  
	float rmin = aspect - aspect*error;
	float rmax = aspect + aspect*error;

	int area = mr.size.height * mr.size.width;
	float r = (float)mr.size.width / (float)mr.size.height;
	if (r<1)
		r = (float)mr.size.height / (float)mr.size.width;

	if ((area < min || area > max) || (r < rmin || r > rmax)) {
		return false;
	}
	else {
		return true;
	}

}

//直方图均衡化  
Mat histeq(Mat in)
{
	Mat out(in.size(), in.type());
	if (in.channels() == 3) {
		Mat hsv;
		vector<Mat> hsvSplit;
		cvtColor(in, hsv, CV_BGR2HSV);
		split(hsv, hsvSplit);
		equalizeHist(hsvSplit[2], hsvSplit[2]);
		merge(hsvSplit, hsv);
		cvtColor(hsv, out, CV_HSV2BGR);
	}
	else if (in.channels() == 1) {
		equalizeHist(in, out);
	}

	return out;

}
int main() {
	Mat srcImg = imread("catPlate//carPlate.jpg",1);
	//预处理
	//灰度化
	Mat srcGray;
	cvtColor(srcImg, srcGray, CV_BGR2GRAY);
	//二值化
	Mat srcHSV;
	cvtColor(srcImg, srcHSV, CV_BGR2HSV);
	vector<Mat> hsvImg;
	Mat hImg, sImg, vImg,threshImg;
	split(srcHSV, hsvImg);
	inRange(hsvImg[0], 94, 115, hImg);
	inRange(hsvImg[1], 90, 255, sImg);
	inRange(hsvImg[2], 36, 255, vImg);
	bitwise_and(hImg, sImg, threshImg);
	bitwise_and(threshImg, vImg, threshImg);
	imshow("二值图像", threshImg);
	//数学形态学除噪
	Mat mathProImg;
	Mat element(1, 1, CV_8U, cv::Scalar(1));
	erode(threshImg, mathProImg, element);
	Mat element2(2, 1, CV_8U, cv::Scalar(1));
	dilate(mathProImg, mathProImg, element2);
	imshow("数学处理后的图像", mathProImg);

	//用大津法进行车牌提取
	Mat ostuImg;
	threshold(srcGray, ostuImg, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	imshow("大津法", ostuImg);

	waitKey(0);
}
