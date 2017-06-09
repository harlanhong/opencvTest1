#include <opencv2/opencv.hpp>   

#include<iostream>    
#define PI 3.1415926    
#define R 150    
using namespace cv;
using namespace std;
int main() {
	Mat t = imread("win10.jpg", 1);
	imshow("dfsd",t);
	waitKey(0);
}