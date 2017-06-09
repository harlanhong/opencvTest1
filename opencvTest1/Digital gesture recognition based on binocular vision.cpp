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
string imgPath = "imgTrain";
string filePath = "config";


//==========the variable of single camera calibrate and the related method================
const int imageWidth = 640;                             //摄像头的分辨率  
const int imageHeight = 480;
const int boardWidth = 8;                               //横向的角点数目  
const int boardHeight = 5;                              //纵向的角点数据  
const int boardCorner = boardWidth * boardHeight;       //总的角点数据  
const int frameNumber = 14;                             //相机标定时需要采用的图像帧数  
const int squareSize = 32;                              //标定板黑白格子的大小 单位mm  
const Size boardSize = Size(boardWidth, boardHeight);   //  
Mat intrinsic;                                          //相机内参数  
Mat distortion_coeff;                                   //相机畸变参数  
vector<Mat> rvecs;                                        //旋转向量  
vector<Mat> tvecs;                                        //平移向量  
vector<vector<Point2f>> corners;                        //各个图像找到的角点的集合 和objRealPoint 一一对应  
vector<vector<Point3f>> objRealPoint;                   //各副图像的角点的实际物理坐标集合  
vector<Point2f> corner;                                   //某一副图像找到的角点  
Mat rgbImage, grayImage;

Mat disparityImage;//视差图
Mat gestureL;//左摄像头拍摄到的手势图像;

			 /*计算标定板上模块的实际物理坐标*/
void calRealPoint(vector<vector<Point3f>>& obj, int boardwidth, int boardheight, int imgNumber, int squaresize)
{
	//  Mat imgpoint(boardheight, boardwidth, CV_32FC3,Scalar(0,0,0));  
	vector<Point3f> imgpoint;
	for (int rowIndex = 0; rowIndex < boardheight; rowIndex++)
	{
		for (int colIndex = 0; colIndex < boardwidth; colIndex++)
		{
			//  imgpoint.at<Vec3f>(rowIndex, colIndex) = Vec3f(rowIndex * squaresize, colIndex*squaresize, 0);  
			imgpoint.push_back(Point3f(colIndex * squaresize, rowIndex * squaresize, 0));
		}
	}
	for (int imgIndex = 0; imgIndex < imgNumber; imgIndex++)
	{
		obj.push_back(imgpoint);
	}
}

/*设置相机的初始参数 也可以不估计*/
void CalibrationEvaluate(void)//标定结束后进行评价
{
	double err = 0;
	double total_err = 0;
	//calibrateCamera(objRealPoint, corners, Size(imageWidth, imageHeight), intrinsic, distortion_coeff, rvecs, tvecs, 0);
	cout << "每幅图像的定标误差：" << endl;
	for (int i = 0; i < corners.size(); i++)
	{
		vector<Point2f> image_points2;
		vector<Point3f> tempPointSet = objRealPoint[i];
		projectPoints(tempPointSet, rvecs[i], tvecs[i], intrinsic, distortion_coeff, image_points2);
		vector<Point2f> tempImagePoint = corners[i];
		Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
		Mat image_points2Mat = Mat(1, image_points2.size(), CV_32FC2);
		for (int j = 0; j < tempImagePoint.size(); j++)
		{
			image_points2Mat.at<Vec2f>(0, j) = Vec2f(image_points2[j].x, image_points2[j].y);
			tempImagePointMat.at<Vec2f>(0, j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
		}
		err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
		total_err = err + total_err;
		cout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
	}
	cout << "总体平均误差：" << total_err / (corners.size() + 1) << "像素" << endl;
}
/*猜测相机参数*/
void guessCameraParam(void)
{
	/*分配内存*/
	intrinsic.create(3, 3, CV_64FC1);
	distortion_coeff.create(5, 1, CV_64FC1);

	/*
	fx 0 cx
	0 fy cy
	0 0  1
	*/
	intrinsic.at<double>(0, 0) = 256.8093262;   //fx         
	intrinsic.at<double>(0, 2) = 160.2826538;   //cx  
	intrinsic.at<double>(1, 1) = 254.7511139;   //fy  
	intrinsic.at<double>(1, 2) = 127.6264572;   //cy  

	intrinsic.at<double>(0, 1) = 0;
	intrinsic.at<double>(1, 0) = 0;
	intrinsic.at<double>(2, 0) = 0;
	intrinsic.at<double>(2, 1) = 0;
	intrinsic.at<double>(2, 2) = 1;

	/*
	k1 k2 p1 p2 p3
	*/
	distortion_coeff.at<double>(0, 0) = -0.193740;  //k1  
	distortion_coeff.at<double>(1, 0) = -0.378588;  //k2  
	distortion_coeff.at<double>(2, 0) = 0.028980;   //p1  
	distortion_coeff.at<double>(3, 0) = 0.008136;   //p2  
	distortion_coeff.at<double>(4, 0) = 0;          //p3  
}
void outputCameraParam(string fileName)
{
	/*保存数据*/
	FileStorage fs(filePath+"\\" + fileName + ".yml", FileStorage::WRITE);   //存yml檔
	fs << "intrinsic" << intrinsic;
	fs << "distcoeff" << distortion_coeff;
	/*输出数据*/
	cout << "fx :" << intrinsic.at<double>(0, 0) << endl << "fy :" << intrinsic.at<double>(1, 1) << endl;
	cout << "cx :" << intrinsic.at<double>(0, 2) << endl << "cy :" << intrinsic.at<double>(1, 2) << endl;

	cout << "k1 :" << distortion_coeff.at<double>(0, 0) << endl;
	cout << "k2 :" << distortion_coeff.at<double>(1, 0) << endl;
	cout << "p1 :" << distortion_coeff.at<double>(2, 0) << endl;
	cout << "p2 :" << distortion_coeff.at<double>(3, 0) << endl;
	cout << "p3 :" << distortion_coeff.at<double>(4, 0) << endl;
}
void singleCameraCalibrate(int cN, string fN) {
	Mat img;
	int goodFrameCount = 0;
	namedWindow("chessboard");
	cout << "按Q退出 ..." << endl;
	VideoCapture capture(cN);
	vector<Mat> imgseq;
	while (goodFrameCount < frameNumber)
	{
		capture >> rgbImage;
		imshow("input", rgbImage);
		int key = waitKey(30);
		if (key == 32) {
			Mat temp;
			rgbImage.copyTo(temp);

			cvtColor(rgbImage, grayImage, CV_BGR2GRAY);

			bool isFind = findChessboardCorners(rgbImage, boardSize, corner, 0);
			if (isFind == true) //所有角点都被找到 说明这幅图像是可行的  
			{
				imgseq.push_back(temp);
				goodFrameCount++;
				/*
				Size(5,5) 搜索窗口的一半大小
				Size(-1,-1) 死区的一半尺寸
				TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1)迭代终止条件
				*/
				cornerSubPix(grayImage, corner, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));
				drawChessboardCorners(rgbImage, boardSize, corner, isFind);
				imshow("chessboard", rgbImage);
				corners.push_back(corner);
				cout << "The image is good" << endl;
			}
			else
			{
				cout << "The image is bad please try again" << endl;
			}

		}
		else if (key == 27)
			return;
	}

	/*
	图像采集完毕 接下来开始摄像头的校正
	calibrateCamera()
	输入参数 objectPoints  角点的实际物理坐标
	imagePoints   角点的图像坐标
	imageSize     图像的大小
	输出参数
	cameraMatrix  相机的内参矩阵
	distCoeffs    相机的畸变参数
	rvecs         旋转矢量(外参数)
	tvecs         平移矢量(外参数）
	*/

	/*设置实际初始参数 根据calibrateCamera来 如果flag = 0 也可以不进行设置*/
	guessCameraParam();
	cout << "guess successful" << endl;
	/*计算实际的校正点的三维坐标*/
	calRealPoint(objRealPoint, boardWidth, boardHeight, frameNumber, squareSize);
	cout << "cal real successful" << endl;
	/*标定摄像头*/
	calibrateCamera(objRealPoint, corners, Size(imageWidth, imageHeight), intrinsic, distortion_coeff, rvecs, tvecs, 0);
	cout << "calibration successful" << endl;
	/*保存并输出参数*/
	outputCameraParam(fN);
	CalibrationEvaluate();
	cout << "out successful" << endl;
	FileStorage fs(filePath+"\\" + fN + ".yml", FileStorage::WRITE);   //存yml檔
	fs << "intrinsic" << intrinsic;
	fs << "distcoeff" << distortion_coeff;
	/*显示畸变校正效果*/
	Mat cImage;
	undistort(rgbImage, cImage, intrinsic, distortion_coeff);
	imshow("Corret Image", cImage);
	cout << "Correct Image" << endl;
	cout << "Wait for Key" << endl;

	/********图像坐标到世界坐标*******/
	int  image_idx = 9;
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
	Rodrigues(rvecs[image_idx], rotation_matrix);

	////自己加了一个旋转矩阵
	//Mat rotation = (Mat_<double>(3, 3) << 0,1,0,-1,0,0,0,0,1);
	//rotation_matrix = rotation_matrix*rotation;
	////自己加了一个旋转矩阵，完

	cv::Mat H(3, 3, CV_32FC1, Scalar::all(0));
	cv::Mat translation_ve;
	tvecs[image_idx].copyTo(translation_ve);
	rotation_matrix.copyTo(H);
	H.at<double>(0, 2) = translation_ve.at<double>(0, 0);
	H.at<double>(1, 2) = translation_ve.at<double>(1, 0);
	H.at<double>(2, 2) = translation_ve.at<double>(2, 0);
	cv::Mat hu;
	hu = intrinsic*H;
	cv::Mat hu2 = hu.inv();
	double a1, a2, a3, a4, a5, a6, a7, a8, a9;
	a1 = hu2.at<double>(0, 0);
	a2 = hu2.at<double>(0, 1);
	a3 = hu2.at<double>(0, 2);
	a4 = hu2.at<double>(1, 0);
	a5 = hu2.at<double>(1, 1);
	a6 = hu2.at<double>(1, 2);
	a7 = hu2.at<double>(2, 0);
	a8 = hu2.at<double>(2, 1);
	a9 = hu2.at<double>(2, 2);



	/*显示一张原图，矫正之后，再寻找角点*/
	Mat show_tuxiang_gray;//imread("e:\\opencv\\calibrate\\chess" + to_string(image_idx) + ".bmp", 0);//显示图像角点
	cvtColor(imgseq[image_idx], show_tuxiang_gray, CV_RGB2GRAY);
	Mat show_tuxiang_rgb = imgseq[image_idx];// ("e:\\opencv\\calibrate\\chess" + to_string(image_idx) + ".bmp", 1);//显示图像角点
	Mat show_gray;
	Mat show_rgb;
	undistort(show_tuxiang_gray, show_gray, intrinsic, distortion_coeff);
	undistort(show_tuxiang_rgb, show_rgb, intrinsic, distortion_coeff);
	imshow("show_gray", show_gray);
	corner.clear();
	bool isFind = findChessboardCorners(show_gray, boardSize, corner, 0);
	//bool isFind = findChessboardCorners(show_gray, boardSize, corner, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
	//cornerSubPix(show_gray, corner, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));
	cornerSubPix(show_gray, corner, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
	//drawChessboardCorners(show_rgb, boardSize, corner, isFind);
	Point tem;
	for (int i = 0; i < boardCorner; i++)
	{
		tem.x = cvRound(corner[i].x);
		tem.y = cvRound(corner[i].y);

		circle(show_rgb, tem, 3, Scalar(0, 0, 255), -1, 8);
	}
	imshow("图像坐标系", show_rgb);

	/*相机矫正后的结果*/
	Point2f kk;
	vector<Point2f>shijie;
	for (int i = 0; i < boardCorner; i++)
	{
		int xe = corner[i].x;//图像中点坐标x
		int ye = corner[i].y;//图像中点坐标y
		kk.x = (a1*xe + a2*ye + a3) / (a7*xe + a8*ye + a9);//世界坐标中x值
		kk.y = (a4*xe + a5*ye + a6) / (a7*xe + a8*ye + a9);//世界坐标中Y值
		shijie.push_back(kk);

	}
	Mat show_shijie = Mat::zeros(480, 640, CV_8UC3);
	for (int i = 0; i < boardCorner; i++)
	{
		tem.x = cvRound(shijie[i].x) + 220;
		tem.y = cvRound(shijie[i].y) + 120;

		circle(show_shijie, tem, 3, Scalar(0, 0, 255), -1, 8);
	}
	imshow("返回到世界坐标系—有矫正", show_shijie);


	/*对比没有进行相机矫正的结果*/
	shijie.clear();
	for (int i = 0; i < boardCorner; i++)
	{
		int xe = corners[image_idx][i].x;//图像中点坐标x
		int ye = corners[image_idx][i].y;//图像中点坐标y
		kk.x = (a1*xe + a2*ye + a3) / (a7*xe + a8*ye + a9);//世界坐标中x值
		kk.y = (a4*xe + a5*ye + a6) / (a7*xe + a8*ye + a9);//世界坐标中Y值
		shijie.push_back(kk);

	}
	Mat show_shijie2 = Mat::zeros(480, 640, CV_8UC3);
	for (int i = 0; i < boardCorner; i++)
	{
		tem.x = cvRound(shijie[i].x) + 220;
		tem.y = cvRound(shijie[i].y) + 120;

		circle(show_shijie2, tem, 3, Scalar(0, 0, 255), -1, 8);
	}
	imshow("返回到世界坐标系-无矫正", show_shijie2);
	/*************图像坐标到世界坐标结束***********************/
	capture.release();
	return;
}

//==========the variable of two camera calibrate and the related method================


Size imageSize = Size(imageWidth, imageHeight);

Mat R, T, E, F;											//R 旋转矢量 T平移矢量 E本征矩阵 F基础矩阵
vector<vector<Point2f>> imagePointL;				    //左边摄像机所有照片角点的坐标集合
vector<vector<Point2f>> imagePointR;					//右边摄像机所有照片角点的坐标集合
vector<Point2f> cornerL;								//左边摄像机某一照片角点坐标集合
vector<Point2f> cornerR;								//右边摄像机某一照片角点坐标集合
Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;
Mat Rl, Rr, Pl, Pr, Q;									//校正旋转矩阵R，投影矩阵P 重投影矩阵Q (下面有具体的含义解释）	
Mat mapLx, mapLy, mapRx, mapRy;							//映射表
Rect validROIL, validROIR;								//图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域
Mat cameraMatrixL;
Mat distCoeffL;
Mat cameraMatrixR;
Mat distCoeffR;

//用单目标定得到的值填充
void getCameraMatrixAndDistCoeff(string left, string right) {
	FileStorage frL(filePath+"\\" + left + ".yml", FileStorage::READ);   //讀yml檔
	frL["intrinsic"] >> cameraMatrixL;
	frL["distcoeff"] >> distCoeffL;
	FileStorage frR(filePath+"\\" + right + ".yml", FileStorage::READ);   //讀yml檔
	frR["intrinsic"] >> cameraMatrixR;
	frR["distcoeff"] >> distCoeffR;
}

//初始化块匹配参数
Ptr<StereoBM> bm = StereoBM::create(16, 9);
int blockSize = 8;
int uniquenessRatio = 6;
int numDisparities = 5;

Mat GrectifyImageL, GrectifyImageR;
//三维坐标
Mat xyz;
//提取手势阈值
int thresholdV = 170;
bool is_thresh = false;
int getMaxValue(Mat src) {
	int max = 0;
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			if (src.at<int>(i, j) > max) {
				max = src.at<int>(i, j);
			}
		}
	}
	return max;
}
void initStereoBM(int, void*) {

	bm->setBlockSize(2 * blockSize + 5);     //SAD窗口大小，5~21之间为宜
	bm->setROI1(validROIL);
	bm->setROI2(validROIR);
	bm->setPreFilterCap(31);
	bm->setMinDisparity(0);  //最小视差，默认值为0, 可以是负值，int型
	bm->setNumDisparities(numDisparities * 16 + 16);//视差窗口，即最大视差值与最小视差值之差,窗口大小必须是16的整数倍，int型
	bm->setTextureThreshold(10);
	bm->setUniquenessRatio(uniquenessRatio);//uniquenessRatio主要可以防止误匹配
	bm->setSpeckleWindowSize(100);
	bm->setSpeckleRange(32);
	bm->setDisp12MaxDiff(-1);
	Mat disp, disp8;
	bm->compute(GrectifyImageL, GrectifyImageR, disp);//输入图像必须为灰度图
	disp.convertTo(disp8, CV_8U, 255 / ((numDisparities * 16 + 16)*16.));//计算出的视差是CV_16S格式
	reprojectImageTo3D(disp, xyz, Q, true); //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
	xyz = xyz * 16;
	if (is_thresh) {
		//double max = getMaxValue(disp8);
		threshold(disp8, disp8, 240,0, THRESH_TOZERO_INV);
		threshold(disp8, disp8, thresholdV, 255, THRESH_BINARY);
	}
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(disp8, disp8, element);
	erode(disp8, disp8, element);
	dilate(disp8, disp8, element);
	imshow("disparity", disp8);
	disp8.copyTo(disparityImage);
	
}

void outputCameraParam2(string intrinsics, string extrinsics)
{
	/*保存数据*/
	/*输出数据*/
	FileStorage fs(filePath+"\\" + intrinsics + ".yml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "cameraMatrixL" << cameraMatrixL << "cameraDistcoeffL" << distCoeffL << "cameraMatrixR" << cameraMatrixR << "cameraDistcoeffR" << distCoeffR;
		fs.release();
		cout << "cameraMatrixL=:" << cameraMatrixL << endl << "cameraDistcoeffL=:" << distCoeffL << endl << "cameraMatrixR=:" << cameraMatrixR << endl << "cameraDistcoeffR=:" << distCoeffR << endl;
	}
	else
	{
		cout << "Error: can not save the intrinsics!!!!!" << endl;
	}

	fs.open(filePath+"\\" + extrinsics + ".yml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "R" << R << "T" << T << "Rl" << Rl << "Rr" << Rr << "Pl" << Pl << "Pr" << Pr << "Q" << Q << "ROIL" << validROIL << "ROIR" << validROIR;
		cout << "R=" << R << endl << "T=" << T << endl << "Rl=" << Rl << endl << "Rr=" << Rr << endl << "Pl=" << Pl << endl << "Pr=" << Pr << endl << "Q=" << Q << endl;
		fs.release();
	}
	else
		cout << "Error: can not save the extrinsic parameters\n";
}
/*****描述：鼠标操作回调*****/
bool selectObject = false;
Rect selection;
Point origin;
static void onMouse(int event, int x, int y, int, void*)
{
	if (selectObject)
	{
		selection.x = MIN(x, origin.x);
		selection.y = MIN(y, origin.y);
		selection.width = std::abs(x - origin.x);
		selection.height = std::abs(y - origin.y);
	}

	switch (event)
	{
	case EVENT_LBUTTONDOWN:   //鼠标左按钮按下的事件
		origin = Point(x, y);
		selection = Rect(x, y, 0, 0);
		selectObject = true;
		cout << origin << "in world coordinate is: " << xyz.at<Vec3f>(origin) << endl;
		break;
	case EVENT_LBUTTONUP:    //鼠标左按钮释放的事件
		selectObject = false;
		if (selection.width > 0 && selection.height > 0)
			break;
	}
}

void TwoCameraCalibrate(int cameraL, string left, int cameraR, string right, string intrinsics, string extrinsics) {
	rvecs.clear();
	tvecs.clear();
	objRealPoint.clear();
	Mat img;
	int goodFrameCount = 0;
	namedWindow("ImageL");
	namedWindow("ImageR");
	getCameraMatrixAndDistCoeff(left, right);
	cout << "按Q退出 ..." << endl;
	VideoCapture capture1(cameraL);
	VideoCapture capture2(cameraR);
	while (goodFrameCount < frameNumber)
	{
		capture1 >> rgbImageL;
		capture2 >> rgbImageR;
		imshow("L", rgbImageL);
		imshow("R", rgbImageR);
		int key = waitKey(30);
		if (key == 32) {
			/*读取左边的图像*/
			cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);
			/*读取右边的图像*/
			cvtColor(rgbImageR, grayImageR, CV_BGR2GRAY);
			bool isFindL = findChessboardCorners(rgbImageL, boardSize, cornerL, 0);
			bool isFindR = findChessboardCorners(rgbImageR, boardSize, cornerR, 0);
			if (isFindL == true && isFindR == true)	 //如果两幅图像都找到了所有的角点 则说明这两幅图像是可行的
			{
				/*
				Size(5,5) 搜索窗口的一半大小
				Size(-1,-1) 死区的一半尺寸
				TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1)迭代终止条件
				*/
				cornerSubPix(grayImageL, cornerL, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));
				drawChessboardCorners(rgbImageL, boardSize, cornerL, isFindL);
				imshow("chessboardL", rgbImageL);
				imagePointL.push_back(cornerL);
				cornerSubPix(grayImageR, cornerR, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));
				drawChessboardCorners(rgbImageR, boardSize, cornerR, isFindR);
				imshow("chessboardR", rgbImageR);
				imagePointR.push_back(cornerR);
				goodFrameCount++;
				cout << "The image is good" << endl;
			}
			else
			{
				cout << "The image is bad please try again" << endl;
			}
		}
		else if (key == 27)
			return;
	}

	/*
	计算实际的校正点的三维坐标
	根据实际标定格子的大小来设置
	*/
	calRealPoint(objRealPoint, boardWidth, boardHeight, frameNumber, squareSize);
	cout << "cal real successful" << endl;

	/*
	标定摄像头
	由于左右摄像机分别都经过了单目标定
	所以在此处选择flag = CALIB_USE_INTRINSIC_GUESS
	*/
	double rms = stereoCalibrate(objRealPoint, imagePointL, imagePointR,
		cameraMatrixL, distCoeffL,
		cameraMatrixR, distCoeffR,
		Size(imageWidth, imageHeight), R, T, E, F,
		CALIB_USE_INTRINSIC_GUESS,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));

	cout << "Stereo Calibration done with RMS error = " << rms << endl;

	/*
	立体校正的时候需要两幅图像共面并且行对准 以使得立体匹配更加的可靠
	使得两幅图像共面的方法就是把两个摄像头的图像投影到一个公共成像面上，这样每幅图像从本图像平面投影到公共图像平面都需要一个旋转矩阵R
	stereoRectify 这个函数计算的就是从图像平面投影都公共成像平面的旋转矩阵Rl,Rr。 Rl,Rr即为左右相机平面行对准的校正旋转矩阵。
	左相机经过Rl旋转，右相机经过Rr旋转之后，两幅图像就已经共面并且行对准了。
	其中Pl,Pr为两个相机的投影矩阵，其作用是将3D点的坐标转换到图像的2D点的坐标:P*[X Y Z 1]' =[x y w]
	Q矩阵为重投影矩阵，即矩阵Q可以把2维平面(图像平面)上的点投影到3维空间的点:Q*[x y d 1] = [X Y Z W]。其中d为左右两幅图像的时差
	*/
	stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q,
		CALIB_ZERO_DISPARITY, -1, imageSize, &validROIL, &validROIR);
	/*
	根据stereoRectify 计算出来的R 和 P 来计算图像的映射表 mapx,mapy
	mapx,mapy这两个映射表接下来可以给remap()函数调用，来校正图像，使得两幅图像共面并且行对准
	ininUndistortRectifyMap()的参数newCameraMatrix就是校正后的摄像机矩阵。在openCV里面，校正后的计算机矩阵Mrect是跟投影矩阵P一起返回的。
	所以我们在这里传入投影矩阵P，此函数可以从投影矩阵P中读出校正后的摄像机矩阵
	*/
	initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pr, imageSize, CV_32FC1, mapLx, mapLy);
	initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);
	Mat rectifyImageL, rectifyImageR;
	cvtColor(grayImageL, rectifyImageL, CV_GRAY2BGR);
	cvtColor(grayImageR, rectifyImageR, CV_GRAY2BGR);
	imshow("Rectify Before", rectifyImageL);
	/*
	经过remap之后，左右相机的图像已经共面并且行对准了
	*/
	remap(rectifyImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
	remap(rectifyImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);
	imshow("ImageL", rectifyImageL);
	imshow("ImageR", rectifyImageR);
	/*保存并输出数据*/
	outputCameraParam2(intrinsics, extrinsics);
	//进行测试
	cout << "请输入图像" << endl;

	Mat testL, testR;
	while (1) {
		int key = waitKey(30);
		capture1 >> testL;
		capture2 >> testR;
		imshow("testL", testL);
		imshow("testR", testR);
		if (key == 32) {
			break;
		}
	}
	imshow("testL", testL);
	imshow("testR", testR);
	remap(testL, testL, mapLx, mapLy, INTER_LINEAR);
	remap(testR, testR, mapRx, mapRy, INTER_LINEAR);
	//进行深度测距===================================
	cvtColor(testL, GrectifyImageL, CV_BGR2GRAY);
	cvtColor(testR, GrectifyImageR, CV_BGR2GRAY);
	// 创建 CvStereoBMState 类的一个实例 BMState，进行双目匹配
	namedWindow("disparity", CV_WINDOW_AUTOSIZE);
	// 创建SAD窗口 Trackbar
	createTrackbar("BlockSize:\n", "disparity", &blockSize, 8, initStereoBM);
	// 创建视差唯一性百分比窗口 Trackbar
	createTrackbar("UniquenessRatio:\n", "disparity", &uniquenessRatio, 50, initStereoBM);
	// 创建视差窗口 Trackbar
	createTrackbar("NumDisparities:\n", "disparity", &numDisparities, 16, initStereoBM);
	// 创建二值化窗口 Trackbar
	createTrackbar("Threshold:\n", "disparity", &thresholdV, 255, initStereoBM);
	setMouseCallback("disparity", onMouse, 0);
	initStereoBM(0, 0);
	//===============================================

	/*
	把校正结果显示出来
	把左右两幅图像显示到同一个画面上
	这里只显示了最后一副图像的校正结果。并没有把所有的图像都显示出来
	*/
	Mat canvas;
	double sf;
	int w, h;
	sf = 600. / MAX(imageSize.width, imageSize.height);
	w = cvRound(imageSize.width * sf);
	h = cvRound(imageSize.height * sf);
	canvas.create(h, w * 2, CV_8UC3);

	/*左图像画到画布上*/
	Mat canvasPart = canvas(Rect(w * 0, 0, w, h));								//得到画布的一部分
	resize(rectifyImageL, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);		//把图像缩放到跟canvasPart一样大小
	Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),				//获得被截取的区域	
		cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));
	rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);						//画上一个矩形

	cout << "Painted ImageL" << endl;

	/*右图像画到画布上*/
	canvasPart = canvas(Rect(w, 0, w, h));										//获得画布的另一部分
	resize(rectifyImageR, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
	Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y*sf),
		cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
	rectangle(canvasPart, vroiR, Scalar(0, 255, 0), 3, 8);

	cout << "Painted ImageR" << endl;

	/*画上对应的线条*/
	for (int i = 0; i < canvas.rows; i += 16)
		line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);

	imshow("rectified", canvas);

	capture1.release();
	capture2.release();
	return;
}


//测试代码
//单目进行测试
void singleTest(int cNo, string fName) {
	//
	Mat m_K1, m_K2, m_K3;
	Mat m_DC1, m_DC2, m_DC3;
	FileStorage fr(filePath+"\\" + fName + ".yml", FileStorage::READ);   //讀yml檔
	fr["intrinsic"] >> m_K1;
	fr["distcoeff"] >> m_DC1;

	VideoCapture capture(cNo);
	while (1) {
		Mat temp;
		capture >> temp;
		imshow("1", temp);
		int key = waitKey(30);
		if (key == 32) {
			imshow("init", temp);
			Mat cImage1;
			undistort(temp, cImage1, m_K1, m_DC1);
			imshow("crrect4", cImage1);
		}
		else if (key == 27)
			break;
	}
	capture.release();
	return;
}

//双目测试
void twoTest(string intrinsics, string extrinsics, int cameraL, int cameraR) {

	/*保存数据*/
	/*输出数据*/
	FileStorage fs(filePath+"\\" + intrinsics + ".yml", FileStorage::READ);
	if (fs.isOpened())
	{

		fs["cameraMatrixL"] >> cameraMatrixL;
		fs["cameraDistcoeffL"] >> distCoeffL;
		fs["cameraMatrixR"] >> cameraMatrixR;
		fs["cameraDistcoeffR"] >> distCoeffR;
		fs.release();
		cout << "cameraMatrixL=:" << cameraMatrixL << endl << "cameraDistcoeffL=:" << distCoeffL << endl << "cameraMatrixR=:" << cameraMatrixR << endl << "cameraDistcoeffR=:" << distCoeffR << endl;
	}
	else
	{
		cout << "Error: can not save the intrinsics!!!!!" << endl;
	}

	fs.open(filePath+"\\" + extrinsics + ".yml", FileStorage::READ);
	if (fs.isOpened())
	{
		fs["R"] >> R;
		fs["T"] >> T;
		fs["Rl"] >> Rl;
		fs["Rr"] >> Rr;
		fs["Pl"] >> Pl;
		fs["Pr"] >> Pr;
		fs["Q"] >> Q;
		fs["ROIL"] >> validROIL;
		fs["ROIR"] >> validROIR;
		cout << "R=" << R << endl << "T=" << T << endl << "Rl=" << Rl << endl << "Rr=" << Rr << endl << "Pl=" << Pl << endl << "Pr=" << Pr << endl << "Q=" << Q << endl;
		cout << validROIL;
		fs.release();
	}
	else
		cout << "Error: can not save the extrinsic parameters\n";
	Size imageSize = Size(640, 480);
	Mat mapLx, mapLy, mapRx, mapRy;
	initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pr, imageSize, CV_32FC1, mapLx, mapLy);
	initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);
	VideoCapture captureL(cameraL);
	VideoCapture captureR(cameraR);
	Mat imageL, imageR;
	namedWindow("disparity", 1);
	// 创建 CvStereoBMState 类的一个实例 BMState，进行双目匹配
	namedWindow("disparity", CV_WINDOW_AUTOSIZE);
	// 创建SAD窗口 Trackbar
	createTrackbar("BlockSize:\n", "disparity", &blockSize, 10, initStereoBM);
	// 创建视差唯一性百分比窗口 Trackbar
	createTrackbar("UniquenessRatio:\n", "disparity", &uniquenessRatio, 50, initStereoBM);
	// 创建视差窗口 Trackbar
	createTrackbar("NumDisparities:\n", "disparity", &numDisparities, 16, initStereoBM);
	createTrackbar("threshold", "disparity", &thresholdV, 255, initStereoBM);
	setMouseCallback("disparity", onMouse, 0);
	while (1) {
		captureL >> imageL;
		captureR >> imageR;
		imshow("imageL", imageL);
		imshow("imageR", imageR);
		int key = waitKey(30);
		if (key == 32) {

			remap(imageL, imageL, mapLx, mapLy, INTER_LINEAR);
			remap(imageR, imageR, mapRx, mapRy, INTER_LINEAR);
			cvtColor(imageL, GrectifyImageL, CV_BGR2GRAY);
			cvtColor(imageR, GrectifyImageR, CV_BGR2GRAY);
			initStereoBM(0, 0);

		}
		else if (key == 27)
			break;
	}
	captureL.release();
	captureR.release();
	return;
}

//手势识别代码


//肤色提取1：hsv
int Hlow, Hhigh;
void skinDetectionHSV(Mat src, Mat& dst) {
	Mat hsv;
	cvtColor(src, hsv, CV_BGR2HSV);
	Mat output_mask = Mat::zeros(src.size(), CV_8UC1);

	for (int i = 0; i < hsv.rows; ++i) {
		uchar* p = (uchar*)output_mask.ptr<uchar>(i);
		for (int j = 0; j < hsv.cols; ++j) {
			if (hsv.at<cv::Vec3b>(i, j)[0]>Hlow && hsv.at<cv::Vec3b>(i, j)[0] < Hhigh)
				p[j] = 255;
		}
	}
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	morphologyEx(output_mask, output_mask, MORPH_CLOSE, element);
	dst = output_mask;
	Mat detected_edges;
	/// 使用 3x3内核降噪
	cv::erode(dst, dst, cv::Mat());
	cv::erode(dst, dst, cv::Mat());
	blur(dst, dst, Size(3, 3));

	///// 运行Canny算子
	//Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, 3);
	//imshow("canny", detected_edges);
}

//肤色检测：hsv椭圆
void skinDetectionYCbCr2(Mat src, Mat& dst) {
	vector<Mat> YCbCr;
	Mat ycrcb_image;
	cvtColor(src, ycrcb_image, CV_BGR2YCrCb); //首先转换成到YCrCb空间 
	split(ycrcb_image, YCbCr);
	Mat output_mask = Mat::zeros(src.size(), CV_8UC1);
	for (int i = 0; i < src.rows; i++) //利用椭圆皮肤模型进行皮肤检测  
	{
		uchar* p = (uchar*)output_mask.ptr<uchar>(i);
		Vec3b* ycrcb = (Vec3b*)ycrcb_image.ptr<Vec3b>(i);
		for (int j = 0; j < src.cols; j++)
		{
			if (ycrcb[j][1] >= 133 && ycrcb[j][1] <= 173 && ycrcb[j][2] >= 77 && ycrcb[j][2] <= 127)
				p[j] = 255;
		}
	}
	dst = output_mask;
}

//肤色检测：阈值分割
int thresholdDetectP = 149;
int thresholdDetectV = 114;
void thresholdDivision(Mat src, Mat& dst) {
	Mat temp1, temp2, temp3;
	cvtColor(src, temp1, CV_BGR2GRAY);
	//src.copyTo(temp1);
	threshold(temp1, temp2, thresholdDetectV, 255, THRESH_BINARY);
	threshold(temp1, temp3, thresholdDetectP, 255, THRESH_BINARY_INV);
	bitwise_and(temp3, temp2, dst);
}
void nothing(int, void*) {
}
//获取训练数据
void getTrainData2( Mat &train_data, Mat &train_label) {
	int totalSample = 421;
	train_data = Mat(totalSample, 1764, CV_32FC1, Scalar::all(1)); //初始化标注
	HOGDescriptor *hog = new HOGDescriptor(cvSize(64, 64), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
	//train_data.convertTo(train_data, CV_32FC1);
	Mat image;
	for (int i = 0; i < totalSample; ++i) {
		vector<float> descriptors;
		image = imread(imgPath+"\\"+ to_string(i) + ".jpg", 1);
		Size sz(64, 64);

		resize(image, image, sz);
		cvtColor(image, image, CV_BGR2GRAY);
		//Mat drawing = Mat::zeros(image.size(), CV_8UC1);

		hog->compute(image, descriptors, Size(1, 1), Size(0, 0));
		cout << descriptors.size() << endl;
		for (int j = 0; j < descriptors.size(); ++j)
			train_data.at<float>(i, j) = descriptors[j];
		descriptors.clear();
		cout << "i:" << i << endl;
	}
	train_label = Mat(totalSample, 1, CV_32S, Scalar::all(1)); //初始化标注
	for (int i = 0; i < totalSample; ++i) {
		train_label.at<int>(i, 0) = i % 10;
	}
}
//进行预测。
bool have_model = true;
float svm(Mat img)
{
	Ptr<SVM> svm = SVM::create();
	if (!have_model) {
		Mat train_data, train_label;
		getTrainData2(train_data, train_label); //

													  // 设置参数		
		svm->setType(SVM::C_SVC);
		svm->setKernel(SVM::LINEAR);

		// 训练分类器
		Ptr<TrainData> tData = TrainData::create(train_data, ROW_SAMPLE, train_label);
		svm->train(tData);
		svm->save(filePath+"\\" + "model_normal.yml");
		have_model = true;
	}
	else {
		svm = StatModel::load<SVM>(filePath + "\\" + "model_normal.yml");
	}

	HOGDescriptor *hog = new HOGDescriptor(cvSize(64, 64), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);


	Size sz(64, 64);

	resize(img, img, sz);

	vector<float> descriptors;
	hog->compute(img, descriptors, Size(1, 1), Size(0, 0));
	Mat predict_data = Mat(1, 1764, CV_32FC1, Scalar::all(1)); //初始化标
	for (int j = 0; j < descriptors.size(); ++j)
		predict_data.at<float>(0, j) = descriptors[j];
	float response = svm->predict(predict_data);
	return response;
}
// 去除二值图像边缘的突出部
//uthreshold、vthreshold分别表示突出部的宽度阈值和高度阈值  
//type代表突出部的颜色，0表示黑色，1代表白色   
void delete_jut(Mat src, Mat& dst, int uthreshold, int vthreshold, int type)
{
	int threshold;
	src.copyTo(dst);
	int height = dst.rows;
	int width = dst.cols;
	int k;  //用于循环计数传递到外部  
	for (int i = 0; i < height - 1; i++)
	{
		uchar* p = dst.ptr<uchar>(i);
		for (int j = 0; j < width - 1; j++)
		{
			if (type == 0)
			{
				//行消除  
				if (p[j] == 255 && p[j + 1] == 0)
				{
					if (j + uthreshold >= width)
					{
						for (int k = j + 1; k < width; k++)
							p[k] = 255;
					}
					else
					{
						for (k = j + 2; k <= j + uthreshold; k++)
						{
							if (p[k] == 255) break;
						}
						if (p[k] == 255)
						{
							for (int h = j + 1; h < k; h++)
								p[h] = 255;
						}
					}
				}
				//列消除  
				if (p[j] == 255 && p[j + width] == 0)
				{
					if (i + vthreshold >= height)
					{
						for (k = j + width; k < j + (height - i)*width; k += width)
							p[k] = 255;
					}
					else
					{
						for (k = j + 2 * width; k <= j + vthreshold*width; k += width)
						{
							if (p[k] == 255) break;
						}
						if (p[k] == 255)
						{
							for (int h = j + width; h < k; h += width)
								p[h] = 255;
						}
					}
				}
			}
			else  //type = 1  
			{
				//行消除  
				if (p[j] == 0 && p[j + 1] == 255)
				{
					if (j + uthreshold >= width)
					{
						for (int k = j + 1; k < width; k++)
							p[k] = 0;
					}
					else
					{
						for (k = j + 2; k <= j + uthreshold; k++)
						{
							if (p[k] == 0) break;
						}
						if (p[k] == 0)
						{
							for (int h = j + 1; h < k; h++)
								p[h] = 0;
						}
					}
				}
				//列消除  
				if (p[j] == 0 && p[j + width] == 255)
				{
					if (i + vthreshold >= height)
					{
						for (k = j + width; k < j + (height - i)*width; k += width)
							p[k] = 0;
					}
					else
					{
						for (k = j + 2 * width; k <= j + vthreshold*width; k += width)
						{
							if (p[k] == 0) break;
						}
						if (p[k] == 0)
						{
							for (int h = j + width; h < k; h += width)
								p[h] = 0;
						}
					}
				}
			}
		}
	}
}
//图片边缘光滑处理  
//size表示取均值的窗口大小，threshold表示对均值图像进行二值化的阈值  
void imageblur(Mat src, Mat& dst, Size size, int threshold)
{
	int height = src.rows;
	int width = src.cols;
	blur(src, dst, size);
	for (int i = 0; i < height; i++)
	{
		uchar* p = dst.ptr<uchar>(i);
		for (int j = 0; j < width; j++)
		{
			if (p[j] < threshold)
				p[j] = 0;
			else p[j] = 255;
		}
	}
}
void fillHole(const Mat srcBw, Mat &dstBw)
{
	Size m_Size = srcBw.size();
	Mat Temp = Mat::zeros(m_Size.height + 2, m_Size.width + 2, srcBw.type());//延展图像

	srcBw.copyTo(Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)));

	cv::floodFill(Temp, Point(0, 0), Scalar(255));

	Mat cutImg;//裁剪延展的图像
	Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)).copyTo(cutImg);

	dstBw = srcBw | (~cutImg);

}
RNG rng(12345);
void gestureDetection(string intrinsics, string extrinsics, int cameraL, int cameraR) {
	namedWindow("thresholdWindow", 1);

	/*保存数据*/
	/*输出数据*/
	FileStorage fs(filePath+"\\" + intrinsics + ".yml", FileStorage::READ);
	if (fs.isOpened())
	{

		fs["cameraMatrixL"] >> cameraMatrixL;
		fs["cameraDistcoeffL"] >> distCoeffL;
		fs["cameraMatrixR"] >> cameraMatrixR;
		fs["cameraDistcoeffR"] >> distCoeffR;
		fs.release();
	}
	else
	{
		cout << "Error: can not save the intrinsics!!!!!" << endl;
	}

	fs.open(filePath+"\\" + extrinsics + ".yml", FileStorage::READ);
	if (fs.isOpened())
	{
		fs["R"] >> R;
		fs["T"] >> T;
		fs["Rl"] >> Rl;
		fs["Rr"] >> Rr;
		fs["Pl"] >> Pl;
		fs["Pr"] >> Pr;
		fs["Q"] >> Q;
		fs["ROIL"] >> validROIL;
		fs["ROIR"] >> validROIR;
		fs.release();
	}
	else
		cout << "Error: can not save the extrinsic parameters\n";
	Size imageSize = Size(640, 480);
	Mat mapLx, mapLy, mapRx, mapRy;
	initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pr, imageSize, CV_32FC1, mapLx, mapLy);
	initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);
	VideoCapture captureL(cameraL);
	VideoCapture captureR(cameraR);
	Mat imageL, imageR;
	namedWindow("disparity", 1);
	// 创建 CvStereoBMState 类的一个实例 BMState，进行双目匹配
	namedWindow("disparity", CV_WINDOW_AUTOSIZE);
	// 创建SAD窗口 Trackbar
	createTrackbar("BlockSize:\n", "disparity", &blockSize, 10, initStereoBM);
	// 创建视差唯一性百分比窗口 Trackbar
	createTrackbar("UniquenessRatio:\n", "disparity", &uniquenessRatio, 50, initStereoBM);
	// 创建视差窗口 Trackbar
	createTrackbar("NumDisparities:\n", "disparity", &numDisparities, 16, initStereoBM);
	createTrackbar("threshold", "disparity", &thresholdV, 255, initStereoBM);
	setMouseCallback("disparity", onMouse, 0);
	int k = 397;
	while (1) {
		captureL >> imageL;
		captureR >> imageR;
		imshow("imageL", imageL);
		imshow("imageR", imageR);
		imageL.copyTo(gestureL);
		remap(imageL, imageL, mapLx, mapLy, INTER_LINEAR);
		remap(imageR, imageR, mapRx, mapRy, INTER_LINEAR);
		cvtColor(imageL, GrectifyImageL, CV_BGR2GRAY);
		cvtColor(imageR, GrectifyImageR, CV_BGR2GRAY);
		initStereoBM(0, 0);
		createTrackbar("low", "thresholdWindow", &thresholdDetectV, 255, nothing);
		createTrackbar("high", "thresholdWindow", &thresholdDetectP, 255, nothing);
		Mat division;
		skinDetectionYCbCr2(gestureL, division);
		imshow("thresholdWindow", division);

		int key = waitKey(30);
		if (key == 32) {

			Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
			dilate(disparityImage, disparityImage, element);
			fillHole(disparityImage, disparityImage);
			delete_jut(disparityImage, disparityImage, 40, 40, 0);
			
			dilate(division, division, element);
			dilate(division, division, element);
			dilate(division, division, element);
			Mat finalyImg;
			bitwise_and(division, disparityImage, finalyImg);
			Mat temp, temp1;
			dilate(finalyImg, temp, element);
			fillHole(temp, temp1);
			delete_jut(temp1, temp, 40, 40, 0);
			medianBlur(temp, temp1, 5);
			bitwise_and(division, temp1, finalyImg);
			medianBlur(finalyImg, finalyImg, 5);
			bitwise_and(finalyImg, division, finalyImg);
			imwrite(imgPath+"\\"+to_string(k)+".jpg", finalyImg);
			k++;
			int num = svm(finalyImg);
			
			putText(finalyImg, to_string(num), Point(20, 100), 7/*font type*/,
				2/*font scale*/,
				255, 7/*thickness*/, 8);
			imshow("gesture", finalyImg);

			cout << "这个手势是：" << num << endl;
		}
		else if (key == 27)
			break;
	}
	captureL.release();
	captureR.release();
	return;
}

//CS===========================================
void getRandomDic(Mat& R, int rows, int cols);
void getWavelet(Mat img, int depth, Mat& wavelet);
void getWavelet(Mat img, int depth, Mat& wavelet) {
	int Height = img.cols;
	int Width = img.rows;
	int depthcount = 1;
	Mat tmp = Mat::ones(Width, Height, CV_32FC1);
	wavelet = Mat::ones(Width, Height, CV_32FC1);
	Mat imgtmp = img.clone();
	namedWindow("imgtmp", 1);
	imshow("imgtmp", imgtmp);
	imgtmp.convertTo(imgtmp, CV_32FC1);

	while (depthcount <= depth) {
		Width = img.rows / depthcount;
		Height = img.cols / depthcount;

		for (int i = 0; i < Width; i++) {
			for (int j = 0; j < Height / 2; j++) {
				tmp.at<float>(i, j) = (imgtmp.at<float>(i, 2 * j) + imgtmp.at<float>(i, 2 * j + 1)) / 2;
				tmp.at<float>(i, j + Height / 2) = (imgtmp.at<float>(i, 2 * j) - imgtmp.at<float>(i, 2 * j + 1)) / 2;
			}
		}
		for (int i = 0; i < Width / 2; i++) {
			for (int j = 0; j < Height; j++) {
				wavelet.at<float>(i, j) = (tmp.at<float>(2 * i, j) + tmp.at<float>(2 * i + 1, j)) / 2;
				wavelet.at<float>(i + Width / 2, j) = (tmp.at<float>(2 * i, j) - tmp.at<float>(2 * i + 1, j)) / 2;
			}
		}
		imgtmp = wavelet;
		depthcount++;
	}

}

void getRandomDic(Mat& R, int rows, int cols) {
	R.create(rows, cols, CV_8UC1);
	srand(1);
	for (int i = 0; i < R.rows; ++i) {
		for (int j = 0; j < R.cols; ++j) {
			int temp = (rand() % (255 - 0 + 1)) + 0;
			R.at<uchar>(i, j) = temp;
		}
	}
}

int main() {
	have_model = true;
	is_thresh = false;
	//singleCameraCalibrate(1, "R-17.6.9-1");
	//waitKey(0);
	//destroyAllWindows();
	//R-17.5.17-2,
	//现在是R7，L7最好
	//TwoCameraCalibrate(2,"L-17.6.9-1",1,"R-17.6.9-1","in-17.6.9-1","ex-17.6.9-1");
	//waitKey(0);
	//destroyAllWindows();
	//singleTest(2, "L-17.6.9-1");
	//waitKey(0);
	//destroyAllWindows();
	
	//twoTest("in-17.5.27-1","ex-17.5.27-1",2,1);
	//waitKey(0);
	//destroyAllWindows();
	
	gestureDetection("in-17.6.9-1", "ex-17.6.9-1", 2, 1);
	//VideoCapture capture(0);
	//while (1) {
	//	Mat t;
	//	capture >> t;
	//	imshow("t", t);
	//	waitKey(30);
	//}
}