//
//#include "opencv2/core.hpp"
//#include "opencv2/imgproc.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/videoio.hpp"
//#include <iostream>
//#include "opencv2/opencv.hpp"
//#include "opencv2/core.hpp"
//#include "opencv2/imgproc.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/videoio.hpp"
//using namespace cv::ml;
//using namespace std;
//using namespace cv;
//void fillHole(const Mat srcBw, Mat &dstBw);
//void delete_jut(Mat src, Mat& dst, int uthreshold, int vthreshold, int type);
//void delete_jut(Mat src, Mat& dst, int uthreshold, int vthreshold, int type);
//void getTrainData_cs(Mat &train_data, Mat &train_label);
//void getTrainData2(Mat &train_data, Mat &train_label);
//void getRandomDic(Mat& R, int rows, int cols);
//void getWavelet(Mat img, int depth, Mat& wavelet);
////获取训练数据
//Mat R;
//bool have_model = true;
//void getTrainData2(Mat &train_data, Mat &train_label) {
//	int totalSample = 300;
//	train_data = Mat(totalSample, 1764, CV_32FC1, Scalar::all(1)); //初始化标注
//	HOGDescriptor *hog = new HOGDescriptor(cvSize(64, 64), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
//	//train_data.convertTo(train_data, CV_32FC1);
//	Mat image;
//	for (int i = 0; i < totalSample; ++i) {
//		vector<float> descriptors;
//		image = imread("imgTrain\\"+ to_string(i) + ".jpg", 1);
//		Size sz(64, 64);
//
//		resize(image, image, sz);
//		cvtColor(image, image, CV_BGR2GRAY);
//		//Mat drawing = Mat::zeros(image.size(), CV_8UC1);
//
//		hog->compute(image, descriptors, Size(1, 1), Size(0, 0));
//		cout << descriptors.size() << endl;
//		for (int j = 0; j < descriptors.size(); ++j)
//			train_data.at<float>(i, j) = descriptors[j];
//		descriptors.clear();
//		cout << "i:" << i << endl;
//	}
//	train_label = Mat(totalSample, 1, CV_32S, Scalar::all(1)); //初始化标注
//	for (int i = 0; i < totalSample; ++i) {
//		train_label.at<int>(i, 0) = i % 10;
//	}
//}
//float svm(Mat img)
//{
//	Ptr<SVM> svm = SVM::create();
//	if (!have_model) {
//		Mat train_data, train_label;
//		getTrainData2(train_data, train_label); //
//
//													  // 设置参数		
//		svm->setType(SVM::C_SVC);
//		svm->setKernel(SVM::LINEAR);
//
//		// 训练分类器
//		Ptr<TrainData> tData = TrainData::create(train_data, ROW_SAMPLE, train_label);
//		svm->train(tData);
//		svm->save("config\\model_normal.yml");
//		have_model = true;
//	}
//	else {
//		svm = StatModel::load<SVM>("config\\model_normal.yml");
//	}
//	HOGDescriptor *hog = new HOGDescriptor(cvSize(64, 64), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
//	Size sz(64, 64);
//	resize(img, img, sz);
//	vector<float> descriptors;
//	hog->compute(img, descriptors, Size(1, 1), Size(0, 0));
//	Mat predict_data = Mat(1, 1764, CV_32FC1, Scalar::all(1)); //初始化标
//	for (int j = 0; j < descriptors.size(); ++j)
//		predict_data.at<float>(0, j) = descriptors[j];
//	float response = svm->predict(predict_data);
//	return response;
//}
//void getTrainData_cs(Mat &train_data, Mat &train_label) {
//	int totalSample = 300;
//	train_data = Mat(totalSample, 1764, CV_32FC1, Scalar::all(1)); //初始化标注
//	HOGDescriptor *hog = new HOGDescriptor(cvSize(64, 16), cvSize(16, 4), cvSize(8, 2), cvSize(8, 2), 9);
//	//train_data.convertTo(train_data, CV_32FC1);
//	Mat image;
//	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
//
//	getRandomDic(R, 16, 64);
//	R.convertTo(R, CV_32FC1);
//	FileStorage fs("config\\randomMat.yml", FileStorage::WRITE);
//	fs << "R" << R;
//	fs.release();
//	for (int i = 0; i < 200; ++i) {
//		vector<float> descriptors;
//		image = imread("imgTrain\\" + to_string(i) + ".jpg", 0);
//		
//		Size sz(64, 64);
//
//		resize(image, image, sz);
//	
//		//Mat drawing = Mat::zeros(image.size(), CV_8UC1);
//		Mat wavelet;
//		getWavelet(image, 3, wavelet);
//		Mat y = R*wavelet;
//		y.convertTo(y, CV_8U);
//		hog->compute(y, descriptors, Size(1,1), Size(0, 0));
//		cout << descriptors.size() << endl;
//		for (int j = 0; j < descriptors.size(); ++j)
//			train_data.at<float>(i, j) = descriptors[j];
//		descriptors.clear();
//		cout << "i:" << i << endl;
//	}
//	train_label = Mat(totalSample, 1, CV_32S, Scalar::all(1)); //初始化标注
//	for (int i = 0; i < totalSample; ++i) {
//		train_label.at<int>(i, 0) = i % 10;
//	}
//}
////进行预测。
//
//float svmCS(Mat img)
//{
//	Ptr<SVM> svm = SVM::create();
//	if (!have_model) {
//		Mat train_data, train_label;
//		getTrainData_cs(train_data, train_label); //
//
//													  // 设置参数		
//		svm->setType(SVM::C_SVC);
//		svm->setKernel(SVM::LINEAR);
//
//		// 训练分类器
//		Ptr<TrainData> tData = TrainData::create(train_data, ROW_SAMPLE, train_label);
//		svm->train(tData);
//		svm->save("config\\17.5.27-modelCS.yml");
//		have_model = true;
//	}
//	else {
//		svm = StatModel::load<SVM>("config\\17.5.27-modelCS.yml");
//	}
//
//	HOGDescriptor *hog = new HOGDescriptor(cvSize(64, 16), cvSize(16, 4), cvSize(8, 2), cvSize(8, 2), 9);
//
//	FileStorage fs("config\\randomMat.yml", FileStorage::READ);
//	fs["R"] >> R;
//	Size sz(64, 64);
//
//	resize(img, img, sz);
//
//	vector<float> descriptors;
//	Mat wavelet;
//	getWavelet(img, 3, wavelet);
//	Mat y = R*wavelet;
//	y.convertTo(y, CV_8U);
//	hog->compute(y, descriptors, Size(1, 1), Size(0, 0));
//
//	Mat predict_data = Mat(1, descriptors.size(), CV_32FC1, Scalar::all(1)); //初始化标
//	for (int j = 0; j < descriptors.size(); ++j)
//		predict_data.at<float>(0, j) = descriptors[j];
//	float response = svm->predict(predict_data);
//	return response;
//}
//// 去除二值图像边缘的突出部
////uthreshold、vthreshold分别表示突出部的宽度阈值和高度阈值  
////type代表突出部的颜色，0表示黑色，1代表白色   
//void delete_jut(Mat src, Mat& dst, int uthreshold, int vthreshold, int type)
//{
//	int threshold;
//	src.copyTo(dst);
//	int height = dst.rows;
//	int width = dst.cols;
//	int k;  //用于循环计数传递到外部  
//	for (int i = 0; i < height - 1; i++)
//	{
//		uchar* p = dst.ptr<uchar>(i);
//		for (int j = 0; j < width - 1; j++)
//		{
//			if (type == 0)
//			{
//				//行消除  
//				if (p[j] == 255 && p[j + 1] == 0)
//				{
//					if (j + uthreshold >= width)
//					{
//						for (int k = j + 1; k < width; k++)
//							p[k] = 255;
//					}
//					else
//					{
//						for (k = j + 2; k <= j + uthreshold; k++)
//						{
//							if (p[k] == 255) break;
//						}
//						if (p[k] == 255)
//						{
//							for (int h = j + 1; h < k; h++)
//								p[h] = 255;
//						}
//					}
//				}
//				//列消除  
//				if (p[j] == 255 && p[j + width] == 0)
//				{
//					if (i + vthreshold >= height)
//					{
//						for (k = j + width; k < j + (height - i)*width; k += width)
//							p[k] = 255;
//					}
//					else
//					{
//						for (k = j + 2 * width; k <= j + vthreshold*width; k += width)
//						{
//							if (p[k] == 255) break;
//						}
//						if (p[k] == 255)
//						{
//							for (int h = j + width; h < k; h += width)
//								p[h] = 255;
//						}
//					}
//				}
//			}
//			else  //type = 1  
//			{
//				//行消除  
//				if (p[j] == 0 && p[j + 1] == 255)
//				{
//					if (j + uthreshold >= width)
//					{
//						for (int k = j + 1; k < width; k++)
//							p[k] = 0;
//					}
//					else
//					{
//						for (k = j + 2; k <= j + uthreshold; k++)
//						{
//							if (p[k] == 0) break;
//						}
//						if (p[k] == 0)
//						{
//							for (int h = j + 1; h < k; h++)
//								p[h] = 0;
//						}
//					}
//				}
//				//列消除  
//				if (p[j] == 0 && p[j + width] == 255)
//				{
//					if (i + vthreshold >= height)
//					{
//						for (k = j + width; k < j + (height - i)*width; k += width)
//							p[k] = 0;
//					}
//					else
//					{
//						for (k = j + 2 * width; k <= j + vthreshold*width; k += width)
//						{
//							if (p[k] == 0) break;
//						}
//						if (p[k] == 0)
//						{
//							for (int h = j + width; h < k; h += width)
//								p[h] = 0;
//						}
//					}
//				}
//			}
//		}
//	}
//}
////图片边缘光滑处理  
////size表示取均值的窗口大小，threshold表示对均值图像进行二值化的阈值  
//void imageblur(Mat src, Mat& dst, Size size, int threshold)
//{
//	int height = src.rows;
//	int width = src.cols;
//	blur(src, dst, size);
//	for (int i = 0; i < height; i++)
//	{
//		uchar* p = dst.ptr<uchar>(i);
//		for (int j = 0; j < width; j++)
//		{
//			if (p[j] < threshold)
//				p[j] = 0;
//			else p[j] = 255;
//		}
//	}
//}
//void fillHole(const Mat srcBw, Mat &dstBw)
//{
//	Size m_Size = srcBw.size();
//	Mat Temp = Mat::zeros(m_Size.height + 2, m_Size.width + 2, srcBw.type());//延展图像
//
//	srcBw.copyTo(Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)));
//
//	cv::floodFill(Temp, Point(0, 0), Scalar(255));
//
//	Mat cutImg;//裁剪延展的图像
//	Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)).copyTo(cutImg);
//
//	dstBw = srcBw | (~cutImg);
//
//}
//void getWavelet(Mat img,int depth,Mat& wavelet) {
//	int Height = img.cols;
//	int Width = img.rows;
//	int depthcount = 1;
//	Mat tmp = Mat::ones(Width, Height, CV_32FC1);
//	wavelet = Mat::ones(Width, Height, CV_32FC1);
//	Mat imgtmp = img.clone();
//	namedWindow("imgtmp", 1);
//	imshow("imgtmp", imgtmp);
//	imgtmp.convertTo(imgtmp, CV_32FC1);
//
//	while (depthcount <= depth) {
//		Width = img.rows / depthcount;
//		Height = img.cols / depthcount;
//
//		for (int i = 0; i < Width; i++) {
//			for (int j = 0; j < Height / 2; j++) {
//				tmp.at<float>(i, j) = (imgtmp.at<float>(i, 2 * j) + imgtmp.at<float>(i, 2 * j + 1)) / 2;
//				tmp.at<float>(i, j + Height / 2) = (imgtmp.at<float>(i, 2 * j) - imgtmp.at<float>(i, 2 * j + 1)) / 2;
//			}
//		}
//		for (int i = 0; i < Width / 2; i++) {
//			for (int j = 0; j < Height; j++) {
//				wavelet.at<float>(i, j) = (tmp.at<float>(2 * i, j) + tmp.at<float>(2 * i + 1, j)) / 2;
//				wavelet.at<float>(i + Width / 2, j) = (tmp.at<float>(2 * i, j) - tmp.at<float>(2 * i + 1, j)) / 2;
//			}
//		}
//		imgtmp = wavelet;
//		depthcount++;
//	}
//	
//}
//RNG rng(12345);
//void getRandomDic(Mat& R, int rows,int cols) {
//	R.create(rows, cols, CV_8UC1);
//	srand(1);
//	for (int i = 0; i < R.rows; ++i) {
//		for (int j = 0; j < R.cols; ++j) {
//			int temp = (rand() % (255 - 0 + 1)) + 0;
//			R.at<uchar>(i, j) = temp;
//		}
//	}
//}
//int main() {
//	have_model = false;
//	int correctCounter_cs = 0;
//	int correctCounter_normal = 0;
//	for (int i = 300; i < 410; ++i) {
//		Mat img = imread("imgTrain\\"+to_string(i)+".jpg", 0);
//		int result = svmCS(img);
//		if (result == i % 10)
//			correctCounter_cs++;
//		cout << result << endl;
//	}
//	cout << "correctCounter_cs:" << correctCounter_cs << endl;
//	//cout << "correctCounter_normal:" << correctCounter_normal << endl;
//}