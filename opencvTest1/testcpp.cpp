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
//////// ȥ����ֵͼ���Ե��ͻ����
////////uthreshold��vthreshold�ֱ��ʾͻ�����Ŀ����ֵ�͸߶���ֵ  
////////type����ͻ��������ɫ��0��ʾ��ɫ��1�����ɫ   
//////void delete_jut(Mat src, Mat& dst, int uthreshold, int vthreshold, int type)
//////{
//////	int threshold;
//////	src.copyTo(dst);
//////	int height = dst.rows;
//////	int width = dst.cols;
//////	int k;  //����ѭ���������ݵ��ⲿ  
//////	for (int i = 0; i < height - 1; i++)
//////	{
//////		uchar* p = dst.ptr<uchar>(i);
//////		for (int j = 0; j < width - 1; j++)
//////		{
//////			if (type == 0)
//////			{
//////				//������  
//////				if (p[j] == 255 && p[j + 1] == 0)
//////				{
//////					if (j + uthreshold >= width)
//////					{
//////						for (int k = j + 1; k < width; k++)
//////							p[k] = 255;
//////					}
//////					else
//////					{
//////						for (k = j + 2; k <= j + uthreshold; k++)
//////						{
//////							if (p[k] == 255) break;
//////						}
//////						if (p[k] == 255)
//////						{
//////							for (int h = j + 1; h < k; h++)
//////								p[h] = 255;
//////						}
//////					}
//////				}
//////				//������  
//////				if (p[j] == 255 && p[j + width] == 0)
//////				{
//////					if (i + vthreshold >= height)
//////					{
//////						for (k = j + width; k < j + (height - i)*width; k += width)
//////							p[k] = 255;
//////					}
//////					else
//////					{
//////						for (k = j + 2 * width; k <= j + vthreshold*width; k += width)
//////						{
//////							if (p[k] == 255) break;
//////						}
//////						if (p[k] == 255)
//////						{
//////							for (int h = j + width; h < k; h += width)
//////								p[h] = 255;
//////						}
//////					}
//////				}
//////			}
//////			else  //type = 1  
//////			{
//////				//������  
//////				if (p[j] == 0 && p[j + 1] == 255)
//////				{
//////					if (j + uthreshold >= width)
//////					{
//////						for (int k = j + 1; k < width; k++)
//////							p[k] = 0;
//////					}
//////					else
//////					{
//////						for (k = j + 2; k <= j + uthreshold; k++)
//////						{
//////							if (p[k] == 0) break;
//////						}
//////						if (p[k] == 0)
//////						{
//////							for (int h = j + 1; h < k; h++)
//////								p[h] = 0;
//////						}
//////					}
//////				}
//////				//������  
//////				if (p[j] == 0 && p[j + width] == 255)
//////				{
//////					if (i + vthreshold >= height)
//////					{
//////						for (k = j + width; k < j + (height - i)*width; k += width)
//////							p[k] = 0;
//////					}
//////					else
//////					{
//////						for (k = j + 2 * width; k <= j + vthreshold*width; k += width)
//////						{
//////							if (p[k] == 0) break;
//////						}
//////						if (p[k] == 0)
//////						{
//////							for (int h = j + width; h < k; h += width)
//////								p[h] = 0;
//////						}
//////					}
//////				}
//////			}
//////		}
//////	}
//////}
//////void fillHole(const Mat srcBw, Mat &dstBw)
//////{
//////	Size m_Size = srcBw.size();
//////	Mat Temp = Mat::zeros(m_Size.height + 2, m_Size.width + 2, srcBw.type());//��չͼ��
//////
//////	srcBw.copyTo(Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)));
//////
//////	cv::floodFill(Temp, Point(0, 0), Scalar(255));
//////
//////	Mat cutImg;//�ü���չ��ͼ��
//////	Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)).copyTo(cutImg);
//////
//////	dstBw = srcBw | (~cutImg);
//////
//////}
//////int main() {
//////	Mat disparityImage = imread("2_17.jpg",0);
//////	imshow("ǰ", disparityImage);
//////	Mat element = getStructuringElement(MORPH_RECT, Size(2, 2), Point(-1, -1));
//////	medianBlur(disparityImage, disparityImage, 5);
//////	dilate(disparityImage, disparityImage, element);
//////	imshow("��ֵ�˲������ͺ��ͼ��", disparityImage);
//////	fillHole(disparityImage, disparityImage);
//////	imshow("���ն����ͼ��", disparityImage);
//////	delete_jut(disparityImage, disparityImage, 5, 5, 0);
//////	medianBlur(disparityImage, disparityImage, 5);
//////	imshow("������Ե��ͻ�������ٴ���ֵ�˲����ͼ��", disparityImage);
//////	imwrite("hhh.jpg", disparityImage);
//////	
//////	waitKey(0);
//////}
////#include <cv.h>  
////#include <highgui.h>  
////#include <opencv2/imgproc/imgproc.hpp>    
////#include <opencv2/highgui/highgui.hpp>    
////#include <iostream>    
////#include <vector>    
////
////
////using namespace cv;
////using namespace std;
////
////void RemoveSmallRegion(Mat& Src, Mat& Dst, int AreaLimit = 50, int CheckMode = 1, int NeihborMode = 0);
////
////int main()
////{
////	double t = (double)getTickCount();
////
////	char* imagePath = "imgTrain - ����\\2.jpg";
////	char* OutPath = "E:\\�ֲ�_ȥ���׶�.jpg";
////
////	Mat Src = imread(imagePath, CV_LOAD_IMAGE_GRAYSCALE);
////	Mat Dst = Mat::zeros(Src.size(), CV_8UC1);
////
////
////	//��ֵ������  
////	for (int i = 0; i < Src.rows; ++i)
////	{
////		uchar* iData = Src.ptr<uchar>(i);
////		for (int j = 0; j < Src.cols; ++j)
////		{
////			if (iData[j] == 0 || iData[j] == 255) continue;
////			else if (iData[j] < 10)
////			{
////				iData[j] = 0;
////				//cout<<'#';  
////			}
////			else if (iData[j] > 10)
////			{
////				iData[j] = 255;
////				//cout<<'!';  
////			}
////		}
////	}
////	cout << "Image Binary processed." << endl;
////
////	RemoveSmallRegion(Src, Dst, 100, 1, 1);
////	//RemoveSmallRegion(Dst, Dst, 20, 0, 0);
////	cout << "Done!" << endl;
////	imwrite(OutPath, Dst);
////
////	t = ((double)getTickCount() - t) / getTickFrequency();
////	cout << "Time cost: " << t << " sec." << endl;
////
////	return 0;
////}
////
//////CheckMode: 0����ȥ��������1����ȥ��������; NeihborMode��0����4����1����8����;  
////void RemoveSmallRegion(Mat& Src, Mat& Dst, int AreaLimit, int CheckMode, int NeihborMode)
////{
////	int RemoveCount = 0;       //��¼��ȥ�ĸ���  
////							   //��¼ÿ�����ص����״̬�ı�ǩ��0����δ��飬1�������ڼ��,2�����鲻�ϸ���Ҫ��ת��ɫ����3������ϸ������  
////	Mat Pointlabel = Mat::zeros(Src.size(), CV_8UC1);
////
////	if (CheckMode == 1)
////	{
////		cout << "Mode: ȥ��С����. ";
////		for (int i = 0; i < Src.rows; ++i)
////		{
////			uchar* iData = Src.ptr<uchar>(i);
////			uchar* iLabel = Pointlabel.ptr<uchar>(i);
////			for (int j = 0; j < Src.cols; ++j)
////			{
////				if (iData[j] < 10)
////				{
////					iLabel[j] = 3;
////				}
////			}
////		}
////	}
////	else
////	{
////		cout << "Mode: ȥ���׶�. ";
////		for (int i = 0; i < Src.rows; ++i)
////		{
////			uchar* iData = Src.ptr<uchar>(i);
////			uchar* iLabel = Pointlabel.ptr<uchar>(i);
////			for (int j = 0; j < Src.cols; ++j)
////			{
////				if (iData[j] > 10)
////				{
////					iLabel[j] = 3;
////				}
////			}
////		}
////	}
////
////	vector<Point2i> NeihborPos;  //��¼�����λ��  
////	NeihborPos.push_back(Point2i(-1, 0));
////	NeihborPos.push_back(Point2i(1, 0));
////	NeihborPos.push_back(Point2i(0, -1));
////	NeihborPos.push_back(Point2i(0, 1));
////	if (NeihborMode == 1)
////	{
////		cout << "Neighbor mode: 8����." << endl;
////		NeihborPos.push_back(Point2i(-1, -1));
////		NeihborPos.push_back(Point2i(-1, 1));
////		NeihborPos.push_back(Point2i(1, -1));
////		NeihborPos.push_back(Point2i(1, 1));
////	}
////	else cout << "Neighbor mode: 4����." << endl;
////	int NeihborCount = 4 + 4 * NeihborMode;
////	int CurrX = 0, CurrY = 0;
////	//��ʼ���  
////	for (int i = 0; i < Src.rows; ++i)
////	{
////		uchar* iLabel = Pointlabel.ptr<uchar>(i);
////		for (int j = 0; j < Src.cols; ++j)
////		{
////			if (iLabel[j] == 0)
////			{
////				//********��ʼ�õ㴦�ļ��**********  
////				vector<Point2i> GrowBuffer;                                      //��ջ�����ڴ洢������  
////				GrowBuffer.push_back(Point2i(j, i));
////				Pointlabel.at<uchar>(i, j) = 1;
////				int CheckResult = 0;                                               //�����жϽ�����Ƿ񳬳���С����0Ϊδ������1Ϊ����  
////
////				for (int z = 0; z<GrowBuffer.size(); z++)
////				{
////
////					for (int q = 0; q<NeihborCount; q++)                                      //����ĸ������  
////					{
////						CurrX = GrowBuffer.at(z).x + NeihborPos.at(q).x;
////						CurrY = GrowBuffer.at(z).y + NeihborPos.at(q).y;
////						if (CurrX >= 0 && CurrX<Src.cols&&CurrY >= 0 && CurrY<Src.rows)  //��ֹԽ��  
////						{
////							if (Pointlabel.at<uchar>(CurrY, CurrX) == 0)
////							{
////								GrowBuffer.push_back(Point2i(CurrX, CurrY));  //��������buffer  
////								Pointlabel.at<uchar>(CurrY, CurrX) = 1;           //���������ļ���ǩ�������ظ����  
////							}
////						}
////					}
////
////				}
////				if (GrowBuffer.size()>AreaLimit) CheckResult = 2;                 //�жϽ�����Ƿ񳬳��޶��Ĵ�С����1Ϊδ������2Ϊ����  
////				else { CheckResult = 1;   RemoveCount++; }
////				for (int z = 0; z<GrowBuffer.size(); z++)                         //����Label��¼  
////				{
////					CurrX = GrowBuffer.at(z).x;
////					CurrY = GrowBuffer.at(z).y;
////					Pointlabel.at<uchar>(CurrY, CurrX) += CheckResult;
////				}
////				//********�����õ㴦�ļ��**********  
////
////
////			}
////		}
////	}
////
////	CheckMode = 255 * (1 - CheckMode);
////	//��ʼ��ת�����С������  
////	for (int i = 0; i < Src.rows; ++i)
////	{
////		uchar* iData = Src.ptr<uchar>(i);
////		uchar* iDstData = Dst.ptr<uchar>(i);
////		uchar* iLabel = Pointlabel.ptr<uchar>(i);
////		for (int j = 0; j < Src.cols; ++j)
////		{
////			if (iLabel[j] == 2)
////			{
////				iDstData[j] = CheckMode;
////			}
////			else if (iLabel[j] == 3)
////			{
////				iDstData[j] = iData[j];
////			}
////		}
////	}
////
////	cout << RemoveCount << " objects removed." << endl;
////}
//int main() {
//	Mat src = imread("222.jpg", 1);
//	imshow("src", src);
//	Size sz(450, 300);
//	resize(src, src, sz);
//	imshow("src1", src);
//	imwrite("���֤.jpg", src);
//	waitKey(0);
//}