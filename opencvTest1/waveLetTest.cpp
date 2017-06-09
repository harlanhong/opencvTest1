///*************************************************
//Copyright:zhuchen
//Author: zhuchen
//Date:2016-01-10
//Description:�༶haarС���任
//**************************************************/
//
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
//
//int SIZE = 20;
//void reconstruction(Mat img, int depth);
//void OrthMatchPursuit(
//	vector<Mat>& dic,//�ֵ�  
//	Mat& target,
//	float min_residual,
//	int sparsity,
//	Mat& x,  //����ÿ��ԭ�Ӷ�Ӧ��ϵ��;  
//	vector<int>& patch_indices  //����ѡ����ԭ�����  
//);
//void getRandomDic(vector<Mat>& dic,Mat& R, int hight);
//void sharpenImage1(const cv::Mat &image, cv::Mat &result);
//void omp(Mat R, Mat y, int len, Mat& x, vector<int>& patch_indices, float min_residual);
//cv::Mat ompSparseL2(const cv::Mat Dict, const cv::Mat Y, const int K);
//void getColDictFormIndex(const cv::Mat Dict, const vector<int> index, cv::Mat &res);
//float OMP(Mat dic, Mat signal, float min_residual, int sparsity, Mat& coe, vector<int>& atom_index);
//int main() {
//	Mat img = imread("win10.jpg", 0);
//	Size size(int(img.cols / 8) * 8, int(img.rows / 8) * 8);
//
//	resize(img, img, Size(128,128));
//	int Height = img.cols;
//	int Width = img.rows;
//	int depth = 4;    //����ֽ����
//	int depthcount = 1;
//	Mat tmp = Mat::ones(Width, Height, CV_32FC1);
//	Mat wavelet = Mat::ones(Width, Height, CV_32FC1);
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
//	Mat R;
//	vector<Mat> dic(wavelet.rows);
//	getRandomDic(dic,R,SIZE);
//	cout << R << endl;
//	Mat temp = R*wavelet;
//	cout << R.size() << endl;
//	cout << wavelet.size() << endl;
//	cout << temp.size()<<" "<<temp.rows<<" "<<temp.cols<<endl;
//	Mat recorver;
//	recorver.create(wavelet.rows, wavelet.cols, CV_32FC1);
//	for (int i = 0; i < temp.cols; ++i) {
//		Mat x;
//		vector<int> patch_indices;
//		//OrthMatchPursuit(dic, temp.col(i), 0.9, wavelet.rows, x, patch_indices);
//		//omp(R, temp.col(i), wavelet.rows, x, patch_indices, 0.9);
//		//x = ompSparseL2(R, temp.col(i), wavelet.rows);
//		OMP(R, temp.col(i), 0.9, wavelet.rows, x, patch_indices);
//		recorver.col(i) = Mat(x);
//		cout << i << endl;
//	}
//	namedWindow("jpg", 1);
//	reconstruction(recorver, depth);
//	wavelet.convertTo(wavelet, CV_8UC1);
//	//wavelet += 50;            //ͼ�񰵶ȹ��ͣ����������Ҽ���50
//	imshow("jpg", wavelet);
//	waitKey(0);
//	return 0;
//}
//RNG rng(12345);
//void getRandomDic(vector<Mat>& dic,Mat& R,int hight) {
//	R.create(hight, dic.size(), CV_32FC1);
//	srand((unsigned)time(NULL));
//	for (int i = 0; i < R.rows; ++i) {
//		for (int j = 0; j < R.cols; ++j) {
//			R.at<int>(i,j) = (rand() % (255 - 0 + 1)) + 0;
//		}
//	}
//}
//void reconstruction(Mat img, int depth) {
//
//	int Height = img.cols;
//	int Width = img.rows;
//	int depthcount = depth;
//	Mat tmp = Mat::ones(Width, Height, CV_32FC1);
//	Mat wavelet = Mat::ones(Width, Height, CV_32FC1);
//	Mat imgtmp = img.clone();
//	while (depthcount >= 1) {
//		Width = img.rows / depthcount;
//		Height = img.cols / depthcount;
//		for (int i = 0; i < Width / 2; i++) {
//			for (int j = 0; j < Height; j++) {
//				tmp.at<float>(2 * i, j) = imgtmp.at<float>(i, j) + imgtmp.at<float>(i + Width / 2, j);
//				tmp.at<float>(2 * i + 1, j) = imgtmp.at<float>(i, j) - imgtmp.at<float>(i + Width / 2, j);
//			}
//		}
//		for (int i = 0; i < Width; i++) {
//			for (int j = 0; j < Height / 2; j++) {
//				wavelet.at<float>(i, 2 * j) = tmp.at<float>(i, j) + tmp.at<float>(i, j + Height / 2);
//				wavelet.at<float>(i, 2 * j + 1) = tmp.at<float>(i, j) - tmp.at<float>(i, j + Height / 2);
//			}
//		}
//		//row reconstruction
//		imgtmp = wavelet;
//		depthcount--;
//	}
//	wavelet.convertTo(wavelet, CV_8UC1);
//	cv::Mat result1;
//	result1.create(wavelet.size(), wavelet.type());
//	sharpenImage1(wavelet, result1);
//	namedWindow("waveletReconstruct", 1);
//	imshow("waveletReconstruct", result1);
//}
//void OrthMatchPursuit(
//	vector<Mat>& dic,//�ֵ�  
//	Mat& target,
//	float min_residual,
//	int sparsity,
//	Mat& x,  //����ÿ��ԭ�Ӷ�Ӧ��ϵ��;  
//	vector<int>& patch_indices  //����ѡ����ԭ�����  
//	)
//{
//	Mat residual = target.clone();
//	
//	Mat phi;//������ѡ����ԭ������  
//	x.create(0, 1, CV_32FC1);
//	float max_coefficient;
//	unsigned int patch_index;
//	for (;;){
//		max_coefficient = 0;
//		for (int i = 0; i < dic.size(); i++){
//			float coefficient = (float)dic[i].dot(residual);
//			if (abs(coefficient) > abs(max_coefficient)){
//				max_coefficient = coefficient;
//				patch_index = i;
//			}
//		}
//		patch_indices.push_back(patch_index); //���ѡ����ԭ�����  
//		Mat& matched_patch = dic[patch_index];
//		if (phi.cols == 0)
//			phi = matched_patch;
//		else
//			hconcat(phi, matched_patch, phi); //����ԭ�Ӻϲ���ԭ�Ӽ����У�������������  
//		
//		x.push_back(0.0f);//��ϵ�������¼�һ��  
//		solve(phi, target, x, DECOMP_SVD);//�����С��������  
//		residual = target - phi*x;  //���²в�  
//		float res_norm = (float)norm(residual);
//		if (x.rows >= sparsity || res_norm <= min_residual) //����в�С����ֵ��ﵽҪ���ϡ��ȣ��ͷ���  
//			return;
//		}
//}
//void sharpenImage1(const cv::Mat &image, cv::Mat &result)
//{
//	//��������ʼ���˲�ģ��
//	cv::Mat kernel(3, 3, CV_32F, cv::Scalar(0));
//	kernel.at<float>(1, 1) = 5.0;
//	kernel.at<float>(0, 1) = -1.0;
//	kernel.at<float>(1, 0) = -1.0;
//	kernel.at<float>(1, 2) = -1.0;
//	kernel.at<float>(2, 1) = -1.0;
//
//	result.create(image.size(), image.type());
//
//	//��ͼ������˲�
//	cv::filter2D(image, result, image.depth(), kernel);
//}
////RΪ�۲����yΪ����ֵ��lenΪx�����ĳ���
//void omp(Mat R, Mat y, int len, Mat& x, vector<int>& patch_indices, float min_residual) {
//	Mat residual = y.clone();
//	Mat phi;//������ѡ����ԭ������  
//	x.create(0, 1, CV_32FC1);
//	float max_coefficient;
//	unsigned int patch_index;
//	for (;;) {
//		max_coefficient = 0;
//		for (int i = 0; i < R.cols; i++) {
//			
//			float coefficient = (float)R.col(i).dot(residual);
//			if (abs(coefficient) > abs(max_coefficient)) {
//				max_coefficient = coefficient;
//				patch_index = i;
//			}
//		}
//		patch_indices.push_back(patch_index); //���ѡ����ԭ�����  
//		Mat matched_patch = R.col(patch_index);
//		if (phi.cols == 0)
//			phi = matched_patch;
//		else
//			hconcat(phi, matched_patch, phi); //����ԭ�Ӻϲ���ԭ�Ӽ����У�������������  
//		for (int i = 0; i<R.rows; ++i) {
//			R.at<float>(i, patch_index) = 0;
//		}
//		x.push_back(0.0f);//��ϵ�������¼�һ��  
//		solve(phi, y, x, DECOMP_SVD);//�����С��������  
//		residual = y - phi*x;  //���²в�  
//		float res_norm = (float)norm(residual);
//		if (x.rows >= len || res_norm <= min_residual) //����в�С����ֵ��ﵽҪ���ϡ��ȣ��ͷ���  
//			return;
//	}
//}
//cv::Mat ompSparseL2(const cv::Mat Dict, const cv::Mat Y, const int K)
//{
//	int r = Dict.rows;
//	int c = Dict.cols;
//	int n = Y.cols;
//	cv::Mat ERR(r, 1, CV_32F);
//	ERR = Y;
//	int size[] = { c,n };
//	cv::Mat A = cv::Mat(2, size, CV_32F, cv::Scalar(0.f));
//	vector<int> index;
//	cv::Mat U = cv::Mat::ones(1, c, CV_32F);
//	cv::Mat tmpA;
//	for (int i = 0; i<K; i++)
//	{
//		cv::Mat S = ERR.t()*Dict;
//		cv::pow(S, 2, S);
//		if (S.rows != 1)
//			cv::reduce(S, S, 0, CV_REDUCE_SUM);
//		cv::sqrt(S, S);
//		S = S / U;
//		if (i != 0)
//		{
//			for (int j = 0; j<index.size(); j++)
//			{
//				S.at<float>(0, index[j]) = 0.f;
//			}
//		}
//
//		cv::Point maxLoc;
//		cv::minMaxLoc(S, NULL, NULL, NULL, &maxLoc);
//		int pos = maxLoc.x;
//		index.push_back(pos);
//
//		cv::Mat subDict;
//		getColDictFormIndex(Dict, index, subDict);
//
//		cv::Mat invSubDict;
//		cv::invert(subDict, invSubDict, cv::DECOMP_SVD);
//
//		tmpA = invSubDict*Y;
//		ERR = Y - subDict*tmpA;
//
//		cv::Mat Dict_T_Dict;
//		cv::mulTransposed(subDict, Dict_T_Dict, 1);
//		cv::Mat invDict_T_Dict;
//		cv::invert(Dict_T_Dict, invDict_T_Dict, cv::DECOMP_SVD);
//
//		cv::Mat P = (cv::Mat::eye(r, r, CV_32F) - subDict*invDict_T_Dict*subDict.t())*Dict;
//		cv::pow(P, 2, P);
//		cv::reduce(P, P, 0, CV_REDUCE_SUM);
//		cv::sqrt(P, U);
//	}
//	for (int i = 0; i<K; i++)
//	{
//		int tmpC = index[i];
//		const float *inP = tmpA.ptr<float>(i);
//		float *outP = A.ptr<float>(tmpC);
//		for (int j = 0; j<n; j++)
//		{
//			outP[j] = inP[j];
//		}
//	}
//	return A;
//}
//void getColDictFormIndex(const cv::Mat Dict, const vector<int> index, cv::Mat &res)
//{
//	if (index.size() == 0)
//		return;
//	if (!Dict.data)
//		return;
//
//	int r = Dict.rows;
//	int c = index.size();
//
//	cv::Mat Dict_T;
//	cv::transpose(Dict, Dict_T);
//
//	cv::Mat res_T = cv::Mat(c, r, Dict.type());
//
//	for (int i = 0; i<index.size(); i++)
//	{
//		int tmpC = index[i];
//		const float *inP = Dict_T.ptr<float>(tmpC);
//		float *outP = res_T.ptr<float>(i);
//		for (int j = 0; j<r; j++)
//		{
//			outP[j] = inP[j];
//		}
//	}
//	cv::transpose(res_T, res);
//	res_T.release();
//	Dict_T.release();
//}
////dic: �ֵ����
////signal :���ع��źţ�һ��ֻ���ع�һ���źţ���һ��������
////min_residual: ��С�в�
////sparsity:ϡ���
////coe:�ع�ϵ��
////atom_index:�ֵ�ԭ��ѡ�����
////�������Ĳв�
//float OMP(Mat dic, Mat signal, float min_residual, int sparsity, Mat& coe, vector<int>& atom_index)
//
//{
//	if (signal.cols>1)
//	{
//		cout << "wrong signal" << endl;
//		return -1;
//	}
//	signal = signal / norm(signal);  //�źŵ�λ��
//	Mat temp(1, dic.cols, 5);
//	for (int i = 0; i<dic.cols; i++)
//	{
//		temp.col(i) = norm(dic.col(i));  //ÿ��ԭ�ӵ�ģ��
//	}
//	divide(dic, repeat(temp, dic.rows, 1), dic); //�ֵ�ԭ�ӵ�λ��
//	Mat residual = signal.clone();  //��ʼ���в�
//	coe.create(0, 1, CV_32FC1);  //��ʼ��ϵ��
//	Mat phi;    //������ѡ����ԭ������
//	float max_coefficient;
//	unsigned int atom_id;  //ÿ����ѡ���ԭ�ӵ����
//
//	for (;;)
//	{
//		max_coefficient = 0;
//		//ȡ���ڻ������
//		for (int i = 0; i <dic.cols; i++)
//		{
//			float coefficient = (float)dic.col(i).dot(residual);
//		
//			if (abs(coefficient) > abs(max_coefficient))
//			{
//				max_coefficient = coefficient;
//				atom_id = i;
//			}
//		}
//		atom_index.push_back(atom_id); //���ѡ����ԭ�����        
//		Mat& temp_atom = dic.col(atom_id); //ȡ����ԭ��
//		if (phi.cols == 0)
//			phi = temp_atom;
//		else
//			hconcat(phi, temp_atom, phi); //����ԭ�Ӻϲ���ԭ�Ӽ����У�������������
//
//		coe.push_back(0.0f);    //��ϵ�������¼�һ��
//		solve(phi, signal, coe, DECOMP_SVD);    //�����С��������
//
//		residual = signal - phi*coe;  //���²в�
//		float res_norm = (float)norm(residual);
//
//		if (coe.rows >= sparsity || res_norm <= min_residual) //����в�С����ֵ��ﵽҪ���ϡ��ȣ��ͷ���
//		{
//			return res_norm;
//		}
//	}
//	
//}