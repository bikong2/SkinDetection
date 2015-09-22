// Skin spots detection and skin ranking
// @author: lixihua@126.com
// @date:   2015/09/20
// pixel based image segmentatin, wavelet features
//

#include "stdafx.h"
#include "opencv2/opencv.hpp"
#include "Wavelet.h"
#include "FilesRead.h"
#include "LearnAlgorithms.h"

using namespace std;
using namespace cv;
const string DATASPATH = "../datas";
const string INFOPATH  = "../datas/标注相关信息.txt";
const int H = 2592/4;
const int W = 3872/4;
const int NLAYER = 1;

int _tmain(int argc, _TCHAR* argv[])
{
	// Info Map
	map<string, InfoStruct> InfoMap;
	GetLabelInfo(INFOPATH, InfoMap);

	// All Files
	vector<string> files;
	GetFiles(DATASPATH, "", files);

	for (int i = 0; i < files.size(); i++)
	{
		string filename = files[i].c_str();
		if (-1 != filename.find(".txt"))  continue;
		if (-1 != filename.find(".xlsx")) continue;
		vector<string> splited;
		splited = str_split(filename, "/");
		map<string, InfoStruct>::iterator it = InfoMap.find(splited[3]);
		if (it != InfoMap.end()) {
			if (-1 != filename.find(".JPG")) {
				it->second.OriImage = filename;
			}
			if (-1 != filename.find("主要区域")) {
				it->second.SpotImage = filename;
			}
			if (-1 != filename.find("正常肤色")) {
				it->second.NormImage = filename;
			}
			if (-1 != filename.find("最深区域") || -1 != filename.find("最深部位")) {
				it->second.DarkImage = filename;
			}
		}
	}

	// Training Data
	Mat data_mat = Mat::zeros(H*W*50/4, 12, CV_32FC1);
	Mat res_mat = Mat::zeros(H*W*50/4, 1, CV_32FC1);
	Wavelet wavelet(NLAYER); // 初始化小波类
	Mat ori, resized, des, spot, gray;
	int j = 0;
	Mat lena = imread("lena.jpg");
	Mat show = wavelet.WaveletImage(lena);
	namedWindow("TEST", CV_WINDOW_NORMAL);
	imshow("TEST", show);
	waitKey();
	for (map<string, InfoStruct>::iterator it = InfoMap.begin(); it != InfoMap.end(); ++it) {
		// wavelet transform
		ori = imread(it->second.OriImage, CV_LOAD_IMAGE_COLOR);
		cv::resize(ori, resized, Size(W, H), 0, 0, CV_INTER_LINEAR);
		des = wavelet.WaveletImage(resized);
		imshow("TEST", resized);
		waitKey();
		imshow("TEST", des);
		waitKey();
		for (int h = 0; h < des.rows/2; h++) {
			for (int w = 0; w < des.cols/2; w++) {
				data_mat.at<float>(j, 0) = float(des.at<Vec3b>(h, w)[0])/255;
				data_mat.at<float>(j, 1) = float(des.at<Vec3b>(h, w)[1])/255;
				data_mat.at<float>(j, 2) = float(des.at<Vec3b>(h, w)[2])/255;
				data_mat.at<float>(j, 3) = float(des.at<Vec3b>(h + H/2, w)[0])/255;
				data_mat.at<float>(j, 4) = float(des.at<Vec3b>(h + H/2, w)[1])/255;
				data_mat.at<float>(j, 5) = float(des.at<Vec3b>(h + H/2, w)[2])/255;
				data_mat.at<float>(j, 6) = float(des.at<Vec3b>(h, w + W/2)[0])/255;
				data_mat.at<float>(j, 7) = float(des.at<Vec3b>(h, w + W/2)[1])/255;
				data_mat.at<float>(j, 8) = float(des.at<Vec3b>(h, w + W/2)[2])/255;
				data_mat.at<float>(j, 9) = float(des.at<Vec3b>(h + H/2, w + W/2)[0])/255;
				data_mat.at<float>(j, 10) = float(des.at<Vec3b>(h + H/2, w + W/2)[1])/255;
				data_mat.at<float>(j, 11) = float(des.at<Vec3b>(h + H/2, w + W/2)[2])/255;
				for (int i = 0; i < 12; i++) cout << data_mat.at<float>(j, i) << endl;
				j++;
			}
		}

		// segmentation info
		ori = imread(it->second.SpotImage);
		resize(ori, spot, Size(W/2, H/2), 0, 0, CV_INTER_LINEAR);
		cvtColor(spot, gray, CV_BGR2GRAY);
		j = 0;
		for (int h = 0; h < gray.rows/2; h++) {
			for (int w = 0; w < gray.cols/2; w++) {
				if (gray.at<uchar>(h, w) > 0) res_mat.at<float>(j, 0) = 1;
				else if (gray.at<uchar>(h, w) == 0) res_mat.at<float>(j, 0) = 0;
				j++;
			}
		}
	}

	// Model Training	
	CvSVMParams params;
	params.svm_type    = CvSVM::C_SVC;
	params.kernel_type = CvSVM::RBF;
	params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	string save_name = "svm_model.xml";
	TrainModel(data_mat, res_mat, params, save_name);

	CvSVM svm;
	svm.load(save_name.c_str());
	int false_alarm = 0;
	int miss = 0;
	int pos = 0;
	int neg = 0;
	for (int i = 0; i < data_mat.rows; i++) {
		int ret = int(svm.predict(data_mat.row(i)));
		if (int(res_mat.at<float>(i, 0))) {
			pos++;
			if (ret) {;}
			else miss++;
		} else {
			neg++;
			if (ret) false_alarm++;
			else {;}
		}
	}
	cout << "miss: " << float(miss)/pos << endl;
	cout << "false alarm: " << float(false_alarm)/neg << endl;

	return 0;
}

