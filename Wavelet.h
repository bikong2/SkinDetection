// Wavelet Related functions
// @author: lixihua9@126.com
// @date:   2015/09/20

#include "stdafx.h"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

class Wavelet
{ 
public:
	Wavelet(int layer);
	~Wavelet();
	Mat WaveletImage(Mat src);               // 获取小波变化之后的图像

private:
	int _layer;
	void DWT(IplImage *pImage, int nLayer);  // 二维离散小波变换（单通道浮点图像）
	void IDWT(IplImage *pImage, int nLayer); // 二维离散小波恢复（单通道浮点图像）
};