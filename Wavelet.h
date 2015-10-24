// Wavelet Related functions
// @author: lixihua9@126.com
// @date:   2015/09/20

#include "stdafx.h"
#include <vector>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

class Wavelet
{ 
public:
	Wavelet(int layer);
	~Wavelet();
	Mat WaveletImage(Mat src);               // ��ȡС���仯֮���ͼ��

private:
	int _layer;
	void DWT(IplImage *pImage, int nLayer);  // ��ά��ɢС���任����ͨ������ͼ��
	void IDWT(IplImage *pImage, int nLayer); // ��ά��ɢС���ָ�����ͨ������ͼ��
};