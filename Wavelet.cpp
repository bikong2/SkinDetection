// Wavelet Related functions
// @author: lixihua9@126.com
// @date:   2015/09/20

#include "stdafx.h"
#include "Wavelet.h"

Wavelet::Wavelet(int layer)
{
	_layer = layer;
}

Wavelet::~Wavelet()
{}

Mat Wavelet::WaveletImage(Mat src)
{
	Mat equal_src;
	vector<Mat> splitBGR(src.channels()); // ��ɫͼ��ֱ��ͼ����
	split(src, splitBGR);
	for (int i = 0; i < src.channels(); i++)
	{
		equalizeHist(splitBGR[i], splitBGR[i]);
	}
	merge(splitBGR, equal_src);

	int nLayer = _layer;
	IplImage pSrc = equal_src;
	CvSize size = cvGetSize(&pSrc); // ����С��ͼ���С
	IplImage *pWavelet = cvCreateImage(size, IPL_DEPTH_32F, pSrc.nChannels); // ����С��ͼ��
	if (pWavelet)
	{
		// С��ͼ��ֵ
		cvConvertScale(&pSrc, pWavelet, 1, -128);
		// ��ɫͼ��С���任
		IplImage *pImage = cvCreateImage(cvGetSize(pWavelet), IPL_DEPTH_32F, 1);
		if (pImage)
		{
			for (int i = 1; i <= pWavelet->nChannels; i++)
			{
				cvSetImageCOI(pWavelet, i);
				cvCopy(pWavelet, pImage, NULL);

				// ��ά��ɢС���任
				DWT(pImage, nLayer);

				// ��ά��ɢС���ָ�
				// IDWT(pImage, nLayer);
				cvCopy(pImage, pWavelet, NULL);
			}
			cvSetImageCOI(pWavelet, 0);
			cvReleaseImage(&pImage);
		}
		// С���任ͼ�� 
		cvConvertScale(pWavelet, &pSrc, 1, 128);
		cvReleaseImage(&pWavelet);
	}
	Mat des(&pSrc, true);
	return des;
}


// ��ά��ɢС���任����ͨ������ͼ��
void Wavelet::DWT(IplImage *pImage, int nLayer)
{
	// ִ������
	if (pImage)
	{
		if (pImage->nChannels == 1 &&
			pImage->depth == IPL_DEPTH_32F &&
			((pImage->width >> nLayer) << nLayer) == pImage->width &&
			((pImage->height >> nLayer) << nLayer) == pImage->height)
		{
			int i, x, y, n; 
			float fValue = 0;
			float fRadius = sqrt(2.0f);
			int nWidth = pImage->width; 
			int nHeight = pImage->height; 
			int nHalfW = nWidth/2; 
			int nHalfH = nHeight/2;
			float **pData = new float*[pImage->height]; 
			float *pRow = new float[pImage->width]; 
			float *pColumn = new float[pImage->height];
			for (i = 0; i < pImage->height; i++) 
			{ 
				pData[i] = (float*) (pImage->imageData + pImage->widthStep * i);
			}
			// ���С���任 
			for (n = 0; n < nLayer; n++, nWidth /= 2, nHeight /= 2, nHalfW /= 2, nHalfH /= 2)
			{ 
				// ˮƽ�任 
				for (y = 0; y < nHeight; y++)
				{
					// ��ż����
					memcpy(pRow, pData[y], sizeof(float) * nWidth);
					for (i = 0; i < nHalfW; i++) 
					{ 
						x = i * 2; pData[y][i] = pRow[x]; 
						pData[y][nHalfW + i] = pRow[x + 1]; 
					} 
					// ����С���任 
					for (i = 0; i < nHalfW - 1; i++) 
					{ 
						fValue = (pData[y][i] + pData[y][i + 1]) / 2; 
						pData[y][nHalfW + i] -= fValue;
					}
					fValue = (pData[y][nHalfW - 1] + pData[y][nHalfW - 2]) / 2;
					pData[y][nWidth - 1] -= fValue;
					fValue = (pData[y][nHalfW] + pData[y][nHalfW + 1]) / 4;
					pData[y][0] += fValue; 
					for (i = 1; i < nHalfW; i++)
					{ 
						fValue = (pData[y][nHalfW + i] + pData[y][nHalfW + i - 1]) / 4; 
						pData[y][i] += fValue;
					}
					// Ƶ��ϵ�� 
					for (i = 0; i < nHalfW; i++) 
					{
						pData[y][i] *= fRadius; pData[y][nHalfW + i] /= fRadius;
					} 
				}
				// ��ֱ�任 
				for (x = 0; x < nWidth; x++)
				{ 
					// ��ż����
					for (i = 0; i < nHalfH; i++) 
					{ 
						y = i * 2; 
						pColumn[i] = pData[y][x]; 
						pColumn[nHalfH + i] = pData[y + 1][x];
					}
					for (i = 0; i < nHeight; i++)
					{ 
						pData[i][x] = pColumn[i]; 
					} 
					// ����С���任
					for (i = 0; i < nHalfH - 1; i++)
					{ 
						fValue = (pData[i][x] + pData[i + 1][x]) / 2;
						pData[nHalfH + i][x] -= fValue;
					}
					fValue = (pData[nHalfH - 1][x] + pData[nHalfH - 2][x]) / 2;
					pData[nHeight - 1][x] -= fValue;
					fValue = (pData[nHalfH][x] + pData[nHalfH + 1][x]) / 4; 
					pData[0][x] += fValue; 
					for (i = 1; i < nHalfH; i++) 
					{ 
						fValue = (pData[nHalfH + i][x] + pData[nHalfH + i - 1][x]) / 4; 
						pData[i][x] += fValue;
					}
					// Ƶ��ϵ�� 
					for (i = 0; i < nHalfH; i++) 
					{
						pData[i][x] *= fRadius;
						pData[nHalfH + i][x] /= fRadius;
					}
				}
			}
			delete[] pData; 
			delete[] pRow; 
			delete[] pColumn;
		}
	}
}


// ��ά��ɢС���ָ�����ͨ������ͼ��
void Wavelet::IDWT(IplImage *pImage, int nLayer)
{
	// ִ������
	if (pImage)
	{
		if (pImage->nChannels == 1 &&
			pImage->depth == IPL_DEPTH_32F &&
			((pImage->width >> nLayer) << nLayer) == pImage->width &&
			((pImage->height >> nLayer) << nLayer) == pImage->height)
		{
			int i, x, y, n;
			float fValue = 0;
			float fRadius = sqrt(2.0f);
			int nWidth = pImage->width >> (nLayer - 1);
			int nHeight = pImage->height >> (nLayer - 1);
			int nHalfW = nWidth/2;
			int nHalfH = nHeight/2;
			float **pData = new float*[pImage->height];
			float *pRow = new float[pImage->width];
			float *pColumn  = new float[pImage->height];
			for (i = 0; i < pImage->height; i++)
			{
				pData[i] = (float*) (pImage->imageData + pImage->widthStep * i);
			}
			// ���С���ָ�
			for (n = 0; n < nLayer; n++, nWidth *= 2, nHeight *= 2, nHalfW *= 2, nHalfH *= 2)
			{
				// ��ֱ�ָ�
				for (x = 0; x < nWidth; x++)
				{
					// Ƶ��ϵ��
					for (i = 0; i < nHalfH; i++)
					{
						pData[i][x] /= fRadius;
						pData[nHalfH + i][x] *= fRadius;
					}
					// ����С���ָ�
					fValue = (pData[nHalfH][x] + pData[nHalfH + 1][x]) / 4;
					pData[0][x] -= fValue;
					for (i = 1; i < nHalfH; i++)
					{
						fValue = (pData[nHalfH + i][x] + pData[nHalfH + i - 1][x]) / 4;
						pData[i][x] -= fValue;
					}
					for (i = 0; i < nHalfH - 1; i++)
					{
						fValue = (pData[i][x] + pData[i + 1][x]) / 2;
						pData[nHalfH + i][x] += fValue;
					}
					fValue = (pData[nHalfH - 1][x] + pData[nHalfH - 2][x]) / 2;
					pData[nHeight - 1][x] += fValue;
					// ��ż�ϲ�
					for (i = 0; i < nHalfH; i++)
					{
						y = i * 2;
						pColumn[y] = pData[i][x];
						pColumn[y + 1] = pData[nHalfH + i][x];
					}
					for (i = 0; i < nHeight; i++)
					{
						pData[i][x] = pColumn[i];
					}
				}
				// ˮƽ�ָ�
				for (y = 0; y < nHeight; y++)
				{
					// Ƶ��ϵ��
					for (i = 0; i < nHalfW; i++)
					{
						pData[y][i] /= fRadius;
						pData[y][nHalfW + i] *= fRadius;
					}
					// ����С���ָ�
					fValue = (pData[y][nHalfW] + pData[y][nHalfW + 1]) / 4;
					pData[y][0] -= fValue;
					for (i = 1; i < nHalfW; i++)
					{
						fValue = (pData[y][nHalfW + i] + pData[y][nHalfW + i - 1]) / 4;
						pData[y][i] -= fValue;
					}
					for (i = 0; i < nHalfW - 1; i++)
					{
						fValue = (pData[y][i] + pData[y][i + 1]) / 2;
						pData[y][nHalfW + i] += fValue;
					}
					fValue = (pData[y][nHalfW - 1] + pData[y][nHalfW - 2]) / 2;
					pData[y][nWidth - 1] += fValue;
					// ��ż�ϲ�
					for (i = 0; i < nHalfW; i++)
					{
						x = i * 2;
						pRow[x] = pData[y][i];
						pRow[x + 1] = pData[y][nHalfW + i];
					}
					memcpy(pData[y], pRow, sizeof(float) * nWidth);
				}
			}
			delete[] pData;
			delete[] pRow;
			delete[] pColumn;
		}
	}
}
