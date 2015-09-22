// Machine Learning functions
// @author: lixihua9@126.com
// @date:   2015/09/20

#pragma once
#include <string>
#include <map>
#include <math.h>
#include <io.h>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace std;

// train model: svm
void TrainModel(Mat features, Mat labels, CvSVMParams param, string savefile);