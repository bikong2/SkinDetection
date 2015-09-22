// Machine Learning functions
// @author: lixihua9@126.com
// @date:   2015/09/20

#include "stdafx.h"
#include "LearnAlgorithms.h"

void TrainModel(Mat features, Mat labels, CvSVMParams param, string savefile)
{
	CvSVM svm;
	svm.train(features, labels, Mat(), Mat(), param);
	svm.save(savefile.c_str());
}