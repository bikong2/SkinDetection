// Machine Learning functions
// @author: lixihua9@126.com
// @date:   2015/09/20

#include "stdafx.h"
#include "LearnAlgorithms.h"

void TrainModel(Mat features, Mat labels, CvSVMParams param, string savefile)
{
	CvSVM svm;
	//svm.train(features, labels, Mat(), Mat(), param);
	///*
	CvParamGrid nuGrid     = CvParamGrid(1, 1, 0.0);
	CvParamGrid coeffGrid  = CvParamGrid(1, 1, 0.0);
	CvParamGrid degreeGrid = CvParamGrid(1, 1, 0.0);
	svm.train_auto(features, labels, Mat(), Mat(), param, 10,
		svm.get_default_grid(CvSVM::C),
		svm.get_default_grid(CvSVM::GAMMA),
		svm.get_default_grid(CvSVM::P),
		nuGrid, coeffGrid, degreeGrid, false);
	//*/
	svm.save(savefile.c_str());
}