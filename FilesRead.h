// Files Reading functions
// @author: lixihua9@126.com
// @date:   2015/09/20

#pragma once
#include <string>
#include <map>
#include <io.h>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>

using namespace std;

// label struct from information data
struct InfoStruct
{
	string Index;
	string OriImage;
	string SpotImage;
	string DarkImage;
	string NormImage;
	int    grade;
};

void GetFiles(string path, string exd, vector<string>& files);
void GetLabelInfo(string path, map<string, InfoStruct>& infos);
vector<string> str_split(string str, string pattern);