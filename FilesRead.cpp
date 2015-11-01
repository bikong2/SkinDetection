// Files Reading functions
// @author: lixihua9@126.com
// @date:   2015/09/20

#include "stdafx.h"
#include "FilesRead.h"

void GetFiles(string path, string exd, vector<string>& files)
{
	long hFile = 0;
	struct _finddata_t fileinfo;
	string pathName, exdName;
	if (0 != strcmp(exd.c_str(), ""))
	{
		exdName = "/*" + exd;
	}
	else
	{
		exdName = "/*";
	}
 	if ((hFile = long(_findfirst(pathName.assign(path).append(exdName).c_str(), &fileinfo))) !=  -1)
	{
		do
		{
			if((fileinfo.attrib & _A_SUBDIR))
			{
				if(strcmp(fileinfo.name, ".") != 0  &&  strcmp(fileinfo.name, "..") != 0)
					GetFiles(pathName.assign(path).append("/").append(fileinfo.name), exd, files);
			}
			else
			{
				if(strcmp(fileinfo.name, ".") != 0  &&  strcmp(fileinfo.name, "..") != 0)
					//cout << pathName.assign(path).append("/").append(fileinfo.name) << endl;
					files.push_back(pathName.assign(path).append("/").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

void GetLabelInfo(string path, map<string, InfoStruct>& infos)
{
	ifstream fin(path, ios::in);
	char line[1024] = {0};
	string personindex;
	InfoStruct tmp_info;
	while (fin.getline(line, sizeof(line)))
	{
		stringstream word(line);
		word >> personindex;
		if (-1 == personindex.find("Sub.")) continue;
		tmp_info.Index = personindex;
		word >> tmp_info.OriImage;
		word >> tmp_info.grade;
		infos.insert(pair<string, InfoStruct>(tmp_info.Index, tmp_info));
	}
	fin.clear();
	fin.close();
}

vector<string> str_split(string str, string pattern)
{
	vector<string> ret;
	if (pattern.empty()) return ret;
	size_t start = 0;
	size_t index = str.find_first_of(pattern, 0);
	while (index != str.npos) {
		if (start != index) {
			ret.push_back(str.substr(start, index - start));
			start = index + 1;
			index = str.find_first_of(pattern, start);
		}
	}
	if (!str.substr(start).empty()) ret.push_back(str.substr(start));
	return ret;
}