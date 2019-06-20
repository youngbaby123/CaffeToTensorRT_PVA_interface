
#ifndef __FACEVISA_COLOR_TOOL_HRADER__
#define __FACEVISA_COLOR_TOOL_HRADER__

#include <io.h>
#include <vector>
#include <string>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

void checkName(std::string& name);
void createFolder(std::string fodler_path);

int read_files(const std::string &path, std::vector<std::string> &files_name, int *files_number);

int detect_crop(const cv::Mat &img, cv::Mat &crop);

//int classify_color_tail(const cv::Mat &img, ct_rasult *result);

int save_crop_img(std::string &src_dir, std::string &sub_dir, std::string &date, std::string &dst_dir);

void readTxt(std::string &file, std::vector<std::string> &img_name, std::vector<std::string> &label);

void ftostring(float number, std::string &output);

void drawTextImg(cv::Mat &img, std::string text, cv::Point &origin, cv::Scalar &scalar, int thickness, int lineType);

template <typename Dtype>
void numbertostring(Dtype number, std::string &output);

template <typename Dtype>
std::string numbertostring(Dtype number);

#endif