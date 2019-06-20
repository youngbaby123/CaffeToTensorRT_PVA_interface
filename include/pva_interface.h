#ifndef _FACEVISA_CLASSFICATION_INTERFACE_H_
#define _FACEVISA_CLASSFICATION_INTERFACE_H_

#include <stdio.h>
#include <opencv2/core/core.hpp>

#ifdef FACEVISA_CLASSFICATION_INTERFACE_H
#define FACEVISA_CLASSFICATION_API __declspec(dllexport)
#else
#define FACEVISA_CLASSFICATION_API __declspec(dllimport)
#endif

#define       FACEVISA_OK                     0x11120000
#define       FACEVISA_ALLOC_MEMORY_ERROR     0x11120001
#define       FACEVISA_PARAMETER_ERROR        0x11120002

typedef void * Facevisa_TensorRT_handle;

typedef struct _Facevisa_TensorRT_bbox_
{
	int cls;									// 框的类别
	float score;								// 框的得分
	cv::Rect BBox;								// 框的坐标
}RT_bbox;

//单batch 输出
typedef struct _Facevisa_TensorRT_result_single_
{
	std::vector<RT_bbox> det_res;				// 所有检测目标构成的vector
}Facevisa_TensorRT_result_s;

int Facevisa_Engine_Create(Facevisa_TensorRT_handle *handle, int device_id);

int Facevisa_Engine_Inference(Facevisa_TensorRT_handle handle, cv::Mat &image, Facevisa_TensorRT_result_s &results);

int Facevisa_Engine_Release(Facevisa_TensorRT_handle handle);

#endif !_FACEVISA_CLASSFICATION_INTERFACE_H_