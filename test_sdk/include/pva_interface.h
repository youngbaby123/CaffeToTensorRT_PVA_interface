#ifndef _FACEVISA_PVA_TENSORRT_INTERFACE_H_
#define _FACEVISA_PVA_TENSORRT_INTERFACE_H_

#include <stdio.h>
#include <opencv2/core/core.hpp>
#ifdef FACEVISA_PVA_TENSORRT_INTERFACE_H
#define FACEVISA_PVA_TENSORRT_API __declspec(dllexport)
#else
#define FACEVISA_PVA_TENSORRT_API __declspec(dllimport)
#endif

#define       FACEVISA_OK                     0x11120000
#define       FACEVISA_ALLOC_MEMORY_ERROR     0x11120001
#define       FACEVISA_PARAMETER_ERROR        0x11120002

typedef void * Facevisa_TensorRT_handle;

typedef struct _Facevisa_TensorRT_bbox_
{
	int index;									// ������
	float score;								// ��ĵ÷�
	cv::Rect BBox;								// �������
}RT_bbox;

//��batch ���
typedef struct _Facevisa_TensorRT_PVA_result_
{
	std::vector<RT_bbox> det_res;				// ���м��Ŀ�깹�ɵ�vector
	std::vector<RT_bbox> cls_res;				// ���м��Ŀ�깹�ɵ�vector
}Facevisa_TensorRT_PVA_result;

FACEVISA_PVA_TENSORRT_API int Facevisa_Engine_Create(Facevisa_TensorRT_handle *handle, int device_id);

FACEVISA_PVA_TENSORRT_API int Facevisa_Engine_Inference(Facevisa_TensorRT_handle handle, cv::Mat &image, Facevisa_TensorRT_PVA_result &results);

FACEVISA_PVA_TENSORRT_API int Facevisa_Engine_Release(Facevisa_TensorRT_handle handle);

FACEVISA_PVA_TENSORRT_API int Facevisa_CaffeModelSerialize(std::string deployFile, std::string modelFile, const std::vector<std::string>& outputs, unsigned int maxBatchSize, std::string& engine_resialize_save);

#endif !_FACEVISA_PVA_TENSORRT_INTERFACE_H_