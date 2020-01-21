#include "pva_interface.h"
#include "tools.h"
#include <iostream>
#include <ctime>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> 

#include "windows.h"

typedef void *HANDLE;

// 前向测试demo
#define Test_SDK					(1)
// 序列化demo
#define MODL_SERIALIZE				(0)

// 单batch 测试
int main(int argc, char** argv)
{
#if Test_SDK
	Facevisa_TensorRT_handle handle = NULL;
	Facevisa_Engine_Create(&handle, 0);
	// 图片路径
	std::string root_dir = R"(D:\Workspace\task\001_tensorRT\factory\pva_cascade_interface\data_sc\)";
	std::vector<std::string> files_name;
	int files_number;
	read_files(root_dir + "*.bmp", files_name, &files_number);
	read_files(root_dir + "*.png", files_name, &files_number);
	read_files(root_dir + "*.jpg", files_name, &files_number);
	std::vector<int> compression_params;
	compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
	compression_params.push_back(100);


	static HANDLE topdetectionevent = 0;
	if (topdetectionevent == 0)
	{
		topdetectionevent = CreateEvent(NULL, FALSE, TRUE, NULL);
	}

#pragma omp parallel for
	for (int ssss = 0; ssss < 10; ssss++) {
		for (int index = 0; index < files_name.size(); index++)
		{
			std::cout << "[" << index << "/" << files_name.size() << "]" << " img name is: " << root_dir + files_name[index] << std::endl;
			//*it = "original_1106_1_D10502_55_72_20180621233457.jpg";
			std::string::size_type idx = files_name[index].rfind("/");
			std::string img_name = files_name[index].substr(idx + 1);
			cv::Mat img = cv::imread(root_dir + files_name[index]);
			if (0 == img.rows || 0 == img.cols || NULL == img.data)
				continue;

			WaitForSingleObject(topdetectionevent, INFINITE);
			// 环形油污检测
			Facevisa_TensorRT_PVA_result results;
			double start = clock();
			if (FACEVISA_OK != Facevisa_Engine_Inference(handle, img, results)) {
				std::cout << "检测失败！！！ " << std::endl;
				continue;
			}
			double end = clock();
			std::cout << "whloe time is: " << end - start << " ms!" << std::endl;
			SetEvent(topdetectionevent);

			for (int box_i = 0; box_i < results.det_res.size(); box_i++) {
				std::cout << "框的类别： " << results.det_res[box_i].index << std::endl;
				std::cout << "框的得分： " << results.det_res[box_i].score << std::endl;
				cv::rectangle(img, results.det_res[box_i].BBox, cv::Scalar(0, 255, 255), 5, 8, 0);
				std::stringstream os;
				os << results.det_res[box_i].score;//bboxes[i][5]存放label
				std::string characters = os.str();
				cv::putText(img, characters, cv::Point(results.det_res[box_i].BBox.x, results.det_res[box_i].BBox.y), CV_FONT_HERSHEY_SIMPLEX, 4, cv::Scalar(0, 255, 0), 2);

				if (results.cls_res[box_i].index == 1)
				{
					std::stringstream os_cls;
					os_cls << results.cls_res[box_i].score;//bboxes[i][5]存放label
					std::string characters_cls = os_cls.str();
					cv::putText(img, characters_cls, cv::Point(results.det_res[box_i].BBox.x, results.det_res[box_i].BBox.y + 0.5* results.det_res[box_i].BBox.height), CV_FONT_HERSHEY_SIMPLEX, 4, cv::Scalar(0, 0, 255), 2);
				}
			}


			//cv::namedWindow("PVA", 0);
			//cv::imshow("PVA", img);
			//cv::waitKey(0);
			std::cout << "---------------------" << std::endl;
		}
	}

	Facevisa_Engine_Release(handle);
#endif //Test_SDK

#if MODL_SERIALIZE
	std::string root_dir = R"(D:\Workspace\task\00_svn\pva_cascaded\makefile\Facevisa_interface_tensorRT_PVA\x64\Release\templates\)";
	std::string prototxt = root_dir + "pva_model.prototxt";
	std::string caffemodel = root_dir + "pva_model.caffemodel";
	std::string engine_resialize_save = root_dir + "pva_model.bin";

	// batch size 一定要设对  需要与你的工程配套
	int batch_size = 1;

	// 输出层设置
	//det  "bbox_pred", "cls_prob", "rois"   // cls "prob"
	//// cls
	//std::vector<std::string> blob_names_ = { "prob" }; 
	// det
	std::vector<std::string> blob_names_ = { "bbox_pred", "cls_prob", "rois" };

	Facevisa_CaffeModelSerialize(prototxt, caffemodel, blob_names_, batch_size, engine_resialize_save);

#endif // MODL_SERIALIZE

	system("pause");
	return 0;
}




