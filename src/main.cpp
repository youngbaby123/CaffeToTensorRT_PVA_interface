#include "pva_interface.h"
#include "tools.h"
#include <iostream>
#include <ctime>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> 

// 单batch 测试
int main(int argc, char** argv)
{
	Facevisa_TensorRT_handle handle;
	Facevisa_Engine_Create(&handle, 0);
	// 图片路径
	std::string root_dir = R"(D:\Workspace\task\001_tensorRT\pva_interface\data\)";
	std::vector<std::string> files_name;
	int files_number;
	read_files(root_dir + "*.bmp", files_name, &files_number);
	read_files(root_dir + "*.png", files_name, &files_number); 
	read_files(root_dir + "*.jpg", files_name, &files_number);
	std::vector<int> compression_params;
	compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
	compression_params.push_back(100);

	for (int index = 0; index < files_name.size(); index++)
	{
		std::cout << "[" << index << "/" << files_name.size() << "]" << " img name is: " << root_dir + files_name[index] << std::endl;
		//*it = "original_1106_1_D10502_55_72_20180621233457.jpg";
		std::string::size_type idx = files_name[index].rfind("/");
		std::string img_name = files_name[index].substr(idx + 1);
		cv::Mat img = cv::imread(root_dir + files_name[index]);
		if (0 == img.rows || 0 == img.cols || NULL == img.data)
			continue;

		// 环形油污检测
		Facevisa_TensorRT_result_s results;
		double start = clock();
		if (FACEVISA_OK != Facevisa_Engine_Inference(handle, img, results)) {
			std::cout << "检测失败！！！ " << std::endl;
			continue;
		}
		double end = clock();
		std::cout << "whloe time is: " << end - start << " ms!" << std::endl;
		for (int box_i = 0; box_i < results.det_res.size(); box_i++) {
			std::cout << "框的类别： " << results.det_res[box_i].cls << std::endl;
			std::cout << "框的得分： " << results.det_res[box_i].score << std::endl;
			cv::rectangle(img, results.det_res[box_i].BBox, cv::Scalar(0, 255, 255), 5, 8, 0);
			std::stringstream os;
			os << results.det_res[box_i].score;//bboxes[i][5]存放label
			std::string characters = os.str();
			cv::putText(img, characters, cv::Point(results.det_res[box_i].BBox.x, results.det_res[box_i].BBox.y), CV_FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0));
		}
		cv::namedWindow("PVA", 0);
		cv::imshow("PVA", img);
		cv::waitKey(0);
		std::cout << "---------------------" << std::endl;


	}
	Facevisa_Engine_Release(handle);
	return 0;
}




