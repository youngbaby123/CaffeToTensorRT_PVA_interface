#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "common.h"
#include "data_tools.h"
#include "pva_interface.h"

#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> 
#include <windows.h>
#include <algorithm>
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;
// Network details

#define ENGINE_CREATE_WITH_Serialization		(true)		// 是否使用序列化模型进行初始化（备注：序列化模型是指 前向之前已经将模型进行优化，可以节省初始化时间）
#define INPUT_DATA_TYPE_GPU						(true)		// 是否使用GPU做数据预处理
#define IF_CASCADE_ALL_BOXES					(1)			// 对于级联是否完全检测所有框

static const int kINPUT_BATCH_SIZE_DET	= 1;				// Input image batch size
static const int kINPUT_BATCH_SIZE_CLS	= 20;				// Input image batch size
static const float NMS_THRESHOLD		= 0.3f;				// outpot bbox nms threshold
static const float SCORE_THRESHOLD		= 0.1f;				// outpot score pva threshold


static Logger gLogger;

typedef struct TensorRTCaffeCantainerBaby_ {
	int buffers_size;								// 输入输出数据维度
	std::vector<DimsNCHW> buffers_dims;				// 2个维度的输入dims，3个维度的输出dims
	void** buffers;									// 输入输出数据，TensorRT 中前向函数需要
	unsigned char *cv_input_gpu;					// GPU存储前处理数据
	nvinfer1::ICudaEngine* engine;
	IExecutionContext* context;
}TensorRTCaffeBaby;

// 工程相关变量容器
typedef struct TensorRTCaffeCantainer_ {
	int gpu_id;										// 配置 GPU 的 index	
	cudaStream_t stream;
	TensorRTCaffeBaby DET;
	TensorRTCaffeBaby CLS;
}TensorRTCaffeCantainer;

// 单 batch 
static int processImg(cv::Mat &img, int inputchannels, float *imgData, cv::Scalar_<float> meanValue) {
	int shift_data = 0;
	cv::Mat float_img;
	std::vector<cv::Mat> splitchannles = std::vector<cv::Mat>(inputchannels);
	if (3 == inputchannels) {
		if (1 == img.channels())
			cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
		img.convertTo(float_img, CV_32F);

		cv::Mat mean_(img.size(), CV_32FC3, meanValue);
		cv::subtract(float_img, mean_, float_img);
		cv::split(float_img, splitchannles);
	}
	else if (1 == inputchannels) {
		if (3 == img.channels())
			cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
		img.convertTo(float_img, CV_32F);

		cv::Mat mean_(img.size(), CV_32FC1, meanValue);
		cv::subtract(float_img, mean_, float_img);
		splitchannles.emplace_back(float_img);
	}
	else {
		return FACEVISA_PARAMETER_ERROR;
	}
	shift_data = sizeof(float) * img.rows * img.cols;
	for (size_t i = 0; i < inputchannels; i++) {
		memcpy(imgData, splitchannles[i].data, shift_data);
		imgData += img.rows * img.cols;
	}

	return FACEVISA_OK;
}

// 多 batch 
static int processImg(std::vector<cv::Mat> &imgs, int inputchannels, float *imgData, cv::Scalar_<float> meanValue) {
	int shift_data = 0;
	for (size_t index = 0; index < imgs.size(); index++) {
		cv::Mat float_img;
		cv::Mat img = imgs[index];
		std::vector<cv::Mat> splitchannles = std::vector<cv::Mat>(inputchannels);
		if (3 == inputchannels) {
			if (1 == img.channels())
				cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
			img.convertTo(float_img, CV_32F);
			cv::Mat mean_(img.size(), CV_32FC3, meanValue);
			cv::subtract(float_img, mean_, float_img);
			cv::split(float_img, splitchannles);
		}
		else if (1 == inputchannels) {
			if (3 == img.channels())
				cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
			img.convertTo(float_img, CV_32F);
			cv::Mat mean_(img.size(), CV_32FC1, meanValue);
			cv::subtract(float_img, mean_, float_img);
			splitchannles.emplace_back(float_img);
		}
		else {
			return FACEVISA_PARAMETER_ERROR;
		}
		shift_data = sizeof(float) * img.rows * img.cols;
		for (size_t i = 0; i < inputchannels; i++) {
			memcpy(imgData, splitchannles[i].data, shift_data);
			imgData += img.rows * img.cols;
		}
	}
	return FACEVISA_OK;
}

// 将输出的框进行尺度转换，转换到原图大小
void bboxTransformInvAndClip(std::vector<float>& rois, std::vector<float>& deltas, std::vector<float>& predBBoxes, float* imInfo,
	const int N, const int nmsMaxOut, const int numCls)
{
	for (int i = 0; i < N * nmsMaxOut; ++i)
	{
		float width = rois[i * 4 + 2] - rois[i * 4] + 1;
		float height = rois[i * 4 + 3] - rois[i * 4 + 1] + 1;
		float ctr_x = rois[i * 4] + 0.5f * width;
		float ctr_y = rois[i * 4 + 1] + 0.5f * height;

		float* imInfo_offset = imInfo + i / nmsMaxOut * 3;
		for (int j = 0; j < numCls; ++j)
		{
			float dx = deltas[i * numCls * 4 + j * 4];
			float dy = deltas[i * numCls * 4 + j * 4 + 1];
			float dw = deltas[i * numCls * 4 + j * 4 + 2];
			float dh = deltas[i * numCls * 4 + j * 4 + 3];
			float pred_ctr_x = (dx * width + ctr_x) / imInfo_offset[3];
			float pred_ctr_y = (dy * height + ctr_y) / imInfo_offset[2];
			float pred_w = (exp(dw) * width) / imInfo_offset[3];
			float pred_h = (exp(dh) * height) / imInfo_offset[2];
			predBBoxes[i * numCls * 4 + j * 4] = max(min(pred_ctr_x - 0.5f * pred_w, imInfo_offset[1] - 1.f), 0.f);
			predBBoxes[i * numCls * 4 + j * 4 + 1] = max(min(pred_ctr_y - 0.5f * pred_h, imInfo_offset[0] - 1.f), 0.f);
			predBBoxes[i * numCls * 4 + j * 4 + 2] = max(min(pred_ctr_x + 0.5f * pred_w, imInfo_offset[1] - 1.f), 0.f);
			predBBoxes[i * numCls * 4 + j * 4 + 3] = max(min(pred_ctr_y + 0.5f * pred_h, imInfo_offset[0] - 1.f), 0.f);
		}
	}
}

std::vector<int> nms(std::vector<std::pair<float, int>>& score_index, float* bbox, const int classNum, const int numClasses, const float nms_threshold)
{
	auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
		if (x1min > x2min)
		{
			std::swap(x1min, x2min);
			std::swap(x1max, x2max);
		}
		return x1max < x2min ? 0 : min(x1max, x2max) - x2min;
	};
	auto computeIoU = [&overlap1D](float* bbox1, float* bbox2) -> float {
		float overlapX = overlap1D(bbox1[0], bbox1[2], bbox2[0], bbox2[2]);
		float overlapY = overlap1D(bbox1[1], bbox1[3], bbox2[1], bbox2[3]);
		float area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
		float area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
		float overlap2D = overlapX * overlapY;
		float u = area1 + area2 - overlap2D;
		return u == 0 ? 0 : overlap2D / u;
	};

	std::vector<int> indices;
	for (auto i : score_index)
	{
		const int idx = i.second;
		bool keep = true;
		for (unsigned k = 0; k < indices.size(); ++k)
		{
			if (keep)
			{
				const int kept_idx = indices[k];
				float overlap = computeIoU(&bbox[(idx * numClasses + classNum) * 4],
					&bbox[(kept_idx * numClasses + classNum) * 4]);
				keep = overlap <= nms_threshold;
			}
			else
				break;
		}
		if (keep)
			indices.push_back(idx);
	}
	return indices;
}


// 2020.1.21 by yxx 加入序列化方式进行模型初始化
#if ENGINE_CREATE_WITH_Serialization
// 模型初始化 使用bin方式
int Facevisa_Engine_Create(Facevisa_TensorRT_handle *handle, int device_id) {
	if (NULL != *handle || NULL == handle)
	{
		return FACEVISA_PARAMETER_ERROR;
	}

	char moduleFileName[MAX_PATH];
	GetModuleFileNameA(0, moduleFileName, MAX_PATH);
	char * ptr = strrchr(moduleFileName, '\\');
	ptr++;
	strcpy(ptr, "templates\\");
	std::string root_dir = std::string(moduleFileName);

	std::string rtmodelstr_det = root_dir + "pva_model.bin";
	std::string rtmodelstr_cls = root_dir + "cascade_model.bin";


	// 初始化变量容器
	TensorRTCaffeCantainer *param = new (std::nothrow) TensorRTCaffeCantainer();
	if (NULL == param)
	{
		return FACEVISA_ALLOC_MEMORY_ERROR;
	}

	// 设置指定GPU, 备注未找到TensorRT的封装写法，先用cuda自己的调用方法
	if ((cudaSuccess == cudaSetDevice(device_id)) && (cudaSuccess == cudaFree(0)))
	{
		param->gpu_id = device_id;
	}
	else
	{
		param->gpu_id = -1;
		return FACEVISA_ALLOC_MEMORY_ERROR;
	}

	// Create a stream
	if (0 != cudaStreamCreate(&param->stream)) {
		return FACEVISA_ALLOC_MEMORY_ERROR;
	}

	IHostMemory* trtModelStream{ nullptr };
	initLibNvInferPlugins(&gLogger, "");


	std::vector<std::string> blob_names_cls = { "data", "prob" };
	// ****************************************   CLS  *******************************************
	// Parse the caffe model to populate the network, then set the outputs
	std::ifstream in_file_cls(rtmodelstr_cls.c_str(), std::ios::in | std::ios::binary);
	std::streampos begin_cls, end_cls;
	begin_cls = in_file_cls.tellg();
	in_file_cls.seekg(0, std::ios::end);
	end_cls = in_file_cls.tellg();
	std::size_t size_cls = end_cls - begin_cls;

	in_file_cls.seekg(0, std::ios::beg);
	std::unique_ptr<unsigned char[]> engine_data_cls(new unsigned char[size_cls]);
	in_file_cls.read((char*)engine_data_cls.get(), size_cls);
	in_file_cls.close();

	// Build the engine
	IRuntime* runtime_cls = createInferRuntime(gLogger);
	ICudaEngine* engine_cls = runtime_cls->deserializeCudaEngine((const void*)engine_data_cls.get(), size_cls, nullptr);


	// 设置输入输出的buffers， 
	param->CLS.buffers_size = engine_cls->getNbBindings();
	param->CLS.buffers = (void **)malloc(sizeof(float*) * engine_cls->getNbBindings());
	param->CLS.buffers_dims = vector<DimsNCHW>(engine_cls->getNbBindings());

	param->CLS.engine = engine_cls;
	// assert(param->engine != nullptr);
	param->CLS.context = engine_cls->createExecutionContext();

	// Create GPU buffers and a stream
	for (int name_idx = 0; name_idx < blob_names_cls.size(); name_idx++) {
		int blobIndex = engine_cls->getBindingIndex(blob_names_cls[name_idx].c_str());
		Dims blobDims = engine_cls->getBindingDimensions(blobIndex);
		int dim_out = 0;
		if (blobDims.nbDims == 3) {
			param->CLS.buffers_dims[name_idx].d[0] = engine_cls->getMaxBatchSize();
			dim_out++;
		}
		for (int dim_in = 0; dim_in < blobDims.nbDims; dim_in++, dim_out++) {
			param->CLS.buffers_dims[name_idx].d[dim_out] = blobDims.d[dim_in];
		}
		if (0 != cudaMalloc(&(param->CLS.buffers[blobIndex]), param->CLS.buffers_dims[name_idx].n() * param->CLS.buffers_dims[name_idx].c() * param->CLS.buffers_dims[name_idx].h() * param->CLS.buffers_dims[name_idx].w() * sizeof(float))) {
			runtime_cls->destroy();
			return FACEVISA_ALLOC_MEMORY_ERROR;
		}
	}

	cudaMalloc((void**)&param->CLS.cv_input_gpu, param->CLS.buffers_dims[0].n() * param->CLS.buffers_dims[0].c() * param->CLS.buffers_dims[0].h() * param->CLS.buffers_dims[0].w() * sizeof(unsigned char));


	// release cls
	runtime_cls->destroy();

	// *******************************************   DET   *****************************************
	// Parse the caffe model to populate the network, then set the outputs
	std::vector<std::string> blob_names_det = { "data", "im_info", "bbox_pred", "cls_prob", "rois" };
	std::ifstream in_file_det(rtmodelstr_det.c_str(), std::ios::in | std::ios::binary);
	std::streampos begin_det, end_det;
	begin_det = in_file_det.tellg();
	in_file_det.seekg(0, std::ios::end);
	end_det = in_file_det.tellg();
	std::size_t size_det = end_det - begin_det;

	in_file_det.seekg(0, std::ios::beg);
	std::unique_ptr<unsigned char[]> engine_data_det(new unsigned char[size_det]);
	in_file_det.read((char*)engine_data_det.get(), size_det);
	in_file_det.close();

	// Build the engine
	IRuntime* runtime_det = createInferRuntime(gLogger);
	ICudaEngine* engine_det = runtime_det->deserializeCudaEngine((const void*)engine_data_det.get(), size_det, nullptr);

	// 设置输入输出的buffers， 
	param->DET.buffers_size = engine_det->getNbBindings();
	param->DET.buffers = (void **)malloc(sizeof(float*) * engine_det->getNbBindings());
	param->DET.buffers_dims = vector<DimsNCHW>(engine_det->getNbBindings());

	param->DET.engine = engine_det;
	// assert(param->engine != nullptr);
	param->DET.context = engine_det->createExecutionContext();

	// Create GPU buffers and a stream
	for (int name_idx = 0; name_idx < blob_names_det.size(); name_idx++) {
		int blobIndex = engine_det->getBindingIndex(blob_names_det[name_idx].c_str());
		Dims blobDims = engine_det->getBindingDimensions(blobIndex);
		int dim_out = 0;
		if (blobDims.nbDims == 3) {
			param->DET.buffers_dims[name_idx].d[0] = engine_det->getMaxBatchSize();
			dim_out++;
		}
		for (int dim_in = 0; dim_in < blobDims.nbDims; dim_in++, dim_out++) {
			param->DET.buffers_dims[name_idx].d[dim_out] = blobDims.d[dim_in];
		}
		if (0 != cudaMalloc(&(param->DET.buffers[blobIndex]), param->DET.buffers_dims[name_idx].n() * param->DET.buffers_dims[name_idx].c() * param->DET.buffers_dims[name_idx].h() * param->DET.buffers_dims[name_idx].w() * sizeof(float))) {
			runtime_det->destroy();
			return FACEVISA_ALLOC_MEMORY_ERROR;
		}
	}

	cudaMalloc((void**)&param->DET.cv_input_gpu, param->DET.buffers_dims[0].n() * param->DET.buffers_dims[0].c() * param->DET.buffers_dims[0].h() * param->DET.buffers_dims[0].w() * sizeof(unsigned char));


	// release det
	runtime_det->destroy();

	*handle = param;
	//shutdownProtobufLibrary();
	return FACEVISA_OK;
}
#else
// 模型初始化， 使用prototxt，caffemodel方式
int Facevisa_Engine_Create(Facevisa_TensorRT_handle *handle, int device_id) {
	if (NULL != *handle || NULL == handle)
	{
		return FACEVISA_PARAMETER_ERROR;
	}


	char moduleFileName[MAX_PATH];
	GetModuleFileNameA(0, moduleFileName, MAX_PATH);
	char * ptr = strrchr(moduleFileName, '\\');
	ptr++;
	strcpy(ptr, "templates\\");
	std::string root_dir = std::string(moduleFileName);
	std::string protostr_det = root_dir + "pva_model.prototxt";
	std::string modelstr_det = root_dir + "pva_model.caffemodel";
	std::string protostr_cls = root_dir + "cascade_model.prototxt";
	std::string modelstr_cls = root_dir + "cascade_model.caffemodel";

	// 初始化变量容器
	TensorRTCaffeCantainer *param = new (std::nothrow) TensorRTCaffeCantainer();
	if (NULL == param)
	{
		return FACEVISA_ALLOC_MEMORY_ERROR;
	}

	// 设置指定GPU, 备注未找到TensorRT的封装写法，先用cuda自己的调用方法
	if ((cudaSuccess == cudaSetDevice(device_id)) && (cudaSuccess == cudaFree(0)))
	{
		param->gpu_id = device_id;
	}
	else
	{
		param->gpu_id = -1;
		return FACEVISA_ALLOC_MEMORY_ERROR;
	}

	// Create a stream
	if (0 != cudaStreamCreate(&param->stream)) {
		return FACEVISA_ALLOC_MEMORY_ERROR;
	}

	//IHostMemory* trtModelStream{ nullptr };
	initLibNvInferPlugins(&gLogger, "");
	

	// ****************************************   CLS   *******************************************

	IBuilder* builder = createInferBuilder(gLogger);
	assert(builder != nullptr);

	builder->setMaxWorkspaceSize(100_MB);									// 设置最大使用显存
	builder->allowGPUFallback(true);

	// Parse the caffe model to populate the network, then set the outputs
	INetworkDefinition* network_cls = builder->createNetwork();

	ICaffeParser* parser_cls = createCaffeParser();
	// 读取caffe model
	const IBlobNameToTensor* blobNameToTensor_cls = parser_cls->parse(protostr_cls.c_str(), modelstr_cls.c_str(), *network_cls, DataType::kFLOAT);

	// Specify which tensors are outputs
	// 输出从第三个维度开始
	std::vector<std::string> blob_names_cls = { "data", "prob" };
	for (int name_idx = 1; name_idx < blob_names_cls.size(); name_idx++) {
		network_cls->markOutput(*blobNameToTensor_cls->find(blob_names_cls[name_idx].c_str()));
	}

	// Build the engine
	builder->setMaxBatchSize(kINPUT_BATCH_SIZE_CLS);

	ICudaEngine* engine_cls = builder->buildCudaEngine(*network_cls);

	// 设置输入输出的buffers， 
	param->CLS.buffers_size = engine_cls->getNbBindings();
	param->CLS.buffers = (void **)malloc(sizeof(float*) * engine_cls->getNbBindings());
	param->CLS.buffers_dims = vector<DimsNCHW>(engine_cls->getNbBindings());

	param->CLS.engine = engine_cls;
	//assert(param->engine != nullptr);
	param->CLS.context = engine_cls->createExecutionContext();

	// Create GPU buffers and a stream
	for (int name_idx = 0; name_idx < blob_names_cls.size(); name_idx++) {
		int blobIndex = engine_cls->getBindingIndex(blob_names_cls[name_idx].c_str());
		Dims blobDims = engine_cls->getBindingDimensions(blobIndex);
		int dim_out = 0;
		if (blobDims.nbDims == 3) {
			param->CLS.buffers_dims[name_idx].d[0] = engine_cls->getMaxBatchSize();
			dim_out++;
		}
		for (int dim_in = 0; dim_in < blobDims.nbDims; dim_in++, dim_out++) {
			param->CLS.buffers_dims[name_idx].d[dim_out] = blobDims.d[dim_in];
		}
		if (0 != cudaMalloc(&(param->CLS.buffers[blobIndex]), param->CLS.buffers_dims[name_idx].n() * param->CLS.buffers_dims[name_idx].c() * param->CLS.buffers_dims[name_idx].h() * param->CLS.buffers_dims[name_idx].w() * sizeof(float))) {
			return FACEVISA_ALLOC_MEMORY_ERROR;
		}
	}

	//unsigned char *d_in;
	cudaMalloc((void**)&param->CLS.cv_input_gpu, param->CLS.buffers_dims[0].n() * param->CLS.buffers_dims[0].c() * param->CLS.buffers_dims[0].h() * param->CLS.buffers_dims[0].w() * sizeof(unsigned char));


	// release cls
	network_cls->destroy();
	parser_cls->destroy();


	// *******************************************   DET   *****************************************
	// Parse the caffe model to populate the network, then set the outputs
	INetworkDefinition* network_det = builder->createNetwork();

	ICaffeParser* parser_det = createCaffeParser();
	// 读取caffe model
	const IBlobNameToTensor* blobNameToTensor_det = parser_det->parse(protostr_det.c_str(), modelstr_det.c_str(), *network_det, DataType::kFLOAT);

	// Specify which tensors are outputs
	// 输出从第三个维度开始
	std::vector<std::string> blob_names_det = { "data", "im_info", "bbox_pred", "cls_prob", "rois" };
	for (int name_idx = 2; name_idx < blob_names_det.size(); name_idx++) {
		network_det->markOutput(*blobNameToTensor_det->find(blob_names_det[name_idx].c_str()));
	}

	// Build the engine
	builder->setMaxBatchSize(kINPUT_BATCH_SIZE_DET);


	ICudaEngine* engine_det = builder->buildCudaEngine(*network_det);

	// 设置输入输出的buffers， 
	param->DET.buffers_size = engine_det->getNbBindings();
	param->DET.buffers = (void **)malloc(sizeof(float*) * engine_det->getNbBindings());
	param->DET.buffers_dims = vector<DimsNCHW>(engine_det->getNbBindings());

	param->DET.engine = engine_det;
	//assert(param->engine != nullptr);
	param->DET.context = engine_det->createExecutionContext();

	// Create GPU buffers and a stream
	for (int name_idx = 0; name_idx < blob_names_det.size(); name_idx++) {
		int blobIndex = engine_det->getBindingIndex(blob_names_det[name_idx].c_str());
		Dims blobDims = engine_det->getBindingDimensions(blobIndex);
		int dim_out = 0;
		if (blobDims.nbDims == 3) {
			param->DET.buffers_dims[name_idx].d[0] = engine_det->getMaxBatchSize();
			dim_out++;
		}
		for (int dim_in = 0; dim_in < blobDims.nbDims; dim_in++, dim_out++) {
			param->DET.buffers_dims[name_idx].d[dim_out] = blobDims.d[dim_in];
		}
		if (0 != cudaMalloc(&(param->DET.buffers[blobIndex]), param->DET.buffers_dims[name_idx].n() * param->DET.buffers_dims[name_idx].c() * param->DET.buffers_dims[name_idx].h() * param->DET.buffers_dims[name_idx].w() * sizeof(float))) {
			return FACEVISA_ALLOC_MEMORY_ERROR;
		}
	}

	//unsigned char *d_in;
	cudaMalloc((void**)&param->DET.cv_input_gpu, param->DET.buffers_dims[0].n() * param->DET.buffers_dims[0].c() * param->DET.buffers_dims[0].h() * param->DET.buffers_dims[0].w() * sizeof(unsigned char));


	// release det
	network_det->destroy();
	parser_det->destroy();


	*handle = param;
	builder->destroy();
	//shutdownProtobufLibrary();
	return FACEVISA_OK;
}
#endif //ENGINE_CREATE_WITH_Serialization


// DET网络前向
static int Facevisa_Engine_Forward_DET(Facevisa_TensorRT_handle handle, cv::Mat &data_resize, float* inputImInfo, std::vector<float>& outputBboxPred, std::vector<float>& outputClsProb, std::vector<float>& outputRois) {
	TensorRTCaffeCantainer *param = (TensorRTCaffeCantainer *)handle;
	if (-1 == param->gpu_id) {
		return FACEVISA_PARAMETER_ERROR;
	}
	IExecutionContext& context = *(param->DET.context);
	const ICudaEngine& engine = *(param->DET.engine);

	std::vector<std::string> blob_names = { "data", "im_info", "bbox_pred", "cls_prob", "rois" };
	if (engine.getNbBindings() != blob_names.size()) {
		return FACEVISA_PARAMETER_ERROR;
	}
	int inputIndex0 = engine.getBindingIndex(blob_names[0].c_str()),
		inputIndex1 = engine.getBindingIndex(blob_names[1].c_str()),
		outputIndex0 = engine.getBindingIndex(blob_names[2].c_str()),
		outputIndex1 = engine.getBindingIndex(blob_names[3].c_str()),
		outputIndex2 = engine.getBindingIndex(blob_names[4].c_str());

	float *input_data_ = static_cast<float *>(param->DET.buffers[inputIndex0]);
	
	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
#if INPUT_DATA_TYPE_GPU
	//// GPU input
	float3 mean_;
	mean_.x = 102.9801f;
	mean_.y = 115.9465f;
	mean_.z = 122.7717f;
	cudaMemcpyAsync(param->DET.cv_input_gpu, data_resize.data, param->DET.buffers_dims[0].n() * param->DET.buffers_dims[0].c() * param->DET.buffers_dims[0].h() * param->DET.buffers_dims[0].w() * sizeof(unsigned char), cudaMemcpyHostToDevice, param->stream);
	getImage(param->DET.cv_input_gpu, input_data_, param->DET.buffers_dims[0].n(), param->DET.buffers_dims[0].c(), param->DET.buffers_dims[0].h(), param->DET.buffers_dims[0].w(), mean_, param->stream);
#else
	// CPU input
	cv::Scalar_<float> meanValue = cv::Scalar_<float>(102.9801f, 115.9465f, 122.7717f);
	float* imgData = (float *)malloc(param->DET.buffers_dims[0].n() * param->DET.buffers_dims[0].c() * param->DET.buffers_dims[0].h() * param->DET.buffers_dims[0].w() * sizeof(float));
	processImg(data_resize, 3, imgData, meanValue);
	if (0 != cudaMemcpyAsync(param->DET.buffers[inputIndex0], imgData, param->DET.buffers_dims[0].n() * param->DET.buffers_dims[0].c() * param->DET.buffers_dims[0].h() * param->DET.buffers_dims[0].w() * sizeof(float), cudaMemcpyHostToDevice, param->stream)) {
		return FACEVISA_ALLOC_MEMORY_ERROR;
	}
#endif //INPUT_DATA_TYPE_GPU

//2020.1.21 by yxx 增加出现异常时，也对内存进行释放
	if (0 != cudaMemcpyAsync(param->DET.buffers[inputIndex1], inputImInfo, param->DET.buffers_dims[1].n() * param->DET.buffers_dims[1].c() * param->DET.buffers_dims[1].h() * param->DET.buffers_dims[1].w() * sizeof(float), cudaMemcpyHostToDevice, param->stream)) {
#if !INPUT_DATA_TYPE_GPU
		// cpu input
		free(imgData);
#endif //INPUT_DATA_TYPE_GPU
		return FACEVISA_ALLOC_MEMORY_ERROR;
	}
	context.enqueue(param->DET.buffers_dims[0].n(), param->DET.buffers, param->stream, nullptr);
	if (0 != cudaMemcpyAsync(outputBboxPred.data(), param->DET.buffers[outputIndex0], param->DET.buffers_dims[2].n() * param->DET.buffers_dims[2].c() * param->DET.buffers_dims[2].h() * param->DET.buffers_dims[2].w() * sizeof(float), cudaMemcpyDeviceToHost, param->stream)) {
#if !INPUT_DATA_TYPE_GPU
		// cpu input
		free(imgData);
#endif //INPUT_DATA_TYPE_GPU
		return FACEVISA_ALLOC_MEMORY_ERROR;
	}
	if (0 != cudaMemcpyAsync(outputClsProb.data(), param->DET.buffers[outputIndex1], param->DET.buffers_dims[3].n() * param->DET.buffers_dims[3].c() * param->DET.buffers_dims[3].h() * param->DET.buffers_dims[3].w() * sizeof(float), cudaMemcpyDeviceToHost, param->stream)) {
#if !INPUT_DATA_TYPE_GPU
		// cpu input
		free(imgData);
#endif //INPUT_DATA_TYPE_GPU
		return FACEVISA_ALLOC_MEMORY_ERROR;
	}
	if (0 != cudaMemcpyAsync(outputRois.data(), param->DET.buffers[outputIndex2], param->DET.buffers_dims[4].n() * param->DET.buffers_dims[4].c() * param->DET.buffers_dims[4].h() * param->DET.buffers_dims[4].w() * sizeof(float), cudaMemcpyDeviceToHost, param->stream)) {
#if !INPUT_DATA_TYPE_GPU
		// cpu input
		free(imgData);
#endif //INPUT_DATA_TYPE_GPU
		return FACEVISA_ALLOC_MEMORY_ERROR;
	}
	cudaStreamSynchronize(param->stream);

#if !INPUT_DATA_TYPE_GPU
	// cpu input
	free(imgData);
#endif //INPUT_DATA_TYPE_GPU
	return FACEVISA_OK;
}

static int Facevisa_Engine_Inference_DET(Facevisa_TensorRT_handle handle, cv::Mat &image, Facevisa_TensorRT_PVA_result &results) {
	TensorRTCaffeCantainer *param = (TensorRTCaffeCantainer *)handle;
	
	// data
	int width = param->DET.buffers_dims[0].w();					// 网络输入图像 w 大小
	int height = param->DET.buffers_dims[0].h();				// 网络输入图像 h 大小
	int channels = param->DET.buffers_dims[0].c();				// 网络输入图像 c 大小
	int batch_size = param->DET.buffers_dims[0].n();			// batch size 大小
	int rpn_nms_out = param->DET.buffers_dims[2].n();			// RPN 层的 NMS 个数
	int output_cls_size = param->DET.buffers_dims[3].c();		// 分类类别数 
	int output_bbox_size = param->DET.buffers_dims[2].c();		// bbox 维度大小 = 类别数 * 4

	cv::Mat data_resize;
	cv::resize(image, data_resize, cv::Size(width, height), (0.0), (0.0), cv::INTER_LINEAR);

	// im_info 
	float imInfo[kINPUT_BATCH_SIZE_DET * 4]; // Input im_info
	for (int i = 0; i < kINPUT_BATCH_SIZE_DET; ++i)
	{
		imInfo[i * 3] = float(image.rows);						// Number of rows
		imInfo[i * 3 + 1] = float(image.cols);					// Number of columns
		imInfo[i * 3 + 2] = 1.0 * height / image.rows;          // Image scale h
		imInfo[i * 3 + 3] = 1.0 * width / image.cols;           // Image scale w
	}

	std::vector<float> rois;
	std::vector<float> bboxPreds;
	std::vector<float> clsProbs;
	std::vector<float> predBBoxes;

	// Host memory for outputs
	rois.assign(param->DET.buffers_dims[4].n() * param->DET.buffers_dims[4].c() * param->DET.buffers_dims[4].h() * param->DET.buffers_dims[4].w(), 0);
	bboxPreds.assign(param->DET.buffers_dims[2].n() * param->DET.buffers_dims[2].c() * param->DET.buffers_dims[2].h() * param->DET.buffers_dims[2].w(), 0);
	clsProbs.assign(param->DET.buffers_dims[3].n() * param->DET.buffers_dims[3].c() * param->DET.buffers_dims[3].h() * param->DET.buffers_dims[3].w(), 0);

	// Predicted bounding boxes
	predBBoxes.assign(param->DET.buffers_dims[2].n() * param->DET.buffers_dims[2].c() * param->DET.buffers_dims[2].h() * param->DET.buffers_dims[2].w(), 0);

	// forward
	double start = clock();
	int status = Facevisa_Engine_Forward_DET(handle, data_resize, imInfo, bboxPreds, clsProbs, rois);
	if (FACEVISA_OK != status) {
		return status;
	}
	double end = clock();
	//std::cout << " Forward time is: " << end - start << " ms!" << std::endl;

	// bbox转换到原图尺度上
	bboxTransformInvAndClip(rois, bboxPreds, predBBoxes, imInfo, batch_size, rpn_nms_out, output_cls_size);

	// The sample passes if there is at least one detection for each item in the batch
	for (int i = 0; i < batch_size; ++i)
	{
		float* bbox = predBBoxes.data() + i * rpn_nms_out * output_bbox_size;
		float* scores = clsProbs.data() + i * rpn_nms_out * output_cls_size;
		int numDetections = 0;
		for (int c = 1; c < output_cls_size; ++c) // Skip the background
		{
			std::vector<std::pair<float, int>> score_index;
			for (int r = 0; r < rpn_nms_out; ++r)
			{
				if (scores[r * output_cls_size + c] > SCORE_THRESHOLD)
				{
					score_index.push_back(std::make_pair(scores[r * output_cls_size + c], r));
					std::stable_sort(score_index.begin(), score_index.end(),
						[](const std::pair<float, int>& pair1,
							const std::pair<float, int>& pair2) {
						return pair1.first > pair2.first;
					});
				}
			}

			// Apply NMS algorithm
			std::vector<int> indices = nms(score_index, bbox, c, output_cls_size, NMS_THRESHOLD);

			numDetections += static_cast<int>(indices.size());

			// Show results
			for (unsigned k = 0; k < indices.size(); ++k)
			{
				RT_bbox single_res;
				int idx = indices[k];
				single_res.index = c;
				single_res.score = scores[idx * output_cls_size + c];
				float x = bbox[idx * output_bbox_size + c * 4];
				float y = bbox[idx * output_bbox_size + c * 4 + 1];
				float w = min(max(bbox[idx * output_bbox_size + c * 4 + 2] - x + 1.0f, 1.0f), imInfo[i + 1]);
				float h = min(max(bbox[idx * output_bbox_size + c * 4 + 3] - y + 1.0f, 1.0f), imInfo[i + 0]);
				single_res.BBox = cv::Rect(x, y, w, h);
				results.det_res.push_back(single_res);
			}
		}
	}
}


// CLS网络前向
// 2020.1.21 by yxx 使用model直接进行前向，防止多级联模型时，重复代码，提高代码复用
static int Facevisa_Engine_Forward_CLS(TensorRTCaffeBaby CLS_MODEL, std::vector<cv::Mat > &data_resize, float* detectionOut, cudaStream_t stream) {
	//TensorRTCaffeCantainer *param = (TensorRTCaffeCantainer *)handle;
	//if (-1 == param->gpu_id) {
	//	return FACEVISA_PARAMETER_ERROR;
	//}
	IExecutionContext& context = *(CLS_MODEL.context);
	const ICudaEngine& engine = *(CLS_MODEL.engine);

	std::vector<std::string> blob_names = { "data", "prob"};
	if (engine.getNbBindings() != blob_names.size()) {
		return FACEVISA_PARAMETER_ERROR;
	}
	int inputIndex0 = engine.getBindingIndex(blob_names[0].c_str()),
		outputIndex0 = engine.getBindingIndex(blob_names[1].c_str());

	float *input_data_ = static_cast<float *>(CLS_MODEL.buffers[inputIndex0]);

	
#if INPUT_DATA_TYPE_GPU
	// gpu input
	float3 mean_;
	mean_.x = 0.0f; // 102.9801f;
	mean_.y = 0.0f; // 115.9465f;
	mean_.z = 0.0f; // 122.7717f;
	getImage_batch(data_resize, CLS_MODEL.cv_input_gpu, input_data_, CLS_MODEL.buffers_dims[0].n(), CLS_MODEL.buffers_dims[0].c(), CLS_MODEL.buffers_dims[0].h(), CLS_MODEL.buffers_dims[0].w(), mean_, stream);
#else
	// CPU input
	cv::Scalar_<float> meanValue = cv::Scalar_<float>(0.0f, 0.0f, 0.0f);
	float* imgData = (float *)malloc(CLS_MODEL.buffers_dims[0].n() * CLS_MODEL.buffers_dims[0].c() * CLS_MODEL.buffers_dims[0].h() * CLS_MODEL.buffers_dims[0].w() * sizeof(float));
	processImg(data_resize, 3, imgData, meanValue);
	if (0 != cudaMemcpyAsync(CLS_MODEL.buffers[inputIndex0], imgData, CLS_MODEL.buffers_dims[0].n() * CLS_MODEL.buffers_dims[0].c() * CLS_MODEL.buffers_dims[0].h() * CLS_MODEL.buffers_dims[0].w() * sizeof(float), cudaMemcpyHostToDevice, stream)) {
		return FACEVISA_ALLOC_MEMORY_ERROR;
	}
#endif

	context.enqueue(CLS_MODEL.buffers_dims[0].n(), CLS_MODEL.buffers, stream, nullptr);
	if (0 != cudaMemcpyAsync(detectionOut, CLS_MODEL.buffers[outputIndex0], CLS_MODEL.buffers_dims[1].n() * CLS_MODEL.buffers_dims[1].c() * CLS_MODEL.buffers_dims[1].h() * CLS_MODEL.buffers_dims[1].w() * sizeof(float), cudaMemcpyDeviceToHost, stream)) {
#if !INPUT_DATA_TYPE_GPU
		//cpu input
		free(imgData);
#endif
		return FACEVISA_ALLOC_MEMORY_ERROR;
	}
	cudaStreamSynchronize(stream);
#if !INPUT_DATA_TYPE_GPU
	//cpu input
	free(imgData);
#endif
	return FACEVISA_OK;
}

static int Facevisa_Engine_Inference_CLS(Facevisa_TensorRT_handle handle, std::vector<cv::Mat> &images, Facevisa_TensorRT_PVA_result &results) {
	
	TensorRTCaffeCantainer *param = (TensorRTCaffeCantainer *)handle;
	if (-1 == param->gpu_id) {
		return FACEVISA_PARAMETER_ERROR;
	}

	int width = param->CLS.buffers_dims[0].w();						// w
	int height = param->CLS.buffers_dims[0].h();					// h
	int channels = param->CLS.buffers_dims[0].c();					// c
	int batch_size = param->CLS.buffers_dims[0].n();				// n
	int output_size = param->CLS.buffers_dims[1].c() * param->CLS.buffers_dims[1].h() * param->CLS.buffers_dims[1].h();

	// 输出
	float* detectionOut = (float *)malloc(batch_size * output_size * sizeof(float));

#if  IF_CASCADE_ALL_BOXES
	std::vector<cv::Mat> img_resize(images.size());
	for (int img_idx = 0; (img_idx < images.size())/* && (img_idx < batch_size)*/; img_idx++) {
		cv::resize(images[img_idx], img_resize[img_idx], cv::Size(width, height), (0.0), (0.0), cv::INTER_LINEAR);
	}


	int  batch_total = (img_resize.size() + batch_size - 1) / batch_size;
	for (int batch_index = 0; batch_index < batch_total; batch_index++)
	{
		//std::cout << "batch  :  " << batch_index << std::endl;
		int  start = batch_index  *  batch_size;
		int  end = std::min<int>(start + batch_size - 1, img_resize.size() - 1);
		std::vector<cv::Mat>  one_batch;
		for (int k = start; k <= end; k++)
		{
			one_batch.push_back(img_resize[k]);
		}
		int  status = Facevisa_Engine_Forward_CLS(param->CLS, one_batch, detectionOut, param->stream);
		if (FACEVISA_OK != status)
		{
			free(detectionOut);
			return  status;
		}

		for (int k = 0; k < end - start + 1; k++)
		{
			int  max_ind = 0;
			float  max_score = 0;
			for (int single_idx = 0; single_idx < output_size; single_idx++)
			{
				float  single_score = detectionOut[k  *  output_size + single_idx];
				if (single_score  >  max_score)
				{
					max_ind = single_idx;
					max_score = single_score;
				}
			}
			RT_bbox  cls_result;
			cls_result.BBox = results.det_res[k + start].BBox;
			cls_result.index = max_ind;
			cls_result.score = max_score;
			results.cls_res.push_back(cls_result);
		}
	}
#else
	// forward
	std::vector<cv::Mat> img_resize(images.size());
	for (int img_idx = 0; (img_idx < images.size()) && (img_idx < batch_size); img_idx++) {
		cv::resize(images[img_idx], img_resize[img_idx], cv::Size(width, height), (0.0), (0.0), cv::INTER_LINEAR);
	}


	double start = clock();
	int status = Facevisa_Engine_Forward_CLS(param->CLS, img_resize, detectionOut, param->stream);
	if (FACEVISA_OK != status) {
		free(detectionOut);
		return status;
	}
	double end = clock();
	//std::cout << " Forward time is: " << end - start << " ms!" << std::endl;


	for (int batch_idx = 0; batch_idx < images.size(); batch_idx++) {
		int max_ind = 0;
		float max_score = 0;
		for (int single_idx = 0; single_idx < output_size; single_idx++) {
			float single_score = detectionOut[batch_idx * output_size + single_idx];
			if (single_score > max_score) {
				max_ind = single_idx;
				max_score = single_score;
			}
		}
		RT_bbox cls_result;
		cls_result.BBox = results.det_res[batch_idx].BBox;
		cls_result.index = max_ind;
		cls_result.score = max_score;
		results.cls_res.push_back(cls_result);
	}
#endif
	free(detectionOut);

	return FACEVISA_OK;
}

// 总前向接口（PVA + Cascade）
// 注意输入图像大小固定为prototxt中 data层的dim
int Facevisa_Engine_Inference(Facevisa_TensorRT_handle handle, cv::Mat &image, Facevisa_TensorRT_PVA_result &results)
{
	if (NULL == handle) {
		return FACEVISA_PARAMETER_ERROR;
	}
	// DET
	Facevisa_Engine_Inference_DET(handle, image, results);

	// CLS
	if (0 == results.det_res.size())
	{
		return FACEVISA_OK;
	}

	std::vector<cv::Mat > cls_imgs;
	for (int det_index = 0; det_index < results.det_res.size(); det_index++) {
		// 2020.1.21 by yxx 增加clone  否则 读取内存会写乱
		cls_imgs.push_back(image(results.det_res[det_index].BBox).clone());
	}

	// cls interface
	Facevisa_Engine_Inference_CLS(handle, cls_imgs, results);
	return FACEVISA_OK;
}


// 内存释放
int Facevisa_Engine_Release(Facevisa_TensorRT_handle handle) {
	if (NULL == handle) {
		return FACEVISA_OK;
	}
	TensorRTCaffeCantainer *param = (TensorRTCaffeCantainer *)handle;
	// det
	nvinfer1::ICudaEngine *engine_det = param->DET.engine;
	IExecutionContext *context_det = param->DET.context;
	context_det->destroy();
	engine_det->destroy();
	
	for (size_t i = 0; i < param->DET.buffers_size; i++) {
		cudaFree(param->DET.buffers[i]);
	}
	cudaFree(param->DET.cv_input_gpu);

	// cls
	nvinfer1::ICudaEngine *engine_cls = param->CLS.engine;
	IExecutionContext *context_cls = param->CLS.context;
	context_cls->destroy();
	engine_cls->destroy();

	for (size_t i = 0; i < param->CLS.buffers_size; i++) {
		cudaFree(param->CLS.buffers[i]);
	}
	cudaFree(param->CLS.cv_input_gpu);

	cudaStreamDestroy(param->stream);
	free(param);

	handle = NULL;
	return FACEVISA_OK;
}

// 序列化model
int Facevisa_CaffeModelSerialize(std::string deployFile, std::string modelFile,
	const std::vector<std::string>& outputs, unsigned int maxBatchSize, std::string& engine_resialize_save)
{
	// create the builder
	initLibNvInferPlugins(&gLogger, "");
	IBuilder* builder = createInferBuilder(gLogger);

	IHostMemory *gieModelStream{ nullptr };

	// parse the caffe model to populate the network, then set the outputs
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser* parser = createCaffeParser();
	const IBlobNameToTensor* blobNameToTensor = parser->parse(deployFile.c_str(), modelFile.c_str(), *network, DataType::kFLOAT);

	// specify which tensors are outputs
	for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(400_MB);

	std::cout << "Begin building the engine..." << std::endl;
	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);
	std::cout << "End building the engine..." << std::endl;

	// serialize the engine, then close everything down
	gieModelStream = engine->serialize();

	std::ofstream ofs(engine_resialize_save, std::ios::out | std::ios::binary);
	ofs.write((char*)(gieModelStream->data()), gieModelStream->size());
	ofs.close();

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();
	engine->destroy();
	builder->destroy();
	shutdownProtobufLibrary();
	return 0;
}
