#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "common.h"
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

static const int kINPUT_BATCH_SIZE = 1;			// Input image batch size
static const float NMS_THRESHOLD = 0.3f;		// outpot bbox nms threshold
static const float SCORE_THRESHOLD = 0.1f;		// outpot score threshold


static Logger gLogger;

// 工程相关变量容器
typedef struct TensorRTCaffeCantainer_ {
	int gpu_id;										// 配置 GPU 的 index	
	int buffers_size;								// 输入输出数据维度
	std::vector<DimsNCHW> buffers_dims;				// 2个维度的输入dims，3个维度的输出dims
	void** buffers;									// 输入输出数据，TensorRT 中前向函数需要
	nvinfer1::ICudaEngine* engine;
	IExecutionContext* context;
	cudaStream_t stream;
}TensorRTCaffeCantainer;


// 单batch 图像数据导入 opencv 方式
static int processImg(cv::Mat &img, int inputchannels, float *imgData) {
	int shift_data = 0;
	cv::Mat float_img;
	std::vector<cv::Mat> splitchannles = std::vector<cv::Mat>(inputchannels);
	if (3 == inputchannels) {
		if (1 == img.channels())
			cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
		img.convertTo(float_img, CV_32F);
		cv::Scalar_<float> meanValue = cv::Scalar_<float>(102.9801f, 115.9465f, 122.7717f);
		//cv::Scalar_<float> meanValue = cv::Scalar_<float>(0.0f, 0.0f, 0.0f);
		cv::Mat mean_(img.size(), CV_32FC3, meanValue);
		cv::subtract(float_img, mean_, float_img);
		cv::split(float_img, splitchannles);
	}
	else if (1 == inputchannels) {
		if (3 == img.channels())
			cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
		img.convertTo(float_img, CV_32F);
		cv::Scalar_<float> meanValue = cv::Scalar_<float>(102.9801f, 0.0f, 0.0f);
		//cv::Scalar_<float> meanValue = cv::Scalar_<float>(0.0f, 0.0f, 0.0f);
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

// 三通道数据读取（备注：1.比opencv方式快些1000*1000的图差不多10ms的差距；2.scale默认为1）
void loadImg_ori(cv::Mat &input, int re_width, int re_height, float *data_unifrom, const float3 mean, const float scale)
{
	int i;
	int j;
	int line_offset;
	int offset_g;
	int offset_r;
	cv::Mat dst;

	unsigned char *line = NULL;
	float *unifrom_data = data_unifrom;

	cv::resize(input, dst, cv::Size(re_width, re_height), (0.0), (0.0), cv::INTER_LINEAR);
	offset_g = re_width * re_height;
	offset_r = re_width * re_height * 2;
	for (i = 0; i < re_height; ++i)
	{
		line = dst.ptr< unsigned char >(i);
		line_offset = i * re_width;
		for (j = 0; j < re_width; ++j)
		{
			// b
			unifrom_data[line_offset + j] = ((float)(line[j * 3] - mean.x) /* * scale*/);
			// g
			unifrom_data[offset_g + line_offset + j] = ((float)(line[j * 3 + 1] - mean.y) /* * scale*/);
			// r
			unifrom_data[offset_r + line_offset + j] = ((float)(line[j * 3 + 2] - mean.z)/* * scale*/);
		}
	}
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


// 模型初始化
int Facevisa_Engine_Create(Facevisa_TensorRT_handle *handle, int device_id) {
	if (NULL == *handle || NULL == handle)
	{
		return FACEVISA_PARAMETER_ERROR;
	}


	char moduleFileName[MAX_PATH];
	GetModuleFileNameA(0, moduleFileName, MAX_PATH);
	char * ptr = strrchr(moduleFileName, '\\');
	ptr++;
	strcpy(ptr, "templates\\");
	std::string root_dir = std::string(moduleFileName);
	std::string protostr = root_dir + "test-lite_rt.prototxt";
	std::string modelstr = root_dir + "MS-06.19_iter_14000.caffemodel";

	std::cout << protostr << std::endl;
	std::cout << modelstr << std::endl;
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

	//// 不加会报错，不知道为什么
	IHostMemory* trtModelStream{ nullptr };
	initLibNvInferPlugins(&gLogger.getTRTLogger(), "");
	// Create the builder
	IBuilder* builder = createInferBuilder(gLogger);
	assert(builder != nullptr);

	
	// Parse the caffe model to populate the network, then set the outputs
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser* parser = createCaffeParser();
	// 读取caffe model
	const IBlobNameToTensor* blobNameToTensor = parser->parse(protostr.c_str(), modelstr.c_str(), *network, DataType::kFLOAT);

	// Specify which tensors are outputs
	// 输出从第三个维度开始
	std::vector<std::string> blob_names = { "data", "im_info", "bbox_pred", "cls_prob", "rois" };
	for (int name_idx = 2; name_idx < blob_names.size(); name_idx++) {
		network->markOutput(*blobNameToTensor->find(blob_names[name_idx].c_str()));
	}

	// Build the engine
	builder->setMaxBatchSize(kINPUT_BATCH_SIZE);
	builder->setMaxWorkspaceSize(1_GB);									// 设置最大使用显存
	builder->allowGPUFallback(true);

	ICudaEngine* engine = builder->buildCudaEngine(*network);
	// 设置输入输出的buffers， 
	param->buffers_size = engine->getNbBindings();
	param->buffers = (void **)malloc(sizeof(float) * engine->getNbBindings());
	param->buffers_dims = vector<DimsNCHW>(engine->getNbBindings());

	param->engine = engine;
	//assert(param->engine != nullptr);
	param->context = engine->createExecutionContext();

	// Create GPU buffers and a stream
	for (int name_idx = 0; name_idx < blob_names.size(); name_idx++) {
		int blobIndex = engine->getBindingIndex(blob_names[name_idx].c_str());
		Dims blobDims = engine->getBindingDimensions(blobIndex);
		int dim_out = 0;
		if (blobDims.nbDims == 3) {
			param->buffers_dims[name_idx].d[0] = engine->getMaxBatchSize();
			dim_out++;
		}
		for (int dim_in = 0; dim_in < blobDims.nbDims; dim_in++, dim_out++) {
			param->buffers_dims[name_idx].d[dim_out] = blobDims.d[dim_in];
		}
		if (0 != cudaMalloc(&(param->buffers[blobIndex]), param->buffers_dims[name_idx].n() * param->buffers_dims[name_idx].c() * param->buffers_dims[name_idx].h() * param->buffers_dims[name_idx].w() * sizeof(float))) {
			return FACEVISA_ALLOC_MEMORY_ERROR;
		}
	}

	// Create a stream
	if (0 != cudaStreamCreate(&param->stream)) {
		return FACEVISA_ALLOC_MEMORY_ERROR;
	}
	*handle = param;

	// release
	network->destroy();
	parser->destroy();
	builder->destroy();
	shutdownProtobufLibrary();
	return FACEVISA_OK;
}

// 网络前向
static int Facevisa_Engine_Forward(Facevisa_TensorRT_handle handle, float* inputData, float* inputImInfo, std::vector<float>& outputBboxPred, std::vector<float>& outputClsProb, std::vector<float>& outputRois) {
	TensorRTCaffeCantainer *param = (TensorRTCaffeCantainer *)handle;
	if (-1 == param->gpu_id) {
		return FACEVISA_PARAMETER_ERROR;
	}
	IExecutionContext& context = *(param->context);
	const ICudaEngine& engine = *(param->engine);

	std::vector<std::string> blob_names = { "data", "im_info", "bbox_pred", "cls_prob", "rois" };
	if (engine.getNbBindings() != blob_names.size()) {
		return FACEVISA_PARAMETER_ERROR;
	}
	int inputIndex0 = engine.getBindingIndex(blob_names[0].c_str()),
		inputIndex1 = engine.getBindingIndex(blob_names[1].c_str()),
		outputIndex0 = engine.getBindingIndex(blob_names[2].c_str()),
		outputIndex1 = engine.getBindingIndex(blob_names[3].c_str()),
		outputIndex2 = engine.getBindingIndex(blob_names[4].c_str());

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	if (0 != cudaMemcpyAsync(param->buffers[inputIndex0], inputData, param->buffers_dims[0].n() * param->buffers_dims[0].c() * param->buffers_dims[0].h() * param->buffers_dims[0].w() * sizeof(float), cudaMemcpyHostToDevice, param->stream)){
		return FACEVISA_ALLOC_MEMORY_ERROR;
	}
	if (0 != cudaMemcpyAsync(param->buffers[inputIndex1], inputImInfo, param->buffers_dims[1].n() * param->buffers_dims[1].c() * param->buffers_dims[1].h() * param->buffers_dims[1].w() * sizeof(float), cudaMemcpyHostToDevice, param->stream)) {
		return FACEVISA_ALLOC_MEMORY_ERROR;
	}
	context.enqueue(param->buffers_dims[0].n(), param->buffers, param->stream, nullptr);
	if (0 != cudaMemcpyAsync(outputBboxPred.data(), param->buffers[outputIndex0], param->buffers_dims[2].n() * param->buffers_dims[2].c() * param->buffers_dims[2].h() * param->buffers_dims[2].w() * sizeof(float), cudaMemcpyDeviceToHost, param->stream)) {
		return FACEVISA_ALLOC_MEMORY_ERROR;
	}
	if (0 != cudaMemcpyAsync(outputClsProb.data(), param->buffers[outputIndex1], param->buffers_dims[3].n() * param->buffers_dims[3].c() * param->buffers_dims[3].h() * param->buffers_dims[3].w() * sizeof(float), cudaMemcpyDeviceToHost, param->stream)) {
		return FACEVISA_ALLOC_MEMORY_ERROR;
	}
	if (0 != cudaMemcpyAsync(outputRois.data(), param->buffers[outputIndex2], param->buffers_dims[4].n() * param->buffers_dims[4].c() * param->buffers_dims[4].h() * param->buffers_dims[4].w() * sizeof(float), cudaMemcpyDeviceToHost, param->stream)) {
		return FACEVISA_ALLOC_MEMORY_ERROR;
	}
	cudaStreamSynchronize(param->stream);

	return FACEVISA_OK;
}


// 单batch检测主接口
// 注意输入图像大小固定为prototxt中 data层的dim
int Facevisa_Engine_Inference(Facevisa_TensorRT_handle handle, cv::Mat &image, Facevisa_TensorRT_result_s &results)
{
	TensorRTCaffeCantainer *param = (TensorRTCaffeCantainer *)handle;
	// data
	int width = param->buffers_dims[0].w();					// 网络输入图像 w 大小
	int height = param->buffers_dims[0].h();				// 网络输入图像 h 大小
	int channels = param->buffers_dims[0].c();				// 网络输入图像 c 大小
	int batch_size = param->buffers_dims[0].n();			// batch size 大小
	int rpn_nms_out = param->buffers_dims[2].n();			// RPN 层的 NMS 个数
	int output_cls_size = param->buffers_dims[3].c();		// 分类类别数 
	int output_bbox_size = param->buffers_dims[2].c();		// bbox 维度大小 = 类别数 * 4

	double start_in = clock();
	int shift_data = batch_size * channels * height * width * sizeof(float);
	float *input_data = (float *)malloc(shift_data);
	//processImg(img_resize, channels, input_data);
	float3 mean_;
	mean_.x = 102.9801f;
	mean_.y = 115.9465f;
	mean_.z = 122.7717f;
	loadImg_ori(image, width, height, input_data, mean_, 1);
	double end_in = clock();
	//std::cout << " img_resize time is: " << end_in - start_in << " ms!" << std::endl;

	// im_info 
	float imInfo[kINPUT_BATCH_SIZE * 4]; // Input im_info
	for (int i = 0; i < kINPUT_BATCH_SIZE; ++i)
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
	rois.assign(param->buffers_dims[4].n() * param->buffers_dims[4].c() * param->buffers_dims[4].h() * param->buffers_dims[4].w(), 0);
	bboxPreds.assign(param->buffers_dims[2].n() * param->buffers_dims[2].c() * param->buffers_dims[2].h() * param->buffers_dims[2].w(), 0);
	clsProbs.assign(param->buffers_dims[3].n() * param->buffers_dims[3].c() * param->buffers_dims[3].h() * param->buffers_dims[3].w(), 0);

	// Predicted bounding boxes
	predBBoxes.assign(param->buffers_dims[2].n() * param->buffers_dims[2].c() * param->buffers_dims[2].h() * param->buffers_dims[2].w(), 0);

	// forward
	double start = clock();
	int status = Facevisa_Engine_Forward(handle, input_data, imInfo, bboxPreds, clsProbs, rois);
	if (FACEVISA_OK != status) {
		return status;
	}
	double end = clock();
	//std::cout << " Forward time is: " << end - start << " ms!" << std::endl;

	// bbox转换到原图尺度上
	bboxTransformInvAndClip(rois, bboxPreds, predBBoxes, imInfo, batch_size, rpn_nms_out, output_cls_size);

	// The sample passes if there is at least one detection for each item in the batch
	bool pass = true;
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
				single_res.cls = c;
				single_res.score = scores[idx * output_cls_size + c];
				float x = bbox[idx * output_bbox_size + c * 4];
				float y = bbox[idx * output_bbox_size + c * 4 + 1];
				float w = min(max(bbox[idx * output_bbox_size + c * 4 + 2] - x + 1.0f, 1.0f), imInfo[i + 1]);
				float h = min(max(bbox[idx * output_bbox_size + c * 4 + 3] - y + 1.0f, 1.0f), imInfo[i + 0]);
				single_res.BBox = cv::Rect(x,y,w,h);
				results.det_res.push_back(single_res);
			}
		}
		pass &= numDetections >= 1;
	}
	
	free(input_data);
	
	return FACEVISA_OK;
}

// 内存释放
int Facevisa_Engine_Release(Facevisa_TensorRT_handle handle) {
	TensorRTCaffeCantainer *param = (TensorRTCaffeCantainer *)handle;
	nvinfer1::ICudaEngine *engine = param->engine;
	IExecutionContext *context = param->context;
	context->destroy();
	engine->destroy();
	cudaStreamDestroy(param->stream);
	for (size_t i = 0; i < param->buffers_size; i++) {
		cudaFree(param->buffers[i]);
	}
	free(param);

	handle = NULL;
	return FACEVISA_OK;
}

