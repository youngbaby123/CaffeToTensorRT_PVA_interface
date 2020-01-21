#include "common.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> 

int getImage(unsigned char* image, float *input_data, int n, int c, int h, int w, cudaStream_t stream);
int getImage(unsigned char* d_in, float *input_data, int n, int c, int h, int w, float3 mean_, cudaStream_t stream);

int getImage_batch(std::vector<cv::Mat > &images_cpu, unsigned char* images_gpu, float *input_model_gpu, int n, int c, int h, int w, float3 mean_, cudaStream_t stream);

