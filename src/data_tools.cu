#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> 


__global__ void unchar2float_incuda(unsigned char *d_in, float *d_out, int c, int h, int w)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int idz = blockIdx.z * blockDim.z + threadIdx.z;
	if (idx < h && idy<w)
		d_out[idx * c * w + idy * c + idz] = float(d_in[idx * c * w + idy * c + idz]);
}


int getImage(unsigned char* d_in, float *input_data, int n, int c, int h, int w, cudaStream_t stream) {
	dim3 threadsPerBlock(16, 16, c);
	dim3 blocksPerGrid((h + threadsPerBlock.x - 1) / threadsPerBlock.x, (w + threadsPerBlock.y - 1) / threadsPerBlock.y, 1);
	unchar2float_incuda << <blocksPerGrid, threadsPerBlock, 0, stream >> >(d_in, input_data, c, h, w);

	return 0;
}


__global__ void SubtractMean_incuda(unsigned char *d_in, float *d_out, int c, int h, int w, float3 mean_)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;	// h
	int idy = blockIdx.y * blockDim.y + threadIdx.y;	// w
	int idz = blockIdx.z * blockDim.z + threadIdx.z;	// c
	if (idx < h && idy < w) {
		float mean;
		if (0 == idz) {
			mean = mean_.x;
		}
		else if (1 == idz) {
			mean = mean_.y;
		}
		else {
			mean = mean_.z;
		}
		d_out[idz * h * w + idx * w + idy] = float(d_in[idx * c * w + idy * c + idz]) - mean;
	}
}


int getImage(unsigned char* d_in, float *input_data, int n, int c, int h, int w, float3 mean_, cudaStream_t stream) {
	dim3 threadsPerBlock(4, 4, c);
	dim3 blocksPerGrid((h + threadsPerBlock.x - 1) / threadsPerBlock.x, (w + threadsPerBlock.y - 1) / threadsPerBlock.y, 1);
	SubtractMean_incuda << <blocksPerGrid, threadsPerBlock, 0, stream >> >(d_in, input_data, c, h, w, mean_);

	return 0;
}


__global__ void SubtractMean_incuda_batch(unsigned char *d_in, float *d_out, int n, int c, int h, int w, float3 mean_)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;	// x
	int idn = idx / h;									// n
	int idh = idx % h;									// h
	int idw = blockIdx.y * blockDim.y + threadIdx.y;	// w
	int idc = blockIdx.z * blockDim.z + threadIdx.z;	// c

														// 2019.08.06 yxx: Transboundary treatment
	if (idn >= n || idh >= h || idw >= w || idc >= c)
		return;

	if (idh < h && idw < w) {
		float mean;
		if (0 == idc) {
			mean = mean_.x;
		}
		else if (1 == idc) {
			mean = mean_.y;
		}
		else {
			mean = mean_.z;
		}
		d_out[idn * c * h * w + idc * h * w + idh * w + idw] = float(d_in[idn * c * w * h + idh * c * w + idw * c + idc]) - mean;
	}
}


int getImage_batch(std::vector<cv::Mat > &images_cpu, unsigned char* images_gpu, float *input_model_gpu, int n, int c, int h, int w, float3 mean_, cudaStream_t stream)
{
	for (int batch_index = 0; batch_index < images_cpu.size(); batch_index++)
	{
		cudaMemcpyAsync(images_gpu + batch_index*c*h*w, images_cpu[batch_index].data, c*h*w * sizeof(unsigned char), cudaMemcpyHostToDevice, stream);
	}
	//cudaMemcpyAsync(param->CLS.cv_input_gpu, data_resize.data, param->DET.buffers_dims[0].n() * param->DET.buffers_dims[0].c() * param->DET.buffers_dims[0].h() * param->DET.buffers_dims[0].w() * sizeof(unsigned char), cudaMemcpyHostToDevice, param->stream);
	dim3 threadsPerBlock(4, 4, c);
	dim3 blocksPerGrid((images_cpu.size()*h + threadsPerBlock.x - 1) / threadsPerBlock.x, (w + threadsPerBlock.y - 1) / threadsPerBlock.y, 1);
	SubtractMean_incuda_batch << <blocksPerGrid, threadsPerBlock, 0, stream >> >(images_gpu, input_model_gpu, images_cpu.size(), c, h, w, mean_);

	return 0;
}
