
#include "SiftCameraParams.h"

__constant__ SiftCameraParams c_siftCameraParams;

extern "C" void updateConstantSiftCameraParams(const SiftCameraParams& params) {
	
	size_t size;
	cudaGetSymbolSize(&size, c_siftCameraParams);
	cudaMemcpyToSymbol(c_siftCameraParams, &params, size, 0, cudaMemcpyHostToDevice);
	
#ifdef DEBUG
	cudaDeviceSynchronize();
#endif

}