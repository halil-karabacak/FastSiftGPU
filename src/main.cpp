#include "SiftGPU.h"

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>

static SiftGPU* g_sift = nullptr;

int RunFeatureDetection(float* d_intensitySift, const float* d_inputDepth)
{
    bool success = g_sift->RunSIFT(d_intensitySift, d_inputDepth);
    return success ? 1 : 0;
}

unsigned int GetKeyPointsAndDescriptorsCUDA(
    SIFTImageGPU& siftImage,
    const float* d_depthDataFullRes,
    unsigned int maxNumKeyPoints)
{
    unsigned int numKeypoints = g_sift->GetKeyPointsAndDescriptorsCUDA(siftImage, d_depthDataFullRes, maxNumKeyPoints);
    return numKeypoints;
}

#define CUDA_CHECK(ans) do { cudaAssert((ans), __FILE__, __LINE__); } while(0)
inline void cudaAssert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code)
            << " at " << file << ":" << line << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv)
{
    try {
        const std::string imgPath = "D:/dev/SiftGPU/data/640-1.jpg";

        cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);

        const int W = img.cols;
        const int H = img.rows;

        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        
        cv::Mat gray32f;
        gray.convertTo(gray32f, CV_32F);

        cv::Mat depth32f(H, W, CV_32F, cv::Scalar(100.0f));

        const size_t numPx = W * H;
        const size_t bytes = numPx * sizeof(float);

        float* d_intensity = nullptr;
        float* d_depth = nullptr;

        CUDA_CHECK(cudaMalloc(&d_intensity, bytes));
        CUDA_CHECK(cudaMalloc(&d_depth, bytes));

        CUDA_CHECK(cudaMemcpy(d_intensity, gray32f.ptr<float>(0), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_depth, depth32f.ptr<float>(0), bytes, cudaMemcpyHostToDevice));

        g_sift = new SiftGPU();
        g_sift->SetParams(W, H, false, 150, -10.0f, 110.0f);
        g_sift->InitSiftGPU();

        RunFeatureDetection(d_intensity, d_depth);

        SIFTImageGPU siftImage;
        const unsigned int maxKp = 2000;
        unsigned int numKp = GetKeyPointsAndDescriptorsCUDA(siftImage, d_depth, maxKp);

        std::cout << "Detected keypoints (no depth eliminated with [90,110]): " << numKp << std::endl;

        CUDA_CHECK(cudaFree(d_intensity));
        CUDA_CHECK(cudaFree(d_depth));
        delete g_sift;
        g_sift = nullptr;

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
