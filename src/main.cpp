#include "SiftGPU.h"

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>
#include <SiftCameraParams.h>
extern "C" void updateConstantSiftCameraParams(const SiftCameraParams & params);

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

int main(int argc, char** argv)
{
    try {
        // C:/Users/Kivi Technologies/Documents/KIVI_IOS/Data/halil_upper/Inputs/20/bgr.png
        const std::string imgPath = "../data/640-1.jpg";

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

        cudaMalloc(&d_intensity, bytes);
        cudaMalloc(&d_depth, bytes);

        cudaMemcpy(d_intensity, gray32f.ptr<float>(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_depth, depth32f.ptr<float>(), bytes, cudaMemcpyHostToDevice);

        g_sift = new SiftGPU();
        g_sift->SetParams(W, H, false, 150, -10.0f, 110.0f);
        g_sift->InitSiftGPU();

        const unsigned int maxKp = 2000;

        SIFTImageGPU siftImage;
        SIFTKeyPoint* d_allKeypoints = nullptr;
        SIFTKeyPointDesc* d_allDescs = nullptr;
        cudaMalloc(&d_allKeypoints, maxKp * sizeof(SIFTKeyPoint));
        cudaMalloc(&d_allDescs, maxKp * sizeof(SIFTKeyPointDesc));


        siftImage.d_keyPoints = d_allKeypoints;
        siftImage.d_keyPointDescs = d_allDescs;

        SiftCameraParams siftCameraParams;
        siftCameraParams.m_depthWidth = W;
        siftCameraParams.m_depthHeight = H;
        siftCameraParams.m_intensityWidth = W;
        siftCameraParams.m_intensityHeight = H;
        siftCameraParams.m_minKeyScale = 0.000001f;
        updateConstantSiftCameraParams(siftCameraParams);

        
        int success = RunFeatureDetection(d_intensity, d_depth);
        if (!success)
        {
            throw std::exception("Error running SIFT detection");
        }

        unsigned int numKp = GetKeyPointsAndDescriptorsCUDA(siftImage, d_depth, maxKp);

        std::cout << "Detected keypoints: " << numKp << std::endl;

        cudaFree(d_intensity);
        cudaFree(d_depth);
        delete g_sift;
        g_sift = nullptr;

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
