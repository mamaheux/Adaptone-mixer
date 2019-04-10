#include <SignalProcessing/Cuda/Processing/MixProcessing.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

template<class T>
__global__ void processMixKernel(T* inputFrame, T* outputFrame, T* gains, std::size_t frameSampleCount,
    std::size_t inputChannelCount, std::size_t outputChannelCount)
{
    processMix(inputFrame, outputFrame, gains, frameSampleCount, inputChannelCount, outputChannelCount);
}

TEST(MixProcessingTests, processMix_shouldMixTheInput)
{
    size_t frameSampleCount = 4;
    size_t inputChannelCount = 3;
    size_t outputChannelCount = 2;
    float* inputFrame;
    float* outputFrame;
    float* gains;

    cudaMallocManaged(reinterpret_cast<void**>(&inputFrame), frameSampleCount * inputChannelCount * sizeof(float));
    cudaMallocManaged(reinterpret_cast<void**>(&outputFrame), frameSampleCount * outputChannelCount * sizeof(float));
    cudaMallocManaged(reinterpret_cast<void**>(&gains), inputChannelCount * outputChannelCount * sizeof(float));

    inputFrame[0] = -128;
    inputFrame[1] = 1;
    inputFrame[2] = 127;
    inputFrame[3] = 16;

    inputFrame[4] = 64;
    inputFrame[5] = -64;
    inputFrame[6] = 32;
    inputFrame[7] = 32;

    inputFrame[8] = -32;
    inputFrame[9] = 127;
    inputFrame[10] = 64;
    inputFrame[11] = 64;

    gains[0] = 0.5;
    gains[1] = 2;
    gains[2] = -0.5;

    gains[3] = 0.25;
    gains[4] = 1.5;
    gains[5] = -1.5;

    processMixKernel<<<1, 256>>>(inputFrame, outputFrame, gains, frameSampleCount, inputChannelCount, outputChannelCount);
    cudaDeviceSynchronize();

    EXPECT_EQ(outputFrame[0], 80);
    EXPECT_EQ(outputFrame[1], -191);
    EXPECT_EQ(outputFrame[2], 95.5);
    EXPECT_EQ(outputFrame[3], 40);

    EXPECT_EQ(outputFrame[4], 112);
    EXPECT_EQ(outputFrame[5], -286.25);
    EXPECT_EQ(outputFrame[6], -16.25);
    EXPECT_EQ(outputFrame[7], -44);

    cudaFree(inputFrame);
    cudaFree(outputFrame);
    cudaFree(gains);
}