#include <SignalProcessing/Cuda/Processing/GainProcessing.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

template<class T>
__global__ void processGainKernel(T* inputFrames, T* outputFrames, T* gains, std::size_t frameSampleCount,
    std::size_t channelCount)
{
    processGain(inputFrames, outputFrames, gains, frameSampleCount, channelCount);
}

TEST(GainProcessingTests, processGain_shouldApplyGainToInput)
{
    size_t frameSampleCount = 3;
    size_t channelCount = 2;
    float* inputFrames;
    float* outputFrames;
    float* gains;

    cudaMallocManaged(reinterpret_cast<void**>(&inputFrames), frameSampleCount * channelCount * sizeof(float));
    cudaMallocManaged(reinterpret_cast<void**>(&outputFrames), frameSampleCount * channelCount * sizeof(float));
    cudaMallocManaged(reinterpret_cast<void**>(&gains), frameSampleCount * sizeof(float));

    inputFrames[0] = -128;
    inputFrames[1] = 1;
    inputFrames[2] = 127;

    inputFrames[3] = 64;
    inputFrames[4] = -64;
    inputFrames[5] = 32;

    gains[0] = 0.5;
    gains[1] = 1;
    gains[2] = 1.5;

    processGainKernel<<<1, 256>>>(inputFrames, outputFrames, gains, frameSampleCount, channelCount);
    cudaDeviceSynchronize();

    EXPECT_EQ(outputFrames[0], -64);
    EXPECT_EQ(outputFrames[1], 1);
    EXPECT_EQ(outputFrames[2], 190.5);
    EXPECT_EQ(outputFrames[3], 32);
    EXPECT_EQ(outputFrames[4], -64);
    EXPECT_EQ(outputFrames[5], 48);

    cudaFree(inputFrames);
    cudaFree(outputFrames);
    cudaFree(gains);
}