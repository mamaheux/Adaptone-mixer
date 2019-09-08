#include <SignalProcessing/Cuda/Processing/GainProcessing.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

template<class T>
__global__ void processGainKernel(T* inputFrame, T* outputFrame, T* gains, std::size_t frameSampleCount,
    std::size_t channelCount)
{
    processGain(inputFrame, outputFrame, gains, frameSampleCount, channelCount);
}

TEST(GainProcessingTests, processGain_shouldApplyGainToInput)
{
    constexpr size_t FrameSampleCount = 3;
    constexpr  size_t ChannelCount = 2;
    float* inputFrame;
    float* outputFrame;
    float* gains;

    cudaMallocManaged(reinterpret_cast<void**>(&inputFrame), FrameSampleCount * ChannelCount * sizeof(float));
    cudaMallocManaged(reinterpret_cast<void**>(&outputFrame), FrameSampleCount * ChannelCount * sizeof(float));
    cudaMallocManaged(reinterpret_cast<void**>(&gains), ChannelCount * sizeof(float));

    inputFrame[0] = -128;
    inputFrame[1] = 1;
    inputFrame[2] = 127;

    inputFrame[3] = 64;
    inputFrame[4] = -64;
    inputFrame[5] = 32;

    gains[0] = 0.5;
    gains[1] = 2;

    processGainKernel<<<1, 256>>>(inputFrame, outputFrame, gains, FrameSampleCount, ChannelCount);
    cudaDeviceSynchronize();

    EXPECT_EQ(outputFrame[0], -64);
    EXPECT_EQ(outputFrame[1], 0.5);
    EXPECT_EQ(outputFrame[2], 63.5);

    EXPECT_EQ(outputFrame[3], 128);
    EXPECT_EQ(outputFrame[4], -128);
    EXPECT_EQ(outputFrame[5], 64);

    cudaFree(inputFrame);
    cudaFree(outputFrame);
    cudaFree(gains);
}
