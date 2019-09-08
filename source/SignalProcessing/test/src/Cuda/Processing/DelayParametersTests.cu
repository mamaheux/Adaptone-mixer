#include <SignalProcessing/Cuda/Processing/GainProcessing.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

template<class T>
__global__ void processGainKernel(T* inputFrame,
    T* delayedOutputFrames,
    std::size_t* delays,
    std::size_t frameSampleCount,
    std::size_t channelCount,
    std::size_t currentDelayedOutputFrameIndex,
    std::size_t delayedOutputFrameCount)
{
    processDelay(inputFrame,
        delayedOutputFrames,
        delays,
        frameSampleCount,
        channelCount,
        currentDelayedOutputFrameIndex,
        delayedOutputFrameCount);
}

TEST(GainProcessingTests, processDelay_shouldDelayTheInput)
{
    constexpr size_t FrameSampleCount = 3;
    constexpr size_t ChannelCount = 4;
    constexpr size_t DelayedOutputFrameCount = 3;
    float* inputFrame;
    float* delayedOutputFrames;
    size_t* delays;

    cudaMallocManaged(reinterpret_cast<void**>(&inputFrame), FrameSampleCount * ChannelCount * sizeof(float));
    cudaMallocManaged(reinterpret_cast<void**>(&delaydOutputFrames),
        DelayedOutputFrameCount * FrameSampleCount * ChannelCount * sizeof(float));
    cudaMallocManaged(reinterpret_cast<void**>(&delays), ChannelCount * sizeof(size_t));

    inputFrame[0] = 0;
    inputFrame[1] = 1;
    inputFrame[2] = 2;

    inputFrame[3] = 10;
    inputFrame[4] = 11;
    inputFrame[5] = 12;

    inputFrame[6] = 20;
    inputFrame[7] = 21;
    inputFrame[8] = 22;

    inputFrame[9] = 30;
    inputFrame[10] = 31;
    inputFrame[11] = 32;

    delays[0] = 0;
    delays[1] = 1;
    delays[2] = 2;
    delays[3] = 3;

    cudaMemset(reinterpret_cast<void**>(&delaydOutputFrames), 0,
        DelayedOutputFrameCount * FrameSampleCount * ChannelCount * sizeof(float));

    size_t currentDelayedOutputFrameIndex = 0;
    processGainKernel<<<1, 256>>>(inputFrame,
        delayedOutputFrames,
        delays,
        FrameSampleCount,
        ChannelCount,
        currentDelayedOutputFrameIndex,
        DelayedOutputFrameCount);
    cudaDeviceSynchronize();

    EXPECT_EQ(delayedOutputFrames[0], 0);
    EXPECT_EQ(delayedOutputFrames[1], 1);
    EXPECT_EQ(delayedOutputFrames[2], 2);

    EXPECT_EQ(delayedOutputFrames[4], 10);
    EXPECT_EQ(delayedOutputFrames[5], 11);
    EXPECT_EQ(delayedOutputFrames[FrameSampleCount * ChannelCount + 3], 12);

    EXPECT_EQ(delayedOutputFrames[8], 20);
    EXPECT_EQ(delayedOutputFrames[FrameSampleCount * ChannelCount + 6], 21);
    EXPECT_EQ(delayedOutputFrames[FrameSampleCount * ChannelCount + 7], 22);

    EXPECT_EQ(delayedOutputFrames[FrameSampleCount * ChannelCount + 9], 30);
    EXPECT_EQ(delayedOutputFrames[FrameSampleCount * ChannelCount + 10], 31);
    EXPECT_EQ(delayedOutputFrames[FrameSampleCount * ChannelCount + 11], 32);

    currentDelayedOutputFrameIndex = 2;
    processGainKernel<<<1, 256>>>(inputFrame,
        delayedOutputFrames,
        delays,
        FrameSampleCount,
        ChannelCount,
        currentDelayedOutputFrameIndex,
        DelayedOutputFrameCount);
    cudaDeviceSynchronize();

    EXPECT_EQ(delayedOutputFrames[2 * FrameSampleCount * ChannelCount + 0], 0);
    EXPECT_EQ(delayedOutputFrames[2 * FrameSampleCount * ChannelCount + 1], 1);
    EXPECT_EQ(delayedOutputFrames[2 * FrameSampleCount * ChannelCount + 2], 2);

    EXPECT_EQ(delayedOutputFrames[2 * FrameSampleCount * ChannelCount + 4], 10);
    EXPECT_EQ(delayedOutputFrames[2 * FrameSampleCount * ChannelCount + 5], 11);
    EXPECT_EQ(delayedOutputFrames[3], 12);

    EXPECT_EQ(delayedOutputFrames[2 * FrameSampleCount * ChannelCount + 8], 20);
    EXPECT_EQ(delayedOutputFrames[6], 21);
    EXPECT_EQ(delayedOutputFrames[7], 22);

    EXPECT_EQ(delayedOutputFrames[9], 30);
    EXPECT_EQ(delayedOutputFrames[10], 31);
    EXPECT_EQ(delayedOutputFrames[11], 32);

    cudaFree(inputFrame);
    cudaFree(delayedOutputFrames);
    cudaFree(delays);
}
