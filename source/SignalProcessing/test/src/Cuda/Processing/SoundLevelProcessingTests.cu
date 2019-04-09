#include <SignalProcessing/Cuda/Processing/SoundLevelProcessing.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

template<class T>
__global__ void processSoundLevelKernel(CudaSoundLevelBuffers<T> buffers, T* currentFrame)
{
    processSoundLevel(buffers, currentFrame);
}

TEST(SoundLevelProcessingTests, processSoundLevel_shouldUpdateSoundLevelsAppropriately)
{
    size_t frameSampleCount = 3;
    size_t channelCount = 2;
    float* currentFrame;
    CudaSoundLevelBuffers<float> soundLevelsBuffers(channelCount, frameSampleCount);
    vector<float> soundLevels(channelCount);

    cudaMallocManaged(reinterpret_cast<void**>(&currentFrame), frameSampleCount * channelCount * sizeof(float));

    currentFrame[0] = -128;
    currentFrame[1] = 0;
    currentFrame[2] = 127;

    currentFrame[3] = 64;
    currentFrame[4] = -64;
    currentFrame[5] = 32;

    processSoundLevelKernel<<<1, 256>>>(soundLevelsBuffers, currentFrame);
    soundLevelsBuffers.toVector(soundLevels);

    EXPECT_EQ(soundLevels[0], 128);
    EXPECT_EQ(soundLevels[1], 64);

    cudaFree(currentFrame);
}