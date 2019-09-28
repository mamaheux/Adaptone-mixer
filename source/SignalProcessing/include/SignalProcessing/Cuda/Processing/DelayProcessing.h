#ifndef SIGNAL_PROCESSING_CUDA_PROCESSING_DELAY_PROCESSING_H
#define SIGNAL_PROCESSING_CUDA_PROCESSING_DELAY_PROCESSING_H

#include <cstddef>
#include <cstdint>

namespace adaptone
{
    template<class T>
    __device__ void processDelay(T* inputFrame,
        T* delayedOutputFrames,
        std::size_t* delays,
        std::size_t frameSampleCount,
        std::size_t channelCount,
        std::size_t currentDelayedOutputFrameIndex,
        std::size_t delayedOutputFrameCount)
    {
        std::size_t startIndex = blockIdx.x * blockDim.x + threadIdx.x;
        std::size_t stride = blockDim.x * gridDim.x;
        std::size_t n = frameSampleCount * channelCount;

        for (std::size_t i = startIndex; i < n; i += stride)
        {
            std::size_t channelIndex = i / frameSampleCount;
            std::size_t inputSampleIndex = i % frameSampleCount;

            std::size_t delayedIndex = inputSampleIndex + delays[channelIndex];
            std::size_t outputFrameDelay = delayedIndex / frameSampleCount;
            std::size_t outputFrameIndex = (currentDelayedOutputFrameIndex + outputFrameDelay) % delayedOutputFrameCount;
            std::size_t outputSampleIndex = delayedIndex % frameSampleCount;

            std::size_t delayedOutputIndex = outputFrameIndex * n + channelIndex * frameSampleCount + outputSampleIndex;
            delayedOutputFrames[delayedOutputIndex] = inputFrame[i];
        }
    }
}

#endif
