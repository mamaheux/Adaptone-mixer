#ifndef SIGNAL_PROCESSING_CUDA_PROCESSING_GAIN_PROCESSING_H
#define SIGNAL_PROCESSING_CUDA_PROCESSING_GAIN_PROCESSING_H

#include <SignalProcessing/Cuda/CudaEqBuffers.h>

#include <cstddef>
#include <cstdint>

namespace adaptone
{
    template<class T>
    __device__ void processGain(T* inputFrames, T* outputFrames, T* gains, std::size_t frameSampleCount,
        std::size_t channelCount)
    {
        std::size_t startIndex = threadIdx.x;
        std::size_t stride = blockDim.x;
        std::size_t n = frameSampleCount * channelCount;

        for (std::size_t i = startIndex; i < n; i += stride)
        {
            outputFrames[i] = inputFrames[i] * gains[i % frameSampleCount];
        }
    }
}

#endif
