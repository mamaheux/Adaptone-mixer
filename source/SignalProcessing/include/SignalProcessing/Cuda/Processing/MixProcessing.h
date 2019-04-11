#ifndef SIGNAL_PROCESSING_CUDA_PROCESSING_MIX_PROCESSING_H
#define SIGNAL_PROCESSING_CUDA_PROCESSING_MIX_PROCESSING_H

#include <cstddef>
#include <cstdint>

namespace adaptone
{
    template<class T>
    __device__ void processMix(T* inputFrame, T* outputFrame, T* gains, std::size_t frameSampleCount,
        std::size_t inputChannelCount, std::size_t outputChannelCount)
    {
        std::size_t startIndex = threadIdx.x;
        std::size_t stride = blockDim.x;
        std::size_t n = frameSampleCount * outputChannelCount;

        for (std::size_t outputIndex = startIndex; outputIndex < n; outputIndex += stride)
        {
            std::size_t outputChannel = outputIndex / frameSampleCount;
            std::size_t sampleIndex = outputIndex % frameSampleCount;

            outputFrame[outputIndex] = 0;
            for (std::size_t inputChannel = 0; inputChannel < inputChannelCount; inputChannel++)
            {
                outputFrame[outputIndex] += inputFrame[inputChannel * frameSampleCount + sampleIndex] *
                    gains[outputChannel * inputChannelCount + inputChannel];
            }
        }
    }
}

#endif
