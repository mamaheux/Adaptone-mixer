#ifndef SIGNAL_PROCESSING_CUDA_PROCESSING_MIX_PROCESSING_H
#define SIGNAL_PROCESSING_CUDA_PROCESSING_MIX_PROCESSING_H

#include <SignalProcessing/Cuda/CudaEqBuffers.h>

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
        std::size_t outputN = frameSampleCount * outputChannelCount;
        std::size_t inputN = frameSampleCount * inputChannelCount;

        for (std::size_t outputIndex = startIndex; outputIndex < outputN; outputIndex += stride)
        {
            outputFrame[outputIndex] = 0;
            for (std::size_t inputIndex = outputIndex; inputIndex < inputN; inputIndex += frameSampleCount)
            {
                outputFrame[outputIndex] += inputFrame[inputIndex] *
                    gains[(outputIndex % outputChannelCount) * inputChannelCount + (inputIndex % frameSampleCount)];
            }
        }
    }
}

#endif
