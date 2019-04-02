#ifndef SIGNAL_PROCESSING_CUDA_PROCESSING_EQ_PROCESSING_H
#define SIGNAL_PROCESSING_CUDA_PROCESSING_EQ_PROCESSING_H

#include <SignalProcessing/Cuda/CudaEqBuffers.h>

#include <Utils/Exception/NotSupportedException.h>

#include <cstddef>
#include <cstdint>

namespace adaptone
{
    template<class T>
    inline __device__ std::size_t getFirstFilterOutputIndex(CudaEqBuffers<T>& buffers, std::size_t currentFrameIndex,
        std::size_t sampleIndex)
    {
        std::size_t filterOutputFrameOffset = buffers.frameSampleCount() * buffers.channelCount() *
            buffers.filterCountPerChannel() * currentFrameIndex;
        std::size_t specificChannelFilterOutputFrameOffset = filterOutputFrameOffset +
            channelIndex * buffers.frameSampleCount() * buffers.filterCountPerChannel();
        return specificChannelFilterOutputFrameOffset + sampleIndex;
    }

    template<class T>
    __device__ void calcultateCurentOutputFrameFromFilterOutputs(CudaEqBuffers<T>& buffers,
        T* currentInputFrame, T* currentOutputFrame, std::size_t currentFrameIndex)
    {
        std::size_t startIndex = threadIdx.x;
        std::size_t stride = blockDim.x;
        std::size_t n = buffers.frameSampleCount() * buffers.channelCount();

        for (std::size_t i = startIndex; i < n; i += stride)
        {
            std::size_t channelIndex = i / buffers.channelCount();
            std::size_t sampleIndex = i % buffers.channelCount();

            T outputSample = buffers.d0()[channelIndex] *
                currentInputFrame[channelIndex * buffers.frameSampleCount() + sampleIndex];

            std::size_t filterOutputIndex = getFirstFilterOutputIndex(buffers, currentFrameIndex, sampleIndex);
            for (std::size_t j = 0; j < buffers.filterCountPerChannel(); j++)
            {
                outputSample += buffers.filterOutputs()[filterOutputIndex];
                filterOutputIndex += buffers.frameSampleCount();
            }

            currentOutputFrame[channelIndex * buffers.frameSampleCount() + sampleIndex] = outputSample;
        }
    }

    template<class T>
    __device__ void processEq(CudaEqBuffers& buffers, T* inputFrames, T* currentOutputFrame,
        std::size_t currentFrameIndex)
    {

        __syncthreads();
        T* currentInputFrame = inputFrames +
        calcultateCurentOutputFrameFromFilterOutputs(buffers, currentInputFrame, currentOutputFrame, currentFrameIndex);
    }
}

#endif
