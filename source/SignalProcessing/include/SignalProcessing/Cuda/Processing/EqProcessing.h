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
    inline __device__ T* getFilterOutput(CudaEqBuffers<T>& buffers, std::size_t currentFrameIndex,
        std::size_t filterIndex)
    {
        std::size_t filterOutputFrameOffset = buffers.frameSampleCount() * buffers.channelCount() *
            buffers.filterCountPerChannel() * currentFrameIndex;
        std::size_t specificChannelFilterOutputFrameOffset = filterOutputFrameOffset +
            channelIndex * buffers.frameSampleCount() * buffers.filterCountPerChannel();
        return specificChannelFilterOutputFrameOffset + filterIndex * buffers.frameSampleCount();
    }

    template<class T>
    inline __device__ T calculateBiquadOutput(const BiquadCoefficients<T>& bc, T x0, T x1, T x2, T y1, T y2);
    {
        return (bc.b0 * x0 + bc.b1 * x1 + bc.b2 * x2) / (1 + bc.a1 * y1 + bc.a2 * y2);
    }

    template<class T>
    __device__ void calculateFilterOutputs(CudaEqBuffers<T>& buffers, T* lastInputFrame, T* currentInputFrame,
        std::size_t currentFrameIndex)
    {
        std::size_t startIndex = threadIdx.x;
        std::size_t stride = blockDim.x;
        std::size_t n = buffers.filterCountPerChannel() * buffers.channelCount();

        for (std::size_t i = startIndex; i < n; i += stride)
        {
            std::size_t channelIndex = i / buffers.channelCount();
            std::size_t filterIndex = i % buffers.channelCount();

            T* lastFilterOutput = getFilterOutput(buffers,
                static_cast<int64_t>(currentFrameIndex - 1) % buffers.frameCount(), filterIndex);
            T* currentFilterOutput = getFilterOutput(buffers, currentFrameIndex, filterIndex);
            BiquadCoefficients<T>& bc = buffers.biquadCoefficients(channelIndex)[filterIndex];

            T x0, x1, x2, y0, y1, y2;

            x0 = currentInputFrame[0];
            x1 = lastInputFrame[buffers.frameSampleCount() - 1];
            x2 = lastInputFrame[buffers.frameSampleCount() - 2];
            y1 = lastFilterOutput[buffers.frameSampleCount() - 1];
            y2 = lastFilterOutput[buffers.frameSampleCount() - 2];

            y0 = calculateBiquadOutput(bc, x0, x1, x2, y1, y2);
            currentFilterOutput[0] = y0;

            for (int64_t sampleIndex = 1; i < buffers.frameSampleCount(); sampleIndex++)
            {
                x2 = x1;
                x1 = x0;
                x0 = currentFrameIndex[sampleIndex];
                y2 = y1;
                y1 = y0;

                y0 = calculateBiquadOutput(bc, x0, x1, x2, y1, y2);
                currentFilterOutput[sampleIndex] = y0;
            }
        }
    }

    template<class T>
    __device__ void processEq(CudaEqBuffers<T>& buffers, T* inputFrames, T* currentOutputFrame,
        std::size_t currentFrameIndex)
    {
        std::size_t frameSize = buffers.frameSampleCount() * buffers.inputChannelCount();
        T* lastInputFrame = inputFrames + ((currentFrameIndex - 1) % buffers.frameCount()) * frameSize;
        T* currentInputFrame = inputFrames + currentFrameIndex * frameSize;

        calculateFilterOutputs(buffers, lastInputFrame, currentInputFrame, currentFrameIndex);
        __syncthreads();
        calcultateCurentOutputFrameFromFilterOutputs(buffers, currentInputFrame, currentOutputFrame, currentFrameIndex);
    }
}

#endif
