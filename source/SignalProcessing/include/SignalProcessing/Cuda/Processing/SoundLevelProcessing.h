#ifndef SIGNAL_PROCESSING_CUDA_PROCESSING_SOUND_LEVEL_PROCESSING_H
#define SIGNAL_PROCESSING_CUDA_PROCESSING_SOUND_LEVEL_PROCESSING_H

#include <SignalProcessing/Cuda/CudaSoundLevelBuffers.h>

#include <Utils/Exception/NotSupportedException.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace adaptone
{
    template<class T>
    __device__ void calculateSoundLevel(CudaSoundLevelBuffers<T>& soundLevelBuffer, T* currentInputFrame)
    {
        std::size_t startIndex = threadIdx.x;
        std::size_t stride = blockDim.x;
        std::size_t n = soundLevelBuffer.channelCount();

        for (std::size_t i = startIndex; i < n; i+= stride) {
            std::size_t channelIndex = i;

            if (abs(currentInputFrame[channelIndex]) > soundLevelBuffer[channelIndex])
            {
                currentInputFrame[channelIndex] = soundLevelBuffer[channelIndex];
            }
        }
    }

    template<class T>
    __device__ void processSoundLevel(CudaSoundLevelBuffers<T>& soundLevelBuffer, T* inputFrame)
    {
        calculateSoundLevel(soundLevelBuffer, inputFrame);
    }
}

#endif
