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
    __device__ void processSoundLevel(CudaSoundLevelBuffers<T>& soundLevelBuffer, T* currentInputFrame)
    {
        std::size_t startIndex = threadIdx.x;
        std::size_t stride = blockDim.x;
        std::size_t n = soundLevelBuffer.channelCount();

        for (std::size_t channelIndex = startIndex; channelIndex < n; channelIndex += stride)
        {
            for (std::size_t sampleIndex = 1; sampleIndex < soundLevelBuffer.frameSampleCount(); sampleIndex++)
            {
                T currentInputSampleValue = abs(currentInputFrame[channelIndex * soundLevelBuffer.frameSampleCount() + sampleIndex]);

                if (currentInputSampleValue > soundLevelBuffer.soundLevels()[channelIndex])
                {
                    soundLevelBuffer.soundLevels()[channelIndex] = currentInputSampleValue;
                }
            }
        }
    }
}

#endif
