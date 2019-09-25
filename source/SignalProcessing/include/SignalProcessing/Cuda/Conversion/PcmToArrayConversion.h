#ifndef SIGNAL_PROCESSING_CUDA_CONVERSION_PCM_TO_ARRAY_CONVERSION_H
#define SIGNAL_PROCESSING_CUDA_CONVERSION_PCM_TO_ARRAY_CONVERSION_H

#include <Utils/Data/PcmAudioFrameFormat.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace adaptone
{
    template<class T, class PcmT>
    __device__ void signedPcmToArray(const uint8_t* inputBytes, T* output, std::size_t frameSampleCount,
        std::size_t channelCount)
    {
        std::size_t startIndex = threadIdx.x;
        std::size_t stride = blockDim.x;
        std::size_t n = frameSampleCount * channelCount;

        const PcmT* input = reinterpret_cast<const PcmT*>(inputBytes);

        for (std::size_t i = startIndex; i < n; i += stride)
        {
            std::size_t channelIndex = i % channelCount;
            std::size_t sampleIndex = i / channelCount;

            T sample = static_cast<T>(input[i]) / std::numeric_limits<PcmT>::max();
            output[channelIndex * frameSampleCount + sampleIndex] = sample;
        }
    }

    template<class T>
    __device__ void signed24PcmToArray(const uint8_t* inputBytes, T* output, std::size_t frameSampleCount,
        std::size_t channelCount)
    {
        std::size_t startIndex = threadIdx.x;
        std::size_t stride = blockDim.x;
        std::size_t n = frameSampleCount * channelCount;

        constexpr T AbsMin = 1 << 23;

        for (std::size_t i = startIndex; i < n; i += stride)
        {
            std::size_t channelIndex = i % channelCount;
            std::size_t sampleIndex = i / channelCount;

            uint32_t b0 = inputBytes[3 * i];
            uint32_t b1 = inputBytes[3 * i + 1];
            uint32_t b2 = static_cast<int8_t>(inputBytes[3 * i + 2]);
            int32_t sampleInteger = b0 | (b1 << 8) | (b2 << 16);

            T sample = static_cast<T>(sampleInteger) / AbsMin;
            output[channelIndex * frameSampleCount + sampleIndex] = sample;
        }
    }

    template<class T>
    __device__ void signedPadded24PcmToArray(const uint8_t* inputBytes, T* output, std::size_t frameSampleCount,
        std::size_t channelCount)
    {
        constexpr T AbsMin = 1 << 23;

        std::size_t startIndex = threadIdx.x;
        std::size_t stride = blockDim.x;
        std::size_t n = frameSampleCount * channelCount;

        const int32_t* input = reinterpret_cast<const int32_t*>(inputBytes);

        for (std::size_t i = startIndex; i < n; i += stride)
        {
            std::size_t channelIndex = i % channelCount;
            std::size_t sampleIndex = i / channelCount;

            T sample = static_cast<T>(input[i]) / AbsMin;
            output[channelIndex * frameSampleCount + sampleIndex] = sample;
        }
    }

    template<class T, class PcmT>
    __device__ void unsignedPcmToArray(const uint8_t* inputBytes, T* output, std::size_t frameSampleCount,
        std::size_t channelCount)
    {
        std::size_t startIndex = threadIdx.x;
        std::size_t stride = blockDim.x;
        std::size_t n = frameSampleCount * channelCount;

        const PcmT* input = reinterpret_cast<const PcmT*>(inputBytes);

        for (std::size_t i = startIndex; i < n; i += stride)
        {
            std::size_t channelIndex = i % channelCount;
            std::size_t sampleIndex = i / channelCount;

            T sample = 2 * static_cast<T>(input[i]) / std::numeric_limits<PcmT>::max() - 1;
            output[channelIndex * frameSampleCount + sampleIndex] = sample;
        }
    }

    template<class T>
    __device__ void unsigned24PcmToArray(const uint8_t* inputBytes, T* output, std::size_t frameSampleCount,
        std::size_t channelCount)
    {
        constexpr T Max = (1 << 24) - 1;

        std::size_t startIndex = threadIdx.x;
        std::size_t stride = blockDim.x;
        std::size_t n = frameSampleCount * channelCount;

        for (std::size_t i = startIndex; i < n; i += stride)
        {
            std::size_t channelIndex = i % channelCount;
            std::size_t sampleIndex = i / channelCount;

            uint32_t b0 = inputBytes[3 * i];
            uint32_t b1 = inputBytes[3 * i + 1];
            uint32_t b2 = inputBytes[3 * i + 2];
            uint32_t sampleInteger = b0 | (b1 << 8) | (b2 << 16);

            T sample = 2 * static_cast<T>(sampleInteger) / Max - 1;
            output[channelIndex * frameSampleCount + sampleIndex] = sample;
        }
    }

    template<class T>
    __device__ void unsignedPadded24PcmToArray(const uint8_t* inputBytes, T* output, std::size_t frameSampleCount,
        std::size_t channelCount)
    {
        constexpr T Max = (1 << 24) - 1;

        std::size_t startIndex = threadIdx.x;
        std::size_t stride = blockDim.x;
        std::size_t n = frameSampleCount * channelCount;

        const uint32_t* input = reinterpret_cast<const uint32_t*>(inputBytes);

        for (std::size_t i = startIndex; i < n; i += stride)
        {
            std::size_t channelIndex = i % channelCount;
            std::size_t sampleIndex = i / channelCount;

            T sample = 2 * static_cast<T>(input[i]) / Max - 1;
            output[channelIndex * frameSampleCount + sampleIndex] = sample;
        }
    }

    template<class T, class PcmT>
    __device__ void floatingPointPcmToArray(const uint8_t* inputBytes, T* output, std::size_t frameSampleCount,
        std::size_t channelCount)
    {
        std::size_t startIndex = threadIdx.x;
        std::size_t stride = blockDim.x;
        std::size_t n = frameSampleCount * channelCount;

        const PcmT* input = reinterpret_cast<const PcmT*>(inputBytes);

        for (std::size_t i = startIndex; i < n; i += stride)
        {
            std::size_t channelIndex = i % channelCount;
            std::size_t sampleIndex = i / channelCount;

            output[channelIndex * frameSampleCount + sampleIndex] = static_cast<T>(input[i]);
        }
    }

    template<class T>
    __device__ void convertPcmToArray(const uint8_t* inputBytes, T* output, std::size_t frameSampleCount,
        std::size_t channelCount, PcmAudioFrameFormat format)
    {
        switch (format)
        {
            case PcmAudioFrameFormat::Signed8:
                signedPcmToArray<T, int8_t>(inputBytes, output, frameSampleCount, channelCount);
                break;
            case PcmAudioFrameFormat::Signed16:
                signedPcmToArray<T, int16_t>(inputBytes, output, frameSampleCount, channelCount);
                break;
            case PcmAudioFrameFormat::Signed24:
                signed24PcmToArray<T>(inputBytes, output, frameSampleCount, channelCount);
                break;
            case PcmAudioFrameFormat::SignedPadded24:
                signedPadded24PcmToArray<T>(inputBytes, output, frameSampleCount, channelCount);
                break;
            case PcmAudioFrameFormat::Signed32:
                signedPcmToArray<T, int32_t>(inputBytes, output, frameSampleCount, channelCount);
                break;

            case PcmAudioFrameFormat::Unsigned8:
                unsignedPcmToArray<T, uint8_t>(inputBytes, output, frameSampleCount, channelCount);
                break;
            case PcmAudioFrameFormat::Unsigned16:
                unsignedPcmToArray<T, uint16_t>(inputBytes, output, frameSampleCount, channelCount);
                break;
            case PcmAudioFrameFormat::Unsigned24:
                unsigned24PcmToArray<T>(inputBytes, output, frameSampleCount, channelCount);
                break;
            case PcmAudioFrameFormat::UnsignedPadded24:
                unsignedPadded24PcmToArray<T>(inputBytes, output, frameSampleCount, channelCount);
                break;
            case PcmAudioFrameFormat::Unsigned32:
                unsignedPcmToArray<T, uint32_t>(inputBytes, output, frameSampleCount, channelCount);
                break;

            case PcmAudioFrameFormat::Float:
                floatingPointPcmToArray<T, float>(inputBytes, output, frameSampleCount, channelCount);
                break;
            case PcmAudioFrameFormat::Double:
                floatingPointPcmToArray<T, double>(inputBytes, output, frameSampleCount, channelCount);
                break;
        }
    }
}

#endif
