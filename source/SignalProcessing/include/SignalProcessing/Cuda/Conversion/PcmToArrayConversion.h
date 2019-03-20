#ifndef SIGNAL_PROCESSING_CUDA_CONVERSION_PCM_TO_ARRAY_CONVERSION_H
#define SIGNAL_PROCESSING_CUDA_CONVERSION_PCM_TO_ARRAY_CONVERSION_H

#include <Utils/Data/PcmAudioFrame.h>
#include <Utils/Exception/NotSupportedException.h>

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

            T sample = -static_cast<T>(input[i]) / std::numeric_limits<PcmT>::min();
            output[channelIndex * channelCount + sampleIndex] = sample;
        }
    };

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
            output[channelIndex * channelCount + sampleIndex] = sample;
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
            output[channelIndex * channelCount + sampleIndex] = sample;
        }
    };

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
            output[channelIndex * channelCount + sampleIndex] = sample;
        }
    };

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
            output[channelIndex * channelCount + sampleIndex] = sample;
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
            output[channelIndex * channelCount + sampleIndex] = sample;
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

            output[channelIndex * channelCount + sampleIndex] = static_cast<T>(input[i]);
        }
    };

    template<class T>
    using PcmToArrayConversionFunctionPointer = void (*)(const uint8_t*, T*, std::size_t, std::size_t);

    template<class T, class PcmT>
    __global__ void getSignedPcmToArrayAddress(PcmToArrayConversionFunctionPointer<T>* pointer)
    {
        *pointer = signedPcmToArray<T, PcmT>;
    }

    template<class T>
    __global__ void  getSigned24PcmToArrayAddress(PcmToArrayConversionFunctionPointer<T>* pointer)
    {
        *pointer = signed24PcmToArray<T>;
    }

    template<class T>
    __global__ void getSignedPadded24PcmToArrayAddress(PcmToArrayConversionFunctionPointer<T>* pointer)
    {
        *pointer = signedPadded24PcmToArray<T>;
    }

    template<class T, class PcmT>
    __global__ void getUnsignedPcmToArrayAddress(PcmToArrayConversionFunctionPointer<T>* pointer)
    {
        *pointer = unsignedPcmToArray<T, PcmT>;
    }

    template<class T>
    __global__ void getUnsigned24PcmToArrayAddress(PcmToArrayConversionFunctionPointer<T>* pointer)
    {
        *pointer = unsigned24PcmToArray<T>;
    }

    template<class T>
    __global__ void getUnsignedPadded24PcmToArrayAddress(PcmToArrayConversionFunctionPointer<T>* pointer)
    {
        *pointer = unsignedPadded24PcmToArray<T>;
    }

    template<class T, class PcmT>
    __global__ void getFloatingPointPcmToArrayAddress(PcmToArrayConversionFunctionPointer<T>* pointer)
    {
        *pointer = floatingPointPcmToArray<T, PcmT>;
    }

    template<class T>
    PcmToArrayConversionFunctionPointer<T> getPcmToArrayConversionFunctionPointer(PcmAudioFrame::Format format)
    {
        PcmToArrayConversionFunctionPointer<T> gpuFunctionPointer;
        PcmToArrayConversionFunctionPointer<T>* d_gpuFunctionPointer;
        cudaMalloc(&d_gpuFunctionPointer, sizeof(PcmToArrayConversionFunctionPointer<T>*));

        bool supported = true;
        switch (format)
        {
            case PcmAudioFrame::Format::Signed8:
                getSignedPcmToArrayAddress<T, int8_t><<<1, 1>>>(d_gpuFunctionPointer);
                break;
            case PcmAudioFrame::Format::Signed16:
                getSignedPcmToArrayAddress<T, int16_t><<<1, 1>>>(d_gpuFunctionPointer);
                break;
            case PcmAudioFrame::Format::Signed24:
                getSigned24PcmToArrayAddress<T><<< 1, 1 >>>(d_gpuFunctionPointer);
                break;
            case PcmAudioFrame::Format::SignedPadded24:
                getSignedPadded24PcmToArrayAddress<T><<<1, 1>>>(d_gpuFunctionPointer);
                break;
            case PcmAudioFrame::Format::Signed32:
                getSignedPcmToArrayAddress<T, int32_t><<<1, 1>>>(d_gpuFunctionPointer);
                break;

            case PcmAudioFrame::Format::Unsigned8:
                getUnsignedPcmToArrayAddress<T, uint8_t> <<<1, 1>>>(d_gpuFunctionPointer);
                break;
            case PcmAudioFrame::Format::Unsigned16:
                getUnsignedPcmToArrayAddress<T, uint16_t><<<1, 1>>>(d_gpuFunctionPointer);
                break;
            case PcmAudioFrame::Format::Unsigned24:
                getUnsigned24PcmToArrayAddress<T><<<1, 1>>>(d_gpuFunctionPointer);
                break;
            case PcmAudioFrame::Format::UnsignedPadded24:
                getUnsignedPadded24PcmToArrayAddress<T> <<<1, 1>>>(d_gpuFunctionPointer);
                break;
            case PcmAudioFrame::Format::Unsigned32:
                getUnsignedPcmToArrayAddress<T, uint32_t><<<1, 1>>>(d_gpuFunctionPointer);
                break;

            case PcmAudioFrame::Format::Float:
                getFloatingPointPcmToArrayAddress<T, float><<<1, 1>>>(d_gpuFunctionPointer);
                break;
            case PcmAudioFrame::Format::Double:
                getFloatingPointPcmToArrayAddress<T, double><<<1, 1>>>(d_gpuFunctionPointer);
                break;

            default:
                supported = false;
                break;
        }

        cudaMemcpy(&gpuFunctionPointer, d_gpuFunctionPointer, sizeof(PcmToArrayConversionFunctionPointer<T>*),
            cudaMemcpyDeviceToHost);
        cudaFree(d_gpuFunctionPointer);

        if (supported)
        {
            return gpuFunctionPointer;
        }
        else
        {
            THROW_NOT_SUPPORTED_EXCEPTION("Not supported pcm format.");
        }
    }
}

#endif
