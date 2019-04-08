#ifndef SIGNAL_PROCESSING_CUDA_CONVERSION_ARRAY_TO_PCM_CONVERSION_H
#define SIGNAL_PROCESSING_CUDA_CONVERSION_ARRAY_TO_PCM_CONVERSION_H

#include <Utils/Data/PcmAudioFrame.h>
#include <Utils/Exception/NotSupportedException.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace adaptone
{
    template<class T, class PcmT>
    __device__ void arrayToSignedPcm(const T* input, uint8_t* output, std::size_t frameSampleCount,
        std::size_t channelCount)
    {
        std::size_t startIndex = threadIdx.x;
        std::size_t stride = blockDim.x;
        std::size_t n = frameSampleCount * channelCount;

        for (std::size_t i = startIndex; i < n; i += stride)
        {
            std::size_t channelIndex = i % channelCount;
            std::size_t sampleIndex = i / channelCount;

            PcmT sample = -static_cast<PcmT>(input[channelIndex * frameSampleCount + sampleIndex]) * std::numeric_limits<PcmT>::min();
            output[i] = sample;
        }
    };

    template<class T>
    __device__ void arrayToSigned24Pcm(const T* input, uint8_t* output, std::size_t frameSampleCount,
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

            int32_t sampleInteger = static_cast<int32_t>(input[channelIndex * frameSampleCount + sampleIndex]) * AbsMin;
            uint32_t b0 = sampleInteger & 0xff;
            uint32_t b1 = (sampleInteger >> 8) & 0xff;
            uint32_t b2 = (sampleInteger >> 16) & 0xff;

            output[3 * i] = static_cast<uint8_t>(b0);
            output[3 * i + 1] = static_cast<uint8_t>(b1);
            output[3 * i + 2] = static_cast<uint8_t>(b2);
        }
    }

    template<class T>
    __device__ void arrayToSignedPadded24Pcm(const T* input, uint8_t* output, std::size_t frameSampleCount,
        std::size_t channelCount)
    {
        constexpr T AbsMin = 1 << 23;

        std::size_t startIndex = threadIdx.x;
        std::size_t stride = blockDim.x;
        std::size_t n = frameSampleCount * channelCount;

        for (std::size_t i = startIndex; i < n; i += stride)
        {
            std::size_t channelIndex = i % channelCount;
            std::size_t sampleIndex = i / channelCount;

            uint8_t sample = static_cast<uint8_t>(input[channelIndex * frameSampleCount + sampleIndex]) * AbsMin;
            output[i] = sample;
        }
    };

    template<class T, class PcmT>
    __device__ void arrayToUnsignedPcm(const T* input, uint8_t* output, std::size_t frameSampleCount,
        std::size_t channelCount)
    {
        std::size_t startIndex = threadIdx.x;
        std::size_t stride = blockDim.x;
        std::size_t n = frameSampleCount * channelCount;

        for (std::size_t i = startIndex; i < n; i += stride)
        {
            std::size_t channelIndex = i % channelCount;
            std::size_t sampleIndex = i / channelCount;

            PcmT sample = 2 * static_cast<PcmT>(input[channelIndex * frameSampleCount + sampleIndex]) * std::numeric_limits<PcmT>::max() - 1;
            output[i] = sample;
        }
    };

    template<class T>
    __device__ void arrayToUnsigned24Pcm(const T* input, uint8_t* output, std::size_t frameSampleCount,
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

            uint32_t sampleInteger = static_cast<uint32_t>(input[channelIndex * frameSampleCount + sampleIndex]) * Max + 1;
            uint32_t b0 = sampleInteger & 0xff;
            uint32_t b1 = (sampleInteger >> 8) & 0xff;
            uint32_t b2 = (sampleInteger >> 16) & 0xff;

            output[3 * i] = static_cast<uint8_t>(b0);
            output[3 * i + 1] = static_cast<uint8_t>(b1);
            output[3 * i + 2] = static_cast<uint8_t>(b2);
        }
    }

    template<class T>
    __device__ void arrayToUnsignedPadded24Pcm(const T* input, uint8_t* output, std::size_t frameSampleCount,
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

            uint8_t sample = 2 * static_cast<uint8_t>(input[channelIndex * frameSampleCount + sampleIndex]) * Max + 1;
            output[i] = sample;
        }
    }

    template<class T, class PcmT>
    __device__ void arrayToFloatingPointPcm(const T* input, uint8_t* output, std::size_t frameSampleCount,
        std::size_t channelCount)
    {
        std::size_t startIndex = threadIdx.x;
        std::size_t stride = blockDim.x;
        std::size_t n = frameSampleCount * channelCount;

        for (std::size_t i = startIndex; i < n; i += stride)
        {
            std::size_t channelIndex = i % channelCount;
            std::size_t sampleIndex = i / channelCount;

            output[i] = static_cast<PcmT>(input[channelIndex * frameSampleCount + sampleIndex]);
        }
    };

    template<class T>
    using ArrayToPcmConversionFunctionPointer = void (*)(const T*, uint8_t*, std::size_t, std::size_t);

    template<class T, class PcmT>
    __global__ void getArrayToSignedPcmAddress(ArrayToPcmConversionFunctionPointer<T>* pointer)
    {
        *pointer = arrayToSignedPcm<T, PcmT>;
    }

    template<class T>
    __global__ void  getArrayToSigned24PcmAddress(ArrayToPcmConversionFunctionPointer<T>* pointer)
    {
        *pointer = arrayToSigned24Pcm<T>;
    }

    template<class T>
    __global__ void getArrayToSignedPadded24Address(ArrayToPcmConversionFunctionPointer<T>* pointer)
    {
        *pointer = arrayToSignedPadded24Pcm<T>;
    }

    template<class T, class PcmT>
    __global__ void getArrayToUnsignedPcmAddress(ArrayToPcmConversionFunctionPointer<T>* pointer)
    {
        *pointer = arrayToUnsignedPcm<T, PcmT>;
    }

    template<class T>
    __global__ void getArrayToUnsigned24PcmAddress(ArrayToPcmConversionFunctionPointer<T>* pointer)
    {
        *pointer = arrayToUnsigned24Pcm<T>;
    }

    template<class T>
    __global__ void getArrayToUnsignedPadded24PcmAddress(ArrayToPcmConversionFunctionPointer<T>* pointer)
    {
        *pointer = arrayToUnsignedPadded24Pcm<T>;
    }

    template<class T, class PcmT>
    __global__ void getArrayToFloatingPointPcmAddress(ArrayToPcmConversionFunctionPointer<T>* pointer)
    {
        *pointer = arrayToFloatingPointPcm<T, PcmT>;
    }

    template<class T>
    ArrayToPcmConversionFunctionPointer<T> getArrayToPcmConversionFunctionPointer(PcmAudioFrame::Format format)
    {
        ArrayToPcmConversionFunctionPointer<T> gpuFunctionPointer;
        ArrayToPcmConversionFunctionPointer<T>* d_gpuFunctionPointer;
        cudaMalloc(&d_gpuFunctionPointer, sizeof(ArrayToPcmConversionFunctionPointer<T>*));

        bool supported = true;
        switch (format)
        {
            case PcmAudioFrame::Format::Signed8:
                getArrayToSignedPcmAddress<T, int8_t><<<1, 1>>>(d_gpuFunctionPointer);
                break;
            case PcmAudioFrame::Format::Signed16:
                getArrayToSignedPcmAddress<T, int16_t><<<1, 1>>>(d_gpuFunctionPointer);
                break;
            case PcmAudioFrame::Format::Signed24:
                getArrayToSigned24PcmAddress<T><<< 1, 1 >>>(d_gpuFunctionPointer);
                break;
            case PcmAudioFrame::Format::SignedPadded24:
                getArrayToSignedPadded24Address<T><<<1, 1>>>(d_gpuFunctionPointer);
                break;
            case PcmAudioFrame::Format::Signed32:
                getArrayToSignedPcmAddress<T, int32_t><<<1, 1>>>(d_gpuFunctionPointer);
                break;

            case PcmAudioFrame::Format::Unsigned8:
                getArrayToUnsignedPcmAddress<T, uint8_t> <<<1, 1>>>(d_gpuFunctionPointer);
                break;
            case PcmAudioFrame::Format::Unsigned16:
                getArrayToUnsignedPcmAddress<T, uint16_t><<<1, 1>>>(d_gpuFunctionPointer);
                break;
            case PcmAudioFrame::Format::Unsigned24:
                getArrayToUnsigned24PcmAddress<T><<<1, 1>>>(d_gpuFunctionPointer);
                break;
            case PcmAudioFrame::Format::UnsignedPadded24:
                getArrayToUnsignedPadded24PcmAddress<T> <<<1, 1>>>(d_gpuFunctionPointer);
                break;
            case PcmAudioFrame::Format::Unsigned32:
                getArrayToUnsignedPcmAddress<T, uint32_t><<<1, 1>>>(d_gpuFunctionPointer);
                break;

            case PcmAudioFrame::Format::Float:
                getArrayToFloatingPointPcmAddress<T, float><<<1, 1>>>(d_gpuFunctionPointer);
                break;
            case PcmAudioFrame::Format::Double:
                getArrayToFloatingPointPcmAddress<T, double><<<1, 1>>>(d_gpuFunctionPointer);
                break;

            default:
                supported = false;
                break;
        }

        cudaMemcpy(&gpuFunctionPointer, d_gpuFunctionPointer, sizeof(ArrayToPcmConversionFunctionPointer<T>*),
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
