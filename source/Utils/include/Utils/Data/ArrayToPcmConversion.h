#ifndef UTILS_DATA_ARRAY_TO_PCM_CONVERSION_H
#define UTILS_DATA_ARRAY_TO_PCM_CONVERSION_H

#include <Utils/Data/PcmAudioFrame.h>
#include <Utils/TypeTraits.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace adaptone
{

    template<class T>
    inline T round(T value)
    {
        return floor(value + 0.5);
    }

    template<class T>
    inline T saturateOutput(T value, T min, T max)
    {
        if (value > max)
        {
            return max;
        }
        if (value < min)
        {
            return min;
        }

        return value;
    }

    template<class T, class PcmT>
    void arrayToSignedPcm(const T* input, uint8_t* outputBytes, std::size_t frameSampleCount,
        std::size_t channelCount)
    {
        std::size_t n = frameSampleCount * channelCount;

        PcmT* output = reinterpret_cast<PcmT*>(outputBytes);

        for (std::size_t i = 0; i < n; i++)
        {
            std::size_t channelIndex = i % channelCount;
            std::size_t sampleIndex = i / channelCount;

            T scaledSample = -input[channelIndex * frameSampleCount + sampleIndex] * std::numeric_limits<PcmT>::min();
            T sample = saturateOutput(scaledSample,
                                      static_cast<T>(std::numeric_limits<PcmT>::min()),
                                      static_cast<T>(std::numeric_limits<PcmT>::max()));
            output[i] = static_cast<PcmT>(round(sample));
        }
    };

    template<class T>
    void arrayToSigned24Pcm(const T* input, uint8_t* output, std::size_t frameSampleCount,
        std::size_t channelCount)
    {
        constexpr T AbsMin = 1 << 23;

        std::size_t n = frameSampleCount * channelCount;

        for (std::size_t i = 0; i < n; i++)
        {
            std::size_t channelIndex = i % channelCount;
            std::size_t sampleIndex = i / channelCount;

            T scaledSample = input[channelIndex * frameSampleCount + sampleIndex] * AbsMin;
            T sample = saturateOutput(scaledSample, -AbsMin, AbsMin - 1);

            int32_t sampleInteger = static_cast<int32_t>(round(sample));
            uint32_t b0 = sampleInteger & 0xff;
            uint32_t b1 = (sampleInteger >> 8) & 0xff;
            uint32_t b2 = (sampleInteger >> 16) & 0xff;

            output[3 * i] = static_cast<uint8_t>(b0);
            output[3 * i + 1] = static_cast<uint8_t>(b1);
            output[3 * i + 2] = static_cast<uint8_t>(b2);
        }
    }

    template<class T>
    void arrayToSignedPadded24Pcm(const T* input, uint8_t* outputBytes, std::size_t frameSampleCount,
        std::size_t channelCount)
    {
        constexpr T AbsMin = 1 << 23;

        std::size_t n = frameSampleCount * channelCount;

        int32_t* output = reinterpret_cast<int32_t*>(outputBytes);

        for (std::size_t i = 0; i < n; i++)
        {
            std::size_t channelIndex = i % channelCount;
            std::size_t sampleIndex = i / channelCount;

            T scaledSample = input[channelIndex * frameSampleCount + sampleIndex] * AbsMin;
            T sample = saturateOutput(scaledSample, -AbsMin, AbsMin - 1);
            output[i] = static_cast<int32_t>(round(sample));
        }
    };

    template<class T, class PcmT>
    void arrayToUnsignedPcm(const T* input, uint8_t* outputBytes, std::size_t frameSampleCount,
        std::size_t channelCount)
    {
        std::size_t n = frameSampleCount * channelCount;

        PcmT* output = reinterpret_cast<PcmT*>(outputBytes);

        for (std::size_t i = 0; i < n; i++)
        {
            std::size_t channelIndex = i % channelCount;
            std::size_t sampleIndex = i / channelCount;

            T scaledSample = (input[channelIndex * frameSampleCount + sampleIndex] + 1) / 2 *
                             std::numeric_limits<PcmT>::max();
            T sample = saturateOutput(scaledSample,
                                      static_cast<T>(std::numeric_limits<PcmT>::min()),
                                      static_cast<T>(std::numeric_limits<PcmT>::max()));
            output[i] = static_cast<PcmT>(round(sample));
        }
    };

    template<class T>
    void arrayToUnsigned24Pcm(const T* input, uint8_t* output, std::size_t frameSampleCount,
        std::size_t channelCount)
    {
        constexpr T Max = (1 << 24) - 1;

        std::size_t n = frameSampleCount * channelCount;

        for (std::size_t i = 0; i < n; i++)
        {
            std::size_t channelIndex = i % channelCount;
            std::size_t sampleIndex = i / channelCount;

            T scaledSample = (input[channelIndex * frameSampleCount + sampleIndex] + 1) / 2 * Max;
            T sample = saturateOutput(scaledSample, static_cast<T>(0), Max);

            uint32_t sampleInteger = static_cast<uint32_t>(round(sample));
            uint32_t b0 = sampleInteger & 0xff;
            uint32_t b1 = (sampleInteger >> 8) & 0xff;
            uint32_t b2 = (sampleInteger >> 16) & 0xff;

            output[3 * i] = static_cast<uint8_t>(b0);
            output[3 * i + 1] = static_cast<uint8_t>(b1);
            output[3 * i + 2] = static_cast<uint8_t>(b2);
        }
    }

    template<class T>
    void arrayToUnsignedPadded24Pcm(const T* input, uint8_t* outputBytes, std::size_t frameSampleCount,
        std::size_t channelCount)
    {
        constexpr T Max = (1 << 24) - 1;

        std::size_t n = frameSampleCount * channelCount;

        uint32_t* output = reinterpret_cast<uint32_t*>(outputBytes);

        for (std::size_t i = 0; i < n; i++)
        {
            std::size_t channelIndex = i % channelCount;
            std::size_t sampleIndex = i / channelCount;

            T scaledSample = (input[channelIndex * frameSampleCount + sampleIndex] + 1) / 2 * Max;
            T sample = saturateOutput(scaledSample, static_cast<T>(0), Max);
            output[i] = static_cast<uint32_t>(round(sample));
        }
    }

    template<class T, class PcmT>
    void arrayToFloatingPointPcm(const T* input, uint8_t* outputBytes, std::size_t frameSampleCount,
        std::size_t channelCount)
    {
        std::size_t n = frameSampleCount * channelCount;

        PcmT* output = reinterpret_cast<PcmT*>(outputBytes);

        for (std::size_t i = 0; i < n; i++)
        {
            std::size_t channelIndex = i % channelCount;
            std::size_t sampleIndex = i / channelCount;

            T sample = saturateOutput(input[channelIndex * frameSampleCount + sampleIndex], static_cast<T>(-1),
                                      static_cast<T>(1));
            output[i] = static_cast<PcmT>(sample);
        }
    };

    template<class T>
    void convertArrayToPcm(const T* input, uint8_t* outputBytes, std::size_t frameSampleCount,
        std::size_t channelCount, PcmAudioFrame::Format format)
    {
        switch (format)
        {
            case PcmAudioFrame::Format::Signed8:
                arrayToSignedPcm<T, int8_t>(input, outputBytes, frameSampleCount, channelCount);
                break;
            case PcmAudioFrame::Format::Signed16:
                arrayToSignedPcm<T, int16_t>(input, outputBytes, frameSampleCount, channelCount);
                break;
            case PcmAudioFrame::Format::Signed24:
                arrayToSigned24Pcm<T>(input, outputBytes, frameSampleCount, channelCount);
                break;
            case PcmAudioFrame::Format::SignedPadded24:
                arrayToSignedPadded24Pcm<T>(input, outputBytes, frameSampleCount, channelCount);
                break;
            case PcmAudioFrame::Format::Signed32:
                arrayToSignedPcm<T, int32_t>(input, outputBytes, frameSampleCount, channelCount);
                break;

            case PcmAudioFrame::Format::Unsigned8:
                arrayToUnsignedPcm<T, uint8_t>(input, outputBytes, frameSampleCount, channelCount);
                break;
            case PcmAudioFrame::Format::Unsigned16:
                arrayToUnsignedPcm<T, uint16_t>(input, outputBytes, frameSampleCount, channelCount);
                break;
            case PcmAudioFrame::Format::Unsigned24:
                arrayToUnsigned24Pcm<T>(input, outputBytes, frameSampleCount, channelCount);
                break;
            case PcmAudioFrame::Format::UnsignedPadded24:
                arrayToUnsignedPadded24Pcm<T>(input, outputBytes, frameSampleCount, channelCount);
                break;
            case PcmAudioFrame::Format::Unsigned32:
                arrayToUnsignedPcm<T, uint32_t>(input, outputBytes, frameSampleCount, channelCount);
                break;

            case PcmAudioFrame::Format::Float:
                arrayToFloatingPointPcm<T, float>(input, outputBytes, frameSampleCount, channelCount);
                break;
            case PcmAudioFrame::Format::Double:
                arrayToFloatingPointPcm<T, double>(input, outputBytes, frameSampleCount, channelCount);
                break;
        }
    }
}

#endif
