#ifndef MIXER_AUDIO_INPUT_ALSA_AUDIO_INPUT_H
#define MIXER_AUDIO_INPUT_ALSA_AUDIO_INPUT_H

#if defined(__unix__) || defined(__linux__)

#include <Mixer/AudioInput/AudioInput.h>
#include <Mixer/Audio/Alsa/AlsaPcmDevice.h>

#include <Utils/ClassMacro.h>

namespace adaptone
{
    class AlsaAudioInput : public AudioInput
    {
        AlsaPcmDevice m_device;

    public:
        AlsaAudioInput(PcmAudioFrameFormat format,
            std::size_t channelCount,
            std::size_t frameSampleCount,
            size_t sampleFrequency,
            const std::string& device);
        ~AlsaAudioInput() override;

        DECLARE_NOT_COPYABLE(AlsaAudioInput);
        DECLARE_NOT_MOVABLE(AlsaAudioInput);

        const PcmAudioFrame& read() override;
        bool hasNext() override;
    };
}

#else

#error "Invalid include file"

#endif

#endif
