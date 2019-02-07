#ifndef MIXER_AUDIO_OUTPUT_ALSA_AUDIO_OUTPUT_H
#define MIXER_AUDIO_OUTPUT_ALSA_AUDIO_OUTPUT_H

#include <Mixer/AudioOutput/AudioOutput.h>
#include <Mixer/Audio/Alsa/AlsaPcmDevice.h>

#include <Utils/ClassMacro.h>

namespace adaptone
{
    class AlsaAudioOutput : public AudioOutput
    {
        AlsaPcmDevice m_device;

    public:
        AlsaAudioOutput(PcmAudioFrame::Format format,
            std::size_t channelCount,
            std::size_t frameSampleCount,
            size_t sampleFrequency,
            const std::string& device);
        ~AlsaAudioOutput() override;

        DECLARE_NOT_COPYABLE(AlsaAudioOutput);
        DECLARE_NOT_MOVABLE(AlsaAudioOutput);

        void write(const PcmAudioFrame& frame) override;
    };
}

#endif
