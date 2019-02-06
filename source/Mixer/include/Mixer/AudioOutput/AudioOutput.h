#ifndef MIXER_AUDIO_OUTPUT_AUDIO_OUTPUT_H
#define MIXER_AUDIO_OUTPUT_AUDIO_OUTPUT_H

#include <Utils/ClassMacro.h>
#include <Utils/Data/PcmAudioFrame.h>

namespace adaptone
{
    class AudioOutput
    {
    protected:
        PcmAudioFrame::Format m_format;
        std::size_t m_channelCount;
        std::size_t m_frameSampleCount;

    public:
        AudioOutput(PcmAudioFrame::Format format, std::size_t channelCount, std::size_t frameSampleCount);
        virtual ~AudioOutput();

        DECLARE_NOT_COPYABLE(AudioOutput);
        DECLARE_NOT_MOVABLE(AudioOutput);

        virtual void write(const PcmAudioFrame& frame) = 0;

        virtual bool hasGainControl();
        virtual void setGain(std::size_t channelIndex, uint8_t gain);
    };
}

#endif