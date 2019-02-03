#ifndef MIXER_AUDIO_INPUT_AUDIO_INPUT_H
#define MIXER_AUDIO_INPUT_AUDIO_INPUT_H

#include <Utils/ClassMacro.h>
#include <Utils/Data/PcmAudioFrame.h>

namespace adaptone
{
    class AudioInput
    {
    protected:
        PcmAudioFrame m_frame;
    public:
        AudioInput(PcmAudioFrame::Format format, std::size_t frameSampleCount, std::size_t sampleCount);
        virtual ~AudioInput();

        DECLARE_NOT_COPYABLE(AudioInput);
        DECLARE_NOT_MOVABLE(AudioInput);

        virtual const PcmAudioFrame& read() = 0;
        virtual bool hasNext() = 0;
    };
}

#endif