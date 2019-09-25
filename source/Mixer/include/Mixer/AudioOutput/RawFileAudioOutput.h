#ifndef MIXER_AUDIO_OUTPUT_RAW_FILE_AUDIO_OUTPUT_H
#define MIXER_AUDIO_OUTPUT_RAW_FILE_AUDIO_OUTPUT_H

#include <Mixer/AudioOutput/AudioOutput.h>

#include <Utils/ClassMacro.h>

#include <fstream>
#include <memory>

namespace adaptone
{
    class RawFileAudioOutput : public AudioOutput
    {
        std::ofstream m_fileStream;

    public:
        RawFileAudioOutput(PcmAudioFrameFormat format,
            std::size_t channelCount,
            std::size_t frameSampleCount,
            const std::string& filename);
        ~RawFileAudioOutput() override;

        DECLARE_NOT_COPYABLE(RawFileAudioOutput);
        DECLARE_NOT_MOVABLE(RawFileAudioOutput);

        void write(const PcmAudioFrame& frame) override;
    };
}

#endif
