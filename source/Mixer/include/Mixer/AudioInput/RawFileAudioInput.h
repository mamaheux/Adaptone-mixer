#ifndef MIXER_AUDIO_INPUT_RAW_FILE_AUDIO_INPUT_H
#define MIXER_AUDIO_INPUT_RAW_FILE_AUDIO_INPUT_H

#include <Mixer/AudioInput/AudioInput.h>

#include <Utils/ClassMacro.h>

#include <fstream>
#include <memory>

namespace adaptone
{
    class RawFileAudioInput : public AudioInput
    {
        std::unique_ptr<std::ifstream> m_fileStream;
        bool m_looping;
        std::size_t m_fileSize;

    public:
        RawFileAudioInput(PcmAudioFrameFormat format,
            std::size_t channelCount,
            std::size_t frameSampleCount,
            const std::string& filename,
            bool looping);
        ~RawFileAudioInput() override;

        DECLARE_NOT_COPYABLE(RawFileAudioInput);
        DECLARE_NOT_MOVABLE(RawFileAudioInput);

        const PcmAudioFrame& read() override;
        bool hasNext() override;
    };
}

#endif
