#ifndef MIXER_CONFIGURATION_AUDIO_INPUT_CONFIGURATION_H
#define MIXER_CONFIGURATION_AUDIO_INPUT_CONFIGURATION_H

#include <Utils/Configuration/Properties.h>
#include <Utils/Data/RawAudioFrame.h>

namespace adaptone
{
    class AudioInputConfiguration
    {
    public:
        enum class Type
        {
            RawFile
        };

    private:
        Type m_type;
        RawAudioFrame::Format m_format;

        //Raw file
        std::string m_filename;

    public:
        explicit AudioInputConfiguration(const Properties& properties);
        virtual ~AudioInputConfiguration();

        Type type() const;
        RawAudioFrame::Format format() const;

        const std::string& filename() const;
    };

    inline AudioInputConfiguration::Type AudioInputConfiguration::type() const
    {
        return m_type;
    }

    inline RawAudioFrame::Format AudioInputConfiguration::format() const
    {
        return m_format;
    }

    inline const std::string& AudioInputConfiguration::filename() const
    {
        return m_filename;
    }
}

#endif