#ifndef MIXER_CONFIGURATION_AUDIO_INPUT_CONFIGURATION_H
#define MIXER_CONFIGURATION_AUDIO_INPUT_CONFIGURATION_H

#include <Utils/Configuration/Properties.h>
#include <Utils/Data/PcmAudioFrame.h>

namespace adaptone
{
    class AudioInputConfiguration
    {
    public:
        enum class Type
        {
            RawFile,
            Alsa
        };

    private:
        Type m_type;
        PcmAudioFrame::Format m_format;

        //Raw file
        std::string m_filename;
        bool m_looping;

        //Alsa
        std::string m_device;

    public:
        explicit AudioInputConfiguration(const Properties& properties);
        virtual ~AudioInputConfiguration();

        Type type() const;
        PcmAudioFrame::Format format() const;

        const std::string& filename() const;
        bool looping() const;

        const std::string& device() const;
    };

    inline AudioInputConfiguration::Type AudioInputConfiguration::type() const
    {
        return m_type;
    }

    inline PcmAudioFrame::Format AudioInputConfiguration::format() const
    {
        return m_format;
    }

    inline const std::string& AudioInputConfiguration::filename() const
    {
        return m_filename;
    }

    inline bool AudioInputConfiguration::looping() const
    {
        return m_looping;
    }

    inline const std::string& AudioInputConfiguration::device() const
    {
        return m_device;
    }
}

#endif
