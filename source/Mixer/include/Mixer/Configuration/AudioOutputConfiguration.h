#ifndef MIXER_CONFIGURATION_AUDIO_OUTPUT_CONFIGURATION_H
#define MIXER_CONFIGURATION_AUDIO_OUTPUT_CONFIGURATION_H

#include <Utils/Configuration/Properties.h>
#include <Utils/Data/PcmAudioFrame.h>

namespace adaptone
{
    class AudioOutputConfiguration
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

        //Alsa
        std::string m_device;

    public:
        explicit AudioOutputConfiguration(const Properties& properties);
        virtual ~AudioOutputConfiguration();

        Type type() const;
        PcmAudioFrame::Format format() const;

        const std::string& filename() const;

        const std::string& device() const;
    };

    inline AudioOutputConfiguration::Type AudioOutputConfiguration::type() const
    {
        return m_type;
    }

    inline PcmAudioFrame::Format AudioOutputConfiguration::format() const
    {
        return m_format;
    }

    inline const std::string& AudioOutputConfiguration::filename() const
    {
        return m_filename;
    }

    inline const std::string& AudioOutputConfiguration::device() const
    {
        return m_device;
    }
}

#endif
