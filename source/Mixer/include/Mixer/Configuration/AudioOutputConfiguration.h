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
#if defined(__unix__) || defined(__linux__)
            Alsa
#endif
        };

    private:
        Type m_type;
        PcmAudioFrameFormat m_format;
        double m_hardwareDelay;

        //Raw file
        std::string m_filename;

#if defined(__unix__) || defined(__linux__)
        //Alsa
        std::string m_device;
#endif

    public:
        explicit AudioOutputConfiguration(const Properties& properties);
        virtual ~AudioOutputConfiguration();

        Type type() const;
        PcmAudioFrameFormat format() const;
        const double hardwareDelay() const;

        const std::string& filename() const;

#if defined(__unix__) || defined(__linux__)
        const std::string& device() const;
#endif
    };

    inline AudioOutputConfiguration::Type AudioOutputConfiguration::type() const
    {
        return m_type;
    }

    inline PcmAudioFrameFormat AudioOutputConfiguration::format() const
    {
        return m_format;
    }

    inline const double AudioOutputConfiguration::hardwareDelay() const
    {
        return m_hardwareDelay;
    }

    inline const std::string& AudioOutputConfiguration::filename() const
    {
        return m_filename;
    }

#if defined(__unix__) || defined(__linux__)

    inline const std::string& AudioOutputConfiguration::device() const
    {
        return m_device;
    }

#endif

}

#endif
