#ifndef MIXER_CONFIGURATION_CONFIGURATION_H
#define MIXER_CONFIGURATION_CONFIGURATION_H

#include <Mixer/Configuration/LoggerConfiguration.h>
#include <Mixer/Configuration/AudioConfiguration.h>
#include <Mixer/Configuration/AudioInputConfiguration.h>
#include <Mixer/Configuration/AudioOutputConfiguration.h>

#include <Utils/Configuration/Properties.h>

namespace adaptone
{
    class Configuration
    {
        LoggerConfiguration m_loggerConfiguration;

        AudioConfiguration m_audioConfiguration;
        AudioInputConfiguration m_audioInputConfiguration;
        AudioOutputConfiguration m_audioOutputConfiguration;

    public:
        explicit Configuration(const Properties& properties);
        virtual ~Configuration();

        const LoggerConfiguration& logger() const;
        const AudioConfiguration& audio() const;
        const AudioInputConfiguration& audioInput() const;
        const AudioOutputConfiguration& audioOutput() const;
    };

    inline const LoggerConfiguration& Configuration::logger() const
    {
        return m_loggerConfiguration;
    }

    inline const AudioConfiguration& Configuration::audio() const
    {
        return m_audioConfiguration;
    }

    inline const AudioInputConfiguration& Configuration::audioInput() const
    {
        return m_audioInputConfiguration;
    }

    inline const AudioOutputConfiguration& Configuration::audioOutput() const
    {
        return m_audioOutputConfiguration;
    }
}

#endif
