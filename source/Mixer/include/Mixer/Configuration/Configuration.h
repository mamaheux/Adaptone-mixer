#ifndef MIXER_CONFIGURATION_CONFIGURATION_H
#define MIXER_CONFIGURATION_CONFIGURATION_H

#include <Mixer/Configuration/LoggerConfiguration.h>
#include <Mixer/Configuration/AudioConfiguration.h>
#include <Mixer/Configuration/AudioInputConfiguration.h>

#include <Utils/Configuration/Properties.h>

namespace adaptone
{
    class Configuration
    {
        LoggerConfiguration m_loggerConfiguration;
        AudioConfiguration m_audioConfiguration;
        AudioInputConfiguration m_audioInputConfiguration;

    public:
        explicit Configuration(const Properties& properties);
        virtual ~Configuration();

        const LoggerConfiguration& logger() const;
        const AudioConfiguration& audio() const;
        const AudioInputConfiguration& audioInput() const;
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
}

#endif
