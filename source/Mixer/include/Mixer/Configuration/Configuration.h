#ifndef MIXER_CONFIGURATION_CONFIGURATION_H
#define MIXER_CONFIGURATION_CONFIGURATION_H

#include <Mixer/Configuration/LoggerConfiguration.h>

#include <Utils/Configuration/Properties.h>

namespace adaptone
{
    class Configuration
    {
        LoggerConfiguration m_loggerConfiguration;

    public:
        explicit Configuration(const Properties& properties);
        virtual ~Configuration();

        const LoggerConfiguration& logger() const;
    };

    inline const LoggerConfiguration& Configuration::logger() const
    {
        return m_loggerConfiguration;
    }
}

#endif