#ifndef MIXER_CONFIGURATION_LOGGER_CONFIGURATION_H
#define MIXER_CONFIGURATION_LOGGER_CONFIGURATION_H

#include <Utils/Configuration/Properties.h>

#include <string>

namespace adaptone
{
    class LoggerConfiguration
    {
    public:
        enum class Type
        {
            Console,
            File
        };

    private:
        Type m_type;
        std::string m_filename;

    public:
        explicit LoggerConfiguration(const Properties& properties);
        virtual ~LoggerConfiguration();

        Type type() const;
        const std::string& filename() const;
    };

    inline LoggerConfiguration::Type LoggerConfiguration::type() const
    {
        return m_type;
    }

    inline const std::string& LoggerConfiguration::filename() const
    {
        return m_filename;
    }
}

#endif