#ifndef MIXER_CONFIGURATION_LOGGER_CONFIGURATION_H
#define MIXER_CONFIGURATION_LOGGER_CONFIGURATION_H

#include <Utils/Configuration/Properties.h>
#include <Utils/Logger/Logger.h>

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
        Logger::Level m_level;
        Type m_type;
        std::string m_filename;

    public:
        explicit LoggerConfiguration(const Properties& properties);
        virtual ~LoggerConfiguration();

        Logger::Level level() const;
        Type type() const;
        const std::string& filename() const;
    };

    inline Logger::Level LoggerConfiguration::level() const
    {
        return m_level;
    }

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
