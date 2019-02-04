#include <Mixer/Configuration/LoggerConfiguration.h>

#include <Utils/Exception/InvalidValueException.h>

using namespace adaptone;
using namespace std;

LoggerConfiguration::LoggerConfiguration(const Properties& properties)
{
    constexpr const char* LoggerTypePropertyKey = "logger.type";
    constexpr const char* LoggerFilenamePropertyKey = "logger.filename";

    string type = properties.get<string>(LoggerTypePropertyKey);

    if (type == "console")
    {
        m_type = LoggerConfiguration::Type::Console;
    }
    else if (type == "file")
    {
        m_type = LoggerConfiguration::Type::File;
        m_filename = properties.get<string>(LoggerFilenamePropertyKey);
    }
    else
    {
        THROW_INVALID_VALUE_EXCEPTION(LoggerTypePropertyKey, type);
    }
}

LoggerConfiguration::~LoggerConfiguration()
{
}
