#include <Utils/Logger/Logger.h>

#include <Utils/Exception/InvalidValueException.h>

#include <unordered_map>

using namespace adaptone;
using namespace std;

Logger::Level Logger::parseLevel(const string& level)
{
    static const unordered_map<string, Level> Mapping(
        {
            { "debug", Level::Debug },
            { "information", Level::Information },
            { "warning", Level::Warning },
            { "error", Level::Error }
        });

    auto it = Mapping.find(level);
    if (it != Mapping.end())
    {
        return it->second;
    }

    THROW_INVALID_VALUE_EXCEPTION("Logger::Level", level);
}

Logger::Logger() : m_level(Level::Debug)
{
}

Logger::Logger(Level level) : m_level(level)
{
}

Logger::~Logger()
{
}
