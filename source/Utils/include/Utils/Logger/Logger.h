#ifndef UTILS_LOGGER_LOGGER_H
#define UTILS_LOGGER_LOGGER_H

#include <Utils/ClassMacro.h>

#include <string>
#include <sstream>

#include <exception>

namespace adaptone
{
    class Logger
    {
    public:
        enum class Level
        {
            Debug,
            Information,
            Warning,
            Error,
            Performance
        };

        Logger();

        virtual ~Logger();

        DECLARE_NOT_COPYABLE(Logger);
        DECLARE_NOT_MOVABLE(Logger);

        void log(Level level, const std::string& message);

        void log(Level level, const std::exception& exception);

        void log(Level level, const std::exception& exception, const std::string& message);

    protected:
        void writeLevelName(std::stringstream& ss, Logger::Level level);

        virtual void logMessage(const std::string& message) = 0;
    };

    inline void Logger::log(Logger::Level level, const std::string& message)
    {
        std::stringstream ss;

        writeLevelName(ss, level);
        ss << " --> " << message;

        logMessage(ss.str());
    }

    inline void Logger::log(Logger::Level level, const std::exception& exception)
    {
        std::stringstream ss;

        writeLevelName(ss, level);
        ss << " --> " << exception.what();

        logMessage(ss.str());
    }

    inline void Logger::log(Logger::Level level, const std::exception& exception, const std::string& message)
    {
        std::stringstream ss;

        writeLevelName(ss, level);
        ss << " --> " << exception.what() << " --> " << message;

        logMessage(ss.str());
    }

    inline void Logger::writeLevelName(std::stringstream& ss, Logger::Level level)
    {
        switch (level)
        {
            case Logger::Level::Information:
                ss << "Information";
                break;

            case Logger::Level::Debug:
                ss << "Debug";
                break;

            case Logger::Level::Warning:
                ss << "Warning";
                break;

            case Logger::Level::Error:
                ss << "Error";
                break;

            case Logger::Level::Performance:
                ss << "Performance";
                break;
        }
    }
}

#endif
