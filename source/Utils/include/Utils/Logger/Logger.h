#ifndef UTILS_LOGGER_LOGGER_H
#define UTILS_LOGGER_LOGGER_H

#include <Utils/ClassMacro.h>

#include <ctime>
#include <chrono>
#include <iomanip>

#include <exception>
#include <string>
#include <sstream>

namespace adaptone
{
    class Logger
    {
    public:
        enum class Level
        {
            Debug = 0,
            Information = 1,
            Warning = 2,
            Error = 3
        };

        static Level parseLevel(const std::string& level);

    private:
        Level m_level;

    public:
        Logger();
        Logger(Level level);
        virtual ~Logger();

        DECLARE_NOT_COPYABLE(Logger);
        DECLARE_NOT_MOVABLE(Logger);

        void log(Level level, const std::string& message);
        void log(Level level, const std::exception& exception);
        void log(Level level, const std::exception& exception, const std::string& message);

    protected:
        void writeTime(std::stringstream& ss);
        void writeLevelName(std::stringstream& ss, Logger::Level level);
        virtual void logMessage(const std::string& message) = 0;
    };

    inline void Logger::log(Logger::Level level, const std::string& message)
    {
        if (level < m_level)
        {
            return;
        }

        std::stringstream ss;

        writeTime(ss);
        writeLevelName(ss, level);
        ss << " --> " << message;

        logMessage(ss.str());
    }

    inline void Logger::log(Logger::Level level, const std::exception& exception)
    {
        if (level < m_level)
        {
            return;
        }

        std::stringstream ss;

        writeTime(ss);
        writeLevelName(ss, level);
        ss << " --> " << exception.what();

        logMessage(ss.str());
    }

    inline void Logger::log(Logger::Level level, const std::exception& exception, const std::string& message)
    {
        if (level < m_level)
        {
            return;
        }

        std::stringstream ss;

        writeTime(ss);
        writeLevelName(ss, level);
        ss << " --> " << exception.what() << " --> " << message;

        logMessage(ss.str());
    }

    inline void Logger::writeTime(std::stringstream& ss)
    {
        std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        ss << std::put_time(std::localtime(&now), "%F %T") << " ";
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
        }
    }
}

#endif
