#ifndef UTILS_LOGGER_FILE_LOGGER_H
#define UTILS_LOGGER_FILE_LOGGER_H

#include <Utils/Logger/Logger.h>

#include <fstream>

namespace adaptone
{
    class FileLogger : public Logger
    {
        std::ofstream m_stream;
    public:
        explicit FileLogger(const std::string& filename);

        ~FileLogger() override;

        DECLARE_NOT_COPYABLE(FileLogger);
        DECLARE_NOT_MOVABLE(FileLogger);

    protected:
        void logMessage(const std::string& message) override;
    };
}

#endif
