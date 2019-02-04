#ifndef UTILS_LOGGER_CONSOLE_LOGGER_H
#define UTILS_LOGGER_CONSOLE_LOGGER_H

#include <Utils/Logger/Logger.h>

namespace adaptone
{
    class ConsoleLogger : public Logger
    {
    public:
        ConsoleLogger();
        ~ConsoleLogger() override;

        DECLARE_NOT_COPYABLE(ConsoleLogger);
        DECLARE_NOT_MOVABLE(ConsoleLogger);

    protected:
        void logMessage(const std::string& message) override;
    };
}

#endif
