#ifndef UTILS_NETWORK_EXCEPTION_H
#define UTILS_NETWORK_EXCEPTION_H

#include <Utils/Exception/LoggedException.h>

#define THROW_NETWORK_EXCEPTION(message) \
    throw adaptone::NetworkException(__FILENAME__, __LOGGED_FUNCTION__, __LINE__, (message))

namespace adaptone
{
    class NetworkException : public LoggedException
    {
    public:
        NetworkException(const std::string& filename,
            const std::string& function,
            int line,
            const std::string& message);

        ~NetworkException() override;
    };
}

#endif
