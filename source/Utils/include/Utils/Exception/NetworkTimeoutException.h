#ifndef UTILS_NETWORK_TIMEOUT_EXCEPTION_H
#define UTILS_NETWORK_TIMEOUT_EXCEPTION_H

#include <Utils/Exception/LoggedException.h>

#define THROW_NETWORK_TIMEOUT_EXCEPTION(message) \
    throw adaptone::NetworkTimeoutException(__FILENAME__, __LOGGED_FUNCTION__, __LINE__, (message))

namespace adaptone
{
    class NetworkTimeoutException : public LoggedException
    {
    public:
        NetworkTimeoutException(const std::string& filename,
            const std::string& function,
            int line,
            const std::string& message);

        ~NetworkTimeoutException() override;
    };
}

#endif
