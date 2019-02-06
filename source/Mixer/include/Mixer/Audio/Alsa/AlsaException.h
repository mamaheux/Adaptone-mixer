#ifndef MIXER_AUDIO_ALSA_ALSA_EXCEPTION_H
#define MIXER_AUDIO_ALSA_ALSA_EXCEPTION_H

#if defined(__unix__) || defined(__linux__)

#include <Utils/Exception/LoggedException.h>

#define THROW_ALSA_EXCEPTION(message, errorCode, errorDescription) \
    throw adaptone::AlsaException(__FILENAME__, \
        __LOGGED_FUNCTION__, \
        __LINE__, \
        (message), \
        (errorCode), \
        (errorDescription))

namespace adaptone
{
    class AlsaException : public LoggedException
    {
    public:
        AlsaException(const std::string& filename,
            const std::string& function,
            int line,
            const std::string& message,
            int errorCode,
            const std::string& errorDescription);

        virtual ~AlsaException();
    };
}

#else

#error "Invalid include file"

#endif

#endif
