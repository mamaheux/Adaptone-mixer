#if defined(__unix__) || defined(__linux__)

#include <Mixer/Audio/Alsa/AlsaException.h>

using namespace adaptone;
using namespace std;

AlsaException::AlsaException(const std::string& filename,
    const std::string& function,
    int line,
    const std::string& message,
    int errorCode,
    const std::string& errorDescription) :
    LoggedException(filename,
        function,
        line,
        "AlsaException: " + message + " (" + to_string(errorCode) + ": " + errorDescription + ")")
{
}

AlsaException::~AlsaException()
{
}

#endif
