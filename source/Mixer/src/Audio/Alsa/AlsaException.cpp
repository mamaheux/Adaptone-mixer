#if defined(__unix__) || defined(__linux__)

#include <Mixer/Audio/Alsa/AlsaException.h>

using namespace adaptone;
using namespace std;

AlsaException::AlsaException(const string& filename,
    const string& function,
    int line,
    const string& message,
    int errorCode,
    const string& errorDescription) :
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
