#include <Utils/Exception/NetworkException.h>

using namespace adaptone;
using namespace std;

NetworkException::NetworkException(const string& filename,
    const string& function,
    int line,
    const string& message) :
    LoggedException(filename, function, line, "NetworkException: " + message)
{
}

NetworkException::~NetworkException()
{
}
