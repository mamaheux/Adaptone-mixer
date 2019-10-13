#include <Utils/Exception/NetworkTimeoutException.h>

using namespace adaptone;
using namespace std;

NetworkTimeoutException::NetworkTimeoutException(const string& filename,
    const string& function,
    int line,
    const string& message) :
    LoggedException(filename, function, line, "NetworkTimeoutException: " + message)
{
}

NetworkTimeoutException::~NetworkTimeoutException()
{
}
