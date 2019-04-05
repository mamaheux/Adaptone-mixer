#include <Utils/Logger/FileLogger.h>

using namespace adaptone;
using namespace std;

FileLogger::FileLogger(const std::string& filename) : m_stream(filename, ofstream::out | ofstream::app)
{
}

FileLogger::~FileLogger()
{
}


void FileLogger::logMessage(const string& message)
{
    m_stream << message << endl;
    m_stream.flush();
}
