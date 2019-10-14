#include <Utils/Logger/FileLogger.h>

using namespace adaptone;
using namespace std;

FileLogger::FileLogger(const string& filename) : m_stream(filename, ofstream::out | ofstream::app)
{
}

FileLogger::FileLogger(Level level, const std::string& filename) :
    Logger(level),
    m_stream(filename, ofstream::out | ofstream::app)
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
