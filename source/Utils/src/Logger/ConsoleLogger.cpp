#include <Utils/Logger/ConsoleLogger.h>

#include <iostream>

using namespace adaptone;
using namespace std;

ConsoleLogger::ConsoleLogger()
{
}

ConsoleLogger::ConsoleLogger(Level level) : Logger(level)
{
}

ConsoleLogger::~ConsoleLogger()
{
}


void ConsoleLogger::logMessage(const string& message)
{
    cout << message << endl;
}
