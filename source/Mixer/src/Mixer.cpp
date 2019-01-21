#include <Mixer/Mixer.h>

#include <Utils/Exception/NotSupportedException.h>

#include <Utils/Logger/ConsoleLogger.h>
#include <Utils/Logger/FileLogger.h>

using namespace adaptone;
using namespace std;

Mixer::Mixer(const Configuration& configuration) : m_configuration(configuration)
{
    std::shared_ptr<Logger> logger = createLogger();

    //Create all members, then assign them to the attributes to prevent memory leaks
    m_logger = logger;
}

Mixer::~Mixer()
{
}

int Mixer::run()
{
    try
    {
        return 0;
    }
    catch (exception& ex)
    {
        m_logger->log(Logger::Level::Error, ex);
        return -1;
    }
}

std::shared_ptr<Logger> Mixer::createLogger()
{
    switch (m_configuration.logger().type())
    {
        case LoggerConfiguration::Type::Console:
            return make_shared<ConsoleLogger>();

        case LoggerConfiguration::Type::File:
            return make_shared<FileLogger>(m_configuration.logger().filename());
    }

    THROW_NOT_SUPPORTED_EXCEPTION("Not supported logger type.");
}