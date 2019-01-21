#include <Mixer/Configuration/Configuration.h>

using namespace adaptone;

Configuration::Configuration(const Properties& properties) :
    m_loggerConfiguration(properties)
{
}

Configuration::~Configuration()
{
}