#include <Mixer/Configuration/Configuration.h>

using namespace adaptone;

Configuration::Configuration(const Properties& properties) :
    m_loggerConfiguration(properties),
    m_audioConfiguration(properties),
    m_audioInputConfiguration(properties)
{
}

Configuration::~Configuration()
{
}