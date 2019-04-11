#include <Communication/Messages/Initialization/ConfigurationPosition.h>

using namespace adaptone;

ConfigurationPosition::ConfigurationPosition() :
    m_x(0),
    m_y(0),
    m_type(ConfigurationPosition::Type::Speaker)
{
}

ConfigurationPosition::ConfigurationPosition(double x, double y, ConfigurationPosition::Type type) :
    m_x(x),
    m_y(y),
    m_type(type)
{
}

ConfigurationPosition::~ConfigurationPosition()
{
}
