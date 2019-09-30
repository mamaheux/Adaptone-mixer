#include <Communication/Messages/Initialization/ConfigurationPosition.h>

using namespace adaptone;

ConfigurationPosition::ConfigurationPosition() :
    m_x(0),
    m_y(0),
    m_type(PositionType::Speaker)
{
}

ConfigurationPosition::ConfigurationPosition(double x, double y, PositionType type) :
    m_x(x),
    m_y(y),
    m_type(type)
{
}

ConfigurationPosition::~ConfigurationPosition()
{
}
