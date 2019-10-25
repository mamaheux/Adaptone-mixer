#include <Communication/Messages/Initialization/ConfigurationPosition.h>

using namespace adaptone;

ConfigurationPosition::ConfigurationPosition() :
    m_x(0),
    m_y(0),
    m_type(PositionType::Speaker)
{
}

ConfigurationPosition::ConfigurationPosition(double x, double y, PositionType type, uint32_t id) :
    m_x(x),
    m_y(y),
    m_type(type),
    m_id(id)
{
}

ConfigurationPosition::~ConfigurationPosition()
{
}
