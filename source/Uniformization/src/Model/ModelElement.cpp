#include <Uniformization/Model/ModelElement.h>

using namespace adaptone;

ModelElement::ModelElement() : m_x(0), m_y(0), m_z(0), m_id(0)
{
}

ModelElement::ModelElement(double x, double y, uint32_t id) : m_x(x), m_y(y), m_z(0), m_id(id)
{
}

ModelElement::ModelElement(double x, double y, double z, uint32_t id) : m_x(x), m_y(y), m_z(z), m_id(id)
{
}

ModelElement::~ModelElement()
{
}
