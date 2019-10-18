#include <Uniformization/Model/ModelElement.h>

using namespace adaptone;

ModelElement::ModelElement() : m_x(0), m_y(0), m_z(0)
{
}

ModelElement::ModelElement(double x, double y) : m_x(x), m_y(y), m_z(0)
{
}

ModelElement::ModelElement(double x, double y, double z) : m_x(x), m_y(y), m_z(z)
{
}

ModelElement::~ModelElement()
{
}
