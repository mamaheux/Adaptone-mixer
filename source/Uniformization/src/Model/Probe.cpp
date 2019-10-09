#include <Uniformization/Model/Probe.h>

using namespace adaptone;

Probe::Probe()
{
}

Probe::Probe(float x, float y) : m_x(x), m_y(y), m_z(0)
{
}

Probe::Probe(float x, float y, float z) : m_x(x), m_y(y), m_z(z)
{
}

Probe::~Probe()
{
}
