#include "Uniformization/Model/Speaker.h"

using namespace adaptone;

Speaker::Speaker()
{
}

Speaker::Speaker(float x, float y) : m_x(x), m_y(y), m_z(0)
{
}

Speaker::Speaker(float x, float y, float z) : m_x(x), m_y(y), m_z(z)
{
}

Speaker::~Speaker()
{
}
