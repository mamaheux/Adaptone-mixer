#include "Uniformization/Probe.h"

using namespace adaptone;

Probe::Probe()
{
}

Probe::Probe(float x, float y) : m_coord{x, y, 0}
{
}

Probe::Probe(float x, float y, float z) : m_coord{x, y, z}
{
}

Probe::~Probe()
{
}

void Probe::setCoord(const float coord[3])
{
    m_coord[0] = coord[0];
    m_coord[1] = coord[1];
    m_coord[2] = coord[2];
}

void Probe::setX(float x)
{
    m_coord[0] = x;
}

void Probe::setY(float y)
{
    m_coord[1] = y;
}

void Probe::setZ(float z)
{
    m_coord[2] = z;
}

float* Probe::getCoord()
{
    return m_coord;
}

float Probe::getX()
{
    return m_coord[0];
}

float Probe::getY()
{
    return m_coord[1];
}

float Probe::getZ()
{
    return m_coord[2];
}