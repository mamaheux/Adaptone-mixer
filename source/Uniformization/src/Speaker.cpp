#include "Uniformization/Speaker.h"

using namespace adaptone;

Speaker::Speaker()
{
}

Speaker::Speaker(float x, float y) : m_coord{x, y, 0}
{
}

Speaker::Speaker(float x, float y, float z) : m_coord{x, y, z}
{
}

Speaker::~Speaker()
{
}

void Speaker::setCoord(const float coord[3])
{
    m_coord[0] = coord[0];
    m_coord[1] = coord[1];
    m_coord[2] = coord[2];
}

void Speaker::setX(float x)
{
    m_coord[0] = x;
}

void Speaker::setY(float y)
{
    m_coord[1] = y;
}

void Speaker::setZ(float z)
{
    m_coord[2] = z;
}

float* Speaker::getCoord()
{
    return m_coord;
}

float Speaker::getX()
{
    return m_coord[0];
}

float Speaker::getY()
{
    return m_coord[1];
}

float Speaker::getZ()
{
    return m_coord[2];
}