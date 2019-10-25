#include "Uniformization/Model/Speaker.h"

using namespace adaptone;

Speaker::Speaker()
{
}

Speaker::Speaker(double x, double y, uint32_t id) : ModelElement(x, y, id)
{
}

Speaker::Speaker(double x, double y, double z, uint32_t id) : ModelElement(x, y, z, id)
{
}

Speaker::~Speaker()
{
}
