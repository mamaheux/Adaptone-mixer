#include <Uniformization/Model/Probe.h>

using namespace adaptone;

Probe::Probe()
{
}

Probe::Probe(double x, double y, uint32_t id) : ModelElement(x, y, id)
{
}

Probe::Probe(double x, double y, double z, uint32_t id) : ModelElement(x, y, z, id)
{
}

Probe::~Probe()
{
}
